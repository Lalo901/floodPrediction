"""Whitelisted, read-only analytical tools for the Flood Mapper AI assistant.

No arbitrary code execution (`eval`, `exec`, shell, or filesystem browsing).
Every tool invocation is validated, bounded in size, and logged into SQLite `tool_calls`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.security import MAX_TOOL_RESULT_CHARS
from core.analytics import compute_scene_reliability_badge, extract_top_hotspots
from core.db import DEFAULT_DB_PATH, get_connection, init_db
from core.registry import list_models


@dataclass
class AnalysisContext:
    """Decoupled snapshot of the active analysis passed from any UI/CLI into the AI tools."""

    analysis_id: int | None
    scene_id: str
    event_name: str | None
    source: str
    s1_date: str | None
    s2_date: str | None
    orbit: str | None
    cloud_pct: float | None
    lag_days: int | None
    model_name: str
    model_config: str
    elapsed_s: float
    summary: dict[str, Any]
    live_metrics: dict[str, Any] | None
    flood_gdf: Any  # geopandas.GeoDataFrame
    export_files: dict[str, str]
    comparison: dict[str, Any] | None = None
    db_path: Path | str = DEFAULT_DB_PATH


@dataclass
class ToolSpec:
    """Schema + callable pair registered inside `ToolRegistry`."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def log_tool_call(
    *,
    analysis_id: int | None,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Persist a tool invocation record into SQLite `tool_calls` using parameterized SQL."""
    init_db(db_path)
    args_str = json.dumps(args, ensure_ascii=False)[:1000]
    res_str = json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO tool_calls (analysis_id, tool_name, args_json, result_summary)
            VALUES (?, ?, ?, ?);
            """,
            (analysis_id, tool_name, args_str, res_str),
        )
        return int(cur.lastrowid)


def list_tool_calls(
    analysis_id: int | None = None,
    limit: int = 30,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve recent tool call audit logs from SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if analysis_id is not None:
            rows = conn.execute(
                """
                SELECT id, analysis_id, tool_name, args_json, result_summary, created_at
                FROM tool_calls
                WHERE analysis_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (analysis_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, analysis_id, tool_name, args_json, result_summary, created_at
                FROM tool_calls
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


class ToolRegistry:
    """Extensible registry of strictly whitelisted tools."""

    def __init__(self, *, analysis_id: int | None = None, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.analysis_id = analysis_id
        self.db_path = db_path

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_schema() for spec in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a whitelisted tool, enforce safe errors, and log the call to SQLite."""
        args = arguments if isinstance(arguments, dict) else {}
        if name not in self._tools:
            result = {"error": f"Herramienta no permitida o desconocida: `{name}`."}
        else:
            try:
                result = self._tools[name].handler(args)
            except Exception as exc:  # noqa: BLE001
                result = {"error": f"Error ejecutando `{name}`: {exc}"}

        log_tool_call(
            analysis_id=self.analysis_id,
            tool_name=name,
            args=args,
            result=result,
            db_path=self.db_path,
        )
        return result


def build_default_tool_registry(ctx: AnalysisContext) -> ToolRegistry:
    """Construct the standard Flood Mapper MVP tool suite bound to `ctx`."""
    reg = ToolRegistry(analysis_id=ctx.analysis_id, db_path=ctx.db_path)

    def _get_summary(_args: dict[str, Any]) -> dict[str, Any]:
        conf = ctx.summary.get("confidence", {})
        badge = compute_scene_reliability_badge(
            config_used=ctx.model_config,
            cloud_pct=ctx.cloud_pct,
            lag_days=ctx.lag_days,
            uncertain_area_pct=float(conf.get("uncertain_area_pct", 0.0)),
        )
        return {
            "scene_id": ctx.scene_id,
            "event_name": ctx.event_name,
            "source": ctx.source,
            "s1_date": ctx.s1_date,
            "s2_date": ctx.s2_date,
            "orbit": ctx.orbit,
            "cloud_pct": ctx.cloud_pct,
            "s1_s2_lag_days": ctx.lag_days,
            "model_name": ctx.model_name,
            "model_config": ctx.model_config,
            "elapsed_s": ctx.elapsed_s,
            "aoi_ha": ctx.summary.get("aoi_ha"),
            "flooded_ha": ctx.summary.get("flooded_ha"),
            "perm_water_ha": ctx.summary.get("perm_water_ha"),
            "total_water_ha": ctx.summary.get("total_water_ha"),
            "flooded_pct_of_valid": ctx.summary.get("flooded_pct_of_valid"),
            "flood_polygon_count": ctx.summary.get("flood_polygon_count"),
            "confidence": conf,
            "reliability_badge": badge,
            "live_validation_vs_ground_truth": (
                {
                    "flood_iou": round(float(ctx.live_metrics["iou"]), 4),
                    "f1": round(float(ctx.live_metrics["f1"]), 4),
                    "precision": round(float(ctx.live_metrics["precision"]), 4),
                    "recall": round(float(ctx.live_metrics["recall"]), 4),
                }
                if ctx.live_metrics
                else None
            ),
        }

    def _get_top_hotspots(args: dict[str, Any]) -> dict[str, Any]:
        raw_n = args.get("top_n", 5)
        try:
            top_n = max(1, min(int(raw_n), 20))
        except (TypeError, ValueError):
            top_n = 5
        hotspots = extract_top_hotspots(ctx.flood_gdf, top_n=top_n)
        return {"top_n": top_n, "total_polygons": len(ctx.flood_gdf), "hotspots": hotspots}

    def _get_zonal_stats(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "zonal_stats_inside_flood": ctx.summary.get("zonal_stats", {}),
            "note": "Estadísticas de Slope (grados), HAND (metros sobre drenaje) y DEM_rel (m) dentro de la inundación detectada.",
        }

    def _filter_by_min_area(args: dict[str, Any]) -> dict[str, Any]:
        try:
            min_ha = float(args.get("min_area_ha", 1.0))
        except (TypeError, ValueError):
            raise ValueError("`min_area_ha` debe ser un número válido.") from None
        min_ha = max(0.0, min(min_ha, 1000.0))

        gdf = ctx.flood_gdf
        if len(gdf) == 0:
            return {"min_area_ha": min_ha, "remaining_polygons": 0, "remaining_ha": 0.0, "removed_ha": 0.0}

        kept = gdf[gdf["area_ha"] >= min_ha]
        total_ha = float(gdf["area_ha"].sum())
        kept_ha = float(kept["area_ha"].sum()) if len(kept) > 0 else 0.0
        return {
            "min_area_ha": min_ha,
            "original_polygons": len(gdf),
            "remaining_polygons": len(kept),
            "remaining_ha": round(kept_ha, 2),
            "removed_ha": round(total_ha - kept_ha, 2),
            "top_remaining_hotspots": extract_top_hotspots(kept, top_n=3),
        }

    def _compare_models(_args: dict[str, Any]) -> dict[str, Any]:
        if ctx.comparison is not None:
            return {"mode": "same_scene_comparison", "comparison": ctx.comparison}
        models = list_models(db_path=ctx.db_path)
        compact = [
            {
                "name": m["name"],
                "config": m["config"],
                "test_iou": round(float(m["test_iou"]), 4) if m.get("test_iou") is not None else None,
                "bolivia_iou": round(float(m["bolivia_iou"]), 4) if m.get("bolivia_iou") is not None else None,
                "size_mb": m.get("size_mb"),
            }
            for m in models
        ]
        return {
            "mode": "benchmark_registry_comparison",
            "note": "No se activó comparación de 2 modelos en esta escena; se muestran métricas globales de benchmark en SQLite.",
            "active_model": ctx.model_name,
            "registered_models": compact,
        }

    def _list_registered_models(args: dict[str, Any]) -> dict[str, Any]:
        cfg = args.get("config")
        if cfg not in (None, "A", "B", "S1_only"):
            cfg = None
        models = list_models(config=cfg, db_path=ctx.db_path)
        return {
            "count": len(models),
            "otsu_baseline": {"valid_iou": 0.5056, "test_iou": 0.5458, "bolivia_iou": 0.6972},
            "models": [
                {
                    "name": m["name"],
                    "family": m["family"],
                    "config": m["config"],
                    "threshold": m["threshold"],
                    "test_iou": m["test_iou"],
                    "test_f1": m["test_f1"],
                    "bolivia_iou": m["bolivia_iou"],
                    "size_mb": m["size_mb"],
                }
                for m in models
            ],
        }

    def _get_export_info(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "available_exports": ctx.export_files,
            "formats_description": {
                "geotiff": "Raster 2 bandas float32 (flood_mask 0/1, water_proba 0..1)",
                "geojson": "Polígonos EPSG:4326 con atributo area_ha (calculado en UTM)",
                "geopackage": "Capa vectorial estándar SIG (.gpkg) con area_ha",
            },
        }

    reg.register(
        ToolSpec(
            name="get_analysis_summary",
            description="Obtiene el resumen completo del análisis actual: hectáreas inundadas, agua permanente, confianza alta/moderada, metadatos S1/S2, semáforo de confiabilidad y Flood IoU en vivo si existe etiqueta.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_summary,
        )
    )
    reg.register(
        ToolSpec(
            name="get_top_hotspots",
            description="Devuelve los N focos/polígonos más grandes de inundación detectada con área en hectáreas, latitud, longitud y enlace a Google Maps.",
            parameters={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Cantidad de focos críticos a devolver (entre 1 y 20, por defecto 5).",
                    }
                },
                "required": [],
            },
            handler=_get_top_hotspots,
        )
    )
    reg.register(
        ToolSpec(
            name="get_zonal_stats",
            description="Obtiene estadísticas zonales de terreno (pendiente en grados, HAND en metros y DEM relativo) únicamente dentro de la zona inundada.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_zonal_stats,
        )
    )
    reg.register(
        ToolSpec(
            name="filter_by_min_area",
            description="Simula filtrar los polígonos de inundación actuales por un nuevo umbral mínimo en hectáreas (ej. 1.0 ha o 5.0 ha) y reporta cuántos polígonos y hectáreas permanecen.",
            parameters={
                "type": "object",
                "properties": {
                    "min_area_ha": {
                        "type": "number",
                        "description": "Área mínima en hectáreas (entre 0 y 1000).",
                    }
                },
                "required": ["min_area_ha"],
            },
            handler=_filter_by_min_area,
        )
    )
    reg.register(
        ToolSpec(
            name="compare_models",
            description="Compara el modelo principal contra el segundo modelo si se ejecutó comparación en esta escena, o contra el catálogo de modelos registrados en SQLite.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_compare_models,
        )
    )
    reg.register(
        ToolSpec(
            name="list_registered_models",
            description="Lista los modelos entrenados registrados en SQLite con sus métricas oficiales (Test IoU, F1, Bolivia IoU, tamaño MB) y el baseline Otsu.",
            parameters={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "string",
                        "enum": ["A", "B", "S1_only"],
                        "description": "Filtro opcional por configuración de entrada.",
                    }
                },
                "required": [],
            },
            handler=_list_registered_models,
        )
    )
    reg.register(
        ToolSpec(
            name="get_export_info",
            description="Devuelve los formatos y nombres de archivos exportados disponibles para el análisis actual (GeoTIFF, GeoJSON, GeoPackage).",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_get_export_info,
        )
    )
    return reg
