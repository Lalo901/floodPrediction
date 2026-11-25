"""Metric area analytics, terrain zonal stats, S1-S2 lag checks, and analysis DB logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import box, shape

from core.db import DEFAULT_DB_PATH, PROJECT_ROOT, get_connection, init_db
from core.postproc import PostprocResult


def compute_s1_s2_lag_days(s1_date: str | None, s2_date: str | None) -> int | None:
    """Return absolute day difference between S1 and S2 acquisition dates."""
    if not s1_date or not s2_date:
        return None
    d1 = datetime.strptime(s1_date.replace("/", "-")[:10], "%Y-%m-%d")
    d2 = datetime.strptime(s2_date.replace("/", "-")[:10], "%Y-%m-%d")
    return abs((d1 - d2).days)


def check_config_b_eligibility(
    s2_available: bool,
    cloud_pct: float | None = None,
    s1_s2_lag_days: int | None = None,
    max_cloud_pct: float = 30.0,
    max_lag_days: int = 3,
) -> tuple[bool, str]:
    """Validate whether Config B (S1+S2+terrain) can be safely used for a scene."""
    if not s2_available:
        return False, "No hay escena Sentinel-2 disponible para esta zona/fecha. Usa Config A (SAR + Terreno)."
    if s1_s2_lag_days is not None and s1_s2_lag_days > max_lag_days:
        msg = (
            f"Desfase S1–S2 de {s1_s2_lag_days} días (máximo permitido: {max_lag_days}). "
            "Usa Config A para evitar inconsistencias temporales."
        )
        return False, msg
    if cloud_pct is not None and cloud_pct > max_cloud_pct:
        msg = (
            f"Cobertura de nubes en Sentinel-2 de {cloud_pct:.1f}% "
            f"(máximo permitido: {max_cloud_pct:.0f}%). Usa Config A (SAR penetra nubes)."
        )
        return False, msg
    return True, "Sentinel-2 útil y dentro de los límites de nubes y desfase temporal."


def compute_aoi_area_ha(
    bounds: tuple[float, float, float, float],
    crs: Any = "EPSG:4326",
    polygon_geojson: dict[str, Any] | None = None,
) -> float:
    """Compute exact area in hectares of a GeoJSON polygon or bounding box."""
    geom = shape(polygon_geojson) if polygon_geojson is not None else box(*bounds)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326" if polygon_geojson else crs)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    utm = gdf.estimate_utm_crs()
    return float(gdf.to_crs(utm).geometry.area.iloc[0] / 10_000.0)


def compute_zonal_stats(
    flood_mask: np.ndarray,
    terrain: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Compute summary stats of slope, HAND, and DEM_rel inside the flooded area."""
    if terrain is None or terrain.ndim != 3 or terrain.shape[0] < 3:
        return {}

    in_flood = flood_mask > 0
    if not np.any(in_flood):
        return {}

    names = ("slope_deg", "hand_m", "dem_rel_m")
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        vals = terrain[idx][in_flood]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        stats[name] = {
            "mean": round(float(np.mean(vals)), 3),
            "median": round(float(np.median(vals)), 3),
            "p90": round(float(np.percentile(vals, 90)), 3),
            "max": round(float(np.max(vals)), 3),
        }
    return stats


def compute_confidence_breakdown(
    flood_mask: np.ndarray,
    proba_map: np.ndarray,
    valid_mask: np.ndarray,
    flooded_ha: float,
    high_conf_thr: float = 0.80,
) -> dict[str, float]:
    """Split flooded hectares into High Confidence (P >= 0.80) vs Moderate Confidence."""
    in_flood = (flood_mask > 0) & valid_mask
    n_flood = int(np.count_nonzero(in_flood))
    n_valid = max(int(np.count_nonzero(valid_mask)), 1)

    if n_flood == 0:
        high_ratio = 0.0
    else:
        n_high = int(np.count_nonzero(in_flood & (proba_map >= high_conf_thr)))
        high_ratio = n_high / n_flood

    high_ha = flooded_ha * high_ratio
    mod_ha = flooded_ha * (1.0 - high_ratio)
    uncertain_px = int(np.count_nonzero(valid_mask & (proba_map >= 0.35) & (proba_map <= 0.65)))
    return {
        "high_conf_ha": round(high_ha, 2),
        "moderate_conf_ha": round(mod_ha, 2),
        "high_conf_pct": round(high_ratio * 100.0, 1),
        "uncertain_area_pct": round(100.0 * uncertain_px / n_valid, 2),
    }


def extract_top_hotspots(
    flood_gdf: gpd.GeoDataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return top-N largest flood polygons with WGS84 centroid coordinates and Google Maps link."""
    if len(flood_gdf) == 0:
        return []
    ordered = flood_gdf.sort_values("area_ha", ascending=False).head(top_n)
    utm_crs = ordered.estimate_utm_crs()
    centroids_wgs84 = ordered.to_crs(utm_crs).geometry.centroid.to_crs("EPSG:4326")

    hotspots: list[dict[str, Any]] = []
    for rank, ((_, row), pt) in enumerate(zip(ordered.iterrows(), centroids_wgs84, strict=True), start=1):
        lat = round(float(pt.y), 5)
        lon = round(float(pt.x), 5)
        hotspots.append(
            {
                "foco": f"#{rank}",
                "area_ha": round(float(row["area_ha"]), 2),
                "lat": lat,
                "lon": lon,
                "google_maps": f"https://www.google.com/maps?q={lat},{lon}",
            }
        )
    return hotspots


def compute_scene_reliability_badge(
    config_used: str,
    cloud_pct: float | None = None,
    lag_days: int | None = None,
    uncertain_area_pct: float = 0.0,
) -> dict[str, str]:
    """Classify operational confidence level for decision-makers."""
    if config_used == "B" and (cloud_pct is None or cloud_pct <= 15.0) and (lag_days is None or lag_days <= 2):
        return {
            "level": "high",
            "title": "Alta confiabilidad operativa (Config B: S1 + S2 despejado)",
            "detail": "Fusión SAR + óptica con baja nubosidad y desfase mínimo. Máxima precisión esperada.",
        }
    if uncertain_area_pct > 12.0:
        return {
            "level": "warning",
            "title": "Precaución: zona con incertidumbre moderada/alta",
            "detail": (
                f"El {uncertain_area_pct:.1f}% del área presenta probabilidades intermedias (0.35–0.65). "
                "Revisa las capas de Probabilidad y NDWI/MNDWI antes de tomar decisiones."
            ),
        }
    if config_used == "A":
        return {
            "level": "medium",
            "title": "Modo operativo todo-clima (Config A: SAR + Terreno)",
            "detail": (
                "Funciona bajo nubes y tormentas. En zonas áridas planas puede haber falsos positivos en suelo seco; "
                "validar con HAND y, cuando despeje, con Config B."
            ),
        }
    return {
        "level": "medium",
        "title": "Modo referencia (Solo SAR — ablación)",
        "detail": "Sin variables de terreno ni Sentinel-2; úsalo solo como referencia comparativa.",
    }


def summarize_analysis(
    post: PostprocResult,
    valid_mask: np.ndarray,
    bounds: tuple[float, float, float, float],
    crs: Any,
    terrain: np.ndarray | None = None,
    polygon_geojson: dict[str, Any] | None = None,
    proba_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """Produce structured metric area + zonal statistics + hotspots summary for UI and AI tools."""
    aoi_ha = compute_aoi_area_ha(bounds, crs=crs, polygon_geojson=polygon_geojson)
    flooded_ha = float(post.flood_gdf["area_ha"].sum()) if len(post.flood_gdf) > 0 else 0.0
    perm_water_ha = (
        float(post.perm_water_gdf["area_ha"].sum()) if len(post.perm_water_gdf) > 0 else 0.0
    )

    valid_px = int(np.count_nonzero(valid_mask))
    flood_px = int(np.count_nonzero(post.flood_mask > 0))
    flooded_pct = (100.0 * flood_px / valid_px) if valid_px > 0 else 0.0

    conf = (
        compute_confidence_breakdown(post.flood_mask, proba_map, valid_mask, flooded_ha)
        if proba_map is not None
        else {"high_conf_ha": 0.0, "moderate_conf_ha": 0.0, "high_conf_pct": 0.0, "uncertain_area_pct": 0.0}
    )

    return {
        "aoi_ha": round(aoi_ha, 2),
        "flooded_ha": round(flooded_ha, 2),
        "perm_water_ha": round(perm_water_ha, 2),
        "total_water_ha": round(flooded_ha + perm_water_ha, 2),
        "flood_polygon_count": len(post.flood_gdf),
        "largest_polygon_ha": round(float(post.flood_gdf["area_ha"].max()), 2)
        if len(post.flood_gdf) > 0
        else 0.0,
        "flooded_pct_of_valid": round(flooded_pct, 2),
        "valid_pixel_pct": round(100.0 * valid_px / max(valid_mask.size, 1), 2),
        "confidence": conf,
        "hotspots": extract_top_hotspots(post.flood_gdf, top_n=5),
        "zonal_stats": compute_zonal_stats(post.flood_mask, terrain),
    }


def _to_rel_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def save_analysis_record(
    *,
    polygon_geojson: dict[str, Any] | str,
    model_id: int | None,
    flooded_ha: float,
    perm_water_ha: float,
    event_id: int | None = None,
    s1_date: str | None = None,
    s2_date: str | None = None,
    cloud_pct: float | None = None,
    s1_s2_lag_days: int | None = None,
    live_iou: float | None = None,
    raster_path: Path | str | None = None,
    vector_path: Path | str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Persist an analysis execution into SQLite `analyses` table and return its ID."""
    init_db(db_path)
    poly_str = (
        polygon_geojson if isinstance(polygon_geojson, str) else json.dumps(polygon_geojson)
    )
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO analyses (
                event_id, polygon_geojson, s1_date, s2_date, cloud_pct, s1_s2_lag_days,
                model_id, flooded_ha, perm_water_ha, live_iou, raster_path, vector_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                poly_str,
                s1_date,
                s2_date,
                cloud_pct,
                s1_s2_lag_days,
                model_id,
                flooded_ha,
                perm_water_ha,
                live_iou,
                _to_rel_path(raster_path),
                _to_rel_path(vector_path),
            ),
        )
        return int(cur.lastrowid)
