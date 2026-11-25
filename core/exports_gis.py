"""GIS interoperability exports: QGIS (.qml), MapLibre/GeoLibre JSON, UTM safety buffer, and Markdown report.

Zero Streamlit dependencies so exports can be reused from CLI, API, or plugins.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import geopandas as gpd


def build_qgis_qml_style(
    *,
    fill_rgba: str = "0,180,255,166",
    stroke_rgba: str = "0,119,182,255",
    stroke_width_mm: float = 0.4,
) -> str:
    """Generate a QGIS 3.x QML style file matching the Copernicus EMS flood cyan palette."""
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">
    <symbols>
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option type="QString" name="color" value="{fill_rgba}"/>
            <Option type="QString" name="outline_color" value="{stroke_rgba}"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="{stroke_width_mm}"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="style" value="solid"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
</qgis>
"""


def build_maplibre_style_layer(
    geojson_filename: str,
    bounds_wgs84: tuple[float, float, float, float],
    *,
    source_id: str = "flood-mask-source",
    layer_prefix: str = "flood-detected",
) -> str:
    """Generate a MapLibre GL JS / GeoLibre JSON style snippet for the exported GeoJSON."""
    west, south, east, north = bounds_wgs84
    center = [round((west + east) / 2.0, 6), round((south + north) / 2.0, 6)]
    payload = {
        "schema": "maplibre-gl-style-snippet/v1",
        "center": center,
        "bounds": [round(west, 6), round(south, 6), round(east, 6), round(north, 6)],
        "sources": {
            source_id: {
                "type": "geojson",
                "data": f"./{geojson_filename}",
            }
        },
        "layers": [
            {
                "id": f"{layer_prefix}-fill",
                "type": "fill",
                "source": source_id,
                "paint": {
                    "fill-color": "#00B4FF",
                    "fill-opacity": 0.55,
                },
            },
            {
                "id": f"{layer_prefix}-outline",
                "type": "line",
                "source": source_id,
                "paint": {
                    "line-color": "#0077B6",
                    "line-width": 1.5,
                },
            },
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_safety_buffer_geojson(
    flood_gdf: gpd.GeoDataFrame,
    buffer_m: float = 100.0,
) -> tuple[bytes, float]:
    """Compute a metric UTM safety buffer around detected flood polygons and return (geojson_bytes, area_ha)."""
    if buffer_m <= 0 or buffer_m > 5000:
        raise ValueError("El radio de buffer debe estar entre 1 m y 5000 m.")

    if flood_gdf is None or len(flood_gdf) == 0:
        empty = json.dumps({"type": "FeatureCollection", "features": []}).encode("utf-8")
        return empty, 0.0

    gdf = flood_gdf if flood_gdf.crs is not None else flood_gdf.set_crs("EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    buffered_geom = gdf_utm.geometry.buffer(float(buffer_m))
    dissolved = gpd.GeoDataFrame(geometry=buffered_geom, crs=utm_crs).dissolve()
    area_ha = round(float(dissolved.geometry.area.sum()) / 10_000.0, 2)
    dissolved["buffer_m"] = float(buffer_m)
    dissolved["area_ha"] = area_ha

    out_wgs84 = dissolved.to_crs("EPSG:4326")
    return out_wgs84.to_json().encode("utf-8"), area_ha


def build_executive_report_md(
    res: dict[str, Any],
    *,
    res2: dict[str, Any] | None = None,
    badge: dict[str, str] | None = None,
) -> str:
    """Generate a downloadable Markdown operational report for emergency response teams."""
    sc = res["scene"]
    m_row = res["model_row"]
    summary = res["summary"]
    conf = summary["confidence"]
    live_m = res.get("live_metrics")
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    cloud_txt = f"{sc.cloud_pct:.1f}%" if sc.cloud_pct is not None else "N/D"
    lag_val = res.get("lag_days")
    lag_txt = f"{lag_val} d" if lag_val is not None else "N/D"

    lines = [
        f"# Reporte Operativo de Detección de Agua Superficial — {sc.scene_id}",
        "",
        f"- **Generado:** {now_utc}",
        f"- **Evento / Origen:** {sc.event_name} (`{sc.source}`)",
        f"- **Modelo principal (M1):** `{m_row['name']}` (Config `{m_row['config']}`)",
        f"- **Fechas de captura:** Sentinel-1 `{sc.s1_date or 'N/D'}` ({sc.orbit or 'N/D'}) | Sentinel-2 `{sc.s2_date or 'No disponible'}`",
        f"- **Nubosidad S2:** {cloud_txt} | **Desfase S1-S2:** {lag_txt}",
        "",
    ]

    if badge:
        lines.extend(
            [
                "## Confiabilidad Operativa",
                f"**{badge['title']}** — {badge['detail']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Indicadores Clave (CRS Métrico UTM)",
            f"- **Área inundada detectada (nueva):** {summary['flooded_ha']:,.2f} ha ({summary['flooded_pct_of_valid']:.2f}% del área válida)",
            f"- **Alta confianza (P >= 0.80):** {conf['high_conf_ha']:,.2f} ha ({conf['high_conf_pct']:.1f}%)",
            f"- **Confianza moderada:** {conf['moderate_conf_ha']:,.2f} ha",
            f"- **Agua permanente descontada (JRC + MERIT):** {summary['perm_water_ha']:,.2f} ha",
            f"- **Polígonos / Focos detectados:** {summary['flood_polygon_count']:,} (mayor: {summary['largest_polygon_ha']:,.2f} ha)",
            f"- **Área total de interés (AOI):** {summary['aoi_ha']:,.2f} ha",
            "",
        ]
    )

    if live_m:
        lines.extend(
            [
                "## Evaluación contra Ground Truth (LabelHand, ignorando -1)",
                f"- **Flood IoU:** `{live_m['iou']:.4f}` | **F1:** `{live_m['f1']:.4f}` | **Precision:** `{live_m['precision']:.4f}` | **Recall:** `{live_m['recall']:.4f}`",
                "",
            ]
        )

    if res2 is not None:
        m2_row = res2["model_row"]
        sum2 = res2["summary"]
        delta_ha = sum2["flooded_ha"] - summary["flooded_ha"]
        lines.extend(
            [
                "## Comparación con Modelo Secundario (M2)",
                f"- **Modelo 2:** `{m2_row['name']}` (Config `{m2_row['config']}`)",
                f"- **Área inundada M2:** {sum2['flooded_ha']:,.2f} ha (Delta M2 - M1: `{delta_ha:+.2f} ha`)",
                "",
            ]
        )

    hotspots = summary.get("hotspots", [])
    if hotspots:
        lines.extend(
            [
                "## Top 5 Focos Críticos (Priorización en Terreno)",
                "| Foco | Área (ha) | Latitud | Longitud | Ubicación |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for hs in hotspots:
            lines.append(
                f"| {hs['foco']} | {hs['area_ha']:.2f} | {hs['lat']:.5f} | {hs['lon']:.5f} | {hs['google_maps']} |"
            )
        lines.append("")

    zstats = summary.get("zonal_stats", {})
    if zstats:
        lines.extend(
            [
                "## Estadísticas Zonales de Terreno (dentro de la inundación)",
                "| Variable | Mediana | Media | P90 |",
                "|---|---:|---:|---:|",
            ]
        )
        for var_name, vals in zstats.items():
            lines.append(
                f"| {var_name} | {vals.get('median', 'N/D')} | {vals.get('mean', 'N/D')} | {vals.get('p90', 'N/D')} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "**Nota científica:** Este reporte refleja únicamente *agua superficial presente* en la fecha de captura satelital. No constituye predicción de riesgo futuro ni reemplaza estudios hidrológicos o de caudal.",
        ]
    )
    return "\n".join(lines) + "\n"
