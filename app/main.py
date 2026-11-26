"""Flood Mapper MVP — Streamlit application (Offline Sen1Floods11 + Upload GeoTIFF + Live GEE).

Map-first, icon-free desktop-GIS layout with:
  - Compact top menu bar (Mapa, Modelos, Historial, Datos, Herramientas, Complementos, Ayuda, Asistente IA).
  - Always-visible interactive base map on startup with Satellite / Streets / Relief / Dark switcher
    and Country/City/Coordinates geocoding search bar.
  - Slim toggleable left Data dock (1-3) and right scrollable AI Assistant dock.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape as shapely_shape
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.assistant_panel import render_assistant_dock
from app.layers import (
    FLOOD_WATER_RGB,
    MODEL2_WATER_RGB,
    PERM_WATER_RGB,
    render_comparison_rgba,
    render_error_map_rgba,
    render_hand_rgba,
    render_low_hand_rgba,
    render_mask_rgba,
    render_probability_rgba,
    render_s1_rgba,
    render_s2_rgb_rgba,
    render_slope_rgba,
    render_water_index_rgba,
    rgba_to_data_url,
)
from app.plugins_panel import render_plugins_popover_content, render_tools_popover_content
from core.analytics import (
    check_config_b_eligibility,
    compute_s1_s2_lag_days,
    compute_scene_reliability_badge,
    save_analysis_record,
    summarize_analysis,
)
from core.data import (
    SceneBundle,
    SlidingWindowRateLimiter,
    fetch_gee_scene,
    gee_is_configured,
    list_gee_s1_scenes,
    list_offline_chips,
    load_offline_scene,
    load_uploaded_geotiff,
    save_scene_as_demo_stack,
)
from core.db import (
    DEFAULT_DB_PATH,
    DEFAULT_METADATA_GEOJSON,
    get_connection,
    init_db,
    list_events,
    seed_events_from_geojson,
)
from core.geocoding import geocode_place
from core.metrics import compute_binary_metrics
from core.postproc import export_raster_geotiff, export_vector, run_postprocessing
from core.registry import list_models, load_model

st.set_page_config(
    page_title="Flood Mapper MVP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_MODE = os.getenv("APP_MODE", "local")
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
MAX_STORED_OUTPUT_FILES = 60  # ~20 analyses x 3 files to protect ephemeral Space disk

BASEMAP_SPECS: dict[str, dict[str, Any]] = {
    "Satélite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
        "label": "Satélite (Esri)",
        "max_zoom": 18,
    },
    "Calles": {
        "tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap contributors",
        "label": "Calles (OSM)",
        "max_zoom": 19,
    },
    "Relieve": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, National Geographic, Garmin, HERE, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, increment P Corp.",
        "label": "Relieve (Esri NatGeo)",
        "max_zoom": 16,
    },
    "Oscuro": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "reference_tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri Dark Gray Canvas",
        "label": "Oscuro (Esri Dark)",
        "max_zoom": 16,
    },
}

_LAYER_CONTROL_CSS = """
<style>
.leaflet-control-layers {
    background: rgba(18, 22, 29, 0.92) !important;
    color: #E5E7EB !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.45) !important;
}
.leaflet-control-layers-expanded {
    width: 215px !important;
    max-width: 215px !important;
    max-height: 215px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 6px 8px !important;
    font-size: 11.5px !important;
    line-height: 1.35 !important;
    scrollbar-width: thin;
    scrollbar-color: #64748B #111827;
}
.leaflet-control-layers-expanded::-webkit-scrollbar {
    width: 6px;
}
.leaflet-control-layers-expanded::-webkit-scrollbar-track {
    background: #111827;
}
.leaflet-control-layers-expanded::-webkit-scrollbar-thumb {
    background: #64748B;
    border-radius: 3px;
}
.leaflet-control-layers label {
    color: #E5E7EB !important;
    margin-bottom: 3px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.leaflet-control-layers-separator {
    border-top: 1px solid #374151 !important;
    margin: 4px 0 !important;
}
</style>
"""


@st.cache_resource
def _init_database() -> None:
    init_db(DEFAULT_DB_PATH)
    if not list_events(DEFAULT_DB_PATH) and DEFAULT_METADATA_GEOJSON.exists():
        seed_events_from_geojson(DEFAULT_METADATA_GEOJSON, DEFAULT_DB_PATH)


@st.cache_resource
def _cached_load_model(model_name: str):
    return load_model(model_name, DEFAULT_DB_PATH)


@st.cache_resource(max_entries=8)
def _cached_load_offline_scene(stem: str) -> SceneBundle:
    return load_offline_scene(stem, DEFAULT_DB_PATH)


@st.cache_resource
def _get_gee_rate_limiter() -> SlidingWindowRateLimiter:
    max_calls = int(os.getenv("GEE_MAX_DOWNLOADS_PER_10MIN", "5"))
    return SlidingWindowRateLimiter(max_calls=max_calls, window_seconds=600.0)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_geocode(query: str) -> list[dict[str, Any]]:
    return geocode_place(query, limit=5)


def _prune_old_outputs(max_files: int = MAX_STORED_OUTPUT_FILES) -> None:
    if not OUTPUTS_DIR.exists():
        return
    files = sorted(OUTPUTS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[max_files:]:
        if old.is_file():
            old.unlink(missing_ok=True)


_UNIFORM_POPOVER_CSS = """
<style>
[data-testid="stPopoverBody"] {
    width: 390px !important;
    min-width: 390px !important;
    max-width: 390px !important;
    max-height: 460px !important;
    overflow-y: auto !important;
}
</style>
"""


def _safe_output_file(raw_path: str | None) -> Path | None:
    """Resolve a DB-stored export path and ensure it strictly resides inside `data/outputs/`."""
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve()
        if resolved.is_relative_to(OUTPUTS_DIR.resolve()) and resolved.is_file():
            return resolved
    except (OSError, ValueError):
        return None
    return None


def _list_recent_analyses(limit: int = 20) -> list[dict[str, Any]]:
    with get_connection(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.created_at, e.name AS event, a.s1_date, a.s2_date, a.cloud_pct,
                   m.name AS model_name, m.config, a.flooded_ha, a.perm_water_ha, a.live_iou,
                   a.raster_path, a.vector_path
            FROM analyses a
            LEFT JOIN models m ON a.model_id = m.id
            LEFT JOIN events e ON a.event_id = e.id
            ORDER BY a.id DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def _add_basemap_layers(m: folium.Map, active_basemap: str = "Satélite") -> None:
    """Attach the active basemap (switched via toolbar) and inject compact scrollable LayerControl CSS."""
    active = BASEMAP_SPECS.get(active_basemap, BASEMAP_SPECS["Satélite"])
    folium.TileLayer(
        tiles=active["tiles"],
        attr=active["attr"],
        name=active["label"],
        max_zoom=int(active.get("max_zoom", 18)),
        overlay=False,
        control=False,
        show=True,
    ).add_to(m)
    if active.get("reference_tiles"):
        folium.TileLayer(
            tiles=active["reference_tiles"],
            attr=active["attr"],
            name=f"{active['label']} Labels",
            max_zoom=int(active.get("max_zoom", 16)),
            overlay=True,
            control=False,
            show=True,
        ).add_to(m)
    m.get_root().header.add_child(folium.Element(_LAYER_CONTROL_CSS))


def _build_base_explorer_map(
    *,
    active_basemap: str,
    enable_draw: bool = False,
    preview_scene: SceneBundle | None = None,
    geo_target: dict[str, Any] | None = None,
) -> folium.Map:
    """Always-visible interactive base map (supports Satellite toggle, search jump, and GEE Draw)."""
    center = [-14.8, -64.9]
    zoom = 5

    if geo_target is not None:
        center = [geo_target["lat"], geo_target["lon"]]
        zoom = int(geo_target.get("zoom", 11))
    elif preview_scene is not None and not enable_draw:
        west, south, east, north = preview_scene.bounds_wgs84
        center = [(south + north) / 2.0, (west + east) / 2.0]
        zoom = 12

    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    _add_basemap_layers(m, active_basemap=active_basemap)

    if geo_target is not None:
        if geo_target.get("bounds"):
            m.fit_bounds(geo_target["bounds"])
        folium.CircleMarker(
            location=[geo_target["lat"], geo_target["lon"]],
            radius=6,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color="#00B4FF",
            fill_opacity=0.9,
            tooltip=geo_target["display_name"],
        ).add_to(m)
    elif preview_scene is not None and not enable_draw:
        west, south, east, north = preview_scene.bounds_wgs84
        bounds = [[south, west], [north, east]]
        folium.Rectangle(
            bounds=bounds,
            color="#00B4FF",
            weight=2,
            dash_array="6",
            fill=True,
            fill_color="#00B4FF",
            fill_opacity=0.12,
            tooltip=f"Escena seleccionada: {preview_scene.scene_id} ({preview_scene.event_name})",
        ).add_to(m)
        m.fit_bounds(bounds)

    if enable_draw:
        Draw(
            export=False,
            draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False},
        ).add_to(m)

    return m


def _get_cached_result_overlays(
    res: dict[str, Any],
    res2: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Pre-encode raster PNG data URLs once per `(analysis_id, analysis_id_2)` for <20ms map reruns."""
    cache: dict[tuple[Any, Any], dict[str, str]] = st.session_state.setdefault("_overlay_cache", {})
    key = (res.get("analysis_id"), res2.get("analysis_id") if res2 else None)
    if key in cache:
        return cache[key]
    if len(cache) >= 4:
        cache.clear()

    scene: SceneBundle = res["scene"]
    urls: dict[str, str] = {
        "s1": rgba_to_data_url(render_s1_rgba(scene.s1)),
        "flood1": rgba_to_data_url(
            render_mask_rgba(res["post"].flood_mask, rgb=FLOOD_WATER_RGB, alpha=215)
        ),
        "proba": rgba_to_data_url(render_probability_rgba(res["proba"], res["valid_mask"])),
    }
    if scene.s2 is not None:
        urls["s2"] = rgba_to_data_url(render_s2_rgb_rgba(scene.s2))
        urls["mndwi"] = rgba_to_data_url(render_water_index_rgba(scene.s2, index="mndwi"))
        urls["ndwi"] = rgba_to_data_url(render_water_index_rgba(scene.s2, index="ndwi"))
    if scene.terrain is not None:
        urls["hand"] = rgba_to_data_url(render_hand_rgba(scene.terrain))
        urls["low_hand"] = rgba_to_data_url(render_low_hand_rgba(scene.terrain, threshold_m=2.0))
        urls["slope"] = rgba_to_data_url(render_slope_rgba(scene.terrain))

    perm_mask = res["post"].perm_water_mask
    if perm_mask is not None and perm_mask.any():
        urls["perm"] = rgba_to_data_url(render_mask_rgba(perm_mask, rgb=PERM_WATER_RGB, alpha=215))

    if res2 is not None:
        urls["flood2"] = rgba_to_data_url(
            render_mask_rgba(res2["post"].flood_mask, rgb=MODEL2_WATER_RGB, alpha=205)
        )
        urls["comp"] = rgba_to_data_url(
            render_comparison_rgba(res["post"].flood_mask, res2["post"].flood_mask, res["valid_mask"])
        )

    if scene.label is not None:
        gt_bin = (scene.label == 1).astype("uint8")
        urls["gt"] = rgba_to_data_url(render_mask_rgba(gt_bin, rgb=(34, 197, 94), alpha=155))
        urls["err"] = rgba_to_data_url(
            render_error_map_rgba(scene.label, res["post"].raw_water_mask, res["valid_mask"])
        )

    cache[key] = urls
    return urls


def _build_result_map(
    res: dict[str, Any],
    *,
    active_basemap: str,
    res2: dict[str, Any] | None = None,
    geo_target: dict[str, Any] | None = None,
    mask_opacity: float = 0.85,
    focused_hotspot: dict[str, Any] | None = None,
) -> folium.Map:
    scene: SceneBundle = res["scene"]
    west, south, east, north = scene.bounds_wgs84
    center = [(south + north) / 2.0, (west + east) / 2.0]
    zoom_init = 12
    if focused_hotspot is not None:
        center = [focused_hotspot["lat"], focused_hotspot["lon"]]
        zoom_init = 15

    m = folium.Map(location=center, zoom_start=zoom_init, tiles=None, control_scale=True)
    _add_basemap_layers(m, active_basemap=active_basemap)

    img_bounds = [[south, west], [north, east]]
    urls = _get_cached_result_overlays(res, res2=res2)
    op = max(0.30, min(float(mask_opacity), 1.0))

    folium.raster_layers.ImageOverlay(
        image=urls["s1"],
        bounds=img_bounds,
        name="S1 SAR (Falso Color)",
        opacity=0.80,
        show=True,
    ).add_to(m)

    if "s2" in urls:
        folium.raster_layers.ImageOverlay(
            image=urls["s2"], bounds=img_bounds, name="S2 RGB Natural", opacity=0.85, show=False
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=urls["mndwi"], bounds=img_bounds, name="MNDWI (Agua S2)", opacity=0.80, show=False
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=urls["ndwi"], bounds=img_bounds, name="NDWI (Agua S2)", opacity=0.80, show=False
        ).add_to(m)

    if "hand" in urls:
        folium.raster_layers.ImageOverlay(
            image=urls["hand"], bounds=img_bounds, name="HAND (m)", opacity=0.70, show=False
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=urls["low_hand"], bounds=img_bounds, name="Zonas bajas (HAND < 2m)", opacity=0.65, show=False
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=urls["slope"], bounds=img_bounds, name="Pendiente (Slope °)", opacity=0.70, show=False
        ).add_to(m)

    if "perm" in urls:
        folium.raster_layers.ImageOverlay(
            image=urls["perm"], bounds=img_bounds, name="Agua permanente (Azul)", opacity=op, show=True
        ).add_to(m)

    m1_name = res["model_row"]["name"]
    folium.raster_layers.ImageOverlay(
        image=urls["flood1"],
        bounds=img_bounds,
        name=f"Inundación {m1_name} (Cian)",
        opacity=op,
        show=(res2 is None),
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=urls["proba"],
        bounds=img_bounds,
        name=f"Probabilidad {m1_name}",
        opacity=op,
        show=False,
    ).add_to(m)

    if res2 is not None:
        m2_name = res2["model_row"]["name"]
        folium.raster_layers.ImageOverlay(
            image=urls["flood2"],
            bounds=img_bounds,
            name=f"Inundación {m2_name} (Violeta)",
            opacity=op,
            show=False,
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=urls["comp"],
            bounds=img_bounds,
            name="Comparación M1 vs M2",
            opacity=op,
            show=True,
        ).add_to(m)

    if "gt" in urls:
        folium.raster_layers.ImageOverlay(
            image=urls["gt"], bounds=img_bounds, name="Ground Truth (Verde)", opacity=0.75, show=False
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=urls["err"], bounds=img_bounds, name="Errores (TP/FP/FN)", opacity=op, show=False
        ).add_to(m)

    for hs in res["summary"].get("hotspots", [])[:5]:
        is_focused = focused_hotspot is not None and hs["foco"] == focused_hotspot["foco"]
        folium.CircleMarker(
            location=[hs["lat"], hs["lon"]],
            radius=10 if is_focused else 6,
            color="#FFFF00" if is_focused else "#FFFFFF",
            weight=3 if is_focused else 2,
            fill=True,
            fill_color="#FF5722",
            fill_opacity=0.95,
            popup=f"Foco {hs['foco']}: {hs['area_ha']} ha ({hs['lat']}, {hs['lon']})",
            tooltip=f"Foco crítico {hs['foco']} ({hs['area_ha']} ha)",
        ).add_to(m)

    if focused_hotspot is not None:
        pass  # keep exact zoom_init=15 centered on focused hotspot
    elif geo_target is not None and geo_target.get("bounds"):
        m.fit_bounds(geo_target["bounds"], padding=(20, 20))
    elif geo_target is not None:
        m.location = [geo_target["lat"], geo_target["lon"]]
    else:
        m.fit_bounds(img_bounds, padding=(24, 24))

    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _run_analysis_pipeline(
    scene: SceneBundle,
    model_row: dict[str, Any],
    median_size: int,
    min_area_ha: float,
    remove_perm_water: bool,
    *,
    save_to_db: bool = True,
    existing_analysis_id: int | None = None,
    existing_tif: Path | None = None,
    existing_gpkg: Path | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    model = _cached_load_model(model_row["name"])
    proba, raw_mask, valid_mask = model.predict(
        scene.s1,
        terrain=scene.terrain,
        s2=scene.s2,
    )
    post = run_postprocessing(
        raw_mask,
        valid_mask=valid_mask,
        transform=scene.profile["transform"],
        crs=scene.profile["crs"],
        perm_water=scene.perm_water,
        median_size=median_size,
        min_area_ha=min_area_ha,
        remove_perm_water=remove_perm_water,
    )
    summary = summarize_analysis(
        post,
        valid_mask=valid_mask,
        bounds=scene.bounds_wgs84,
        crs="EPSG:4326",
        terrain=scene.terrain,
        polygon_geojson=scene.polygon_geojson,
        proba_map=proba,
    )

    live_metrics = None
    if scene.label is not None:
        live_metrics = compute_binary_metrics(scene.label, post.raw_water_mask, valid_mask=valid_mask)

    lag_days = compute_s1_s2_lag_days(scene.s1_date, scene.s2_date)

    if save_to_db or existing_tif is None or existing_gpkg is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem_out = f"{scene.scene_id}_{model_row['name']}_{ts}"
        tif_path = export_raster_geotiff(post.flood_mask, proba, scene.profile, OUTPUTS_DIR / f"{stem_out}.tif")
        gpkg_path = export_vector(post.flood_gdf, OUTPUTS_DIR / f"{stem_out}.gpkg")
        geojson_path = export_vector(post.flood_gdf, OUTPUTS_DIR / f"{stem_out}.geojson")
        _prune_old_outputs()
    else:
        tif_path = existing_tif
        gpkg_path = existing_gpkg
        candidate_gj = existing_tif.with_suffix(".geojson")
        geojson_path = (
            candidate_gj
            if candidate_gj.is_file()
            else export_vector(post.flood_gdf, candidate_gj)
        )

    if save_to_db:
        west, south, east, north = scene.bounds_wgs84
        poly_geojson = scene.polygon_geojson or {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        }
        analysis_id = save_analysis_record(
            event_id=scene.event_id,
            polygon_geojson=poly_geojson,
            model_id=int(model_row["id"]),
            flooded_ha=summary["flooded_ha"],
            perm_water_ha=summary["perm_water_ha"],
            s1_date=scene.s1_date,
            s2_date=scene.s2_date,
            cloud_pct=scene.cloud_pct,
            s1_s2_lag_days=lag_days,
            live_iou=live_metrics["iou"] if live_metrics else None,
            raster_path=tif_path,
            vector_path=gpkg_path,
        )
    else:
        analysis_id = existing_analysis_id

    return {
        "analysis_id": analysis_id,
        "scene": scene,
        "model_row": model_row,
        "post": post,
        "valid_mask": valid_mask,
        "proba": proba,
        "summary": summary,
        "live_metrics": live_metrics,
        "lag_days": lag_days,
        "elapsed_s": round(time.time() - t0, 2),
        "tif_path": tif_path,
        "gpkg_path": gpkg_path,
        "geojson_path": geojson_path,
    }


def _resolve_model_for_live_scene(
    chosen: dict[str, Any],
    all_models: list[dict[str, Any]],
    b_ok: bool,
    b_msg: str,
) -> dict[str, Any]:
    if chosen["config"] == "B" and not b_ok:
        same_family_a = next(
            (m for m in all_models if m["config"] == "A" and m.get("family") == chosen.get("family")),
            None,
        )
        fallback = same_family_a or next((m for m in all_models if m["config"] == "A"), all_models[0])
        st.info(
            f"`{chosen['name']}` omitido por nubes/desfase ({b_msg}). "
            f"Usando automáticamente **`{fallback['name']}`** (Config A — SAR penetra nubes)."
        )
        return fallback
    return chosen


def _format_model_option(m: dict[str, Any]) -> str:
    tag = "Ablación" if m["config"] == "S1_only" else f"Cfg {m['config']}"
    iou_txt = f"{m['test_iou']:.3f}" if m.get("test_iou") is not None else "N/D"
    return f"{m['name']} ({tag} · {iou_txt})"


def _render_help_popover() -> None:
    st.markdown("**Guía rápida de flujo**")
    st.markdown(
        "1. Usa el **buscador de país/ciudad** sobre el mapa y cambia entre **Satélite / Calles / Relieve / Oscuro**.\n"
        "2. Abre **Datos** para elegir escena (Offline, GeoTIFF o GEE en vivo) y modelo.\n"
        "3. Usa **Herramientas** para buffers UTM, estilo QGIS (.qml), MapLibre (.json) o reporte (.md).\n"
        "4. Abre **Asistente IA** para consultar métricas o adjuntar una imagen procesada."
    )
    st.divider()
    st.markdown("**Leyenda de capas**")
    st.markdown(
        "- **Azul marino:** Agua permanente (JRC + MERIT)\n"
        "- **Azul cian:** Inundación detectada (nueva)\n"
        "- **Puntos naranjas:** Top 3 focos críticos\n"
        "- **Comparación:** Azul = Consenso | Naranja = Solo M1 | Magenta = Solo M2\n"
        "- **Errores:** Verde = TP | Rojo = FP | Amarillo = FN"
    )
    st.divider()
    st.caption(
        "Alcance científico: detecta únicamente agua superficial presente en la fecha satelital. "
        "No predice riesgo futuro ni sustituye estudios hidrológicos."
    )


def _render_map_toolbar() -> tuple[str, dict[str, Any] | None]:
    """Render compact search bar (country/province/city/coords) + basemap switcher above the map."""
    c_form, c_base = st.columns([2.75, 1.95], vertical_alignment="center")

    with c_form, st.form("map_search_form", border=False):
        f_in, f_btn = st.columns([3.8, 1.0], vertical_alignment="center")
        query = f_in.text_input(
            "Buscar ubicación",
            placeholder="País, provincia, ciudad o lat, lon (ej. Salta, Argentina)",
            label_visibility="collapsed",
            key="map_geo_query",
        )
        go_clicked = f_btn.form_submit_button("Buscar", width="stretch")

    active_basemap = c_base.segmented_control(
        "Mapa base",
        options=list(BASEMAP_SPECS.keys()),
        default=st.session_state.get("active_basemap", "Satélite"),
        label_visibility="collapsed",
        key="map_basemap_picker",
    ) or "Satélite"
    st.session_state["active_basemap"] = active_basemap

    if go_clicked and query.strip():
        try:
            with st.spinner(f"Buscando `{query.strip()}`..."):
                matches = _cached_geocode(query.strip())
            if not matches:
                st.warning(f"No se encontraron ubicaciones para `{query.strip()}`.")
                st.session_state["geo_matches"] = []
            else:
                st.session_state["geo_matches"] = matches
                st.session_state["geo_target"] = matches[0]
        except RuntimeError as exc:
            st.error(str(exc))

    matches = st.session_state.get("geo_matches", [])
    if len(matches) > 1:
        m_col, clr_col = st.columns([4.5, 0.8], vertical_alignment="center")
        labels = [m["display_name"] for m in matches]
        picked_lbl = m_col.selectbox(
            "Coincidencias",
            labels,
            index=0,
            label_visibility="collapsed",
            key="geo_match_pick",
        )
        st.session_state["geo_target"] = matches[labels.index(picked_lbl)]
        if clr_col.button("Limpiar", key="geo_clear_btn", width="stretch"):
            st.session_state["geo_matches"] = []
            st.session_state["geo_target"] = None
            st.rerun()
    elif st.session_state.get("geo_target") is not None:
        m_col, clr_col = st.columns([4.5, 0.8], vertical_alignment="center")
        tgt = st.session_state["geo_target"]
        m_col.caption(f"Ubicación centrada: {tgt['display_name']}")
        if clr_col.button("Centrar", key="geo_reset_btn", width="stretch", help="Volver al encuadre de la escena"):
            st.session_state["geo_matches"] = []
            st.session_state["geo_target"] = None
            st.rerun()

    return active_basemap, st.session_state.get("geo_target")


def _render_data_panel(
    models: list[dict[str, Any]],
    *,
    gee_ok: bool,
) -> dict[str, Any]:
    """Slim collapsible left-docked panel (1. Fuente de Datos, 2. Modelo, 3. Postproceso)."""
    hdr_l, hdr_r = st.columns([1.0, 1.25], vertical_alignment="center")
    hdr_l.markdown("**Datos**")
    if hdr_r.button("Cerrar", key="close_data_panel", width="stretch"):
        st.session_state["show_data_panel"] = False
        st.rerun()

    source_options = ["Offline (Sen1Floods11)", "Subir GeoTIFF"]
    if gee_ok:
        source_options.append("Polígono en vivo (GEE)")
    source_mode = st.radio("1. Fuente de Datos", source_options, index=0, key="cfg_source_mode")

    selected_scene: SceneBundle | None = None

    if source_mode == "Offline (Sen1Floods11)":
        split_choice = st.selectbox("Split", ["bolivia", "test", "valid", "demo"], index=0, key="cfg_split")
        chips = list_offline_chips(split_choice)
        if not chips:
            st.warning(f"Sin chips para `{split_choice}`.")
        else:
            events_in_split = sorted({c["event"] for c in chips})
            ev_filter = st.selectbox("Evento", ["Todos"] + events_in_split, key="cfg_ev_filter")
            filtered = [c for c in chips if ev_filter == "Todos" or c["event"] == ev_filter]
            chip_stems = [c["stem"] for c in filtered]
            chosen_stem = st.selectbox("Chip", chip_stems, key="cfg_chip_stem")
            if chosen_stem:
                selected_scene = _cached_load_offline_scene(chosen_stem)

    elif source_mode == "Subir GeoTIFF":
        up_s1 = st.file_uploader("S1 GeoTIFF (VV, VH dB)", type=["tif", "tiff"], key="cfg_up_s1")
        up_s2 = st.file_uploader("S2 GeoTIFF (opcional)", type=["tif", "tiff"], key="cfg_up_s2")
        enrich_gee = st.checkbox("Slope/HAND desde GEE", value=gee_ok, disabled=not gee_ok, key="cfg_enrich")
        if up_s1 is not None:
            selected_scene = load_uploaded_geotiff(
                up_s1.getvalue(),
                filename=up_s1.name,
                s2_bytes=up_s2.getvalue() if up_s2 is not None else None,
                fetch_terrain_from_gee=enrich_gee,
            )

    st.divider()

    config_b_ok = True
    config_b_reason = ""
    if selected_scene is not None:
        lag = compute_s1_s2_lag_days(selected_scene.s1_date, selected_scene.s2_date)
        config_b_ok, config_b_reason = check_config_b_eligibility(
            s2_available=selected_scene.s2 is not None,
            cloud_pct=selected_scene.cloud_pct,
            s1_s2_lag_days=lag,
        )

    eligible_models = []
    for m in models:
        if m["config"] == "B" and selected_scene is not None and not config_b_ok:
            continue
        if m["config"] in ("A", "B") and selected_scene is not None and selected_scene.terrain is None:
            continue
        eligible_models.append(m)

    if selected_scene is not None and not config_b_ok:
        st.warning(f"Config B bloqueada: {config_b_reason}")

    if not eligible_models:
        st.error("Sin modelo compatible para esta escena.")
        return {"error": True}

    model_labels = {_format_model_option(m): m for m in eligible_models}
    label_keys = list(model_labels.keys())
    chosen_model_label = st.selectbox("2. Modelo (M1)", label_keys, key="cfg_m1")
    chosen_model = model_labels[chosen_model_label]

    compare_mode = st.checkbox("Comparar con M2", value=False, key="cfg_compare")
    chosen_model_2: dict[str, Any] | None = None
    if compare_mode:
        other_keys = [k for k in label_keys if k != chosen_model_label]
        if other_keys:
            default_idx = next(
                (i for i, k in enumerate(other_keys) if model_labels[k]["config"] != chosen_model["config"]),
                0,
            )
            chosen_label_2 = st.selectbox("Modelo M2", other_keys, index=default_idx, key="cfg_m2")
            chosen_model_2 = model_labels[chosen_label_2]
        else:
            st.caption("Solo hay un modelo compatible.")

    with st.expander("3. Postproceso", expanded=False):
        remove_perm = st.checkbox("Restar agua permanente", value=True, key="cfg_rm_perm")
        median_size = st.select_slider("Mediana (px)", options=[1, 3, 5], value=3, key="cfg_median")
        min_area_ha = st.slider("Área mínima (ha)", 0.0, 2.0, 0.1, 0.05, key="cfg_min_ha")

    run_btn = False
    if source_mode != "Polígono en vivo (GEE)":
        btn_text = "Ejecutar Comparación" if chosen_model_2 else "Ejecutar Detección"
        run_btn = st.button(btn_text, type="primary", width="stretch", key="cfg_run_btn")
    else:
        st.caption("Busca tu ciudad y dibuja el polígono (<= 20 km) en el mapa.")

    if st.session_state.get("last_result") is not None and st.button(
        "Limpiar resultado",
        width="stretch",
        key="cfg_clear_result",
    ):
        st.session_state["last_result"] = None
        st.session_state["last_result_2"] = None
        st.rerun()

    cfg_state = {
        "error": False,
        "source_mode": source_mode,
        "selected_scene": selected_scene,
        "chosen_model": chosen_model,
        "chosen_model_2": chosen_model_2,
        "remove_perm": remove_perm,
        "median_size": median_size,
        "min_area_ha": min_area_ha,
        "run_btn": run_btn,
    }
    st.session_state["last_cfg_state"] = {
        "source_mode": source_mode,
        "chosen_model": chosen_model,
        "chosen_model_2": chosen_model_2,
        "remove_perm": remove_perm,
        "median_size": median_size,
        "min_area_ha": min_area_ha,
    }
    return cfg_state


def _render_map_workspace(
    *,
    cfg: dict[str, Any],
    models: list[dict[str, Any]],
) -> None:
    source_mode = cfg.get("source_mode", "Offline (Sen1Floods11)")
    selected_scene = cfg.get("selected_scene")
    chosen_model = cfg.get("chosen_model", models[0])
    chosen_model_2 = cfg.get("chosen_model_2")
    median_size = int(cfg.get("median_size", 3))
    min_area_ha = float(cfg.get("min_area_ha", 0.1))
    remove_perm = bool(cfg.get("remove_perm", True))
    run_btn = bool(cfg.get("run_btn", False))

    # Always-present search bar + basemap switcher (Satélite / Calles / Relieve / Oscuro)
    active_basemap, geo_target = _render_map_toolbar()

    # Run detection immediately if requested
    if run_btn and selected_scene is not None:
        st.session_state["geo_target"] = None  # focus the processed scene
        geo_target = None
        with st.spinner(f"Ejecutando {chosen_model['name']} sobre {selected_scene.scene_id}..."):
            st.session_state["last_result"] = _run_analysis_pipeline(
                selected_scene, chosen_model, median_size, min_area_ha, remove_perm
            )
        if chosen_model_2 is not None:
            with st.spinner(f"Comparando con {chosen_model_2['name']}..."):
                st.session_state["last_result_2"] = _run_analysis_pipeline(
                    selected_scene, chosen_model_2, median_size, min_area_ha, remove_perm
                )
        else:
            st.session_state["last_result_2"] = None

    # --- Mode: Live GEE Polygon Drawing ---
    if source_mode == "Polígono en vivo (GEE)":
        g1, g2, g3 = st.columns([1.1, 1.1, 1.6], vertical_alignment="bottom")
        start_d = g1.date_input("Fecha inicio (GEE)", value=date.today() - timedelta(days=30), key="gee_start")
        end_d = g2.date_input("Fecha fin (GEE)", value=date.today(), key="gee_end")

        poly_too_big = False
        poly_status_msg = "Dibuja un polígono (<= 20 x 20 km) con las herramientas a la izquierda del mapa."
        if "gee_polygon" in st.session_state:
            _poly = shapely_shape(st.session_state["gee_polygon"])
            _gdf = gpd.GeoDataFrame(geometry=[_poly], crs="EPSG:4326")
            _bm = _gdf.to_crs(_gdf.estimate_utm_crs()).total_bounds
            _w_km = (_bm[2] - _bm[0]) / 1000.0
            _h_km = (_bm[3] - _bm[1]) / 1000.0
            poly_too_big = _w_km > 20.0 or _h_km > 20.0
            if poly_too_big:
                poly_status_msg = f"Polígono de {_w_km:.1f} x {_h_km:.1f} km excede el máximo de 20 x 20 km."
            else:
                poly_status_msg = f"Polígono válido: {_w_km:.1f} x {_h_km:.1f} km."

        if g3.button(
            "Buscar escenas Sentinel-1 en GEE",
            disabled=(poly_too_big or "gee_polygon" not in st.session_state),
            width="stretch",
            key="gee_search_btn",
        ):
            try:
                with st.spinner("Consultando catálogo COPERNICUS/S1_GRD en GEE..."):
                    st.session_state["gee_scenes"] = list_gee_s1_scenes(
                        st.session_state["gee_polygon"],
                        start_d.strftime("%Y-%m-%d"),
                        end_d.strftime("%Y-%m-%d"),
                    )
                if not st.session_state["gee_scenes"]:
                    st.info("No se encontraron escenas Sentinel-1 en ese rango de fechas.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error consultando GEE: {exc}")

        if poly_too_big:
            st.warning(poly_status_msg)
        else:
            st.caption(poly_status_msg)

        scenes_found = st.session_state.get("gee_scenes", [])
        if scenes_found and not poly_too_big:
            gee_limiter = _get_gee_rate_limiter()
            if APP_MODE != "local":
                st.caption(
                    f"Cuota GEE compartida: {gee_limiter.remaining()}/{gee_limiter.max_calls} "
                    "descargas disponibles en esta ventana de 10 min."
                )
            s_pick, s_run = st.columns([2.2, 1.4], vertical_alignment="bottom")
            scene_opts = {f"{s['date']} ({s['orbit']})": s for s in scenes_found}
            pick = s_pick.selectbox("Escena Sentinel-1 encontrada", list(scene_opts.keys()), key="gee_pick_sc")
            if s_run.button("Descargar GEE y ejecutar", type="primary", width="stretch", key="gee_run_btn"):
                if APP_MODE != "local":
                    allowed, retry_s = gee_limiter.try_acquire()
                    if not allowed:
                        mins, secs = divmod(retry_s, 60)
                        st.warning(
                            f"Límite de protección de cuota alcanzado (máx. {gee_limiter.max_calls} "
                            f"descargas GEE cada 10 min). Intenta nuevamente en {mins} min {secs} s."
                        )
                        return
                sc_meta = scene_opts[pick]
                try:
                    with st.spinner("Descargando S1 + Slope/HAND + S2 (Cloud Score+) desde GEE..."):
                        gee_bundle = fetch_gee_scene(
                            st.session_state["gee_polygon"],
                            s1_date=sc_meta["date"],
                            orbit=sc_meta["orbit"],
                        )
                        lag = compute_s1_s2_lag_days(gee_bundle.s1_date, gee_bundle.s2_date)
                        b_ok, b_msg = check_config_b_eligibility(
                            gee_bundle.s2 is not None, gee_bundle.cloud_pct, lag
                        )
                        m1_run = _resolve_model_for_live_scene(chosen_model, models, b_ok, b_msg)
                        st.session_state["geo_target"] = None
                        st.session_state["last_result"] = _run_analysis_pipeline(
                            gee_bundle, m1_run, median_size, min_area_ha, remove_perm
                        )
                        if chosen_model_2 is not None:
                            m2_run = _resolve_model_for_live_scene(chosen_model_2, models, b_ok, b_msg)
                            st.session_state["last_result_2"] = (
                                _run_analysis_pipeline(
                                    gee_bundle, m2_run, median_size, min_area_ha, remove_perm
                                )
                                if m2_run["name"] != m1_run["name"]
                                else None
                            )
                        else:
                            st.session_state["last_result_2"] = None
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error descargando/procesando desde GEE: {exc}")

        if st.session_state.get("last_result") is None:
            view_tag = (
                f"{active_basemap}_{geo_target['lat']:.4f}_{geo_target['lon']:.4f}"
                if geo_target is not None
                else f"{active_basemap}_default"
            )
            draw_map = _build_base_explorer_map(
                active_basemap=active_basemap,
                enable_draw=True,
                geo_target=geo_target,
            )
            draw_out = st_folium(draw_map, height=540, width=None, key=f"gee_draw_map_{view_tag}")
            last_drawn = (draw_out or {}).get("last_active_drawing")
            if (
                last_drawn
                and "geometry" in last_drawn
                and st.session_state.get("gee_polygon") != last_drawn["geometry"]
            ):
                st.session_state["gee_polygon"] = last_drawn["geometry"]
                st.rerun()
        elif st.button("Dibujar nuevo polígono (GEE)", key="gee_redraw_btn"):
            st.session_state["last_result"] = None
            st.session_state["last_result_2"] = None
            st.rerun()

    res = st.session_state.get("last_result")
    res2 = st.session_state.get("last_result_2")

    # --- Initial Startup State (no detection run yet): Show Base Explorer Map immediately ---
    if res is None:
        if source_mode != "Polígono en vivo (GEE)":
            fallback_tag = selected_scene.scene_id if selected_scene is not None else "world"
            base_tag = (
                f"{active_basemap}_{geo_target['lat']:.4f}_{geo_target['lon']:.4f}"
                if geo_target is not None
                else f"{active_basemap}_{fallback_tag}"
            )
            base_map = _build_base_explorer_map(
                active_basemap=active_basemap,
                enable_draw=False,
                preview_scene=selected_scene,
                geo_target=geo_target,
            )
            st_folium(
                base_map,
                height=545,
                width=None,
                returned_objects=[],
                key=f"base_explorer_map_{base_tag}",
            )
            if selected_scene is not None and geo_target is None:
                st.caption(
                    f"Encuadre previsualizado: `{selected_scene.scene_id}` ({selected_scene.event_name}). "
                    "Pulsa **Ejecutar Detección** en el panel Datos para procesar esta escena."
                )
            elif geo_target is None:
                st.caption(
                    "Mapa base listo. Usa el buscador superior para ir a cualquier país, provincia o ciudad, "
                    "o selecciona una fuente en **Datos**."
                )
        return

    # --- Processed Result View (Map first, then KPIs, Hotspots, Zonal Stats, Export) ---
    sc: SceneBundle = res["scene"]
    m_row = res["model_row"]
    summary = res["summary"]
    conf = summary["confidence"]
    live_m = res["live_metrics"]
    hotspots = summary.get("hotspots", [])

    # Quick controls right above the result map: Opacity (Propuesta 1) + Hotspot Focus (Propuesta 2)
    ctrl_op, ctrl_hs = st.columns([1.45, 2.55], vertical_alignment="center")
    op_map = {"40%": 0.40, "65%": 0.65, "85%": 0.85, "100%": 1.00}
    op_pick = ctrl_op.segmented_control(
        "Opacidad máscara",
        options=list(op_map.keys()),
        default=st.session_state.get("mask_opacity_label", "85%"),
        label_visibility="collapsed",
        key="mask_opacity_picker",
    ) or "85%"
    st.session_state["mask_opacity_label"] = op_pick
    mask_opacity = op_map[op_pick]

    focused_hotspot: dict[str, Any] | None = None
    if hotspots:
        hs_opts = ["Escena"] + [hs["foco"] for hs in hotspots[:5]]
        hs_pick = ctrl_hs.segmented_control(
            "Enfocar foco",
            options=hs_opts,
            default="Escena",
            label_visibility="collapsed",
            key=f"hs_focus_picker_{res.get('analysis_id')}",
        ) or "Escena"
        if hs_pick != "Escena":
            focused_hotspot = next((h for h in hotspots if h["foco"] == hs_pick), None)

    focus_key = (
        f"hs_{focused_hotspot['foco']}"
        if focused_hotspot is not None
        else (
            f"geo_{geo_target['lat']:.4f}_{geo_target['lon']:.4f}"
            if geo_target is not None
            else "scene"
        )
    )
    res_tag = f"{active_basemap}_{op_pick}_{focus_key}_{sc.scene_id}_{m_row['name']}_{res.get('analysis_id')}"
    fmap = _build_result_map(
        res,
        active_basemap=active_basemap,
        res2=res2,
        geo_target=geo_target,
        mask_opacity=mask_opacity,
        focused_hotspot=focused_hotspot,
    )
    st_folium(fmap, height=515, width=None, returned_objects=[], key=f"result_map_{res_tag}")

    legend_parts = [
        f"**Opacidad:** {op_pick}",
        "**Azul marino:** Agua permanente",
        "**Azul cian:** Inundación nueva",
        "**Puntos naranjas:** Top 5 focos",
    ]
    if focused_hotspot is not None:
        legend_parts.append(
            f"**Enfocado:** Foco {focused_hotspot['foco']} ({focused_hotspot['area_ha']} ha)"
        )
    if res2 is not None:
        legend_parts.append(
            f"**Comparación:** Azul = Consenso · Naranja = `{m_row['name']}` · Magenta = `{res2['model_row']['name']}`"
        )
    if sc.label is not None:
        legend_parts.append("**Errores:** Verde = TP · Rojo = FP · Amarillo = FN")
    st.caption(" | ".join(legend_parts))

    badge = compute_scene_reliability_badge(
        config_used=m_row["config"],
        cloud_pct=sc.cloud_pct,
        lag_days=res["lag_days"],
        uncertain_area_pct=conf["uncertain_area_pct"],
    )
    if badge["level"] == "high":
        st.success(f"**{badge['title']}** — {badge['detail']}")
    elif badge["level"] == "warning":
        st.warning(f"**{badge['title']}** — {badge['detail']}")
    else:
        st.info(f"**{badge['title']}** — {badge['detail']}")

    cloud_txt = (
        f"{sc.cloud_pct:.1f}% (polígono)"
        if sc.cloud_pct is not None
        else ("N/D (Sen1Floods11 offline)" if sc.source == "offline" else "N/D")
    )
    st.caption(
        f"Escena: `{sc.scene_id}` ({sc.event_name}) | "
        f"M1: `{m_row['name']}` (Config {m_row['config']}) | "
        f"S1: `{sc.s1_date or 'N/D'}` ({sc.orbit or 'N/D'}) | "
        f"S2: `{sc.s2_date or 'No disponible'}` | "
        f"Nubes S2: `{cloud_txt}` | "
        f"Desfase S1-S2: `{f'{res['lag_days']}d' if res['lag_days'] is not None else 'N/D'}` | "
        f"Tiempo: `{res['elapsed_s']}s`"
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Inundación (ha)",
        f"{summary['flooded_ha']:,.1f}",
        delta=f"Alta conf.: {conf['high_conf_ha']:,.1f} ha ({conf['high_conf_pct']:.0f}%)",
    )
    k2.metric("Agua permanente (ha)", f"{summary['perm_water_ha']:,.1f}")
    k3.metric("% Área válida", f"{summary['flooded_pct_of_valid']:.1f}%")
    k4.metric("Focos / Polígonos", f"{summary['flood_polygon_count']:,}")
    if live_m is not None:
        k5.metric(
            "Flood IoU (vivo)",
            f"{live_m['iou']:.4f}",
            delta=f"F1 {live_m['f1']:.3f}",
            help="Evalúa agua superficial total detectada contra LabelHand, ignorando -1.",
        )
    else:
        k5.metric("AOI total (ha)", f"{summary['aoi_ha']:,.1f}")

    if res2 is not None:
        m2_row = res2["model_row"]
        sum2 = res2["summary"]
        lm2 = res2["live_metrics"]
        st.markdown(f"**Comparación directa: `{m_row['name']}` vs `{m2_row['name']}`**")
        comp_rows = [
            {
                "Modelo": m_row["name"],
                "Config": m_row["config"],
                "Inundación (ha)": summary["flooded_ha"],
                "Alta Conf. (ha)": conf["high_conf_ha"],
                "Polígonos": summary["flood_polygon_count"],
                "Flood IoU": round(live_m["iou"], 4) if live_m else None,
                "F1": round(live_m["f1"], 4) if live_m else None,
                "Tiempo (s)": res["elapsed_s"],
            },
            {
                "Modelo": m2_row["name"],
                "Config": m2_row["config"],
                "Inundación (ha)": sum2["flooded_ha"],
                "Alta Conf. (ha)": sum2["confidence"]["high_conf_ha"],
                "Polígonos": sum2["flood_polygon_count"],
                "Flood IoU": round(lm2["iou"], 4) if lm2 else None,
                "F1": round(lm2["f1"], 4) if lm2 else None,
                "Tiempo (s)": res2["elapsed_s"],
            },
        ]
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)
        delta_ha = sum2["flooded_ha"] - summary["flooded_ha"]
        delta_txt = f"Delta Inundación (M2 - M1): `{delta_ha:+.1f} ha`"
        if live_m and lm2:
            delta_txt += f" | Delta Flood IoU (M2 - M1): `{lm2['iou'] - live_m['iou']:+.4f}`"
        st.caption(delta_txt)

    col_left, col_exp = st.columns([2, 1])
    with col_left:
        st.markdown("#### Top 5 Focos Críticos de Inundación")
        hotspots = summary.get("hotspots", [])
        if hotspots:
            hs_df = pd.DataFrame(hotspots)
            st.dataframe(
                hs_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "area_ha": st.column_config.NumberColumn("Área (ha)", format="%.2f"),
                    "google_maps": st.column_config.LinkColumn("Ubicación", display_text="Abrir mapa"),
                },
            )
        else:
            st.caption("No se detectaron polígonos por encima del área mínima configurada.")

        st.markdown("#### Estadísticas Zonales (dentro de la inundación)")
        zstats = summary.get("zonal_stats", {})
        if zstats:
            z_df = pd.DataFrame(zstats).T.reset_index().rename(columns={"index": "Variable"})
            st.dataframe(z_df, width="stretch", hide_index=True)
        else:
            st.caption("Sin datos de terreno o sin píxeles inundados detectados.")

        if live_m is not None:
            st.caption(
                f"Evaluación agua total vs LabelHand (ignorando -1): "
                f"IoU `{live_m['iou']:.4f}` · F1 `{live_m['f1']:.4f}` · "
                f"Precision `{live_m['precision']:.4f}` · Recall `{live_m['recall']:.4f}`"
            )

    with col_exp:
        st.markdown("#### Exportación")
        st.metric("Alta confianza (P >= 80%)", f"{conf['high_conf_ha']:,.1f} ha")
        st.metric("Confianza moderada", f"{conf['moderate_conf_ha']:,.1f} ha")
        st.download_button(
            "Descargar GeoTIFF (Raster)",
            data=Path(res["tif_path"]).read_bytes(),
            file_name=Path(res["tif_path"]).name,
            mime="image/tiff",
            width="stretch",
        )
        st.download_button(
            "Descargar GeoJSON (Vector)",
            data=Path(res["geojson_path"]).read_bytes(),
            file_name=Path(res["geojson_path"]).name,
            mime="application/geo+json",
            width="stretch",
        )
        st.download_button(
            "Descargar GeoPackage (.gpkg)",
            data=Path(res["gpkg_path"]).read_bytes(),
            file_name=Path(res["gpkg_path"]).name,
            mime="application/geopackage+sqlite3",
            width="stretch",
        )

        if sc.source in ("gee", "upload"):
            st.divider()
            save_name = st.text_input("Nombre del stack offline", value=sc.scene_id)
            if st.button("Guardar recorte en data/stacks/", width="stretch"):
                saved_stem = save_scene_as_demo_stack(sc, stack_name=save_name)
                st.success(f"Guardado como `{saved_stem}` (disponible en Offline -> Split `demo`).")


def _find_matching_offline_chip(raster_path: str | None, model_name: str | None) -> str | None:
    """Safely extract an offline/demo chip stem from a stored raster filename if it still exists."""
    if not raster_path or not model_name:
        return None
    fname = Path(raster_path).stem
    marker = f"_{model_name}_"
    if marker not in fname:
        return None
    candidate_stem = fname.split(marker, 1)[0]
    for split_name in ("bolivia", "test", "valid", "demo"):
        if any(c["stem"] == candidate_stem for c in list_offline_chips(split_name)):
            return candidate_stem
    return None


def _load_saved_result_from_outputs(
    rec: dict[str, Any],
    model_row: dict[str, Any],
    tif_path: Path,
    gpkg_path: Path | None,
) -> dict[str, Any]:
    """Reconstruct a viewable map result directly from a saved 2-band GeoTIFF + GeoPackage."""
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    from core.postproc import PostprocResult

    with rasterio.open(tif_path) as src:
        flood_mask = (src.read(1) > 0).astype(np.uint8)
        proba = src.read(2).astype(np.float32) if src.count >= 2 else flood_mask.astype(np.float32)
        profile = dict(src.profile)
        b = src.bounds
        wgs = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        bounds_wgs84 = (float(wgs[0]), float(wgs[1]), float(wgs[2]), float(wgs[3]))

    valid_mask = np.isfinite(proba)
    if gpkg_path is not None and gpkg_path.is_file():
        flood_gdf = gpd.read_file(gpkg_path)
    else:
        flood_gdf = gpd.GeoDataFrame({"area_ha": []}, geometry=[], crs="EPSG:4326")
    empty_gdf = gpd.GeoDataFrame({"area_ha": []}, geometry=[], crs="EPSG:4326")

    h, w = flood_mask.shape
    dummy_s1 = np.full((2, h, w), np.nan, dtype=np.float32)
    scene_id = tif_path.stem.split(f"_{model_row['name']}_", 1)[0]
    sc = SceneBundle(
        scene_id=scene_id,
        source="history",
        s1=dummy_s1,
        profile=profile,
        bounds_wgs84=bounds_wgs84,
        event_name=rec.get("event") or "Historial",
        s1_date=rec.get("s1_date"),
        s2_date=rec.get("s2_date"),
        cloud_pct=rec.get("cloud_pct"),
    )
    post = PostprocResult(
        raw_water_mask=flood_mask,
        flood_mask=flood_mask,
        perm_water_mask=None,
        flood_gdf=flood_gdf,
        perm_water_gdf=empty_gdf,
    )
    summary = summarize_analysis(
        post,
        valid_mask=valid_mask,
        bounds=bounds_wgs84,
        crs="EPSG:4326",
        proba_map=proba,
    )
    gj_path = tif_path.with_suffix(".geojson")
    if not gj_path.is_file():
        gj_path = export_vector(flood_gdf, gj_path)

    return {
        "analysis_id": rec["id"],
        "scene": sc,
        "model_row": model_row,
        "post": post,
        "valid_mask": valid_mask,
        "proba": proba,
        "summary": summary,
        "live_metrics": None,
        "lag_days": compute_s1_s2_lag_days(rec.get("s1_date"), rec.get("s2_date")),
        "elapsed_s": 0.0,
        "tif_path": tif_path,
        "gpkg_path": gpkg_path or tif_path,
        "geojson_path": gj_path,
    }


def main() -> None:
    _init_database()
    models = list_models(db_path=DEFAULT_DB_PATH)
    gee_ok = gee_is_configured()

    st.markdown(_UNIFORM_POPOVER_CSS, unsafe_allow_html=True)

    if "top_active_view" not in st.session_state:
        st.session_state["top_active_view"] = "Mapa"
    if st.session_state.pop("_switch_to_map", False):
        st.session_state["top_active_view"] = "Mapa"

    if "show_data_panel" not in st.session_state:
        st.session_state["show_data_panel"] = True
    if "show_ai_panel" not in st.session_state:
        st.session_state["show_ai_panel"] = False

    if not models:
        st.error("No hay modelos registrados en `data/flood.db`. Ejecuta `training/train_trees.py` primero.")
        return

    res = st.session_state.get("last_result")
    res2 = st.session_state.get("last_result_2")

    # ---------------- Compact Top Desktop-GIS Menu Bar (Icon-Free, Uniform Popovers) ----------------
    with st.container(border=True):
        m_brand, m_view, m_data, m_tools, m_plug, m_help, m_ai = st.columns(
            [1.45, 2.0, 0.9, 1.15, 1.2, 0.8, 1.2],
            vertical_alignment="center",
        )
        m_brand.markdown(f"**Flood Mapper** `{APP_MODE}`")

        active_view = m_view.segmented_control(
            "Vista",
            options=["Mapa", "Modelos", "Historial"],
            label_visibility="collapsed",
            key="top_active_view",
        ) or "Mapa"

        data_open = st.session_state["show_data_panel"]
        if m_data.button(
            "Datos",
            type="primary" if data_open else "secondary",
            width="stretch",
            key="top_btn_data",
        ):
            st.session_state["show_data_panel"] = not data_open
            st.rerun()

        with m_tools.popover("Herramientas", width="stretch"):
            render_tools_popover_content(res=res, res2=res2)

        with m_plug.popover("Complementos", width="stretch"):
            render_plugins_popover_content(app_mode=APP_MODE)

        with m_help.popover("Ayuda", width="stretch"):
            _render_help_popover()

        ai_open = st.session_state["show_ai_panel"]
        if m_ai.button(
            "Asistente IA",
            type="primary" if ai_open else "secondary",
            width="stretch",
            key="top_btn_ai",
        ):
            st.session_state["show_ai_panel"] = not ai_open
            st.rerun()

    # ---------------- Main Workspace Views ----------------
    if active_view == "Modelos":
        st.subheader("Modelos Entrenados y Registrados en SQLite (`data/flood.db`)")
        df_m = pd.DataFrame(models)[
            [
                "name",
                "family",
                "config",
                "threshold",
                "val_iou",
                "test_iou",
                "test_f1",
                "test_precision",
                "test_recall",
                "bolivia_iou",
                "size_mb",
            ]
        ]
        st.dataframe(df_m, width="stretch", hide_index=True)
        st.caption(
            "Referencia baseline oficial S1 Otsu (ignorando -1): "
            "Valid IoU = 0.5056 | Test IoU = 0.5458 | Bolivia IoU = 0.6972."
        )
        return

    if active_view == "Historial":
        st.subheader("Últimos análisis ejecutados (SQLite)")
        recent = _list_recent_analyses(25)
        if not recent:
            st.info("Todavía no se ejecutó ningún análisis en esta sesión.")
            return

        st.caption(
            "Haz clic en cualquier fila de la tabla para seleccionarla y pulsa **Cargar en el Mapa** "
            "para visualizar esa ejecución nuevamente (sin duplicar registros en SQLite)."
        )
        display_cols = [
            "id",
            "created_at",
            "event",
            "model_name",
            "config",
            "flooded_ha",
            "perm_water_ha",
            "live_iou",
            "s1_date",
            "s2_date",
            "cloud_pct",
        ]
        sel_event = st.dataframe(
            pd.DataFrame(recent)[display_cols],
            width="stretch",
            hide_index=True,
            height=300,
            on_select="rerun",
            selection_mode="single-row",
            key="hist_table_select",
        )
        selected_rows = (
            sel_event.selection.rows
            if sel_event is not None and hasattr(sel_event, "selection") and sel_event.selection.rows
            else []
        )
        selected_idx = selected_rows[0] if selected_rows else 0
        rec = recent[selected_idx]

        tif_safe = _safe_output_file(rec.get("raster_path"))
        gpkg_safe = _safe_output_file(rec.get("vector_path"))
        matched_stem = _find_matching_offline_chip(rec.get("raster_path"), rec.get("model_name"))
        matched_model = next((m for m in models if m["name"] == rec.get("model_name")), models[0])

        with st.container(border=True):
            st.markdown(
                f"**Ejecución seleccionada:** `#{rec['id']}` · "
                f"{rec['event'] or 'Escena'} · `{rec['model_name']}` (Config {rec['config']}) · "
                f"`{rec['flooded_ha']:.1f} ha` · `{rec['created_at']}`"
            )
            h1, h2, h3 = st.columns(3)
            can_load = bool(matched_stem) or (tif_safe is not None)
            if h1.button(
                "Cargar en el Mapa",
                type="primary",
                width="stretch",
                disabled=not can_load,
                key="hist_reopen_btn",
            ):
                with st.spinner(f"Cargando análisis #{rec['id']} en el mapa..."):
                    if matched_stem:
                        sc_reload = _cached_load_offline_scene(matched_stem)
                        st.session_state["last_result"] = _run_analysis_pipeline(
                            sc_reload,
                            matched_model,
                            3,
                            0.1,
                            True,
                            save_to_db=False,
                            existing_analysis_id=rec["id"],
                            existing_tif=tif_safe,
                            existing_gpkg=gpkg_safe,
                        )
                    elif tif_safe is not None:
                        st.session_state["last_result"] = _load_saved_result_from_outputs(
                            rec, matched_model, tif_safe, gpkg_safe
                        )
                    st.session_state["last_result_2"] = None
                    st.session_state["geo_target"] = None
                    st.session_state["_switch_to_map"] = True
                st.rerun()

            if tif_safe is not None:
                h2.download_button(
                    "Descargar GeoTIFF",
                    data=tif_safe.read_bytes(),
                    file_name=tif_safe.name,
                    mime="image/tiff",
                    width="stretch",
                    key=f"hist_dl_tif_{rec['id']}",
                )
            else:
                h2.caption("GeoTIFF ya rotado de `data/outputs/`.")

            if gpkg_safe is not None:
                h3.download_button(
                    "Descargar GeoPackage (.gpkg)",
                    data=gpkg_safe.read_bytes(),
                    file_name=gpkg_safe.name,
                    mime="application/geopackage+sqlite3",
                    width="stretch",
                    key=f"hist_dl_gpkg_{rec['id']}",
                )
            else:
                h3.caption("GeoPackage ya rotado de `data/outputs/`.")
        return

    # --- Active view: Mapa (Map is the primary actor: 82% width with Data open, 100% when closed) ---
    show_data = st.session_state["show_data_panel"]
    show_ai = st.session_state["show_ai_panel"]

    cfg_state: dict[str, Any] = st.session_state.get(
        "last_cfg_state",
        {
            "source_mode": "Offline (Sen1Floods11)",
            "chosen_model": models[0],
            "chosen_model_2": None,
            "remove_perm": True,
            "median_size": 3,
            "min_area_ha": 0.1,
            "run_btn": False,
        },
    )

    if show_data and show_ai:
        col_data, col_map, col_ai = st.columns([0.78, 2.22, 1.00], gap="small")
        with col_data, st.container(border=True):
            cfg_state = _render_data_panel(models, gee_ok=gee_ok)
        with col_map:
            if not cfg_state.get("error"):
                _render_map_workspace(cfg=cfg_state, models=models)
        with col_ai, st.container(border=True):
            render_assistant_dock(
                res=st.session_state.get("last_result"),
                res2=st.session_state.get("last_result_2"),
                app_mode=APP_MODE,
                db_path=DEFAULT_DB_PATH,
            )
    elif show_data:
        col_data, col_map = st.columns([0.78, 3.22], gap="small")
        with col_data, st.container(border=True):
            cfg_state = _render_data_panel(models, gee_ok=gee_ok)
        with col_map:
            if not cfg_state.get("error"):
                _render_map_workspace(cfg=cfg_state, models=models)
    elif show_ai:
        col_map, col_ai = st.columns([2.75, 1.05], gap="small")
        with col_map:
            _render_map_workspace(cfg=cfg_state, models=models)
        with col_ai, st.container(border=True):
            render_assistant_dock(
                res=st.session_state.get("last_result"),
                res2=st.session_state.get("last_result_2"),
                app_mode=APP_MODE,
                db_path=DEFAULT_DB_PATH,
            )
    else:
        _render_map_workspace(cfg=cfg_state, models=models)

    st.caption(
        "Detección de agua superficial presente en escenas Sentinel-1/2 + terreno relativo "
        "(Slope, HAND, DEM). No predice riesgo futuro ni reemplaza estudios hidrológicos."
    )


if __name__ == "__main__":
    main()
