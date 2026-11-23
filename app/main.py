"""Flood Mapper MVP — Streamlit application (Offline Sen1Floods11 + Upload GeoTIFF + Live GEE)."""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.layers import (
    render_error_map_rgba,
    render_hand_rgba,
    render_mask_rgba,
    render_s1_rgba,
    render_s2_rgb_rgba,
    rgba_to_data_url,
)
from core.analytics import (
    check_config_b_eligibility,
    compute_s1_s2_lag_days,
    save_analysis_record,
    summarize_analysis,
)
from core.data import (
    SceneBundle,
    fetch_gee_scene,
    gee_is_configured,
    list_gee_s1_scenes,
    list_offline_chips,
    load_offline_scene,
    load_uploaded_geotiff,
)
from core.db import (
    DEFAULT_DB_PATH,
    DEFAULT_METADATA_GEOJSON,
    get_connection,
    init_db,
    list_events,
    seed_events_from_geojson,
)
from core.metrics import compute_binary_metrics
from core.postproc import export_raster_geotiff, export_vector, run_postprocessing
from core.registry import list_models, load_model

st.set_page_config(page_title="Flood Mapper MVP", page_icon="🌊", layout="wide")

APP_MODE = os.getenv("APP_MODE", "local")
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
MAX_STORED_OUTPUT_FILES = 60  # ~20 analyses x 3 files to protect ephemeral Space disk


@st.cache_resource
def _init_database() -> None:
    init_db(DEFAULT_DB_PATH)
    if not list_events(DEFAULT_DB_PATH) and DEFAULT_METADATA_GEOJSON.exists():
        seed_events_from_geojson(DEFAULT_METADATA_GEOJSON, DEFAULT_DB_PATH)


@st.cache_resource
def _cached_load_model(model_name: str):
    return load_model(model_name, DEFAULT_DB_PATH)


def _prune_old_outputs(max_files: int = MAX_STORED_OUTPUT_FILES) -> None:
    if not OUTPUTS_DIR.exists():
        return
    files = sorted(OUTPUTS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[max_files:]:
        if old.is_file():
            old.unlink(missing_ok=True)


def _list_recent_analyses(limit: int = 20) -> list[dict[str, Any]]:
    with get_connection(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.created_at, e.name AS event, a.s1_date, a.s2_date, a.cloud_pct,
                   m.name AS model_name, m.config, a.flooded_ha, a.perm_water_ha, a.live_iou
            FROM analyses a
            LEFT JOIN models m ON a.model_id = m.id
            LEFT JOIN events e ON a.event_id = e.id
            ORDER BY a.id DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def _build_result_map(
    scene: SceneBundle,
    flood_mask,
    raw_water_mask,
    perm_water_mask,
    valid_mask,
    show_gt: bool = True,
) -> folium.Map:
    west, south, east, north = scene.bounds_wgs84
    center = [(south + north) / 2.0, (west + east) / 2.0]
    m = folium.Map(location=center, zoom_start=12, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite Base (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)

    img_bounds = [[south, west], [north, east]]

    folium.raster_layers.ImageOverlay(
        image=rgba_to_data_url(render_s1_rgba(scene.s1)),
        bounds=img_bounds,
        name="Sentinel-1 (Falso Color SAR)",
        opacity=0.85,
        show=True,
    ).add_to(m)

    if scene.s2 is not None:
        folium.raster_layers.ImageOverlay(
            image=rgba_to_data_url(render_s2_rgb_rgba(scene.s2)),
            bounds=img_bounds,
            name="Sentinel-2 (RGB Natural)",
            opacity=0.85,
            show=False,
        ).add_to(m)

    if scene.terrain is not None:
        folium.raster_layers.ImageOverlay(
            image=rgba_to_data_url(render_hand_rgba(scene.terrain)),
            bounds=img_bounds,
            name="HAND (m, MERIT Hydro)",
            opacity=0.70,
            show=False,
        ).add_to(m)

    if perm_water_mask is not None and perm_water_mask.any():
        folium.raster_layers.ImageOverlay(
            image=rgba_to_data_url(render_mask_rgba(perm_water_mask, rgb=(0, 191, 255), alpha=170)),
            bounds=img_bounds,
            name="Agua Permanente (JRC + MERIT)",
            opacity=0.8,
            show=True,
        ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=rgba_to_data_url(render_mask_rgba(flood_mask, rgb=(220, 38, 38), alpha=190)),
        bounds=img_bounds,
        name="Inundación Detectada (Modelo)",
        opacity=0.85,
        show=True,
    ).add_to(m)

    if show_gt and scene.label is not None:
        gt_bin = (scene.label == 1).astype("uint8")
        folium.raster_layers.ImageOverlay(
            image=rgba_to_data_url(render_mask_rgba(gt_bin, rgb=(34, 197, 94), alpha=150)),
            bounds=img_bounds,
            name="Ground Truth (LabelHand)",
            opacity=0.75,
            show=False,
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=rgba_to_data_url(render_error_map_rgba(scene.label, raw_water_mask, valid_mask)),
            bounds=img_bounds,
            name="Mapa de Errores (TP=Verde, FP=Rojo, FN=Amarillo)",
            opacity=0.85,
            show=False,
        ).add_to(m)

    m.fit_bounds(img_bounds)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _run_analysis_pipeline(
    scene: SceneBundle,
    model_row: dict[str, Any],
    median_size: int,
    min_area_ha: float,
    remove_perm_water: bool,
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
    )

    live_metrics = None
    if scene.label is not None:
        # LabelHand marks all surface water (flood + permanent); evaluate total detected water
        live_metrics = compute_binary_metrics(scene.label, post.raw_water_mask, valid_mask=valid_mask)

    lag_days = compute_s1_s2_lag_days(scene.s1_date, scene.s2_date)

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem_out = f"{scene.scene_id}_{model_row['name']}_{ts}"
    tif_path = export_raster_geotiff(post.flood_mask, proba, scene.profile, OUTPUTS_DIR / f"{stem_out}.tif")
    gpkg_path = export_vector(post.flood_gdf, OUTPUTS_DIR / f"{stem_out}.gpkg")
    geojson_path = export_vector(post.flood_gdf, OUTPUTS_DIR / f"{stem_out}.geojson")
    _prune_old_outputs()

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


def _format_model_option(m: dict[str, Any]) -> str:
    tag = "Solo ablación" if m["config"] == "S1_only" else f"Config {m['config']}"
    iou_txt = f"{m['test_iou']:.3f}" if m.get("test_iou") is not None else "N/D"
    return f"{m['name']}  |  {tag}  (Test IoU: {iou_txt})"


def main() -> None:
    _init_database()
    models = list_models(db_path=DEFAULT_DB_PATH)
    gee_ok = gee_is_configured()

    st.title("🌊 Flood Mapper MVP")
    st.caption(
        f"Modo: `{APP_MODE}` | Detección de **agua superficial presente** en escenas Sentinel-1/2 + "
        "terreno relativo (Slope, HAND, DEM). *No predice riesgo futuro ni reemplaza estudios hidrológicos.*"
    )

    if not models:
        st.error("No hay modelos registrados en `data/flood.db`. Ejecuta `training/train_trees.py` primero.")
        return

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("1. Fuente de Datos")
        source_options = ["Offline (Sen1Floods11)", "Subir GeoTIFF"]
        if gee_ok:
            source_options.append("Polígono en vivo (GEE)")
        source_mode = st.radio("Origen de la escena", source_options, index=0)

        selected_scene: SceneBundle | None = None

        if source_mode == "Offline (Sen1Floods11)":
            split_choice = st.selectbox("Split", ["bolivia", "test", "valid", "demo"], index=0)
            chips = list_offline_chips(split_choice)
            if not chips:
                st.warning(f"No hay chips disponibles para split `{split_choice}`.")
            else:
                events_in_split = sorted({c["event"] for c in chips})
                ev_filter = st.selectbox("Filtrar evento", ["Todos"] + events_in_split)
                filtered = [c for c in chips if ev_filter == "Todos" or c["event"] == ev_filter]
                chip_stems = [c["stem"] for c in filtered]
                chosen_stem = st.selectbox("Chip Sen1Floods11", chip_stems)
                if chosen_stem:
                    selected_scene = load_offline_scene(chosen_stem, DEFAULT_DB_PATH)

        elif source_mode == "Subir GeoTIFF":
            up_s1 = st.file_uploader("Sentinel-1 GeoTIFF (2 bandas VV, VH en dB)", type=["tif", "tiff"])
            up_s2 = st.file_uploader("Sentinel-2 GeoTIFF (opcional, 6 o 13 bandas)", type=["tif", "tiff"])
            enrich_gee = st.checkbox("Descargar Slope/HAND desde GEE", value=gee_ok, disabled=not gee_ok)
            if up_s1 is not None:
                selected_scene = load_uploaded_geotiff(
                    up_s1.getvalue(),
                    filename=up_s1.name,
                    s2_bytes=up_s2.getvalue() if up_s2 is not None else None,
                    fetch_terrain_from_gee=enrich_gee,
                )

        st.divider()
        st.header("2. Modelo")

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
            st.warning(f"**Config B bloqueada:** {config_b_reason}")

        if not eligible_models:
            st.error("Ningún modelo registrado es compatible con las bandas de esta escena.")
            return

        model_labels = {_format_model_option(m): m for m in eligible_models}
        chosen_model_label = st.selectbox("Seleccionar modelo", list(model_labels.keys()))
        chosen_model = model_labels[chosen_model_label]

        st.divider()
        st.header("3. Postproceso")
        remove_perm = st.checkbox("Restar agua permanente (JRC + MERIT)", value=True)
        median_size = st.select_slider("Filtro mediana (px)", options=[1, 3, 5], value=3)
        min_area_ha = st.slider("Área mínima de polígono (ha)", 0.0, 2.0, 0.1, 0.05)

        run_btn = False
        if source_mode != "Polígono en vivo (GEE)":
            run_btn = st.button("🚀 Ejecutar Detección", type="primary", use_container_width=True)
        else:
            st.caption("Dibuja tu polígono y lanza la consulta en el panel principal →")

    # ---------------- Main Tabs ----------------
    tab_map, tab_models, tab_history = st.tabs(
        ["🗺️ Mapa y Resultados", "📊 Comparativa de Modelos", "🕒 Historial (SQLite)"]
    )

    with tab_map:
        if source_mode == "Polígono en vivo (GEE)":
            st.subheader("Dibuja un polígono (máx. 20 km por lado) y busca escenas Sentinel-1")
            c1, c2 = st.columns(2)
            start_d = c1.date_input("Fecha inicio", value=date.today() - timedelta(days=30))
            end_d = c2.date_input("Fecha fin", value=date.today())

            draw_map = folium.Map(location=[-15.0, -65.0], zoom_start=5)
            Draw(
                export=False,
                draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False},
            ).add_to(draw_map)
            draw_out = st_folium(draw_map, height=360, width=None, key="gee_draw_map")

            last_drawn = (draw_out or {}).get("last_active_drawing")
            if last_drawn and "geometry" in last_drawn:
                st.session_state["gee_polygon"] = last_drawn["geometry"]

            if "gee_polygon" in st.session_state:
                if st.button("🔎 Buscar escenas Sentinel-1 en GEE"):
                    with st.spinner("Consultando catálogo COPERNICUS/S1_GRD en GEE..."):
                        st.session_state["gee_scenes"] = list_gee_s1_scenes(
                            st.session_state["gee_polygon"],
                            start_d.strftime("%Y-%m-%d"),
                            end_d.strftime("%Y-%m-%d"),
                        )

                scenes_found = st.session_state.get("gee_scenes", [])
                if scenes_found:
                    scene_opts = {f"{s['date']} ({s['orbit']})": s for s in scenes_found}
                    pick = st.selectbox("Escena Sentinel-1 encontrada", list(scene_opts.keys()))
                    if st.button("⬇️ Descargar recorte GEE y ejecutar modelo", type="primary"):
                        sc_meta = scene_opts[pick]
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
                            model_to_run = chosen_model
                            if model_to_run["config"] == "B" and not b_ok:
                                st.warning(
                                    f"Config B bloqueada en escena en vivo ({b_msg}). "
                                    "Usando fallback automático Config A."
                                )
                                fallback = next((m for m in models if m["config"] == "A"), models[0])
                                model_to_run = fallback

                            st.session_state["last_result"] = _run_analysis_pipeline(
                                gee_bundle, model_to_run, median_size, min_area_ha, remove_perm
                            )

        if run_btn and selected_scene is not None:
            with st.spinner(f"Ejecutando {chosen_model['name']} sobre {selected_scene.scene_id}..."):
                st.session_state["last_result"] = _run_analysis_pipeline(
                    selected_scene, chosen_model, median_size, min_area_ha, remove_perm
                )

        res = st.session_state.get("last_result")
        if res is None:
            st.info("Selecciona una escena y un modelo en el panel izquierdo y ejecuta la detección.")
        else:
            sc: SceneBundle = res["scene"]
            m_row = res["model_row"]
            summary = res["summary"]
            live_m = res["live_metrics"]

            cloud_txt = (
                f"{sc.cloud_pct:.1f}% (polígono)"
                if sc.cloud_pct is not None
                else ("N/D (Sen1Floods11 offline)" if sc.source == "offline" else "N/D")
            )
            st.markdown(
                f"**Escena:** `{sc.scene_id}` ({sc.event_name}) | "
                f"**Modelo:** `{m_row['name']}` (Config **{m_row['config']}**) | "
                f"**S1:** `{sc.s1_date or 'N/D'}` ({sc.orbit or 'N/D'}) | "
                f"**S2:** `{sc.s2_date or 'No disponible'}` | "
                f"**Nubes S2:** `{cloud_txt}` | "
                f"**Desfase S1–S2:** `{f'{res['lag_days']}d' if res['lag_days'] is not None else 'N/D'}` | "
                f"**Tiempo:** `{res['elapsed_s']}s`"
            )

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Inundación (ha)", f"{summary['flooded_ha']:,.1f}")
            k2.metric("Agua permanente (ha)", f"{summary['perm_water_ha']:,.1f}")
            k3.metric("% Área válida inundada", f"{summary['flooded_pct_of_valid']:.1f}%")
            k4.metric("Polígonos", f"{summary['flood_polygon_count']:,}")
            if live_m is not None:
                k5.metric(
                    "Flood IoU (en vivo)",
                    f"{live_m['iou']:.4f}",
                    delta=f"F1 {live_m['f1']:.3f}",
                    help="Evalúa agua superficial total detectada (con agua permanente) contra LabelHand, ignorando -1.",
                )
            else:
                k5.metric("AOI total (ha)", f"{summary['aoi_ha']:,.1f}")

            fmap = _build_result_map(
                sc,
                res["post"].flood_mask,
                res["post"].raw_water_mask,
                res["post"].perm_water_mask,
                res["valid_mask"],
                show_gt=sc.label is not None,
            )
            st_folium(fmap, height=520, width=None, returned_objects=[], key="result_map")

            col_z, col_exp = st.columns([2, 1])
            with col_z:
                st.subheader("Estadísticas Zonales (dentro de la inundación)")
                zstats = summary.get("zonal_stats", {})
                if zstats:
                    z_df = pd.DataFrame(zstats).T.reset_index().rename(columns={"index": "Variable"})
                    st.dataframe(z_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("Sin datos de terreno o sin píxeles inundados detectados.")

                if live_m is not None:
                    st.caption(
                        f"Evaluación agua total vs `LabelHand` (ignorando `-1`, sin restar agua permanente): "
                        f"**IoU** `{live_m['iou']:.4f}` · **F1** `{live_m['f1']:.4f}` · "
                        f"**Precision** `{live_m['precision']:.4f}` · **Recall** `{live_m['recall']:.4f}`"
                    )

            with col_exp:
                st.subheader("Exportación")
                st.download_button(
                    "⬇️ Descargar GeoTIFF (Raster)",
                    data=Path(res["tif_path"]).read_bytes(),
                    file_name=Path(res["tif_path"]).name,
                    mime="image/tiff",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Descargar GeoJSON (Vector)",
                    data=Path(res["geojson_path"]).read_bytes(),
                    file_name=Path(res["geojson_path"]).name,
                    mime="application/geo+json",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Descargar GeoPackage (.gpkg)",
                    data=Path(res["gpkg_path"]).read_bytes(),
                    file_name=Path(res["gpkg_path"]).name,
                    mime="application/geopackage+sqlite3",
                    use_container_width=True,
                )

    with tab_models:
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
        st.dataframe(df_m, use_container_width=True, hide_index=True)
        st.caption(
            "Referencia baseline oficial S1 Otsu (ignorando `-1`): "
            "`Valid IoU = 0.5056` | `Test IoU = 0.5458` | `Bolivia IoU = 0.6972`."
        )

    with tab_history:
        st.subheader("Últimos análisis ejecutados")
        recent = _list_recent_analyses(25)
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
        else:
            st.info("Todavía no se ejecutó ningún análisis en esta sesión.")


if __name__ == "__main__":
    main()
