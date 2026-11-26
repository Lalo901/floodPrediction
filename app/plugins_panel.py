"""Streamlit UI for uniform top-bar popovers: `Herramientas` and `Complementos`.

Clean, icon-free desktop-GIS styling. Both menus open identically as fixed-width
dropdown popovers anchored to the top bar. UTM buffer exports are cached per
`(analysis_id, buffer_m)` so reruns remain instant (<5 ms).
Installation of whitelisted open-source packages is strictly restricted to `APP_MODE=local`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from core.analytics import compute_scene_reliability_badge
from core.exports_gis import (
    build_executive_report_md,
    build_maplibre_style_layer,
    build_qgis_qml_style,
    build_safety_buffer_geojson,
)
from plugins.manager import (
    install_whitelisted_plugin,
    list_local_plugin_scripts,
    list_plugins,
)

CATEGORIES = ["Todas", "Exportación", "Cartografía", "Vector", "Hidrología", "Análisis", "Imágenes"]
_MAX_BUFFER_CACHE_ENTRIES = 12


def _get_cached_safety_buffer(res: dict[str, Any], buf_m: int) -> tuple[bytes, float]:
    """Memoize metric UTM safety buffer per `(analysis_id, buf_m)` to avoid recomputing on every UI click."""
    cache: dict[tuple[Any, int], tuple[bytes, float]] = st.session_state.setdefault("_buffer_cache", {})
    key = (res.get("analysis_id"), buf_m)
    if key not in cache:
        if len(cache) >= _MAX_BUFFER_CACHE_ENTRIES:
            cache.clear()
        cache[key] = build_safety_buffer_geojson(res["post"].flood_gdf, buffer_m=float(buf_m))
    return cache[key]


def render_tools_popover_content(
    *,
    res: dict[str, Any] | None,
    res2: dict[str, Any] | None,
) -> None:
    """Render operational GIS tools inside the top-bar `Herramientas` popover."""
    st.markdown("**Herramientas sobre la escena**")
    if res is None:
        st.caption("Ejecuta primero una detección para habilitar el buffer UTM y los conectores GIS.")
        return

    sc = res["scene"]
    m_row = res["model_row"]
    stem = f"{sc.scene_id}_{m_row['name']}"

    st.markdown("**Buffer de seguridad (UTM)**")
    buf_m = st.select_slider(
        "Radio del buffer (m)",
        options=[50, 100, 250, 500],
        value=100,
        key="tool_buffer_m",
    )
    buf_bytes, buf_ha = _get_cached_safety_buffer(res, int(buf_m))
    st.caption(f"Área total con franja +{buf_m} m: **{buf_ha:,.2f} ha**")
    st.download_button(
        f"Descargar Buffer +{buf_m} m (.geojson)",
        data=buf_bytes,
        file_name=f"{stem}_buffer_{buf_m}m.geojson",
        mime="application/geo+json",
        width="stretch",
        key="dl_safety_buffer",
    )

    st.divider()

    st.markdown("**Conectores GIS y Reportes**")
    qml_str = build_qgis_qml_style()
    st.download_button(
        "Estilo QGIS (.qml)",
        data=qml_str.encode("utf-8"),
        file_name=f"{stem}.qml",
        mime="application/xml",
        width="stretch",
        key="dl_qgis_qml",
    )

    geojson_name = Path(res["geojson_path"]).name
    ml_json = build_maplibre_style_layer(geojson_name, sc.bounds_wgs84)
    st.download_button(
        "Capa MapLibre / GeoLibre (.json)",
        data=ml_json.encode("utf-8"),
        file_name=f"{stem}_maplibre_style.json",
        mime="application/json",
        width="stretch",
        key="dl_maplibre_json",
    )

    badge = compute_scene_reliability_badge(
        config_used=m_row["config"],
        cloud_pct=sc.cloud_pct,
        lag_days=res["lag_days"],
        uncertain_area_pct=res["summary"]["confidence"]["uncertain_area_pct"],
    )
    report_md = build_executive_report_md(res, res2=res2, badge=badge)
    st.download_button(
        "Reporte Ejecutivo (.md)",
        data=report_md.encode("utf-8"),
        file_name=f"{stem}_reporte.md",
        mime="text/markdown",
        width="stretch",
        key="dl_exec_report",
    )


def _render_plugin_card(entry: dict[str, Any], *, app_mode: str, allow_install: bool) -> None:
    spec = entry["spec"]
    installed = bool(entry["installed"])

    with st.container(border=True):
        top_l, top_r = st.columns([2.6, 1.1], vertical_alignment="center")
        status_txt = (
            "Integrado"
            if spec.source_type == "builtin"
            else ("Instalado" if installed else "Disponible")
        )
        top_l.markdown(f"**{spec.name}**  \n`{spec.category}` · {status_txt}")

        if allow_install and not installed:
            if app_mode == "local":
                confirm_key = f"confirm_install_{spec.id}"
                if top_r.button("Instalar", key=f"btn_inst_{spec.id}", width="stretch"):
                    st.session_state[confirm_key] = True
                if st.session_state.get(confirm_key):
                    st.warning(f"Instalar `{spec.pip_package}` en el entorno local?")
                    c_ok, c_cancel = st.columns(2)
                    if c_ok.button("Confirmar", key=f"ok_inst_{spec.id}", type="primary", width="stretch"):
                        with st.spinner(f"Instalando {spec.pip_package}..."):
                            ok, msg = install_whitelisted_plugin(spec.id, app_mode=app_mode)
                        st.session_state[confirm_key] = False
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    if c_cancel.button("Cancelar", key=f"cancel_inst_{spec.id}", width="stretch"):
                        st.session_state[confirm_key] = False
                        st.rerun()
            else:
                top_r.button("Solo local", key=f"btn_dis_{spec.id}", disabled=True, width="stretch")

        st.caption(f"{spec.summary} {spec.value_for_flood_image} ([Web]({spec.homepage_url}))")


def render_plugins_popover_content(*, app_mode: str) -> None:
    """Render the Plugin Manager directly inside the top-bar `Complementos` dropdown popover."""
    st.markdown("**Administrador de complementos**")
    st.caption("Conectores integrados y paquetes open-source en lista blanca.")

    tab_inst, tab_explore, tab_cfg = st.tabs(["Instalados", "Explorar", "Seguridad"])

    with tab_inst:
        s1, c1 = st.columns([1.7, 1.2])
        q_inst = s1.text_input(
            "Buscar instalados",
            key="pm_q_inst",
            label_visibility="collapsed",
            placeholder="Buscar...",
        )
        cat_inst = c1.selectbox("Categoría", CATEGORIES, key="pm_cat_inst", label_visibility="collapsed")
        items = list_plugins(category=cat_inst, query=q_inst, only_installed=True)
        with st.container(height=260, border=False):
            if not items:
                st.caption("Sin coincidencias.")
            else:
                for entry in items:
                    _render_plugin_card(entry, app_mode=app_mode, allow_install=False)

    with tab_explore:
        if app_mode != "local":
            st.caption(f"Modo público (`APP_MODE={app_mode}`): instalación bloqueada por seguridad.")
        s2, c2 = st.columns([1.7, 1.2])
        q_exp = s2.text_input(
            "Buscar en catálogo",
            key="pm_q_exp",
            label_visibility="collapsed",
            placeholder="Buscar...",
        )
        cat_exp = c2.selectbox("Categoría", CATEGORIES, key="pm_cat_exp", label_visibility="collapsed")
        items_exp = list_plugins(category=cat_exp, query=q_exp, only_installed=None)
        with st.container(height=260, border=False):
            if not items_exp:
                st.caption("Sin coincidencias.")
            else:
                for entry in items_exp:
                    _render_plugin_card(entry, app_mode=app_mode, allow_install=True)

    with tab_cfg:
        st.caption(
            f"Modo actual: `{app_mode}`. Solo se permiten paquetes verificados de PyPI "
            "(lista blanca, sin URLs arbitrarias ni `shell=True`)."
        )
        local_scripts = list_local_plugin_scripts()
        st.markdown("**Scripts locales en `plugins/`**")
        if local_scripts:
            for fname in local_scripts:
                st.code(f"plugins/{fname}", language="text")
        else:
            st.caption("No hay scripts personalizados en `plugins/`.")
