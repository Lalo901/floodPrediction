"""Streamlit UI module for the right-docked, scrollable AI Assistant panel.

Clean, icon-free desktop-GIS aesthetic:
  - Provider configuration is ALWAYS accessible (even before running a detection) so
    the user can pick local Ollama / LM Studio or enter a session-only BYOK API key upfront.
  - Processed-image layer selector with live thumbnail preview.
  - Independently scrollable chat container (`st.container(height=360, border=True)`) and
    native inline `st.chat_input`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ai import (
    PROVIDER_PRESETS,
    AnalysisContext,
    FloodAssistant,
    ProviderConfig,
    build_default_tool_registry,
    create_provider,
    list_available_presets,
    list_ollama_models,
    list_tool_calls,
)
from app.layers import render_layer_preview_bytes_and_url
from core.db import DEFAULT_DB_PATH

QUICK_PROMPTS = (
    ("Resumen operativo", "Resume la situación de esta escena para un equipo de Defensa Civil."),
    ("Top 3 focos críticos", "¿Cuáles son los 3 focos críticos más grandes y sus coordenadas?"),
    ("Terreno HAND/Slope", "Explica las estadísticas zonales de HAND, pendiente y DEM relativo."),
    ("Comparar modelos", "Compara este resultado con los otros modelos disponibles."),
)


def _build_analysis_context(
    res: dict[str, Any],
    res2: dict[str, Any] | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> AnalysisContext:
    sc = res["scene"]
    m_row = res["model_row"]
    comparison = None
    if res2 is not None:
        lm1 = res["live_metrics"]
        lm2 = res2["live_metrics"]
        comparison = {
            "model_1": {
                "name": m_row["name"],
                "config": m_row["config"],
                "flooded_ha": res["summary"]["flooded_ha"],
                "high_conf_ha": res["summary"]["confidence"]["high_conf_ha"],
                "live_iou": round(float(lm1["iou"]), 4) if lm1 else None,
                "elapsed_s": res["elapsed_s"],
            },
            "model_2": {
                "name": res2["model_row"]["name"],
                "config": res2["model_row"]["config"],
                "flooded_ha": res2["summary"]["flooded_ha"],
                "high_conf_ha": res2["summary"]["confidence"]["high_conf_ha"],
                "live_iou": round(float(lm2["iou"]), 4) if lm2 else None,
                "elapsed_s": res2["elapsed_s"],
            },
            "delta_flooded_ha_m2_minus_m1": round(
                res2["summary"]["flooded_ha"] - res["summary"]["flooded_ha"], 2
            ),
            "delta_iou_m2_minus_m1": (
                round(float(lm2["iou"] - lm1["iou"]), 4) if (lm1 and lm2) else None
            ),
        }

    return AnalysisContext(
        analysis_id=res.get("analysis_id"),
        scene_id=sc.scene_id,
        event_name=sc.event_name,
        source=sc.source,
        s1_date=sc.s1_date,
        s2_date=sc.s2_date,
        orbit=sc.orbit,
        cloud_pct=sc.cloud_pct,
        lag_days=res.get("lag_days"),
        model_name=m_row["name"],
        model_config=m_row["config"],
        elapsed_s=res["elapsed_s"],
        summary=res["summary"],
        live_metrics=res.get("live_metrics"),
        flood_gdf=res["post"].flood_gdf,
        export_files={
            "geotiff": Path(res["tif_path"]).name,
            "geojson": Path(res["geojson_path"]).name,
            "geopackage": Path(res["gpkg_path"]).name,
        },
        comparison=comparison,
        db_path=db_path,
    )


def _available_image_layers(res: dict[str, Any], res2: dict[str, Any] | None) -> dict[str, str]:
    """Return mapping of `{label: layer_key}` for processed layers available on the active scene."""
    sc = res["scene"]
    m1_name = res["model_row"]["name"]
    opts: dict[str, str] = {
        "Ninguna (solo métricas JSON)": "none",
        f"Modelo 1: {m1_name} (Inundación)": "model1",
    }
    if res2 is not None:
        m2_name = res2["model_row"]["name"]
        opts[f"Modelo 2: {m2_name} (Inundación)"] = "model2"
        opts[f"Comparación ({m1_name} vs {m2_name})"] = "comparison"
    if sc.label is not None:
        opts["Mapa de Errores (TP / FP / FN)"] = "error_map"
    if sc.s2 is not None:
        opts["Índice de Agua MNDWI (Sentinel-2)"] = "mndwi"
    if sc.terrain is not None:
        opts["Terreno HAND (MERIT Hydro)"] = "hand"
    return opts


def render_assistant_dock(
    *,
    res: dict[str, Any] | None,
    res2: dict[str, Any] | None,
    app_mode: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Render the right-docked AI Assistant panel (always configurable, chat active once a result exists)."""
    h_left, h_right = st.columns([2.2, 1.1], vertical_alignment="center")
    h_left.markdown("**Asistente IA**")
    if h_right.button("Cerrar", key="close_ai_dock", width="stretch"):
        st.session_state["show_ai_panel"] = False
        st.rerun()

    # --- Provider configuration: ALWAYS accessible even before running detection (Propuesta 3) ---
    presets = list_available_presets(app_mode=app_mode)
    preset_by_label = {p.label: p for p in presets}

    default_cfg_open = (res is None) or (not st.session_state.get("ai_provider_configured", False))
    with st.expander("Configuración del proveedor", expanded=default_cfg_open):
        if app_mode != "local":
            st.caption(
                "Modo nube (APP_MODE=space): modelos locales (Ollama/LM Studio) "
                "deshabilitados. Usa un proveedor BYOK."
            )

        chosen_label = st.selectbox("Proveedor", list(preset_by_label.keys()), key="ai_provider_label")
        preset = preset_by_label[chosen_label]
        st.caption(preset.notes)

        base_url = st.text_input(
            "URL Base",
            value=preset.default_base_url,
            disabled=(preset.protocol == "anthropic"),
            key=f"ai_url_{preset.id}",
        )

        if preset.id == "ollama":
            col_m, col_r = st.columns([2.6, 1.2], vertical_alignment="bottom")
            if col_r.button("Refrescar", key="ai_refresh_ollama", width="stretch"):
                st.session_state["ollama_models_cache"] = list_ollama_models(base_url, app_mode=app_mode)
            if "ollama_models_cache" not in st.session_state:
                st.session_state["ollama_models_cache"] = list_ollama_models(base_url, app_mode=app_mode)
            detected = st.session_state.get("ollama_models_cache", [])
            if detected:
                options = [*detected, "Escribir otro modelo..."]
                pick_m = col_m.selectbox("Modelo local (Ollama)", options, key="ai_ollama_pick")
                if pick_m == "Escribir otro modelo...":
                    model_name = st.text_input("Nombre del modelo Ollama", value=preset.default_model)
                else:
                    model_name = pick_m
                st.session_state["ai_provider_configured"] = True
            else:
                col_m.warning("No se detectó Ollama en esa URL. Verifica `ollama serve`.")
                model_name = st.text_input("Nombre del modelo Ollama", value=preset.default_model)
        elif preset.suggested_models:
            options = [*preset.suggested_models, "Personalizado..."]
            pick_m = st.selectbox("Modelo", options, key=f"ai_model_pick_{preset.id}")
            if pick_m == "Personalizado...":
                model_name = st.text_input("Nombre exacto del modelo", value="", key=f"ai_model_custom_{preset.id}")
            else:
                model_name = pick_m
        else:
            model_name = st.text_input(
                "Nombre del modelo",
                value=preset.default_model,
                key=f"ai_model_free_{preset.id}",
            )

        api_key = ""
        privacy_ok = True
        if preset.is_local:
            st.caption("Privacidad local: todo se procesa en tu equipo sin envíos externos.")
        else:
            k_col, clr_col = st.columns([2.7, 1.1], vertical_alignment="bottom")
            key_state_name = f"ai_key_{preset.id}"
            api_key = k_col.text_input(
                f"API Key ({preset.id})",
                type="password",
                key=key_state_name,
                help="Vive solo en memoria de sesión. Nunca se guarda en disco ni SQLite.",
            )
            if clr_col.button("Borrar", key=f"ai_clr_{preset.id}", width="stretch"):
                st.session_state[key_state_name] = ""
                st.rerun()
            if api_key.strip():
                st.session_state["ai_provider_configured"] = True

            privacy_ok = st.checkbox(
                "Acepto enviar resúmenes estadísticos a este proveedor externo.",
                value=st.session_state.get("ai_privacy_ack", False),
                key="ai_privacy_ack",
            )

        fast_local_mode = False
        if preset.is_local:
            fast_local_mode = st.checkbox(
                "Modo rápido CPU (sin thinking, 1 pasada; recomendado sin GPU)",
                value=True,
                key="ai_fast_local_mode",
            )

    provider_ready = (
        bool(model_name.strip())
        and (preset.is_local or (bool(api_key.strip()) and privacy_ok))
    )

    if res is None:
        status_msg = (
            f"Proveedor listo: `{preset.id}` / `{model_name}`. "
            if provider_ready
            else "Configura arriba tu proveedor local (Ollama) o tu API key BYOK. "
        )
        st.info(
            f"{status_msg}Ejecuta una detección en el mapa para activar el chat "
            "sobre métricas, focos críticos e imágenes procesadas."
        )
        return

    # Active provider + scene status line
    active_aid = res.get("analysis_id")
    if st.session_state.get("ai_chat_analysis_id") != active_aid:
        st.session_state["ai_chat_analysis_id"] = active_aid
        st.session_state["ai_messages"] = []

    st.caption(
        f"Proveedor: `{preset.id}` / `{model_name}` | "
        f"Análisis `#{active_aid}` (`{res['scene'].scene_id}`, `{res['summary']['flooded_ha']:,.1f} ha`)"
    )

    # --- Processed image selector + thumbnail preview ---
    layer_opts = _available_image_layers(res, res2)
    chosen_img_label = st.selectbox(
        "Imagen procesada a adjuntar (visión)",
        list(layer_opts.keys()),
        index=0,
        key="ai_selected_layer_label",
        help="Selecciona una capa procesada (<=512 px) para enviar junto a tu consulta. Nunca se envían GeoTIFF crudos.",
    )
    chosen_layer_key = layer_opts[chosen_img_label]
    preview_uri: str | None = None
    if chosen_layer_key != "none":
        png_bytes, preview_uri = render_layer_preview_bytes_and_url(
            res,
            chosen_layer_key,
            res2=res2,
            max_side=512,
        )
        st.image(
            png_bytes,
            caption=f"Vista previa adjunta: {chosen_img_label}",
            width=200,
        )

    # --- Quick prompts (compact 2x2 grid) ---
    queued_prompt: str | None = None
    row1 = st.columns(2)
    row2 = st.columns(2)
    for col, (btn_label, prompt_text) in zip([*row1, *row2], QUICK_PROMPTS, strict=True):
        if col.button(btn_label, width="stretch", key=f"qp_{btn_label}"):
            queued_prompt = prompt_text

    # --- Scrollable chat history container ---
    chat_history: list[dict[str, Any]] = st.session_state.get("ai_messages", [])
    chat_box = st.container(height=360, border=True)
    with chat_box:
        if not chat_history:
            st.caption(
                "Escribe una pregunta o usa los accesos rápidos superiores. "
                "Todas las consultas operan con herramientas acotadas y quedan auditadas en SQLite."
            )
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("attached_layer"):
                    st.caption(f"Imagen adjunta: {msg['attached_layer']}")
                if msg.get("tool_trace"):
                    with st.expander(f"Herramientas ejecutadas ({len(msg['tool_trace'])})", expanded=False):
                        if msg.get("used_fallback"):
                            st.caption("Modo directo (herramientas pre-ejecutadas en 1 pasada).")
                        for step in msg["tool_trace"]:
                            st.markdown(f"**`{step['tool_name']}`** — args: `{step['arguments']}`")
                            st.json(step["result"])

    # --- Native inline chat input (placeholder disappears on typing, no overlapping hint) ---
    typed_prompt = st.chat_input(
        "Pregunta sobre hectáreas, focos, terreno...",
        key="ai_dock_chat_input",
    )
    prompt_to_send = queued_prompt or (typed_prompt.strip() if typed_prompt and typed_prompt.strip() else None)

    if prompt_to_send:
        if not privacy_ok:
            st.error("Confirma el aviso de privacidad antes de consultar a un proveedor externo.")
            return
        if PROVIDER_PRESETS[preset.id].requires_api_key and not api_key.strip():
            st.error(f"Ingresa tu API key para `{preset.label}` en Configuración del proveedor.")
            return

        ctx = _build_analysis_context(res, res2=res2, db_path=db_path)
        registry = build_default_tool_registry(ctx)
        cfg = ProviderConfig(
            provider_id=preset.id,
            model=model_name.strip(),
            api_key=api_key.strip(),
            base_url=base_url.strip(),
            max_tokens=1600 if preset.is_local else 1200,
            timeout_s=180.0 if preset.is_local else 60.0,
            disable_thinking=bool(preset.is_local and fast_local_mode),
        )

        try:
            provider = create_provider(cfg, app_mode=app_mode)
            assistant = FloodAssistant(provider=provider, registry=registry)
            plain_history = [{"role": m["role"], "content": m["content"]} for m in chat_history]
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt_to_send)
                with st.chat_message("assistant"):
                    with st.spinner(f"Consultando `{cfg.model}` ({preset.id})..."):
                        reply = assistant.ask(
                            prompt_to_send,
                            history=plain_history,
                            preview_image_data_url=preview_uri,
                            force_single_turn=fast_local_mode,
                        )
                    st.markdown(reply.answer)

            serialized_trace = [
                {"tool_name": t.tool_name, "arguments": t.arguments, "result": t.result}
                for t in reply.tool_trace
            ]
            chat_history.append(
                {
                    "role": "user",
                    "content": prompt_to_send,
                    "attached_layer": chosen_img_label if chosen_layer_key != "none" else None,
                }
            )
            chat_history.append(
                {
                    "role": "assistant",
                    "content": reply.answer,
                    "tool_trace": serialized_trace,
                    "used_fallback": reply.used_fallback,
                }
            )
            st.session_state["ai_messages"] = chat_history
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    # --- Compact audit trail + clear chat ---
    c_aud, c_clr = st.columns([2.4, 1.2])
    if c_clr.button("Limpiar chat", key="ai_clear_chat", width="stretch"):
        st.session_state["ai_messages"] = []
        st.rerun()
    with c_aud.expander("Auditoría SQLite", expanded=False):
        calls = list_tool_calls(analysis_id=active_aid, limit=20, db_path=db_path)
        if calls:
            st.dataframe(pd.DataFrame(calls), width="stretch", hide_index=True)
        else:
            st.caption("Sin llamadas registradas para este análisis.")
