"""Security, functional, and fallback tests for the modular `ai/` assistant package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from ai import (
    AnalysisContext,
    BaseLLMProvider,
    FloodAssistant,
    LLMResponse,
    ProviderConfig,
    ToolCallRequest,
    build_default_tool_registry,
    create_provider,
    list_available_presets,
    list_tool_calls,
    redact_secrets,
    sanitize_user_prompt,
    validate_endpoint_url,
)
from app.layers import render_preview_composite_data_url
from core.db import init_db


def _make_sample_context(db_path: Path) -> AnalysisContext:
    init_db(db_path)
    gdf = gpd.GeoDataFrame(
        {"area_ha": [12.5, 3.2, 0.4]},
        geometry=[
            box(-65.0, -15.0, -64.99, -14.99),
            box(-64.98, -15.0, -64.975, -14.995),
            box(-64.97, -15.0, -64.968, -14.998),
        ],
        crs="EPSG:4326",
    )
    return AnalysisContext(
        analysis_id=None,
        scene_id="Bolivia_103757",
        event_name="Bolivia",
        source="offline",
        s1_date="2018-02-15",
        s2_date="2018-02-15",
        orbit="DESCENDING",
        cloud_pct=None,
        lag_days=0,
        model_name="rf_B",
        model_config="B",
        elapsed_s=0.65,
        summary={
            "aoi_ha": 2620.0,
            "flooded_ha": 16.1,
            "perm_water_ha": 2.3,
            "total_water_ha": 18.4,
            "flooded_pct_of_valid": 14.2,
            "flood_polygon_count": 3,
            "confidence": {
                "high_conf_ha": 13.0,
                "moderate_conf_ha": 3.1,
                "high_conf_pct": 80.7,
                "uncertain_area_pct": 3.5,
            },
            "zonal_stats": {
                "slope_deg": {"mean": 0.8, "median": 0.5, "p90": 1.6, "max": 4.2},
                "hand_m": {"mean": 0.9, "median": 0.7, "p90": 1.8, "max": 3.1},
            },
        },
        live_metrics={"iou": 0.9037, "f1": 0.9494, "precision": 0.9584, "recall": 0.9406},
        flood_gdf=gdf,
        export_files={
            "geotiff": "Bolivia_103757_rf_B.tif",
            "geojson": "Bolivia_103757_rf_B.geojson",
            "geopackage": "Bolivia_103757_rf_B.gpkg",
        },
        db_path=db_path,
    )


# --- 1. Security tests (Anti-SSRF, Secrets, Prompt Limits, APP_MODE) ---

def test_ssrf_and_endpoint_validation():
    # Allowed local Ollama ONLY in APP_MODE=local
    assert validate_endpoint_url("http://localhost:11434/v1", app_mode="local", allow_local=True) == "http://localhost:11434/v1"
    # Allowed HTTPS external endpoints
    assert validate_endpoint_url("https://api.openai.com/v1", app_mode="space", allow_local=False) == "https://api.openai.com/v1"
    assert validate_endpoint_url(
        "https://generativelanguage.googleapis.com/v1beta/openai/", app_mode="space", allow_local=False
    ).startswith("https://generativelanguage.googleapis.com")

    # Blocked in APP_MODE=space or remote
    for bad in (
        "http://localhost:11434/v1",
        "http://127.0.0.1:11434/v1",
        "https://10.0.0.5/v1",
        "https://192.168.1.10/v1",
        "https://169.254.169.254/latest/meta-data/",
        "http://api.openai.com/v1",
        "file:///etc/passwd",
        "https://user:secret@api.openai.com/v1",
        "https://internal.service.local/v1",
    ):
        with pytest.raises(ValueError):
            validate_endpoint_url(bad, app_mode="space", allow_local=False)

    # Cloud metadata blocked even in local mode
    with pytest.raises(ValueError):
        validate_endpoint_url("http://169.254.169.254/latest/meta-data/", app_mode="local", allow_local=True)


def test_secret_redaction_and_provider_policies():
    cfg = ProviderConfig(provider_id="openai", model="gpt-4o-mini", api_key="sk-proj-supersecretkey123456")
    assert "supersecretkey" not in repr(cfg)

    scrubbed = redact_secrets(
        "Failed auth with sk-proj-supersecretkey123456 and Bearer eyJhbGciOiJIUzI1NiJ9.abcdef",
        ("sk-proj-supersecretkey123456",),
    )
    assert "supersecretkey" not in scrubbed
    assert "eyJhbGci" not in scrubbed

    # Local Ollama blocked in APP_MODE=space
    with pytest.raises(PermissionError):
        create_provider(ProviderConfig(provider_id="ollama", model="gemma4:latest"), app_mode="space")

    # External providers require BYOK key
    with pytest.raises(ValueError):
        create_provider(ProviderConfig(provider_id="anthropic", model="claude-3-5-haiku-latest", api_key=""), app_mode="local")

    # Space preset list excludes local providers
    space_ids = {p.id for p in list_available_presets(app_mode="space")}
    assert "ollama" not in space_ids and "lmstudio" not in space_ids
    assert {"anthropic", "gemini", "openai", "openrouter", "groq", "custom_openai"}.issubset(space_ids)


def test_prompt_sanitization():
    assert sanitize_user_prompt("  hola\x00 mundo  ") == "hola mundo"
    with pytest.raises(ValueError):
        sanitize_user_prompt("   ")
    with pytest.raises(ValueError):
        sanitize_user_prompt("x" * 2500)


# --- 2. Tool whitelist, argument bounds, and SQLite audit ---

def test_tool_registry_whitelist_and_audit(tmp_path: Path):
    db_path = tmp_path / "ai_tools.db"
    ctx = _make_sample_context(db_path)
    reg = build_default_tool_registry(ctx)

    summary = reg.execute("get_analysis_summary", {})
    assert summary["flooded_ha"] == 16.1
    assert summary["live_validation_vs_ground_truth"]["flood_iou"] == 0.9037

    filtered = reg.execute("filter_by_min_area", {"min_area_ha": 1.0})
    assert filtered["remaining_polygons"] == 2
    assert filtered["remaining_ha"] == 15.7

    hotspots = reg.execute("get_top_hotspots", {"top_n": 999})  # clamped to 20
    assert hotspots["top_n"] == 20 and len(hotspots["hotspots"]) == 3

    # Disallowed / unknown tool must never execute arbitrary code
    denied = reg.execute("exec_python", {"code": "1+1"})
    assert "error" in denied

    logged = list_tool_calls(db_path=db_path)
    assert len(logged) == 4
    assert {c["tool_name"] for c in logged} == {
        "get_analysis_summary",
        "filter_by_min_area",
        "get_top_hotspots",
        "exec_python",
    }


# --- 3. Agent loop, bounded rounds, and fallback for local models ---

class _FakeToolProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(ProviderConfig(provider_id="openai", model="mock-model", api_key="sk-test000000000000"))
        self.calls = 0

    def chat(self, *, system_prompt: str, messages: list[dict[str, Any]], tools=None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1 and tools:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="get_analysis_summary", arguments={}),
                    ToolCallRequest(id="call_2", name="filter_by_min_area", arguments={"min_area_ha": 1.0}),
                ],
                raw_assistant_message={"role": "assistant", "content": ""},
            )
        return LLMResponse(text="Detectadas 16.1 ha inundadas (15.7 ha en polígonos >= 1 ha).")


class _FakeNoToolsLocalProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(ProviderConfig(provider_id="ollama", model="small-local"))

    def chat(self, *, system_prompt: str, messages: list[dict[str, Any]], tools=None) -> LLMResponse:
        if tools:
            raise RuntimeError("Error del proveedor `ollama`: model does not support tools")
        assert "16.1" in str(messages[-1]["content"])
        return LLMResponse(text="Resumen fallback: 16.1 ha inundadas con IoU 0.9037.")


class _FakeInfiniteToolProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(ProviderConfig(provider_id="openai", model="loop-model", api_key="sk-test000000000000"))

    def chat(self, *, system_prompt: str, messages: list[dict[str, Any]], tools=None) -> LLMResponse:
        if tools:
            return LLMResponse(
                text="",
                tool_calls=[ToolCallRequest(id="c", name="get_zonal_stats", arguments={})],
                raw_assistant_message={"role": "assistant", "content": ""},
            )
        return LLMResponse(text="Respuesta final tras límite de rondas.")


def test_assistant_native_tool_calling_fallback_and_round_limit(tmp_path: Path):
    db_path = tmp_path / "ai_agent.db"
    ctx = _make_sample_context(db_path)

    # Native tool calling
    reg1 = build_default_tool_registry(ctx)
    reply = FloodAssistant(_FakeToolProvider(), reg1).ask("Resume y filtra >1 ha")
    assert not reply.used_fallback
    assert len(reply.tool_trace) == 2
    assert "16.1" in reply.answer

    # Fallback when local model lacks tool support
    reg2 = build_default_tool_registry(ctx)
    fb_reply = FloodAssistant(_FakeNoToolsLocalProvider(), reg2).ask("Resume la escena")
    assert fb_reply.used_fallback
    assert len(fb_reply.tool_trace) == 4
    assert "16.1" in fb_reply.answer

    # Strict round limit prevents infinite loops
    reg3 = build_default_tool_registry(ctx)
    lim_reply = FloodAssistant(_FakeInfiniteToolProvider(), reg3, max_tool_rounds=2).ask("Analiza")
    assert len(lim_reply.tool_trace) == 2
    assert "límite" in lim_reply.answer.lower()


def test_preview_composite_respects_max_side():
    s1 = np.full((2, 640, 640), -14.0, dtype=np.float32)
    mask = np.zeros((640, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    uri = render_preview_composite_data_url(s1, mask, max_side=256)
    assert uri.startswith("data:image/png;base64,")
