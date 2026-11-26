"""Unit and security tests for GIS exports (`core/exports_gis.py`), Plugin Manager (`plugins/manager.py`),
and AI vision fallback (`ai/agent.py`)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

from ai.agent import FloodAssistant
from ai.providers import BaseLLMProvider, LLMResponse, ProviderConfig
from ai.tools import AnalysisContext, build_default_tool_registry
from core.exports_gis import (
    build_executive_report_md,
    build_maplibre_style_layer,
    build_qgis_qml_style,
    build_safety_buffer_geojson,
)
from plugins.manager import (
    PLUGIN_CATALOG,
    install_whitelisted_plugin,
    list_plugins,
)


def _sample_flood_gdf() -> gpd.GeoDataFrame:
    # ~1 km x 1 km box near Bolivia
    poly = box(-64.91, -14.81, -64.90, -14.80)
    return gpd.GeoDataFrame({"area_ha": [110.0]}, geometry=[poly], crs="EPSG:4326")


def test_qgis_qml_style_contains_copernicus_cyan() -> None:
    qml = build_qgis_qml_style()
    assert "<qgis " in qml
    assert "0,180,255,166" in qml
    assert "SimpleFill" in qml


def test_maplibre_style_layer_valid_json() -> None:
    raw = build_maplibre_style_layer("bolivia_flood.geojson", (-64.91, -14.81, -64.90, -14.80))
    parsed = json.loads(raw)
    assert parsed["schema"] == "maplibre-gl-style-snippet/v1"
    assert parsed["sources"]["flood-mask-source"]["data"] == "./bolivia_flood.geojson"
    assert len(parsed["layers"]) == 2
    assert parsed["layers"][0]["paint"]["fill-color"] == "#00B4FF"


def test_safety_buffer_metric_utm_and_validation() -> None:
    gdf = _sample_flood_gdf()
    gj_bytes, area_ha = build_safety_buffer_geojson(gdf, buffer_m=100.0)
    parsed = json.loads(gj_bytes.decode("utf-8"))
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 1
    assert area_ha > 110.0  # buffered area must exceed original box

    # Empty GDF handled cleanly
    empty_gdf = gpd.GeoDataFrame({"area_ha": []}, geometry=[], crs="EPSG:4326")
    empty_bytes, empty_ha = build_safety_buffer_geojson(empty_gdf, buffer_m=50.0)
    assert empty_ha == 0.0
    assert json.loads(empty_bytes.decode("utf-8"))["features"] == []

    # Invalid radii rejected
    with pytest.raises(ValueError):
        build_safety_buffer_geojson(gdf, buffer_m=0.0)
    with pytest.raises(ValueError):
        build_safety_buffer_geojson(gdf, buffer_m=9999.0)


def test_executive_report_md_structure() -> None:
    res = {
        "scene": SimpleNamespace(
            scene_id="Bolivia_103757",
            event_name="Bolivia",
            source="offline",
            s1_date="2018-02-15",
            s2_date="2018-02-16",
            orbit="DESCENDING",
            cloud_pct=2.0,
        ),
        "model_row": {"name": "rf_B", "config": "B"},
        "lag_days": 1,
        "summary": {
            "aoi_ha": 2621.4,
            "flooded_ha": 412.5,
            "perm_water_ha": 38.2,
            "flood_polygon_count": 12,
            "largest_polygon_ha": 180.0,
            "flooded_pct_of_valid": 15.7,
            "confidence": {"high_conf_ha": 350.0, "high_conf_pct": 84.8, "moderate_conf_ha": 62.5},
            "hotspots": [
                {"foco": "#1", "area_ha": 180.0, "lat": -14.805, "lon": -64.905, "google_maps": "https://maps.google.com/?q=-14.805,-64.905"}
            ],
            "zonal_stats": {"HAND_m": {"median": 0.4, "mean": 0.8, "p90": 1.9}},
        },
        "live_metrics": {"iou": 0.8338, "f1": 0.9094, "precision": 0.91, "recall": 0.90},
    }
    md = build_executive_report_md(res)
    assert "Bolivia_103757" in md
    assert "412.50 ha" in md
    assert "0.8338" in md
    assert "agua superficial presente" in md


def test_plugin_manager_security_whitelist_and_space_block() -> None:
    # Cloud/Space mode strictly blocks any installation
    with pytest.raises(PermissionError):
        install_whitelisted_plugin("whitebox", app_mode="space")

    # Arbitrary/non-whitelisted packages are rejected even in local mode
    with pytest.raises(ValueError):
        install_whitelisted_plugin("malicious-package", app_mode="local")
    with pytest.raises(ValueError):
        install_whitelisted_plugin("https://evil.example.com/pkg.whl", app_mode="local")

    # Built-in plugins always report installed
    builtins = list_plugins(only_installed=True)
    builtin_ids = {entry["spec"].id for entry in builtins}
    assert {"qgis_connector", "maplibre_connector", "safety_buffer", "executive_report"}.issubset(builtin_ids)

    # Category & query filters work
    hydro = list_plugins(category="Hidrología")
    assert any(entry["spec"].id == "whitebox" for entry in hydro)
    assert len(PLUGIN_CATALOG) >= 9


class _VisionRejectingProvider(BaseLLMProvider):
    """Stub provider that rejects multimodal image inputs once, then succeeds on text-only retry."""

    def __init__(self) -> None:
        super().__init__(ProviderConfig(provider_id="ollama", model="text-only-stub", disable_thinking=True))
        self.calls: list[list[dict[str, Any]]] = []

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        last_content = messages[-1]["content"]
        has_image_blocks = isinstance(last_content, list)
        if has_image_blocks:
            raise RuntimeError("400 model does not support image inputs")
        return LLMResponse(text="Respuesta en texto sin imagen tras reintento automático.")


def test_agent_single_turn_vision_fallback_retries_text_only(tmp_path) -> None:
    from core.db import init_db

    db_path = tmp_path / "test_vision.db"
    init_db(db_path)

    ctx = AnalysisContext(
        analysis_id=None,
        scene_id="Bolivia_103757",
        event_name="Bolivia",
        source="offline",
        s1_date="2018-02-15",
        s2_date="2018-02-16",
        orbit="DESCENDING",
        cloud_pct=2.0,
        lag_days=1,
        model_name="rf_B",
        model_config="B",
        elapsed_s=1.2,
        summary={
            "aoi_ha": 2600.0,
            "flooded_ha": 320.0,
            "perm_water_ha": 20.0,
            "flood_polygon_count": 5,
            "largest_polygon_ha": 120.0,
            "flooded_pct_of_valid": 12.3,
            "confidence": {"high_conf_ha": 280.0, "high_conf_pct": 87.5, "moderate_conf_ha": 40.0, "uncertain_area_pct": 2.0},
            "hotspots": [],
            "zonal_stats": {},
        },
        live_metrics=None,
        flood_gdf=_sample_flood_gdf(),
        export_files={},
        comparison=None,
        db_path=db_path,
    )
    provider = _VisionRejectingProvider()
    assistant = FloodAssistant(provider=provider, registry=build_default_tool_registry(ctx))
    fake_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

    reply = assistant.ask(
        "Resume esta escena.",
        preview_image_data_url=fake_uri,
        force_single_turn=True,
    )
    assert reply.used_fallback is True
    assert "reintento automático" in reply.answer
    assert len(provider.calls) == 2  # 1st attempt with image, 2nd clean retry text-only


def test_geocoding_direct_coordinates_and_nominatim_mock(monkeypatch) -> None:
    from core import geocoding

    # Offline direct lat, lon parsing
    assert geocoding.parse_direct_coordinates("-14.8, -64.9") == (-14.8, -64.9)
    assert geocoding.parse_direct_coordinates("95.0, 10.0") is None  # invalid lat
    assert geocoding.parse_direct_coordinates("Salta, Argentina") is None

    direct = geocoding.geocode_place("-24.7821, -65.4232")
    assert len(direct) == 1
    assert direct[0]["lat"] == pytest.approx(-24.7821)
    assert direct[0]["lon"] == pytest.approx(-65.4232)
    assert direct[0]["place_type"] == "coordinates"

    # Mocked Nominatim response (boundingbox: [south, north, west, east])
    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, Any]]:
            return [
                {
                    "display_name": "Salta, Argentina",
                    "lat": "-24.7892",
                    "lon": "-65.4103",
                    "boundingbox": ["-24.85", "-24.72", "-65.48", "-65.34"],
                    "type": "city",
                    "addresstype": "city",
                }
            ]

    monkeypatch.setattr(geocoding.requests, "get", lambda *a, **kw: _FakeResp())
    res = geocoding.geocode_place("Salta, Argentina")
    assert len(res) == 1
    assert res[0]["display_name"] == "Salta, Argentina"
    assert res[0]["bounds"] == [[-24.85, -65.48], [-24.72, -65.34]]
    assert res[0]["zoom"] == 12


def test_history_safe_output_file_blocks_path_traversal(tmp_path, monkeypatch) -> None:
    import app.main as app_main

    fake_outputs = tmp_path / "outputs"
    fake_outputs.mkdir()
    ok_file = fake_outputs / "valid.tif"
    ok_file.write_bytes(b"tif")
    secret_file = tmp_path / "secret.env"
    secret_file.write_text("SECRET=1")

    monkeypatch.setattr(app_main, "OUTPUTS_DIR", fake_outputs)
    assert app_main._safe_output_file(str(ok_file)) == ok_file.resolve()
    assert app_main._safe_output_file(str(secret_file)) is None
    assert app_main._safe_output_file("/etc/passwd") is None
    assert app_main._safe_output_file(str(fake_outputs / ".." / "secret.env")) is None


def test_enable_gee_false_disables_init_gee(monkeypatch) -> None:
    import core.data as core_data

    monkeypatch.setattr(core_data, "_EE_INITIALIZED", False)
    monkeypatch.setenv("ENABLE_GEE", "false")
    assert core_data.init_gee() is False
    assert core_data.gee_is_configured() is False


def test_gee_sliding_window_rate_limiter() -> None:
    from core.data import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(max_calls=5, window_seconds=600.0)
    t0 = 1000.0
    for i in range(5):
        allowed, retry = limiter.try_acquire(now=t0 + i)
        assert allowed is True
        assert retry == 0
    assert limiter.remaining(now=t0 + 10) == 0

    # 6th call inside 10-minute window must be blocked with positive retry_after
    allowed, retry = limiter.try_acquire(now=t0 + 10)
    assert allowed is False
    assert 585 <= retry <= 595

    # After 600s elapse from first call, slot frees up cleanly
    allowed_after, retry_after = limiter.try_acquire(now=t0 + 601.0)
    assert allowed_after is True
    assert retry_after == 0
