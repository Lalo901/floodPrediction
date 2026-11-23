"""Unit tests for db, registry, FloodModel save/load + tiling, sanitization, and layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier

from app.layers import render_error_map_rgba, render_s1_rgba, rgba_to_data_url
from core.analytics import save_analysis_record
from core.data import _compute_dem_rel, sanitize_scene_id
from core.db import init_db, list_events
from core.features import FEATURE_NAMES_BY_CONFIG
from core.inference import FloodModel, ModelMetadata, _blend_weight_2d, _tile_starts
from core.registry import list_models, load_model, register_model


def test_db_registry_and_analysis_roundtrip(tmp_path: Path):
    db_path = tmp_path / "test_flood.db"
    init_db(db_path)
    assert list_events(db_path) == []

    clf = DummyClassifier(strategy="constant", constant=1)
    clf.fit(np.zeros((4, 8), dtype=np.float32), np.array([0, 1, 0, 1]))
    meta = ModelMetadata(
        name="dummy_s1",
        family="rf",
        config="S1_only",
        feature_names=FEATURE_NAMES_BY_CONFIG["S1_only"],
        threshold=0.5,
    )
    art_path = tmp_path / "dummy_s1.joblib"
    FloodModel(metadata=meta, estimator=clf).save(art_path)

    row_id = register_model(
        name="dummy_s1",
        family="rf",
        config="S1_only",
        artifact_path=art_path,
        feature_names=meta.feature_names,
        threshold=0.5,
        test_iou=0.61,
        bolivia_iou=0.55,
        db_path=db_path,
    )
    assert row_id >= 1
    models = list_models(db_path=db_path)
    assert len(models) == 1 and models[0]["name"] == "dummy_s1"

    loaded = load_model("dummy_s1", db_path=db_path)
    s1 = np.full((2, 16, 16), -15.0, dtype=np.float32)
    proba, mask, valid = loaded.predict(s1)
    assert proba.shape == (16, 16) and mask.shape == (16, 16) and valid.all()

    aid = save_analysis_record(
        polygon_geojson={"type": "Polygon", "coordinates": []},
        model_id=row_id,
        flooded_ha=12.5,
        perm_water_ha=3.0,
        db_path=db_path,
    )
    assert aid >= 1


def test_onnx_sliding_window_helpers():
    starts = _tile_starts(512, tile=256, stride=224)
    assert starts[0] == 0 and starts[-1] == 256 and len(starts) == 3

    blend = _blend_weight_2d(256, overlap=32)
    assert blend.shape == (256, 256)
    assert float(blend.min()) > 0.0
    assert float(blend[128, 128]) == 1.0


def test_sanitize_scene_id_and_metric_dem_rel():
    assert sanitize_scene_id("../../evil path/Bolivia 103.tif") == "Bolivia_103"
    assert sanitize_scene_id("...tif") == "scene"

    dem = np.ones((64, 64), dtype=np.float32) * 100.0
    dem_rel = _compute_dem_rel(dem, pixel_res_m=20.0, window_meters=510.0)
    assert np.allclose(dem_rel, 0.0, atol=1e-5)


def test_layers_error_map_and_data_url():
    y_true = np.array([[-1, 1], [0, 1]], dtype=np.int16)
    y_pred = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    valid = np.ones((2, 2), dtype=bool)
    err = render_error_map_rgba(y_true, y_pred, valid)
    assert tuple(err[0, 0]) == (0, 0, 0, 0)       # -1 ignored -> transparent
    assert err[0, 1, 1] > 150                     # TP -> green
    assert err[1, 0, 0] > 150 and err[1, 0, 1] < 100  # FP -> red
    assert err[1, 1, 0] > 150 and err[1, 1, 1] > 150  # FN -> yellow

    s1 = np.full((2, 8, 8), -12.0, dtype=np.float32)
    uri = rgba_to_data_url(render_s1_rgba(s1))
    assert uri.startswith("data:image/png;base64,")
