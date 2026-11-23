"""End-to-end integration test for data -> inference -> postproc -> analytics -> export."""

from pathlib import Path

import pytest
import rasterio

from core.analytics import check_config_b_eligibility, compute_s1_s2_lag_days, summarize_analysis
from core.data import HAND_DIR, load_offline_scene
from core.db import PROJECT_ROOT
from core.metrics import compute_binary_metrics
from core.postproc import export_raster_geotiff, export_vector, run_postprocessing
from core.registry import load_model

HAS_LOCAL_E2E_ASSETS = (
    (HAND_DIR / "S1Hand" / "Bolivia_103757_S1Hand.tif").exists()
    and (PROJECT_ROOT / "models" / "lightgbm_B.joblib").exists()
)


def test_config_b_eligibility_rules():
    ok, _ = check_config_b_eligibility(True, cloud_pct=12.0, s1_s2_lag_days=1)
    assert ok

    ok_cloud, reason = check_config_b_eligibility(True, cloud_pct=45.0, s1_s2_lag_days=1)
    assert not ok_cloud and "nubes" in reason.lower()

    ok_lag, reason = check_config_b_eligibility(True, cloud_pct=5.0, s1_s2_lag_days=6)
    assert not ok_lag and "desfase" in reason.lower()

    assert compute_s1_s2_lag_days("2018-02-15", "2018-02-18") == 3


@pytest.mark.skipif(
    not HAS_LOCAL_E2E_ASSETS,
    reason="Requires local Sen1Floods11 dataset and trained models/lightgbm_B.joblib",
)
def test_end_to_end_bolivia_chip_pipeline(tmp_path: Path):
    scene = load_offline_scene("Bolivia_103757")
    assert scene.s1.shape == (2, 512, 512)
    assert scene.terrain is not None and scene.terrain.shape == (4, 512, 512)
    assert scene.s2 is not None and scene.label is not None
    assert scene.cloud_pct is None
    assert scene.event_id is not None

    model = load_model("lightgbm_B")
    proba, mask, valid_mask = model.predict(scene.s1, terrain=scene.terrain, s2=scene.s2)

    raw_metrics = compute_binary_metrics(scene.label, mask, valid_mask=valid_mask)
    assert raw_metrics["iou"] > 0.70

    post = run_postprocessing(
        mask,
        valid_mask=valid_mask,
        transform=scene.profile["transform"],
        crs=scene.profile["crs"],
        perm_water=scene.perm_water,
        median_size=3,
        min_area_ha=0.1,
    )
    summary = summarize_analysis(
        post,
        valid_mask=valid_mask,
        bounds=scene.bounds_wgs84,
        crs="EPSG:4326",
        terrain=scene.terrain,
    )
    assert summary["aoi_ha"] > 2000.0  # ~5.12km x 5.12km ≈ 2620 ha
    assert summary["flooded_ha"] > 0.0
    assert "hand_m" in summary["zonal_stats"]

    tif_out = export_raster_geotiff(post.flood_mask, proba, scene.profile, tmp_path / "flood.tif")
    gpkg_out = export_vector(post.flood_gdf, tmp_path / "flood.gpkg")
    geojson_out = export_vector(post.flood_gdf, tmp_path / "flood.geojson")

    assert tif_out.exists() and gpkg_out.exists() and geojson_out.exists()
    with rasterio.open(tif_out) as src:
        assert src.count == 2
        assert src.shape == (512, 512)
