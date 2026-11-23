"""Unit tests for core.features guaranteeing NaN safety and exact feature contracts."""

import numpy as np

from core.features import FEATURE_NAMES_BY_CONFIG, build_feature_stack


def _synthetic_inputs(h: int = 32, w: int = 32):
    rng = np.random.default_rng(42)
    s1 = np.stack(
        [
            rng.uniform(-20.0, -5.0, size=(h, w)),
            rng.uniform(-28.0, -12.0, size=(h, w)),
        ],
        axis=0,
    ).astype(np.float32)
    # Inject NaNs in top-left corner to ensure no NaN propagation
    s1[:, :4, :4] = np.nan

    terrain = np.stack(
        [
            rng.uniform(0.0, 15.0, size=(h, w)),   # slope
            rng.uniform(0.0, 25.0, size=(h, w)),   # hand
            rng.uniform(-10.0, 10.0, size=(h, w)), # dem_rel
        ],
        axis=0,
    ).astype(np.float32)

    s2 = rng.integers(500, 4000, size=(13, h, w), dtype=np.int16)
    return s1, terrain, s2


def test_config_shapes_and_no_nans():
    s1, terrain, s2 = _synthetic_inputs()

    for cfg in ("S1_only", "A", "B"):
        stack, names, valid = build_feature_stack(cfg, s1, terrain=terrain, s2=s2)
        assert names == FEATURE_NAMES_BY_CONFIG[cfg]
        assert stack.shape == (len(names), 32, 32)
        assert stack.dtype == np.float32
        assert np.all(np.isfinite(stack)), f"Non-finite values in config {cfg}"
        assert valid.shape == (32, 32)
        assert not np.any(valid[:4, :4])
        assert np.all(valid[4:, 4:])


def test_s2_accepts_6_or_13_bands_identically():
    s1, terrain, s2_13 = _synthetic_inputs()
    s2_6 = s2_13[[1, 2, 3, 7, 11, 12]]

    st_13, _, v_13 = build_feature_stack("B", s1, terrain=terrain, s2=s2_13)
    st_6, _, v_6 = build_feature_stack("B", s1, terrain=terrain, s2=s2_6)

    np.testing.assert_allclose(st_13, st_6, atol=1e-6)
    np.testing.assert_array_equal(v_13, v_6)
