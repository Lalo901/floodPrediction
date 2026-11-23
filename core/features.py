"""Deterministic feature engineering shared by training and inference.

Guarantees identical preprocessing across offline Sen1Floods11 chips,
uploaded GeoTIFFs, and live GEE extractions.

Configurations (trees):
  - `S1_only` (8 features): speckle-filtered VV/VH, diff, ratio, 5x5 local mean & std.
  - `A`      (11 features): `S1_only` + relative terrain (`slope`, `hand`, `dem_rel`).
  - `B`      (21 features): `A` + Sentinel-2 L1C key bands (`B2,B3,B4,B8,B11,B12`)
                            + spectral indices (`ndwi`, `mndwi`, `ndvi`, `awei_nsh`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

ConfigName = Literal["S1_only", "A", "B"]

S2_ALL_13_BANDS = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
)
S2_KEY_6_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
S2_KEY_6_INDICES = tuple(S2_ALL_13_BANDS.index(b) for b in S2_KEY_6_BANDS)

S1_FEATURE_NAMES: tuple[str, ...] = (
    "vv",
    "vh",
    "vv_minus_vh",
    "vv_vh_ratio",
    "vv_mean_5",
    "vv_std_5",
    "vh_mean_5",
    "vh_std_5",
)

TERRAIN_FEATURE_NAMES: tuple[str, ...] = (
    "slope",
    "hand",
    "dem_rel",
)

S2_FEATURE_NAMES: tuple[str, ...] = (
    "s2_b2",
    "s2_b3",
    "s2_b4",
    "s2_b8",
    "s2_b11",
    "s2_b12",
    "ndwi",
    "mndwi",
    "ndvi",
    "awei_nsh",
)

FEATURE_NAMES_BY_CONFIG: dict[ConfigName, tuple[str, ...]] = {
    "S1_only": S1_FEATURE_NAMES,
    "A": S1_FEATURE_NAMES + TERRAIN_FEATURE_NAMES,
    "B": S1_FEATURE_NAMES + TERRAIN_FEATURE_NAMES + S2_FEATURE_NAMES,
}

# Sentinel-1 dB safe bounds & neutral fill values used only to stop NaN bleed during spatial filters
S1_CLIP_MIN = -50.0
S1_CLIP_MAX = 5.0
S1_DEFAULT_FILL = (-12.0, -19.0)


def _fill_nonfinite(arr: np.ndarray, fallback: float) -> np.ndarray:
    """Replace NaN/Inf with chip finite median (or fallback if all non-finite)."""
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr.astype(np.float32, copy=False)
    fill_val = float(np.median(arr[finite])) if np.any(finite) else fallback
    out = arr.astype(np.float32, copy=True)
    out[~finite] = fill_val
    return out


def _local_mean_std(arr: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Fast local mean and standard deviation over a square window."""
    mean = uniform_filter(arr, size=window, mode="reflect")
    mean_sq = uniform_filter(arr * arr, size=window, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _extract_s2_six_bands(s2: np.ndarray) -> np.ndarray:
    """Accept either 13-band Sen1Floods11 S2Hand or 6-band (B2,B3,B4,B8,B11,B12) array."""
    if s2.ndim != 3:
        raise ValueError(f"Expected s2 shape (C, H, W), got {s2.shape}")
    if s2.shape[0] == 13:
        sub = s2[list(S2_KEY_6_INDICES)]
    elif s2.shape[0] == 6:
        sub = s2
    else:
        raise ValueError(f"Expected s2 with 13 or 6 bands, got {s2.shape[0]}")

    sub = sub.astype(np.float32, copy=False)
    # Normalize L1C digital numbers (0..10000+) to TOA reflectance ~[0, 1.5]
    if np.nanmax(sub) > 10.0:
        sub = sub / 10000.0
    return np.clip(sub, 0.0, 1.5)


def build_feature_stack(
    config: ConfigName,
    s1: np.ndarray,
    terrain: np.ndarray | None = None,
    s2: np.ndarray | None = None,
    speckle_median_size: int = 3,
    context_window: int = 5,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Build aligned feature stack `(C, H, W)` and boolean `valid_mask (H, W)`.

    Parameters
    ----------
    config : {'S1_only', 'A', 'B'}
        Target feature configuration.
    s1 : np.ndarray
        Sentinel-1 array `(2, H, W)` in dB (`VV`, `VH`).
    terrain : np.ndarray | None
        Terrain array `(>=3, H, W)` with bands `slope`, `hand`, `dem_rel` (+ optional `merit_wat`).
    s2 : np.ndarray | None
        Sentinel-2 L1C array `(13, H, W)` or `(6, H, W)` ordered as `B2,B3,B4,B8,B11,B12`.
    speckle_median_size : int
        Median filter kernel size for SAR speckle suppression (3 recommended; <=1 disables).
    context_window : int
        Square window size for SAR local mean and std (default 5).

    Returns
    -------
    features : np.ndarray
        Float32 array of shape `(n_features, H, W)` with no NaNs/Infs.
    feature_names : tuple[str, ...]
        Exact ordered feature names matching axis 0.
    valid_mask : np.ndarray
        Boolean mask `(H, W)` True where input sensors had valid finite data.
    """
    if s1.ndim != 3 or s1.shape[0] < 2:
        raise ValueError(f"Expected s1 shape (2, H, W), got {s1.shape}")

    vv_raw = s1[0].astype(np.float32, copy=False)
    vh_raw = s1[1].astype(np.float32, copy=False)
    valid_mask = np.isfinite(vv_raw) & np.isfinite(vh_raw)

    vv_clean = np.clip(_fill_nonfinite(vv_raw, S1_DEFAULT_FILL[0]), S1_CLIP_MIN, S1_CLIP_MAX)
    vh_clean = np.clip(_fill_nonfinite(vh_raw, S1_DEFAULT_FILL[1]), S1_CLIP_MIN, S1_CLIP_MAX)

    if speckle_median_size and speckle_median_size > 1:
        vv = median_filter(vv_clean, size=speckle_median_size, mode="reflect")
        vh = median_filter(vh_clean, size=speckle_median_size, mode="reflect")
    else:
        vv, vh = vv_clean, vh_clean

    vv_minus_vh = (vv - vh).astype(np.float32)
    denom = np.where(np.abs(vh) < 1e-3, -1e-3, vh)
    vv_vh_ratio = np.clip(vv / denom, -5.0, 5.0).astype(np.float32)

    vv_mean, vv_std = _local_mean_std(vv, window=context_window)
    vh_mean, vh_std = _local_mean_std(vh, window=context_window)

    channels: list[np.ndarray] = [
        vv,
        vh,
        vv_minus_vh,
        vv_vh_ratio,
        vv_mean,
        vv_std,
        vh_mean,
        vh_std,
    ]

    if config in ("A", "B"):
        if terrain is None or terrain.ndim != 3 or terrain.shape[0] < 3:
            raise ValueError(f"Config '{config}' requires terrain shape (>=3, H, W).")
        if terrain.shape[1:] != vv.shape:
            raise ValueError(f"Terrain spatial shape {terrain.shape[1:]} != S1 {vv.shape}")

        slope = np.clip(_fill_nonfinite(terrain[0], 0.0), 0.0, 90.0)
        hand = np.clip(_fill_nonfinite(terrain[1], 0.0), 0.0, 200.0)
        dem_rel = np.clip(_fill_nonfinite(terrain[2], 0.0), -100.0, 100.0)
        channels.extend([slope, hand, dem_rel])

    if config == "B":
        if s2 is None:
            raise ValueError("Config 'B' requires Sentinel-2 array `s2`.")
        s2_6 = _extract_s2_six_bands(s2)
        if s2_6.shape[1:] != vv.shape:
            raise ValueError(f"S2 spatial shape {s2_6.shape[1:]} != S1 {vv.shape}")

        s2_valid = np.all(np.isfinite(s2_6), axis=0) & (np.sum(s2_6, axis=0) > 1e-4)
        valid_mask = valid_mask & s2_valid

        s2_clean = np.stack([_fill_nonfinite(s2_6[i], 0.0) for i in range(6)], axis=0)
        b2, b3, b4, b8, b11, b12 = (s2_clean[i] for i in range(6))

        eps = 1e-6
        ndwi = np.clip((b3 - b8) / (b3 + b8 + eps), -1.0, 1.0).astype(np.float32)
        mndwi = np.clip((b3 - b11) / (b3 + b11 + eps), -1.0, 1.0).astype(np.float32)
        ndvi = np.clip((b8 - b4) / (b8 + b4 + eps), -1.0, 1.0).astype(np.float32)
        awei_nsh = np.clip(
            4.0 * (b3 - b11) - (0.25 * b8 + 2.75 * b12), -5.0, 5.0
        ).astype(np.float32)

        channels.extend([b2, b3, b4, b8, b11, b12, ndwi, mndwi, ndvi, awei_nsh])

    stack = np.stack(channels, axis=0).astype(np.float32, copy=False)
    return stack, FEATURE_NAMES_BY_CONFIG[config], valid_mask
