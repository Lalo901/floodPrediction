"""Raster-to-RGBA image renderers for Folium ImageOverlay and AI preview thumbnails."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def _stretch_channel(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)
    return (norm * 255.0).astype(np.uint8)


def render_s1_rgba(s1: np.ndarray) -> np.ndarray:
    """False-color SAR composite (R=VV, G=VH, B=VV-VH) with transparent NaN pixels."""
    vv, vh = s1[0], s1[1]
    valid = np.isfinite(vv) & np.isfinite(vh)
    r = _stretch_channel(vv)
    g = _stretch_channel(vh)
    b = _stretch_channel(vv - vh)
    a = np.where(valid, 230, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_s2_rgb_rgba(s2: np.ndarray) -> np.ndarray:
    """Natural-color Sentinel-2 RGB (B4, B3, B2) from 13-band or 6-band arrays."""
    if s2.shape[0] == 13:
        r_raw, g_raw, b_raw = s2[3], s2[2], s2[1]
    else:
        # 6-band order: B2, B3, B4, B8, B11, B12
        r_raw, g_raw, b_raw = s2[2], s2[1], s2[0]
    valid = np.isfinite(r_raw) & np.isfinite(g_raw) & np.isfinite(b_raw) & ((r_raw + g_raw + b_raw) > 0)
    r = _stretch_channel(r_raw.astype(np.float32))
    g = _stretch_channel(g_raw.astype(np.float32))
    b = _stretch_channel(b_raw.astype(np.float32))
    a = np.where(valid, 240, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_hand_rgba(terrain: np.ndarray, max_m: float = 25.0) -> np.ndarray:
    """Render HAND (m) from blue (0 m, low drainage) to tan/brown (>= max_m)."""
    hand = terrain[1]
    valid = np.isfinite(hand)
    norm = np.clip(np.where(valid, hand, 0.0) / max_m, 0.0, 1.0)
    # Simple perceptual ramp: deep blue -> cyan -> yellow -> brown
    r = (np.clip(norm * 1.6, 0, 1) * 210 + 20).astype(np.uint8)
    g = (np.clip(1.0 - np.abs(norm - 0.45) * 1.6, 0, 1) * 190 + 40).astype(np.uint8)
    b = (np.clip(1.0 - norm * 1.5, 0, 1) * 220 + 20).astype(np.uint8)
    a = np.where(valid, 190, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_mask_rgba(
    mask: np.ndarray,
    rgb: tuple[int, int, int] = (30, 144, 255),
    alpha: int = 190,
) -> np.ndarray:
    """Render a binary mask as a colored RGBA overlay."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    on = mask > 0
    out[on, 0] = rgb[0]
    out[on, 1] = rgb[1]
    out[on, 2] = rgb[2]
    out[on, 3] = alpha
    return out


def render_error_map_rgba(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    alpha: int = 195,
) -> np.ndarray:
    """Render TP (green), FP (red), FN (yellow) validation overlay; TN and -1 stay transparent."""
    h, w = y_true.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    usable = valid_mask & (y_true != -1)
    yt = (y_true == 1) & usable
    yp = (y_pred > 0) & usable

    tp = yt & yp
    fp = (~yt) & yp & usable
    fn = yt & (~yp)

    out[tp] = (34, 197, 94, alpha)    # Green: True Positive
    out[fp] = (239, 68, 68, alpha)    # Red: False Positive
    out[fn] = (250, 204, 21, alpha)   # Yellow: False Negative
    return out


def rgba_to_data_url(rgba: np.ndarray) -> str:
    """Encode an `(H, W, 4)` uint8 array as a base64 PNG data URI for Folium."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
