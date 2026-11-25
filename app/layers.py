"""Raster-to-RGBA image renderers for Folium ImageOverlay and AI preview thumbnails.

Standard disaster-response water palette (Copernicus EMS inspired):
  - Permanent water : Deep navy blue  (15, 61, 145)
  - Detected flood  : Bright cyan/blue (0, 180, 255)
  - Error map       : TP green, FP red, FN yellow
  - Model comparison: Consensus blue, Only M1 orange, Only M2 magenta
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

PERM_WATER_RGB = (15, 61, 145)      # #0F3D91 Deep navy
FLOOD_WATER_RGB = (0, 180, 255)     # #00B4FF Electric cyan-blue
MODEL2_WATER_RGB = (168, 85, 247)   # #A855F7 Purple (secondary model overlay)


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


def _extract_s2_b3_b8_b11(s2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if s2.shape[0] == 13:
        b3, b8, b11 = s2[2], s2[7], s2[11]
    else:
        # 6-band order: B2, B3, B4, B8, B11, B12
        b3, b8, b11 = s2[1], s2[3], s2[4]
    return b3.astype(np.float32), b8.astype(np.float32), b11.astype(np.float32)


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
        r_raw, g_raw, b_raw = s2[2], s2[1], s2[0]
    valid = (
        np.isfinite(r_raw)
        & np.isfinite(g_raw)
        & np.isfinite(b_raw)
        & ((r_raw + g_raw + b_raw) > 0)
    )
    r = _stretch_channel(r_raw.astype(np.float32))
    g = _stretch_channel(g_raw.astype(np.float32))
    b = _stretch_channel(b_raw.astype(np.float32))
    a = np.where(valid, 240, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_water_index_rgba(s2: np.ndarray, index: str = "mndwi") -> np.ndarray:
    """Render NDWI or MNDWI on a diverging dry-brown -> white -> water-blue ramp."""
    b3, b8, b11 = _extract_s2_b3_b8_b11(s2)
    second = b11 if index.lower() == "mndwi" else b8
    valid = np.isfinite(b3) & np.isfinite(second) & ((b3 + second) > 1e-4)
    idx = np.where(valid, (b3 - second) / (b3 + second + 1e-6), 0.0)
    # Map [-0.5, +0.5] -> [0, 1]
    t = np.clip((idx + 0.5) / 1.0, 0.0, 1.0)

    # Diverging ramp: t=0 brown (140,81,10) -> t=0.5 off-white (245,245,245) -> t=1 deep blue (0,102,220)
    low = t < 0.5
    f_lo = np.clip(t * 2.0, 0.0, 1.0)
    f_hi = np.clip((t - 0.5) * 2.0, 0.0, 1.0)

    r = np.where(low, 140 + (245 - 140) * f_lo, 245 + (0 - 245) * f_hi).astype(np.uint8)
    g = np.where(low, 81 + (245 - 81) * f_lo, 245 + (102 - 245) * f_hi).astype(np.uint8)
    b = np.where(low, 10 + (245 - 10) * f_lo, 245 + (220 - 245) * f_hi).astype(np.uint8)
    a = np.where(valid, 215, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_hand_rgba(terrain: np.ndarray, max_m: float = 25.0) -> np.ndarray:
    """Render HAND (m) from blue (0 m, low drainage) to tan/brown (>= max_m)."""
    hand = terrain[1]
    valid = np.isfinite(hand)
    norm = np.clip(np.where(valid, hand, 0.0) / max_m, 0.0, 1.0)
    r = (np.clip(norm * 1.6, 0, 1) * 210 + 20).astype(np.uint8)
    g = (np.clip(1.0 - np.abs(norm - 0.45) * 1.6, 0, 1) * 190 + 40).astype(np.uint8)
    b = (np.clip(1.0 - norm * 1.5, 0, 1) * 220 + 20).astype(np.uint8)
    a = np.where(valid, 190, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_low_hand_rgba(terrain: np.ndarray, threshold_m: float = 2.0) -> np.ndarray:
    """Highlight topographically low terrain (`HAND < threshold_m`) as static context."""
    hand = terrain[1]
    low_zone = np.isfinite(hand) & (hand <= threshold_m)
    return render_mask_rgba(low_zone.astype(np.uint8), rgb=(99, 102, 241), alpha=125)


def render_slope_rgba(terrain: np.ndarray, max_deg: float = 20.0) -> np.ndarray:
    """Render terrain slope (deg): green/flat -> yellow -> dark red/steep."""
    slope = terrain[0]
    valid = np.isfinite(slope)
    norm = np.clip(np.where(valid, slope, 0.0) / max_deg, 0.0, 1.0)
    r = np.clip(norm * 2.0 * 220 + 30, 0, 255).astype(np.uint8)
    g = np.clip((1.0 - np.maximum(norm - 0.4, 0.0) * 1.5) * 200, 0, 255).astype(np.uint8)
    b = np.clip((1.0 - norm) * 90, 0, 255).astype(np.uint8)
    a = np.where(valid, 180, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def render_probability_rgba(
    proba: np.ndarray,
    valid_mask: np.ndarray,
    min_prob: float = 0.25,
) -> np.ndarray:
    """Continuous model confidence heatmap (cyan -> deep blue -> indigo for P >= 0.80)."""
    h, w = proba.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    show = valid_mask & np.isfinite(proba) & (proba >= min_prob)
    if not np.any(show):
        return out
    t = np.clip((proba - min_prob) / max(1.0 - min_prob, 1e-6), 0.0, 1.0)
    r = (180 * (1.0 - t) + 15 * t).astype(np.uint8)
    g = (235 * (1.0 - t) + 76 * t).astype(np.uint8)
    b = (255 * (1.0 - t * 0.15)).astype(np.uint8)
    a = (90 + 145 * t).astype(np.uint8)
    out[show, 0] = r[show]
    out[show, 1] = g[show]
    out[show, 2] = b[show]
    out[show, 3] = a[show]
    return out


def render_mask_rgba(
    mask: np.ndarray,
    rgb: tuple[int, int, int] = FLOOD_WATER_RGB,
    alpha: int = 195,
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


def render_comparison_rgba(
    mask1: np.ndarray,
    mask2: np.ndarray,
    valid_mask: np.ndarray,
    alpha: int = 200,
) -> np.ndarray:
    """Render 2-model consensus & disagreement: Both=blue, Only M1=orange, Only M2=magenta."""
    h, w = mask1.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    m1 = (mask1 > 0) & valid_mask
    m2 = (mask2 > 0) & valid_mask

    both = m1 & m2
    only1 = m1 & (~m2)
    only2 = (~m1) & m2

    out[both] = (0, 140, 255, alpha)     # Blue: Consensus
    out[only1] = (249, 115, 22, alpha)   # Orange: Only Model 1
    out[only2] = (217, 70, 239, alpha)   # Magenta: Only Model 2
    return out


def render_preview_composite_data_url(
    s1: np.ndarray,
    flood_mask: np.ndarray,
    *,
    s2: np.ndarray | None = None,
    perm_water_mask: np.ndarray | None = None,
    max_side: int = 512,
) -> str:
    """Create a lightweight RGB+mask PNG data URI (<=512px) for LLM vision — never raw rasters."""
    base = render_s2_rgb_rgba(s2) if s2 is not None else render_s1_rgba(s1)
    comp = base[:, :, :3].astype(np.float32)

    if perm_water_mask is not None:
        pm = perm_water_mask > 0
        comp[pm] = 0.45 * comp[pm] + 0.55 * np.array(PERM_WATER_RGB, dtype=np.float32)

    fm = flood_mask > 0
    comp[fm] = 0.35 * comp[fm] + 0.65 * np.array(FLOOD_WATER_RGB, dtype=np.float32)

    img = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), mode="RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def rgba_to_data_url(rgba: np.ndarray) -> str:
    """Encode an `(H, W, 4)` uint8 array as a base64 PNG data URI for Folium."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _blend_rgba_on_rgb(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32)) / 255.0
    ov_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    out = base_rgb.astype(np.float32) * (1.0 - alpha) + ov_rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_layer_preview_bytes_and_url(
    res: dict,
    layer_key: str,
    *,
    res2: dict | None = None,
    max_side: int = 512,
) -> tuple[bytes, str]:
    """Render a chosen processed layer as compact PNG bytes + data URI (<=512px) for AI vision."""
    sc = res["scene"]
    base_rgba = render_s2_rgb_rgba(sc.s2) if sc.s2 is not None else render_s1_rgba(sc.s1)
    comp = base_rgba[:, :, :3].copy()

    if layer_key == "model1":
        if res["post"].perm_water_mask is not None:
            comp = _blend_rgba_on_rgb(comp, render_mask_rgba(res["post"].perm_water_mask, rgb=PERM_WATER_RGB, alpha=160))
        comp = _blend_rgba_on_rgb(comp, render_mask_rgba(res["post"].flood_mask, rgb=FLOOD_WATER_RGB, alpha=185))
    elif layer_key == "model2" and res2 is not None:
        if res2["post"].perm_water_mask is not None:
            comp = _blend_rgba_on_rgb(comp, render_mask_rgba(res2["post"].perm_water_mask, rgb=PERM_WATER_RGB, alpha=160))
        comp = _blend_rgba_on_rgb(comp, render_mask_rgba(res2["post"].flood_mask, rgb=MODEL2_WATER_RGB, alpha=185))
    elif layer_key == "comparison" and res2 is not None:
        comp = _blend_rgba_on_rgb(
            comp,
            render_comparison_rgba(res["post"].flood_mask, res2["post"].flood_mask, res["valid_mask"], alpha=205),
        )
    elif layer_key == "error_map" and sc.label is not None:
        comp = _blend_rgba_on_rgb(
            comp,
            render_error_map_rgba(sc.label, res["post"].raw_water_mask, res["valid_mask"], alpha=200),
        )
    elif layer_key == "mndwi" and sc.s2 is not None:
        comp = render_water_index_rgba(sc.s2, index="mndwi")[:, :, :3]
    elif layer_key == "hand" and sc.terrain is not None:
        comp = render_hand_rgba(sc.terrain)[:, :, :3]

    img = Image.fromarray(comp, mode="RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return png_bytes, f"data:image/png;base64,{b64}"
