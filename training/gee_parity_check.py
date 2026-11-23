"""GEE Parity Check (Day 3 Decision Gate).

Compares local Sen1Floods11 HandLabeled chips (S1Hand & S2Hand) against
live Google Earth Engine queries (`COPERNICUS/S1_GRD` and `COPERNICUS/S2_HARMONIZED`)
sampled on the exact 512x512 affine grid via `ee.data.computePixels`.

Usage:
    .venv/bin/python -m training.gee_parity_check
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import rasterio
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAND_DIR = PROJECT_ROOT / "dataset" / "data" / "flood_events" / "HandLabeled"
METADATA_GEOJSON = PROJECT_ROOT / "dataset" / "Sen1Floods11_Metadata.geojson"

S2_TEST_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S2_ALL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_BAND_IDX = {b: i for i, b in enumerate(S2_ALL_BANDS)}

TARGET_EVENTS = ["Bolivia", "Ghana", "Spain"]


def load_event_metadata() -> dict[str, dict]:
    with METADATA_GEOJSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    for feat in data["features"]:
        props = feat["properties"]
        meta[props["location"]] = props
    return meta


def pick_sample_chip(event_prefix: str) -> str:
    """Select a chip with low NaN ratio in S1Hand for the given event."""
    candidates = sorted((HAND_DIR / "S1Hand").glob(f"{event_prefix}_*_S1Hand.tif"))
    best_stem = None
    best_valid = -1
    for p in candidates[:10]:
        with rasterio.open(p) as src:
            arr = src.read()
        valid = int(np.isfinite(arr).all(axis=0).sum())
        if valid > best_valid:
            best_valid = valid
            best_stem = p.name.replace("_S1Hand.tif", "")
    if not best_stem:
        raise FileNotFoundError(f"No S1Hand chips found for {event_prefix}")
    return best_stem


def compute_pixels_geotiff(ee_image: ee.Image, ref_profile: dict) -> np.ndarray:
    """Fetch an EE image aligned exactly to a local rasterio chip grid."""
    t = ref_profile["transform"]
    crs_code = ref_profile["crs"].to_string()
    request = {
        "expression": ee_image,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {
                "width": int(ref_profile["width"]),
                "height": int(ref_profile["height"]),
            },
            "affineTransform": {
                "scaleX": float(t.a),
                "shearX": float(t.b),
                "translateX": float(t.c),
                "shearY": float(t.d),
                "scaleY": float(t.e),
                "translateY": float(t.f),
            },
            "crsCode": crs_code,
        },
    }
    raw_bytes = ee.data.computePixels(request)
    with rasterio.MemoryFile(raw_bytes) as mem, mem.open() as src:
        return src.read()


def compare_arrays(local: np.ndarray, gee: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(local) & np.isfinite(gee)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return {"n_valid": n_valid, "corr": float("nan"), "mae": float("nan")}
    a = local[valid].astype(np.float64)
    b = gee[valid].astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else 0.0
    diff = np.abs(a - b)
    return {
        "n_valid": n_valid,
        "local_mean": round(float(a.mean()), 4),
        "gee_mean": round(float(b.mean()), 4),
        "local_std": round(float(a.std()), 4),
        "gee_std": round(float(b.std()), 4),
        "local_p50": round(float(np.percentile(a, 50)), 4),
        "gee_p50": round(float(np.percentile(b, 50)), 4),
        "mae": round(float(diff.mean()), 4),
        "corr": round(corr, 4),
    }


def date_window(date_str: str, pad_days: int = 1) -> tuple[str, str]:
    dt = datetime.strptime(date_str.replace("/", "-"), "%Y-%m-%d")
    start = (dt - timedelta(days=pad_days)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=pad_days + 1)).strftime("%Y-%m-%d")
    return start, end


def run_parity() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    project_id = os.getenv("GEE_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Define GEE_PROJECT_ID en .env antes de ejecutar la prueba de paridad.")
    ee.Initialize(project=project_id)

    event_meta = load_event_metadata()
    results: dict = {"project_id": project_id, "chips": {}}

    for event in TARGET_EVENTS:
        stem = pick_sample_chip(event)
        meta = event_meta[event]
        s1_path = HAND_DIR / "S1Hand" / f"{stem}_S1Hand.tif"
        s2_path = HAND_DIR / "S2Hand" / f"{stem}_S2Hand.tif"

        with rasterio.open(s1_path) as src:
            s1_local = src.read()
            s1_profile = src.profile
            bounds = src.bounds
        with rasterio.open(s2_path) as src:
            s2_local = src.read()
            s2_profile = src.profile

        geom = ee.Geometry.Rectangle([bounds.left, bounds.bottom, bounds.right, bounds.top])

        # --- Sentinel-1 GRD ---
        s1_start, s1_end = date_window(meta["s1_date"], pad_days=1)
        s1_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(s1_start, s1_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        if meta.get("orbit"):
            s1_col = s1_col.filter(ee.Filter.eq("orbitProperties_pass", meta["orbit"]))

        s1_count = int(s1_col.size().getInfo())
        s1_stats = {}
        if s1_count > 0:
            s1_img = s1_col.select(["VV", "VH"]).mosaic().toFloat()
            s1_gee = compute_pixels_geotiff(s1_img, s1_profile)
            s1_stats["VV"] = compare_arrays(s1_local[0], s1_gee[0])
            s1_stats["VH"] = compare_arrays(s1_local[1], s1_gee[1])

        # --- Sentinel-2 L1C (HARMONIZED) ---
        s2_start, s2_end = date_window(meta["s2_date"], pad_days=1)
        s2_col = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(geom)
            .filterDate(s2_start, s2_end)
        )
        s2_count = int(s2_col.size().getInfo())
        s2_stats = {}
        if s2_count > 0:
            s2_img = s2_col.select(S2_TEST_BANDS).mosaic().toFloat()
            s2_gee = compute_pixels_geotiff(s2_img, s2_profile)
            for i, b in enumerate(S2_TEST_BANDS):
                s2_stats[b] = compare_arrays(s2_local[S2_BAND_IDX[b]], s2_gee[i])

        results["chips"][stem] = {
            "event": event,
            "s1_date": meta["s1_date"],
            "s2_date": meta["s2_date"],
            "orbit": meta.get("orbit"),
            "s1_scenes_found": s1_count,
            "s2_scenes_found": s2_count,
            "s1_parity": s1_stats,
            "s2_parity": s2_stats,
        }

        print(f"\n=== {stem} ({event}) | S1={meta['s1_date']} ({s1_count} scenes) | S2={meta['s2_date']} ({s2_count} scenes) ===")
        for band, st in s1_stats.items():
            print(
                f"  S1 {band:<3} | local_mean={st['local_mean']:>8.3f} gee_mean={st['gee_mean']:>8.3f} "
                f"| MAE={st['mae']:>7.3f} | corr={st['corr']:.4f}"
            )
        for band, st in s2_stats.items():
            print(
                f"  S2 {band:<3} | local_mean={st['local_mean']:>8.1f} gee_mean={st['gee_mean']:>8.1f} "
                f"| MAE={st['mae']:>7.1f} | corr={st['corr']:.4f}"
            )

    out_path = PROJECT_ROOT / "data" / "gee_parity_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en: {out_path}")
    return results


if __name__ == "__main__":
    run_parity()
