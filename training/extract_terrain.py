"""Extract aligned terrain layers (Slope, HAND, DEM_rel, MERIT wat) for all 446 HandLabeled chips.

Uses Google Earth Engine `computePixels` on global mosaics BEFORE clipping to avoid
drainage/border artifacts:
  - Band 1: `slope`     (deg)  — ee.Terrain.slope on Copernicus GLO-30 (30 m)
  - Band 2: `hand`      (m)    — MERIT Hydro `hnd` (~93 m, basin-scale HAND)
  - Band 3: `dem_rel`   (m)    — GLO-30 elevation minus local 1 km focal mean
  - Band 4: `merit_wat` (0/1)  — MERIT Hydro permanent water mask `wat`

Outputs compressed float32 GeoTIFFs in `dataset/derived/TerrainHand/{stem}_TerrainHand.tif`.
Resumable: skips chips already present on disk unless `--overwrite` is passed.

Usage:
    .venv/bin/python -m training.extract_terrain --limit 5   # quick smoke test
    .venv/bin/python -m training.extract_terrain             # all 446 chips
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import rasterio
from dotenv import load_dotenv
from scipy.ndimage import uniform_filter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S1_DIR = PROJECT_ROOT / "dataset" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
OUT_DIR = PROJECT_ROOT / "dataset" / "derived" / "TerrainHand"
BAND_NAMES = ("slope", "hand", "dem_rel", "merit_wat")
GEE_RAW_BANDS = ("slope", "hand", "dem", "merit_wat")


def build_terrain_image() -> ee.Image:
    """Build global composite image of Slope, HAND, DEM, and MERIT water (fast in GEE).

    Maps `ee.Terrain.slope` per native 30m GLO-30 tile BEFORE mosaicking so GEE
    does not fall back to the default 1-degree projection.
    """
    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
    dem = dem_col.mosaic().rename("dem")
    slope = dem_col.map(ee.Terrain.slope).mosaic().rename("slope")

    merit = ee.Image("MERIT/Hydro/v1_0_1")
    hand = merit.select("hnd").unmask(0).rename("hand")
    merit_wat = merit.select("wat").unmask(0).gt(0).rename("merit_wat")

    return ee.Image.cat([slope, hand, dem, merit_wat]).toFloat()


def compute_dem_rel(dem: np.ndarray, window_px: int = 51) -> np.ndarray:
    """Compute local relative elevation (~500m window at 10m) locally in milliseconds."""
    finite = np.isfinite(dem)
    if not np.any(finite):
        return np.zeros_like(dem, dtype=np.float32)
    fill = float(np.median(dem[finite]))
    dem_clean = np.where(finite, dem, fill).astype(np.float32)
    local_mean = uniform_filter(dem_clean, size=window_px, mode="reflect")
    return (dem_clean - local_mean).astype(np.float32)


def fetch_chip_terrain(
    s1_path: Path,
    terrain_img: ee.Image,
    out_dir: Path,
    overwrite: bool = False,
    max_retries: int = 3,
) -> tuple[str, str]:
    stem = s1_path.name.replace("_S1Hand.tif", "")
    out_path = out_dir / f"{stem}_TerrainHand.tif"
    if out_path.exists() and not overwrite:
        return stem, "skipped"

    with rasterio.open(s1_path) as src:
        profile = src.profile.copy()
        t = src.transform
        crs_code = src.crs.to_string()
        width, height = src.width, src.height

    request = {
        "expression": terrain_img,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": int(width), "height": int(height)},
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

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_bytes = ee.data.computePixels(request)
            with rasterio.MemoryFile(raw_bytes) as mem, mem.open() as mem_src:
                raw = mem_src.read().astype(np.float32)

            # raw bands: (slope, hand, dem, merit_wat) -> convert dem to dem_rel
            arr = raw.copy()
            arr[2] = compute_dem_rel(raw[2])

            profile.update(
                count=len(BAND_NAMES),
                dtype="float32",
                compress="deflate",
                nodata=None,
            )
            tmp_path = out_path.with_suffix(".tif.tmp")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(arr)
                dst.descriptions = BAND_NAMES
            tmp_path.replace(out_path)
            return stem, "ok"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0 * attempt)

    return stem, f"error: {last_err}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download terrain stacks from GEE for Sen1Floods11")
    parser.add_argument("--project", type=str, default=None, help="GCP Project ID (overrides .env)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent GEE requests")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N chips")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    project_id = args.project or os.getenv("GEE_PROJECT_ID")
    if not project_id:
        raise RuntimeError("Define GEE_PROJECT_ID en .env o pasa --project TU_PROJECT_ID.")
    ee.Initialize(project=project_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_files = sorted(S1_DIR.glob("*_S1Hand.tif"))
    if args.limit:
        s1_files = s1_files[: args.limit]

    terrain_img = build_terrain_image()

    ok_count = 0
    skip_count = 0
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_chip_terrain, p, terrain_img, OUT_DIR, args.overwrite): p
            for p in s1_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Terrain GEE"):
            stem, status = fut.result()
            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skip_count += 1
            else:
                errors.append((stem, status))

    print(f"\nCompletado: ok={ok_count}, skipped={skip_count}, errors={len(errors)}")
    if errors:
        print("Primeros errores:")
        for stem, msg in errors[:10]:
            print(f"  {stem}: {msg}")


if __name__ == "__main__":
    main()
