"""Uniform post-processing and raster/vector export for all flood models.

Steps (Agent/spec.md §6.4):
  1. Mask out invalid input pixels (`~valid_mask`).
  2. Apply median filter (3x3 by default) to remove isolated salt-and-pepper noise.
  3. Separate permanent water (JRC or MERIT `wat`) from flood water.
  4. Vectorize flood mask to GeoDataFrame (`EPSG:4326`), compute exact polygon
     areas in metric UTM CRS, and filter polygons smaller than `min_area_ha`.
  5. Rasterize back the area-filtered polygons so raster and vector exports match 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from scipy.ndimage import median_filter
from shapely.geometry import shape


@dataclass
class PostprocResult:
    """Aligned post-processed masks and vectorized flood polygons."""

    raw_water_mask: np.ndarray        # (H, W) uint8: all detected water (after median)
    perm_water_mask: np.ndarray       # (H, W) uint8: permanent water mask
    flood_mask: np.ndarray            # (H, W) uint8: flood-only after perm water & min_area
    flood_gdf: gpd.GeoDataFrame       # EPSG:4326 polygons with `area_ha` column
    perm_water_gdf: gpd.GeoDataFrame  # EPSG:4326 permanent water polygons with `area_ha`


def _vectorize_binary_mask(
    mask: np.ndarray,
    transform: Any,
    crs: Any,
    min_area_ha: float = 0.0,
) -> gpd.GeoDataFrame:
    """Convert binary uint8 mask to EPSG:4326 GeoDataFrame with metric `area_ha`."""
    mask_u8 = (mask > 0).astype(np.uint8)
    geoms = [
        shape(geom)
        for geom, val in shapes(mask_u8, mask=(mask_u8 == 1), transform=transform)
        if val == 1
    ]
    if not geoms:
        return gpd.GeoDataFrame({"area_ha": pd.Series(dtype="float64")}, geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Compute exact area in metric UTM CRS derived from polygon centroids
    utm_crs = gdf.estimate_utm_crs()
    areas_m2 = gdf.to_crs(utm_crs).geometry.area
    gdf["area_ha"] = (areas_m2 / 10_000.0).astype(float)

    if min_area_ha > 0:
        gdf = gdf[gdf["area_ha"] >= float(min_area_ha)].reset_index(drop=True)
    return gdf


def run_postprocessing(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    transform: Any,
    crs: Any,
    perm_water: np.ndarray | None = None,
    median_size: int = 3,
    min_area_ha: float = 0.1,
    remove_perm_water: bool = True,
) -> PostprocResult:
    """Apply standardized post-processing to any model's binary prediction mask."""
    clean = ((binary_mask > 0) & valid_mask).astype(np.uint8)
    if median_size and median_size > 1:
        clean = median_filter(clean, size=median_size, mode="nearest").astype(np.uint8)
        clean = (clean & valid_mask.astype(np.uint8)).astype(np.uint8)

    if perm_water is not None:
        perm_u8 = ((perm_water > 0) & valid_mask).astype(np.uint8)
    else:
        perm_u8 = np.zeros_like(clean, dtype=np.uint8)

    flood_candidate = (clean & (1 - perm_u8)) if remove_perm_water else clean.copy()

    flood_gdf = _vectorize_binary_mask(flood_candidate, transform, crs, min_area_ha=min_area_ha)
    perm_gdf = _vectorize_binary_mask(perm_u8, transform, crs, min_area_ha=0.05)

    # Re-rasterize filtered polygons so raster and vector outputs are 100% consistent
    if len(flood_gdf) > 0:
        flood_reproj = flood_gdf.to_crs(crs) if crs is not None else flood_gdf
        flood_filtered = rasterize(
            [(geom, 1) for geom in flood_reproj.geometry],
            out_shape=clean.shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    else:
        flood_filtered = np.zeros_like(clean, dtype=np.uint8)

    return PostprocResult(
        raw_water_mask=clean,
        perm_water_mask=perm_u8,
        flood_mask=flood_filtered,
        flood_gdf=flood_gdf,
        perm_water_gdf=perm_gdf,
    )


def export_raster_geotiff(
    flood_mask: np.ndarray,
    proba_map: np.ndarray,
    profile: dict[str, Any],
    out_path: Path | str,
) -> Path:
    """Write a 2-band compressed GeoTIFF: band 1 = flood_mask (0/1), band 2 = water_proba."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_prof = profile.copy()
    out_prof.update(
        driver="GTiff",
        count=2,
        dtype="float32",
        compress="deflate",
        nodata=None,
    )
    with rasterio.open(out_path, "w", **out_prof) as dst:
        dst.write(flood_mask.astype(np.float32), 1)
        dst.write(proba_map.astype(np.float32), 2)
        dst.descriptions = ("flood_mask", "water_proba")
    return out_path


def export_vector(
    flood_gdf: gpd.GeoDataFrame,
    out_path: Path | str,
) -> Path:
    """Export flood polygons to GeoPackage (`.gpkg`) or GeoJSON (`.geojson`)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    driver = "GPKG" if out_path.suffix.lower() == ".gpkg" else "GeoJSON"
    if len(flood_gdf) == 0:
        # Write valid empty FeatureCollection / layer
        empty = gpd.GeoDataFrame({"area_ha": pd.Series(dtype="float64")}, geometry=[], crs="EPSG:4326")
        empty.to_file(out_path, driver=driver)
    else:
        flood_gdf.to_file(out_path, driver=driver)
    return out_path
