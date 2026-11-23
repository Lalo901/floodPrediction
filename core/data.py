"""Unified data source loader for Flood Mapper MVP (Offline, Uploaded GeoTIFF, Live GEE).

Every source returns a `SceneBundle` with aligned arrays, rasterio profile,
EPSG:4326 bounds, and mandatory acquisition metadata (dates, orbit, cloud %).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from scipy.ndimage import uniform_filter
from shapely.geometry import shape

from core.db import DEFAULT_DB_PATH, PROJECT_ROOT, list_events

logger = logging.getLogger(__name__)

HAND_DIR = PROJECT_ROOT / "dataset" / "data" / "flood_events" / "HandLabeled"
TERRAIN_DIR = PROJECT_ROOT / "dataset" / "derived" / "TerrainHand"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits" / "flood_handlabeled"
DEMO_STACKS_DIR = PROJECT_ROOT / "data" / "stacks"

EVENT_PREFIX_TO_DB_NAME = {
    "Mekong": "Cambodia",
}

S2_SIX_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
CLOUD_SCORE_THRESHOLD = 0.60


@dataclass
class SceneBundle:
    """Aligned multi-sensor scene ready for FloodModel inference and Folium display."""

    scene_id: str
    source: str  # offline | upload | gee
    event_name: str | None
    s1: np.ndarray  # (2, H, W) float32 VV, VH in dB
    profile: dict[str, Any]
    bounds_wgs84: tuple[float, float, float, float]  # (west, south, east, north)
    event_id: int | None = None
    polygon_geojson: dict[str, Any] | None = None
    s2: np.ndarray | None = None  # (13, H, W) or (6, H, W)
    terrain: np.ndarray | None = None  # (4, H, W) slope, hand, dem_rel, merit_wat
    perm_water: np.ndarray | None = None  # (H, W) uint8
    label: np.ndarray | None = None  # (H, W) int16 {-1, 0, 1}
    s1_date: str | None = None
    s2_date: str | None = None
    orbit: str | None = None
    cloud_pct: float | None = None


def sanitize_scene_id(raw: str) -> str:
    """Sanitize untrusted filenames into safe identifier tokens for output paths."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(raw).stem).strip("_")
    return cleaned[:80] or "scene"


def _bounds_to_wgs84(bounds: Any, crs: Any) -> tuple[float, float, float, float]:
    if crs is not None and str(crs).upper() != "EPSG:4326":
        w, s, e, n = transform_bounds(
            crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
        )
        return (float(w), float(s), float(e), float(n))
    return (float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top))


# ---------------------------------------------------------------------------
# 1. Offline Sen1Floods11 & prepackaged demo stacks
# ---------------------------------------------------------------------------

def list_offline_chips(split: str = "bolivia") -> list[dict[str, str]]:
    """List available offline HandLabeled chips for a split (`bolivia`, `test`, `valid`, `demo`)."""
    if split == "demo" or not HAND_DIR.exists():
        if not DEMO_STACKS_DIR.exists():
            return []
        items = []
        for p in sorted(DEMO_STACKS_DIR.glob("*_S1Hand.tif")):
            stem = p.name.replace("_S1Hand.tif", "")
            items.append({"stem": stem, "event": stem.split("_")[0], "split": "demo"})
        return items

    csv_map = {
        "bolivia": SPLITS_DIR / "flood_bolivia_data.csv",
        "test": SPLITS_DIR / "flood_test_data.csv",
        "valid": SPLITS_DIR / "flood_valid_data.csv",
        "train": SPLITS_DIR / "flood_train_data.csv",
    }
    csv_path = csv_map.get(split)
    if csv_path is None or not csv_path.exists():
        return []

    items = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                stem = row[0].strip().replace("_S1Hand.tif", "")
                items.append({"stem": stem, "event": stem.split("_")[0], "split": split})
    return items


def _find_offline_file(subdir: str, filename: str, derived: bool = False) -> Path | None:
    if derived:
        p = TERRAIN_DIR / filename
        if p.exists():
            return p
    else:
        p = HAND_DIR / subdir / filename
        if p.exists():
            return p
    demo_p = DEMO_STACKS_DIR / filename
    return demo_p if demo_p.exists() else None


def load_offline_scene(stem: str, db_path: Path | str = DEFAULT_DB_PATH) -> SceneBundle:
    """Load an aligned Sen1Floods11 HandLabeled chip (or demo stack) as a SceneBundle."""
    s1_path = _find_offline_file("S1Hand", f"{stem}_S1Hand.tif")
    if s1_path is None:
        raise FileNotFoundError(f"S1Hand chip not found for stem: {stem}")

    with rasterio.open(s1_path) as src:
        s1 = src.read().astype(np.float32)
        profile = src.profile.copy()
        bounds_wgs84 = _bounds_to_wgs84(src.bounds, src.crs)

    s2_path = _find_offline_file("S2Hand", f"{stem}_S2Hand.tif")
    s2 = None
    if s2_path is not None:
        with rasterio.open(s2_path) as src:
            s2 = src.read()

    terrain_path = _find_offline_file("", f"{stem}_TerrainHand.tif", derived=True)
    terrain = None
    if terrain_path is not None:
        with rasterio.open(terrain_path) as src:
            terrain = src.read().astype(np.float32)

    label_path = _find_offline_file("LabelHand", f"{stem}_LabelHand.tif")
    label = None
    if label_path is not None:
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int16)

    # Combine JRC permanent water with MERIT Hydro permanent channels when both exist
    jrc_mask = None
    jrc_path = _find_offline_file("JRCWaterHand", f"{stem}_JRCWaterHand.tif")
    if jrc_path is not None:
        with rasterio.open(jrc_path) as src:
            jrc_mask = src.read(1) == 1

    merit_mask = (terrain[3] > 0.5) if (terrain is not None and terrain.shape[0] >= 4) else None

    if jrc_mask is not None and merit_mask is not None:
        perm_water = (jrc_mask | merit_mask).astype(np.uint8)
    elif jrc_mask is not None:
        perm_water = jrc_mask.astype(np.uint8)
    elif merit_mask is not None:
        perm_water = merit_mask.astype(np.uint8)
    else:
        perm_water = None

    event_prefix = stem.split("_")[0]
    db_event_name = EVENT_PREFIX_TO_DB_NAME.get(event_prefix, event_prefix)
    ev_meta = next((e for e in list_events(db_path) if e["name"] == db_event_name), None)

    return SceneBundle(
        scene_id=stem,
        source="offline",
        event_name=event_prefix,
        event_id=int(ev_meta["id"]) if ev_meta else None,
        s1=s1,
        s2=s2,
        terrain=terrain,
        perm_water=perm_water,
        label=label,
        profile=profile,
        bounds_wgs84=bounds_wgs84,
        s1_date=ev_meta["s1_date"] if ev_meta else None,
        s2_date=ev_meta["s2_date"] if ev_meta else None,
        orbit=ev_meta["orbit"] if ev_meta else None,
        cloud_pct=None,  # Sen1Floods11 S2Hand does not store per-chip cloud %
    )


# ---------------------------------------------------------------------------
# 2. Uploaded GeoTIFF loader
# ---------------------------------------------------------------------------

def load_uploaded_geotiff(
    s1_bytes: bytes,
    filename: str = "uploaded_s1.tif",
    s2_bytes: bytes | None = None,
    fetch_terrain_from_gee: bool = False,
) -> SceneBundle:
    """Load a user-uploaded 2-band Sentinel-1 GeoTIFF (VV, VH in dB) into a SceneBundle."""
    with rasterio.MemoryFile(s1_bytes) as mem, mem.open() as src:
        s1 = src.read().astype(np.float32)
        profile = src.profile.copy()
        bounds_wgs84 = _bounds_to_wgs84(src.bounds, src.crs)

    if s1.shape[0] < 2:
        raise ValueError("El GeoTIFF de Sentinel-1 debe tener al menos 2 bandas (VV y VH en dB).")
    s1 = s1[:2]

    s2 = None
    if s2_bytes is not None:
        with rasterio.MemoryFile(s2_bytes) as mem, mem.open() as src:
            s2 = src.read()

    terrain = None
    perm_water = None
    if fetch_terrain_from_gee and gee_is_configured():
        terrain = fetch_gee_terrain_for_profile(profile)
        perm_water = (terrain[3] > 0.5).astype(np.uint8)

    return SceneBundle(
        scene_id=sanitize_scene_id(filename),
        source="upload",
        event_name="Custom GeoTIFF",
        s1=s1,
        s2=s2,
        terrain=terrain,
        perm_water=perm_water,
        label=None,
        profile=profile,
        bounds_wgs84=bounds_wgs84,
    )


# ---------------------------------------------------------------------------
# 3. Live Google Earth Engine (S1, S2 L1C + Cloud Score+, GLO-30, MERIT HAND)
# ---------------------------------------------------------------------------

_EE_INITIALIZED = False


def init_gee() -> bool:
    """Initialize Earth Engine using service account JSON (Space) or local OAuth credentials."""
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return True
    try:
        import ee
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        project_id = os.getenv("GEE_PROJECT_ID")
        sa_json_str = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        sa_key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

        if sa_json_str:
            info = json.loads(sa_json_str)
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=sa_json_str)
            ee.Initialize(credentials=creds, project=project_id or info.get("project_id"))
        elif sa_key_path and Path(sa_key_path).exists():
            info = json.loads(Path(sa_key_path).read_text(encoding="utf-8"))
            creds = ee.ServiceAccountCredentials(info["client_email"], key_file=sa_key_path)
            ee.Initialize(credentials=creds, project=project_id or info.get("project_id"))
        else:
            if not project_id:
                logger.warning("GEE_PROJECT_ID no configurado; modo GEE en vivo deshabilitado.")
                return False
            ee.Initialize(project=project_id)

        _EE_INITIALIZED = True
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo inicializar Google Earth Engine: %s", exc)
        return False


def gee_is_configured() -> bool:
    return init_gee()


def _compute_dem_rel(
    dem: np.ndarray,
    pixel_res_m: float = 10.0,
    window_meters: float = 510.0,
) -> np.ndarray:
    """Compute local relative elevation over a fixed metric radius (~510 m = 51 px at 10 m)."""
    finite = np.isfinite(dem)
    if not np.any(finite):
        return np.zeros_like(dem, dtype=np.float32)
    fill = float(np.median(dem[finite]))
    dem_clean = np.where(finite, dem, fill).astype(np.float32)
    win_px = max(3, round(window_meters / max(pixel_res_m, 1.0)) | 1)
    return (dem_clean - uniform_filter(dem_clean, size=win_px, mode="reflect")).astype(np.float32)


def _ee_compute_pixels(ee_img: Any, profile: dict[str, Any]) -> np.ndarray:
    import ee

    t = profile["transform"]
    crs_code = (
        profile["crs"].to_string() if hasattr(profile["crs"], "to_string") else str(profile["crs"])
    )
    request = {
        "expression": ee_img,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": int(profile["width"]), "height": int(profile["height"])},
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
        return src.read().astype(np.float32)


def fetch_gee_terrain_for_profile(
    profile: dict[str, Any],
    pixel_res_m: float = 10.0,
) -> np.ndarray:
    """Download `(slope, hand, dem_rel, merit_wat)` aligned to an arbitrary raster profile."""
    import ee

    init_gee()
    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
    dem = dem_col.mosaic().rename("dem")
    slope = dem_col.map(ee.Terrain.slope).mosaic().rename("slope")
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    hand = merit.select("hnd").unmask(0).rename("hand")
    merit_wat = merit.select("wat").unmask(0).gt(0).rename("merit_wat")

    terrain_img = ee.Image.cat([slope, hand, dem, merit_wat]).toFloat()
    raw = _ee_compute_pixels(terrain_img, profile)
    arr = raw.copy()
    arr[2] = _compute_dem_rel(raw[2], pixel_res_m=pixel_res_m)
    return arr


def list_gee_s1_scenes(
    polygon_geojson: dict[str, Any],
    start_date: str,
    end_date: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List available Sentinel-1 IW (VV+VH) scenes over a drawn polygon."""
    import ee

    if not init_gee():
        raise RuntimeError("Google Earth Engine no está autenticado en este entorno.")

    geom = ee.Geometry(polygon_geojson)
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("system:time_start", False)
        .limit(limit)
    )
    info = col.getInfo()
    scenes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        ts_ms = props.get("system:time_start")
        date_str = (
            datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")
            if ts_ms
            else ""
        )
        orbit = props.get("orbitProperties_pass", "")
        key = (date_str, orbit)
        if key in seen:
            continue
        seen.add(key)
        scenes.append({"date": date_str, "orbit": orbit, "scene_id": feat.get("id", "")})
    return scenes


def fetch_gee_scene(
    polygon_geojson: dict[str, Any],
    s1_date: str,
    orbit: str | None = None,
    max_side_km: float = 20.0,
    target_res_m: float = 10.0,
    max_dim_px: int = 1024,
) -> SceneBundle:
    """Fetch polygon-clipped S1 + Terrain + Cloud-Score+ masked S2 L1C (±3d) from GEE."""
    import ee

    if not init_gee():
        raise RuntimeError("Google Earth Engine no está autenticado en este entorno.")

    poly = shape(polygon_geojson)
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    utm = gdf.estimate_utm_crs()
    bounds_m = gdf.to_crs(utm).total_bounds
    width_km = (bounds_m[2] - bounds_m[0]) / 1000.0
    height_km = (bounds_m[3] - bounds_m[1]) / 1000.0
    if width_km > max_side_km or height_km > max_side_km:
        raise ValueError(
            f"El polígono mide {width_km:.1f}×{height_km:.1f} km "
            f"(máximo permitido: {max_side_km:.0f} km por lado)."
        )

    west, south, east, north = poly.bounds
    width_px = int(np.clip(round((width_km * 1000.0) / target_res_m), 64, max_dim_px))
    height_px = int(np.clip(round((height_km * 1000.0) / target_res_m), 64, max_dim_px))
    eff_res_m = max((width_km * 1000.0) / width_px, (height_km * 1000.0) / height_px)

    transform = from_bounds(west, south, east, north, width_px, height_px)
    profile = {
        "driver": "GTiff",
        "width": width_px,
        "height": height_px,
        "count": 2,
        "dtype": "float32",
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": transform,
    }

    # Exact polygon interior mask so pixels outside the drawn shape are excluded from inference & area
    inside_poly = rasterize(
        [(poly, 1)],
        out_shape=(height_px, width_px),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    ee_geom = ee.Geometry(polygon_geojson)
    dt = datetime.strptime(s1_date[:10], "%Y-%m-%d")
    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee_geom)
        .filterDate(s1_date[:10], (dt + timedelta(days=1)).strftime("%Y-%m-%d"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if orbit:
        s1_col = s1_col.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    s1_img = s1_col.select(["VV", "VH"]).mosaic().toFloat()
    s1 = _ee_compute_pixels(s1_img, profile)
    s1[:, ~inside_poly] = np.nan

    terrain = fetch_gee_terrain_for_profile(profile, pixel_res_m=eff_res_m)
    perm_water = ((terrain[3] > 0.5) & inside_poly).astype(np.uint8)

    # Search nearest S2 L1C within ±3 days and mask clouds inside the polygon via Cloud Score+
    s2_start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
    s2_end = (dt + timedelta(days=4)).strftime("%Y-%m-%d")
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(ee_geom)
        .filterDate(s2_start, s2_end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    s2_info = s2_col.limit(1).getInfo()
    s2 = None
    s2_date = None
    cloud_pct = None
    if s2_info.get("features"):
        feat = s2_info["features"][0]
        props = feat.get("properties", {})
        ts_ms = props.get("system:time_start")
        s2_date = (
            datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")
            if ts_ms
            else None
        )
        s2_img = ee.Image(feat["id"])
        cs_col = (
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            .filterBounds(ee_geom)
            .filterDate(s2_date, (datetime.strptime(s2_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
        )
        cs_img = cs_col.select("cs_cdf").mosaic().unmask(1.0).rename("cs_cdf")
        s2_and_cs = ee.Image.cat([s2_img.select(S2_SIX_BANDS), cs_img]).toFloat()
        s2_raw = _ee_compute_pixels(s2_and_cs, profile)
        s2 = s2_raw[:6]
        cs_cdf = s2_raw[6]

        cloudy = cs_cdf < CLOUD_SCORE_THRESHOLD
        poly_px = max(int(inside_poly.sum()), 1)
        cloud_pct = round(100.0 * float((cloudy & inside_poly).sum()) / poly_px, 1)
        s2[:, cloudy | (~inside_poly)] = np.nan

    return SceneBundle(
        scene_id=f"gee_{s1_date[:10]}",
        source="gee",
        event_name="Polígono GEE en vivo",
        polygon_geojson=polygon_geojson,
        s1=s1,
        s2=s2,
        terrain=terrain,
        perm_water=perm_water,
        label=None,
        profile=profile,
        bounds_wgs84=(float(west), float(south), float(east), float(north)),
        s1_date=s1_date[:10],
        s2_date=s2_date,
        orbit=orbit,
        cloud_pct=cloud_pct,
    )
