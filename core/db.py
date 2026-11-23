"""SQLite database initialization and event seeding for Flood Mapper MVP."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "flood.db"
DEFAULT_METADATA_GEOJSON = PROJECT_ROOT / "dataset" / "Sen1Floods11_Metadata.geojson"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    family TEXT NOT NULL,                 -- lightgbm | rf | xgboost | deeplabv3plus
    config TEXT NOT NULL,                 -- A | B | S1_only
    artifact_path TEXT NOT NULL,
    feature_names_json TEXT NOT NULL,
    threshold REAL NOT NULL DEFAULT 0.5,
    hyperparams_json TEXT,
    val_iou REAL,
    test_iou REAL,
    test_f1 REAL,
    test_precision REAL,
    test_recall REAL,
    bolivia_iou REAL,
    bolivia_f1 REAL,
    size_mb REAL,
    trained_at TEXT NOT NULL DEFAULT (datetime('now')),
    data_version TEXT NOT NULL DEFAULT 'sen1floods11_handlabeled'
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    iso_cc TEXT,
    s1_date TEXT,
    s2_date TEXT,
    orbit TEXT,
    bbox_json TEXT NOT NULL,              -- [min_lon, min_lat, max_lon, max_lat]
    vh_thresh REAL,
    source TEXT NOT NULL DEFAULT 'sen1floods11'
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES events(id),
    polygon_geojson TEXT NOT NULL,
    s1_date TEXT,
    s2_date TEXT,
    cloud_pct REAL,
    s1_s2_lag_days INTEGER,
    model_id INTEGER REFERENCES models(id),
    flooded_ha REAL,
    perm_water_ha REAL,
    live_iou REAL,
    raster_path TEXT,
    vector_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER REFERENCES analyses(id),
    tool_name TEXT NOT NULL,
    args_json TEXT,
    result_summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_models_family_config ON models(family, config);
CREATE INDEX IF NOT EXISTS idx_analyses_model ON analyses(model_id);
CREATE INDEX IF NOT EXISTS idx_analyses_event ON analyses(event_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_analysis ON tool_calls(analysis_id);
"""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row_factory enabled and foreign keys on."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Create the SQLite database and all required tables if they do not exist."""
    db_path = Path(db_path)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    return db_path


def _bbox_from_geometry(geometry: dict[str, Any]) -> list[float]:
    """Compute [min_lon, min_lat, max_lon, max_lat] from a GeoJSON Polygon/MultiPolygon."""
    coords: list[tuple[float, float]] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
            coords.append((float(obj[0]), float(obj[1])))
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(geometry.get("coordinates", []))
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def _normalize_date(raw: str | None) -> str | None:
    """Convert YYYY/MM/DD dates from Sen1Floods11 metadata into ISO YYYY-MM-DD."""
    if not raw:
        return None
    return raw.replace("/", "-")


def seed_events_from_geojson(
    geojson_path: Path | str = DEFAULT_METADATA_GEOJSON,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Insert or update Sen1Floods11 events from the metadata GeoJSON.

    Returns the number of processed events.
    """
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"Metadata GeoJSON not found: {geojson_path}")

    init_db(db_path)
    with geojson_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        bbox = _bbox_from_geometry(feat.get("geometry", {}))
        rows.append(
            (
                props["location"],
                props.get("ISO_CC"),
                _normalize_date(props.get("s1_date")),
                _normalize_date(props.get("s2_date")),
                props.get("orbit"),
                json.dumps(bbox),
                props.get("VH_thresh"),
                "sen1floods11",
            )
        )

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO events (name, iso_cc, s1_date, s2_date, orbit, bbox_json, vh_thresh, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                iso_cc = excluded.iso_cc,
                s1_date = excluded.s1_date,
                s2_date = excluded.s2_date,
                orbit = excluded.orbit,
                bbox_json = excluded.bbox_json,
                vh_thresh = excluded.vh_thresh
            """,
            rows,
        )
    return len(rows)


def list_events(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Return all registered events ordered by name."""
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT * FROM events ORDER BY name;")
        return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    init_db()
    if DEFAULT_METADATA_GEOJSON.exists():
        count = seed_events_from_geojson()
        print(f"DB initialized at {DEFAULT_DB_PATH} with {count} Sen1Floods11 events.")
    else:
        print(f"DB initialized at {DEFAULT_DB_PATH} (metadata geojson not present).")
