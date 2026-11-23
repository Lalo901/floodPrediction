"""Model registry backed by SQLite (`models` table) for Flood Mapper MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.db import DEFAULT_DB_PATH, PROJECT_ROOT, get_connection, init_db
from core.inference import FloodModel


def register_model(
    *,
    name: str,
    family: str,
    config: str,
    artifact_path: Path | str,
    feature_names: list[str] | tuple[str, ...],
    threshold: float,
    hyperparams: dict[str, Any] | None = None,
    val_iou: float | None = None,
    test_iou: float | None = None,
    test_f1: float | None = None,
    test_precision: float | None = None,
    test_recall: float | None = None,
    bolivia_iou: float | None = None,
    bolivia_f1: float | None = None,
    data_version: str = "sen1floods11_handlabeled",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Insert or update a trained model record in SQLite and return its row id."""
    init_db(db_path)
    art = Path(artifact_path)
    size_mb = round(art.stat().st_size / (1024 * 1024), 3) if art.exists() else None
    try:
        rel_path = str(art.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        rel_path = str(art)

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO models (
                name, family, config, artifact_path, feature_names_json, threshold,
                hyperparams_json, val_iou, test_iou, test_f1, test_precision, test_recall,
                bolivia_iou, bolivia_f1, size_mb, data_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                family = excluded.family,
                config = excluded.config,
                artifact_path = excluded.artifact_path,
                feature_names_json = excluded.feature_names_json,
                threshold = excluded.threshold,
                hyperparams_json = excluded.hyperparams_json,
                val_iou = excluded.val_iou,
                test_iou = excluded.test_iou,
                test_f1 = excluded.test_f1,
                test_precision = excluded.test_precision,
                test_recall = excluded.test_recall,
                bolivia_iou = excluded.bolivia_iou,
                bolivia_f1 = excluded.bolivia_f1,
                size_mb = excluded.size_mb,
                trained_at = datetime('now'),
                data_version = excluded.data_version
            """,
            (
                name,
                family,
                config,
                rel_path,
                json.dumps(list(feature_names)),
                float(threshold),
                json.dumps(hyperparams or {}),
                val_iou,
                test_iou,
                test_f1,
                test_precision,
                test_recall,
                bolivia_iou,
                bolivia_f1,
                size_mb,
                data_version,
            ),
        )
        row = conn.execute("SELECT id FROM models WHERE name = ?;", (name,)).fetchone()
        return int(row["id"])


def list_models(
    config: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return registered models ordered by test_iou descending."""
    init_db(db_path)
    query = "SELECT * FROM models"
    params: list[Any] = []
    if config is not None:
        query += " WHERE config = ?"
        params.append(config)
    query += " ORDER BY COALESCE(test_iou, -1) DESC, name ASC;"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def load_model(name: str, db_path: Path | str = DEFAULT_DB_PATH) -> FloodModel:
    """Resolve artifact path from SQLite and load the FloodModel."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT artifact_path FROM models WHERE name = ?;", (name,)).fetchone()
    if row is None:
        raise KeyError(f"Model not registered in DB: {name}")
    art_path = Path(row["artifact_path"])
    if not art_path.is_absolute():
        art_path = PROJECT_ROOT / art_path
    return FloodModel.load(art_path)
