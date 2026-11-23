"""Register exported DeepLabV3+ ONNX models into SQLite (`models` table) and verify inference.

Usage:
    .venv/bin/python -m training.register_onnx models/deeplabv3plus_A.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.data import load_offline_scene
from core.inference import FloodModel
from core.metrics import compute_binary_metrics
from core.registry import register_model


def register_onnx_model(onnx_path: Path, verify_chip: str = "Bolivia_103757") -> None:
    meta_path = onnx_path.with_suffix(".meta.json")
    metrics_path = onnx_path.parent / f"metrics_{onnx_path.stem}.json"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")

    model = FloodModel.load(onnx_path)

    # Smoke-test inference on CPU with a real chip
    scene = load_offline_scene(verify_chip)
    t0 = time.time()
    _, mask, valid_mask = model.predict(scene.s1, terrain=scene.terrain, s2=scene.s2)
    dt = time.time() - t0
    m_chip = compute_binary_metrics(scene.label, mask, valid_mask=valid_mask)
    print(
        f"CPU smoke test ({verify_chip}): {dt:.2f}s | IoU={m_chip['iou']:.4f} | F1={m_chip['f1']:.4f}"
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    val_m = metrics.get("valid", {})
    test_m = metrics.get("test", {})
    bol_m = metrics.get("bolivia", {})

    row_id = register_model(
        name=model.metadata.name,
        family=model.metadata.family,
        config=model.metadata.config,
        artifact_path=onnx_path,
        feature_names=model.metadata.feature_names,
        threshold=model.metadata.threshold,
        hyperparams={"encoder": metrics.get("encoder", "resnet34"), "tile_size": model.metadata.tile_size},
        val_iou=val_m.get("iou"),
        test_iou=test_m.get("iou"),
        test_f1=test_m.get("f1"),
        test_precision=test_m.get("precision"),
        test_recall=test_m.get("recall"),
        bolivia_iou=bol_m.get("iou"),
        bolivia_f1=bol_m.get("f1"),
    )
    print(f"Registered `{model.metadata.name}` in SQLite (id={row_id}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and register ONNX DeepLabV3+ in SQLite")
    parser.add_argument("onnx_files", nargs="+", type=Path)
    args = parser.parse_args()
    for p in args.onnx_files:
        register_onnx_model(p)


if __name__ == "__main__":
    main()
