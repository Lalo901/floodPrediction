"""EDA over the 446 Sen1Floods11 HandLabeled chips + official S1 Otsu baseline.

Usage:
    uv run python -m training.eda_and_baseline
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

from core.metrics import ConfusionStats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAND_DIR = PROJECT_ROOT / "dataset" / "data" / "flood_events" / "HandLabeled"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits" / "flood_handlabeled"

SPLIT_FILES = {
    "train": SPLITS_DIR / "flood_train_data.csv",
    "valid": SPLITS_DIR / "flood_valid_data.csv",
    "test": SPLITS_DIR / "flood_test_data.csv",
    "bolivia": SPLITS_DIR / "flood_bolivia_data.csv",
}


def load_split_pairs(csv_path: Path) -> list[tuple[str, str]]:
    """Read (s1_filename, label_filename) pairs from a Sen1Floods11 split CSV."""
    pairs: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                pairs.append((row[0].strip(), row[1].strip()))
    return pairs


def chip_event_name(filename: str) -> str:
    """Extract event prefix (e.g. 'Bolivia', 'Sri-Lanka') from chip filename."""
    return filename.split("_")[0]


def run_eda_and_otsu() -> dict:
    summary: dict = {"splits": {}, "events": {}, "integrity": {}}
    missing_files: list[str] = []

    event_otsu: dict[str, ConfusionStats] = defaultdict(ConfusionStats)
    event_meta: dict[str, dict[str, int]] = defaultdict(
        lambda: {"chips": 0, "s1_nan_chips": 0, "s2_nan_chips": 0, "total_px": 0, "minus1_px": 0}
    )

    for split_name, csv_path in SPLIT_FILES.items():
        pairs = load_split_pairs(csv_path)
        split_otsu = ConfusionStats()
        split_otsu_s1_valid = ConfusionStats()
        s1_nan_chips = 0
        s2_nan_chips = 0
        s1_nan_pixels = 0
        total_pixels = 0
        minus1_pixels = 0
        water_pixels = 0
        nonwater_pixels = 0

        for s1_name, label_name in pairs:
            stem = s1_name.replace("_S1Hand.tif", "")
            event = chip_event_name(s1_name)

            s1_path = HAND_DIR / "S1Hand" / s1_name
            s2_path = HAND_DIR / "S2Hand" / f"{stem}_S2Hand.tif"
            label_path = HAND_DIR / "LabelHand" / label_name
            otsu_path = HAND_DIR / "S1OtsuLabelHand" / f"{stem}_S1OtsuLabelHand.tif"
            jrc_path = HAND_DIR / "JRCWaterHand" / f"{stem}_JRCWaterHand.tif"

            for p in (s1_path, s2_path, label_path, otsu_path, jrc_path):
                if not p.exists():
                    missing_files.append(str(p))

            with rasterio.open(s1_path) as src:
                s1 = src.read()  # (2, H, W)
            with rasterio.open(s2_path) as src:
                s2 = src.read(1)  # quick check band 1 for NaNs/shape
            with rasterio.open(label_path) as src:
                label = src.read(1)
            with rasterio.open(otsu_path) as src:
                otsu = src.read(1)

            s1_valid = np.all(np.isfinite(s1), axis=0)
            n_s1_nan = int(s1_valid.size - np.count_nonzero(s1_valid))
            if n_s1_nan > 0:
                s1_nan_chips += 1
                s1_nan_pixels += n_s1_nan
                event_meta[event]["s1_nan_chips"] += 1

            if np.isnan(s2).any():
                s2_nan_chips += 1
                event_meta[event]["s2_nan_chips"] += 1

            n_tot = int(label.size)
            n_m1 = int(np.count_nonzero(label == -1))
            n_w = int(np.count_nonzero(label == 1))
            n_nw = int(np.count_nonzero(label == 0))

            total_pixels += n_tot
            minus1_pixels += n_m1
            water_pixels += n_w
            nonwater_pixels += n_nw

            event_meta[event]["chips"] += 1
            event_meta[event]["total_px"] += n_tot
            event_meta[event]["minus1_px"] += n_m1

            # Otsu mask in Sen1Floods11 can contain -1 or 0/1; treat == 1 as water
            otsu_bin = otsu == 1
            split_otsu.update(label, otsu_bin)
            split_otsu_s1_valid.update(label, otsu_bin, valid_mask=s1_valid)
            event_otsu[event].update(label, otsu_bin)

        valid_label_px = water_pixels + nonwater_pixels
        summary["splits"][split_name] = {
            "chips": len(pairs),
            "s1_nan_chips": s1_nan_chips,
            "s2_nan_chips": s2_nan_chips,
            "s1_nan_pct": round(100.0 * s1_nan_pixels / max(total_pixels, 1), 2),
            "label_minus1_pct": round(100.0 * minus1_pixels / max(total_pixels, 1), 2),
            "water_pct_of_valid": round(100.0 * water_pixels / max(valid_label_px, 1), 2),
            "otsu_metrics": split_otsu.to_dict(),
            "otsu_metrics_s1_valid_only": split_otsu_s1_valid.to_dict(),
        }

    for event in sorted(event_otsu.keys()):
        meta = event_meta[event]
        m = event_otsu[event].to_dict()
        summary["events"][event] = {
            "chips": meta["chips"],
            "s1_nan_chips": meta["s1_nan_chips"],
            "label_minus1_pct": round(100.0 * meta["minus1_px"] / max(meta["total_px"], 1), 2),
            "water_pct_of_valid": round(m["gt_water_pct"], 2),
            "otsu_iou": round(m["iou"], 4),
            "otsu_f1": round(m["f1"], 4),
            "otsu_precision": round(m["precision"], 4),
            "otsu_recall": round(m["recall"], 4),
        }

    summary["integrity"] = {
        "missing_files_count": len(missing_files),
        "missing_files_sample": missing_files[:10],
    }
    return summary


def print_report(summary: dict) -> None:
    print("\n=== INTEGRIDAD DEL DATASET (446 HandLabeled) ===")
    print(f"Archivos faltantes: {summary['integrity']['missing_files_count']}")

    print("\n=== RESUMEN POR SPLIT + BASELINE OTSU (ignorando label == -1) ===")
    header = (
        f"{'Split':<10} {'Chips':>5} {'S1_NaN%':>8} {'Lbl_-1%':>8} "
        f"{'Agua%':>7} {'Otsu IoU':>9} {'Otsu F1':>8} {'Prec':>7} {'Rec':>7}"
    )
    print(header)
    print("-" * len(header))
    for split, d in summary["splits"].items():
        m = d["otsu_metrics"]
        print(
            f"{split:<10} {d['chips']:>5} {d['s1_nan_pct']:>7.2f}% {d['label_minus1_pct']:>7.2f}% "
            f"{d['water_pct_of_valid']:>6.2f}% {m['iou']:>9.4f} {m['f1']:>8.4f} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f}"
        )

    print("\n=== DESGLOSE POR EVENTO (TODOS LOS SPLITS, OTSU OFICIAL) ===")
    ev_header = (
        f"{'Evento':<12} {'Chips':>5} {'S1NaNCh':>8} {'Lbl_-1%':>8} "
        f"{'Agua%':>7} {'Otsu IoU':>9} {'Otsu F1':>8} {'Prec':>7} {'Rec':>7}"
    )
    print(ev_header)
    print("-" * len(ev_header))
    for ev, d in summary["events"].items():
        print(
            f"{ev:<12} {d['chips']:>5} {d['s1_nan_chips']:>8} {d['label_minus1_pct']:>7.2f}% "
            f"{d['water_pct_of_valid']:>6.2f}% {d['otsu_iou']:>9.4f} {d['otsu_f1']:>8.4f} "
            f"{d['otsu_precision']:>7.4f} {d['otsu_recall']:>7.4f}"
        )


if __name__ == "__main__":
    report = run_eda_and_otsu()
    print_report(report)
    out_path = PROJECT_ROOT / "data" / "eda_otsu_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResumen guardado en: {out_path}")
