"""Pack the minimal dataset + code needed to train DeepLabV3+ in Google Colab.

Creates `data/colab/sen1floods11_mini.zip` (~2 GB) with:
  code/core/{__init__,features,metrics}.py
  code/training/{__init__,train_deeplab_colab}.py
  data/flood_events/HandLabeled/{S1Hand,S2Hand,LabelHand}/   (446 chips each)
  derived/TerrainHand/                                       (446 chips)
  splits/flood_handlabeled/*.csv

WeaklyLabeled is intentionally excluded (Agent/spec.md §3).

Usage:
    .venv/bin/python -m training.pack_colab
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "dataset"
OUT_ZIP = PROJECT_ROOT / "data" / "colab" / "sen1floods11_mini.zip"

CODE_FILES = [
    "core/__init__.py",
    "core/features.py",
    "core/metrics.py",
    "training/__init__.py",
    "training/train_deeplab_colab.py",
]
DATA_DIRS = [
    "data/flood_events/HandLabeled/S1Hand",
    "data/flood_events/HandLabeled/S2Hand",
    "data/flood_events/HandLabeled/LabelHand",
    "derived/TerrainHand",
    "splits/flood_handlabeled",
]


def main() -> None:
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    # GeoTIFFs are already compressed -> ZIP_STORED is much faster with ~same size
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for rel in CODE_FILES:
            zf.write(PROJECT_ROOT / rel, arcname=f"code/{rel}")
            n_files += 1
        for rel_dir in DATA_DIRS:
            for p in sorted((DATASET / rel_dir).glob("*")):
                if p.is_file():
                    zf.write(p, arcname=f"{rel_dir}/{p.name}")
                    n_files += 1

    size_gb = OUT_ZIP.stat().st_size / (1024**3)
    print(f"OK: {OUT_ZIP} | {n_files} archivos | {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
