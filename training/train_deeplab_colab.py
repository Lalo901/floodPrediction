"""Google Colab GPU training script for DeepLabV3+ (ResNet34) -> ONNX export.

Designed to produce a `.onnx` + `.meta.json` artifact 100% compatible with
`core.inference.FloodModel` without any training-serving skew.

Colab setup (run once in a Colab cell):
    !pip install -q segmentation-models-pytorch onnx onnxruntime rasterio scipy

Expected dataset layout in Colab / Drive (pack only the 446 HandLabeled + TerrainHand ~1.3 GB):
    DATA_ROOT/
      data/flood_events/HandLabeled/{S1Hand,S2Hand,LabelHand}/
      derived/TerrainHand/
      splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv

Usage:
    python training/train_deeplab_colab.py \
        --data-root /content/drive/MyDrive/sen1floods11_mini \
        --config A \
        --epochs 40 \
        --out-dir /content/drive/MyDrive/flood_models
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow running standalone in Colab when repo is cloned or on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.features import ConfigName, FEATURE_NAMES_BY_CONFIG, build_feature_stack
from core.metrics import ConfusionStats


def load_split_stems(splits_dir: Path, split: str) -> list[str]:
    csv_path = splits_dir / f"flood_{split}_data.csv"
    stems: list[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                stems.append(row[0].strip().replace("_S1Hand.tif", ""))
    return stems


def load_chip_stack(
    data_root: Path,
    stem: str,
    config: ConfigName,
    speckle_median_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_dir = data_root / "data" / "flood_events" / "HandLabeled"
    terrain_dir = data_root / "derived" / "TerrainHand"

    with rasterio.open(hand_dir / "S1Hand" / f"{stem}_S1Hand.tif") as src:
        s1 = src.read()
    with rasterio.open(hand_dir / "LabelHand" / f"{stem}_LabelHand.tif") as src:
        label = src.read(1).astype(np.int16)

    terrain = None
    if config in ("A", "B"):
        with rasterio.open(terrain_dir / f"{stem}_TerrainHand.tif") as src:
            terrain = src.read()

    s2 = None
    if config == "B":
        with rasterio.open(hand_dir / "S2Hand" / f"{stem}_S2Hand.tif") as src:
            s2 = src.read()

    stack, _, valid_mask = build_feature_stack(
        config=config,
        s1=s1,
        terrain=terrain,
        s2=s2,
        speckle_median_size=speckle_median_size,
    )
    return stack, label, valid_mask


def compute_train_channel_stats(
    data_root: Path,
    train_stems: list[str],
    config: ConfigName,
    speckle_median_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std strictly over valid pixels in `train`."""
    n_ch = len(FEATURE_NAMES_BY_CONFIG[config])
    sum_c = np.zeros(n_ch, dtype=np.float64)
    sum_sq_c = np.zeros(n_ch, dtype=np.float64)
    count = 0

    for stem in train_stems:
        stack, label, valid_mask = load_chip_stack(data_root, stem, config, speckle_median_size)
        m = valid_mask & (label != -1)
        if not np.any(m):
            continue
        vals = stack[:, m].astype(np.float64)
        sum_c += vals.sum(axis=1)
        sum_sq_c += (vals * vals).sum(axis=1)
        count += vals.shape[1]

    mean = (sum_c / max(count, 1)).astype(np.float32)
    var = np.maximum((sum_sq_c / max(count, 1)) - mean.astype(np.float64) ** 2, 1e-6)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


class FloodChipDataset(Dataset):
    """Random 256x256 crops with D4 augmentations for train; full 512x512 for eval."""

    def __init__(
        self,
        data_root: Path,
        stems: list[str],
        config: ConfigName,
        mean: np.ndarray,
        std: np.ndarray,
        train: bool = True,
        crop_size: int = 256,
        crops_per_chip: int = 4,
        speckle_median_size: int = 0,
    ) -> None:
        self.data_root = data_root
        self.stems = stems
        self.config = config
        self.mean = mean[:, None, None]
        self.std = np.maximum(std, 1e-6)[:, None, None]
        self.train = train
        self.crop_size = crop_size
        self.crops_per_chip = crops_per_chip if train else 1
        self.speckle_median_size = speckle_median_size

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_chip

    def __getitem__(self, idx: int):
        stem = self.stems[idx % len(self.stems)]
        stack, label, valid_mask = load_chip_stack(
            self.data_root, stem, self.config, self.speckle_median_size
        )
        # Merge invalid input pixels into ignore label -1
        target = np.where(valid_mask, label, -1).astype(np.int16)
        x = (stack - self.mean) / self.std

        if self.train:
            _, h, w = x.shape
            cs = self.crop_size
            # Bias 50% of crops toward water pixels when present
            water_coords = np.argwhere(target == 1)
            if len(water_coords) > 0 and np.random.random() < 0.5:
                cy, cx = water_coords[np.random.randint(len(water_coords))]
                y0 = int(np.clip(cy - cs // 2 + np.random.randint(-32, 33), 0, h - cs))
                x0 = int(np.clip(cx - cs // 2 + np.random.randint(-32, 33), 0, w - cs))
            else:
                y0 = np.random.randint(0, h - cs + 1)
                x0 = np.random.randint(0, w - cs + 1)

            x = x[:, y0 : y0 + cs, x0 : x0 + cs]
            target = target[y0 : y0 + cs, x0 : x0 + cs]

            # D4 geometric augmentation
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2))
                target = np.rot90(target, k, axes=(0, 1))
            if np.random.random() < 0.5:
                x = np.flip(x, axis=2)
                target = np.flip(target, axis=1)

            x = np.ascontiguousarray(x)
            target = np.ascontiguousarray(target)

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(target.astype(np.int64)),
            stem,
        )


class MaskedBCEDiceLoss(nn.Module):
    """Binary cross-entropy + soft Dice loss strictly ignoring `target == -1`."""

    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != -1
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits_v = logits.squeeze(1)[valid]
        targets_v = targets[valid].float()

        bce = F.binary_cross_entropy_with_logits(
            logits_v, targets_v, pos_weight=self.pos_weight.to(logits.device)
        )
        probs_v = torch.sigmoid(logits_v)
        inter = (probs_v * targets_v).sum()
        dice_loss = 1.0 - (2.0 * inter + 1.0) / (probs_v.sum() + targets_v.sum() + 1.0)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> tuple[float, dict, dict]:
    model.eval()
    stats_by_thr = {t: ConfusionStats() for t in thresholds}
    per_event_by_thr: dict[float, dict[str, ConfusionStats]] = {
        t: defaultdict(ConfusionStats) for t in thresholds
    }

    for x, target, stems in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        targets_np = target.numpy()

        for i, stem in enumerate(stems):
            ev = stem.split("_")[0]
            for t in thresholds:
                pred = probs[i] >= t
                stats_by_thr[t].update(targets_np[i], pred)
                per_event_by_thr[t][ev].update(targets_np[i], pred)

    best_t = max(thresholds, key=lambda t: stats_by_thr[t].iou)
    best_global = stats_by_thr[best_t].to_dict()
    best_events = {ev: st.to_dict() for ev, st in sorted(per_event_by_thr[best_t].items())}
    return best_t, best_global, best_events


def export_onnx(model: nn.Module, in_channels: int, out_path: Path, tile_size: int = 256) -> None:
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, in_channels, tile_size, tile_size, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def main() -> None:
    import segmentation_models_pytorch as smp

    parser = argparse.ArgumentParser(description="Train DeepLabV3+ (ResNet34) on Sen1Floods11")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", choices=["A", "B"], default="A")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Config: {args.config} | Encoder: {args.encoder}")

    splits_dir = args.data_root / "splits" / "flood_handlabeled"
    train_stems = load_split_stems(splits_dir, "train")
    valid_stems = load_split_stems(splits_dir, "valid")
    test_stems = load_split_stems(splits_dir, "test")
    bolivia_stems = load_split_stems(splits_dir, "bolivia")

    speckle = 0  # Let CNN convolutions learn speckle filtering without blurring thin channels
    mean, std = compute_train_channel_stats(args.data_root, train_stems, args.config, speckle)

    train_ds = FloodChipDataset(
        args.data_root, train_stems, args.config, mean, std, train=True, speckle_median_size=speckle
    )
    valid_ds = FloodChipDataset(
        args.data_root, valid_stems, args.config, mean, std, train=False, speckle_median_size=speckle
    )
    test_ds = FloodChipDataset(
        args.data_root, test_stems, args.config, mean, std, train=False, speckle_median_size=speckle
    )
    bolivia_ds = FloodChipDataset(
        args.data_root, bolivia_stems, args.config, mean, std, train=False, speckle_median_size=speckle
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)
    bolivia_loader = DataLoader(bolivia_ds, batch_size=4, shuffle=False, num_workers=2)

    in_channels = len(FEATURE_NAMES_BY_CONFIG[args.config])
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
    ).to(device)

    criterion = MaskedBCEDiceLoss(bce_weight=0.5, pos_weight=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"deeplabv3plus_{args.config}"
    ckpt_path = args.out_dir / f"{model_name}_best.pt"
    best_val_iou = -1.0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, target, _ in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        scheduler.step()

        _, val_m, _ = evaluate_loader(model, valid_loader, device, thresholds=[0.5])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={running_loss/len(train_loader):.4f} "
            f"| val_iou@0.5={val_m['iou']:.4f} | val_f1={val_m['f1']:.4f}"
        )
        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            torch.save(model.state_dict(), ckpt_path)

    # Load best checkpoint, calibrate threshold ONLY on valid, evaluate once on test & bolivia
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    thr_grid = [round(float(t), 3) for t in np.linspace(0.25, 0.75, 21)]
    best_thr, val_best, _ = evaluate_loader(model, valid_loader, device, thresholds=thr_grid)
    _, test_m, test_events = evaluate_loader(model, test_loader, device, thresholds=[best_thr])
    _, bolivia_m, _ = evaluate_loader(model, bolivia_loader, device, thresholds=[best_thr])

    print(f"\nBEST VALID (thr={best_thr:.3f}): IoU={val_best['iou']:.4f}, F1={val_best['f1']:.4f}")
    print(f"TEST    : IoU={test_m['iou']:.4f}, F1={test_m['f1']:.4f}, Prec={test_m['precision']:.4f}, Rec={test_m['recall']:.4f}")
    print(f"BOLIVIA : IoU={bolivia_m['iou']:.4f}, F1={bolivia_m['f1']:.4f}, Prec={bolivia_m['precision']:.4f}, Rec={bolivia_m['recall']:.4f}")

    onnx_path = args.out_dir / f"{model_name}.onnx"
    export_onnx(model, in_channels, onnx_path, tile_size=256)

    meta = {
        "name": model_name,
        "family": "deeplabv3plus",
        "config": args.config,
        "feature_names": list(FEATURE_NAMES_BY_CONFIG[args.config]),
        "threshold": best_thr,
        "speckle_median_size": speckle,
        "context_window": 5,
        "norm_mean": [float(v) for v in mean],
        "norm_std": [float(v) for v in std],
        "tile_size": 256,
        "data_version": "sen1floods11_handlabeled",
    }
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics_out = {
        "name": model_name,
        "family": "deeplabv3plus",
        "config": args.config,
        "encoder": args.encoder,
        "threshold": best_thr,
        "train_seconds": round(time.time() - t0, 1),
        "valid": val_best,
        "test": test_m,
        "test_by_event": test_events,
        "bolivia": bolivia_m,
    }
    (args.out_dir / f"metrics_{model_name}.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )
    print(f"Exported ONNX: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
