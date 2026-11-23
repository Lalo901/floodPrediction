"""Shared flood segmentation metrics (Flood IoU, F1, Precision, Recall).

All metrics strictly exclude NoData pixels (`label == ignore_index`, default `-1`)
and compute micro-averaged statistics by accumulating TP, FP, FN, TN across chips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

IGNORE_INDEX = -1


@dataclass
class ConfusionStats:
    """Accumulated binary confusion counts over valid pixels."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    ignored: int = 0

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        valid_mask: np.ndarray | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        """Accumulate counts from a single chip or batch.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth integer labels with values in {ignore_index, 0, 1}.
        y_pred : np.ndarray
            Binary predictions with values in {0, 1} (or boolean).
        valid_mask : np.ndarray | None
            Optional boolean mask of valid input pixels (e.g. non-NaN S1/S2).
            Pixels where valid_mask is False are excluded from metrics.
        ignore_index : int
            Label value to ignore (default: -1 for Sen1Floods11).
        """
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

        mask = yt != ignore_index
        if valid_mask is not None:
            vm = np.asarray(valid_mask, dtype=bool)
            if vm.shape != yt.shape:
                raise ValueError(f"Shape mismatch: valid_mask {vm.shape} vs y_true {yt.shape}")
            mask = mask & vm

        self.ignored += int(yt.size - np.count_nonzero(mask))
        if not np.any(mask):
            return

        yt_v = yt[mask] == 1
        yp_v = yp[mask].astype(bool)

        self.tp += int(np.count_nonzero(yt_v & yp_v))
        self.fp += int(np.count_nonzero((~yt_v) & yp_v))
        self.fn += int(np.count_nonzero(yt_v & (~yp_v)))
        self.tn += int(np.count_nonzero((~yt_v) & (~yp_v)))

    def merge(self, other: "ConfusionStats") -> "ConfusionStats":
        """Return a new ConfusionStats combining self and other."""
        return ConfusionStats(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
            ignored=self.ignored + other.ignored,
        )

    @property
    def valid_pixels(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return float((2 * self.tp) / denom) if denom > 0 else 0.0

    @property
    def gt_water_ratio(self) -> float:
        return float((self.tp + self.fn) / self.valid_pixels) if self.valid_pixels > 0 else 0.0

    @property
    def pred_water_ratio(self) -> float:
        return float((self.tp + self.fp) / self.valid_pixels) if self.valid_pixels > 0 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "valid_pixels": self.valid_pixels,
            "ignored_pixels": self.ignored,
            "gt_water_pct": self.gt_water_ratio * 100.0,
            "pred_water_pct": self.pred_water_ratio * 100.0,
        }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, float | int]:
    """Compute Flood IoU, F1, Precision, Recall on a single array pair."""
    stats = ConfusionStats()
    stats.update(y_true, y_pred, valid_mask=valid_mask, ignore_index=ignore_index)
    return stats.to_dict()


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float] | None = None,
    valid_mask: np.ndarray | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, dict[str, float | int]]:
    """Grid-search probability threshold maximizing Flood IoU on validation data."""
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 33)

    best_thr = 0.5
    best_metrics: dict[str, float | int] = {"iou": -1.0}
    for thr in thresholds:
        metrics = compute_binary_metrics(
            y_true,
            y_prob >= float(thr),
            valid_mask=valid_mask,
            ignore_index=ignore_index,
        )
        if metrics["iou"] > best_metrics["iou"]:
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics
