"""Unit tests for core.metrics ensuring -1 NoData handling and global micro-IoU."""

import numpy as np
import pytest

from core.metrics import ConfusionStats, compute_binary_metrics, find_best_threshold


def test_ignore_minus_one_pixels():
    y_true = np.array([-1, -1, 1, 1, 0, 0], dtype=np.int16)
    y_pred = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)

    m = compute_binary_metrics(y_true, y_pred)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["ignored_pixels"] == 2
    assert m["valid_pixels"] == 4
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_valid_mask_excludes_invalid_inputs():
    y_true = np.array([1, 1, 0, 0], dtype=np.int16)
    y_pred = np.array([0, 1, 1, 0], dtype=np.uint8)
    valid_mask = np.array([False, True, False, True])

    m = compute_binary_metrics(y_true, y_pred, valid_mask=valid_mask)
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["ignored_pixels"] == 2
    assert m["iou"] == pytest.approx(1.0)


def test_global_micro_iou_across_chips():
    stats = ConfusionStats()
    # Chip 1: 1 water pixel missed (chip IoU = 0.0)
    stats.update(np.array([1, 0]), np.array([0, 0]))
    # Chip 2: 9 water pixels all hit (chip IoU = 1.0)
    stats.update(np.ones(9, dtype=np.int16), np.ones(9, dtype=np.uint8))

    # Global micro-IoU is 9 / (9 + 0 + 1) = 0.9, NOT mean(0.0, 1.0) = 0.5
    assert stats.iou == pytest.approx(0.9)


def test_find_best_threshold_ignores_nodata():
    y_true = np.array([-1, 0, 0, 1, 1], dtype=np.int16)
    y_prob = np.array([0.99, 0.20, 0.35, 0.60, 0.85], dtype=np.float32)

    thr, m = find_best_threshold(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])
    assert thr == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(1.0)
