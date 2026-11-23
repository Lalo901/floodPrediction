"""Generic Optuna training & evaluation script for tabular flood models.

Supports:
  - Families: `lightgbm`, `rf` (regularized < 50 MB), `xgboost`
  - Configs : `S1_only` (8 features), `A` (11 features), `B` (21 features)

Discipline enforced (Agent/spec.md §7):
  1. Uses ONLY the 446 `HandLabeled` chips with official splits.
  2. Strictly ignores `label == -1` and non-finite input pixels (`~valid_mask`).
  3. Subsamples ONLY `train` (single imbalance strategy: controlled neg:pos ratio,
     no stacked `scale_pos_weight` / `class_weight`).
  4. Tunes hyperparameters & probability threshold ONLY on `valid` maximizing Flood IoU.
  5. Evaluates once on `test` (90 chips) and `bolivia` (15 chips) over 100% of valid pixels.
  6. Saves self-contained `FloodModel` artifact and logs metrics to SQLite (`models` table).

Usage:
    .venv/bin/python -m training.train_trees --family lightgbm --config A --trials 20
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import rasterio
from sklearn.ensemble import RandomForestClassifier

from core.features import ConfigName, FEATURE_NAMES_BY_CONFIG, build_feature_stack
from core.inference import FloodModel, ModelMetadata
from core.metrics import ConfusionStats, find_best_threshold
from core.registry import register_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAND_DIR = PROJECT_ROOT / "dataset" / "data" / "flood_events" / "HandLabeled"
TERRAIN_DIR = PROJECT_ROOT / "dataset" / "derived" / "TerrainHand"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits" / "flood_handlabeled"
MODELS_DIR = PROJECT_ROOT / "models"

SPLIT_CSVS = {
    "train": SPLITS_DIR / "flood_train_data.csv",
    "valid": SPLITS_DIR / "flood_valid_data.csv",
    "test": SPLITS_DIR / "flood_test_data.csv",
    "bolivia": SPLITS_DIR / "flood_bolivia_data.csv",
}


def load_split_stems(split: str) -> list[str]:
    csv_path = SPLIT_CSVS[split]
    stems: list[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                stems.append(row[0].strip().replace("_S1Hand.tif", ""))
    return stems


def load_chip_arrays(
    stem: str,
    config: ConfigName,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    with rasterio.open(HAND_DIR / "S1Hand" / f"{stem}_S1Hand.tif") as src:
        s1 = src.read()
    with rasterio.open(HAND_DIR / "LabelHand" / f"{stem}_LabelHand.tif") as src:
        label = src.read(1)

    terrain = None
    if config in ("A", "B"):
        t_path = TERRAIN_DIR / f"{stem}_TerrainHand.tif"
        if not t_path.exists():
            raise FileNotFoundError(
                f"Missing terrain chip: {t_path}. Run `training.extract_terrain` first."
            )
        with rasterio.open(t_path) as src:
            terrain = src.read()

    s2 = None
    if config == "B":
        with rasterio.open(HAND_DIR / "S2Hand" / f"{stem}_S2Hand.tif") as src:
            s2 = src.read()

    return s1, terrain, s2, label


def build_train_sample(
    stems: list[str],
    config: ConfigName,
    max_pos_per_chip: int = 800,
    neg_pos_ratio: float = 3.0,
    min_neg_per_chip: int = 600,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stratified subsampled training matrix strictly from `train` chips."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for stem in stems:
        s1, terrain, s2, label = load_chip_arrays(stem, config)
        stack, _, valid_mask = build_feature_stack(config, s1, terrain=terrain, s2=s2)
        usable = valid_mask & (label != -1)
        if not np.any(usable):
            continue

        pos_idx = np.flatnonzero((usable & (label == 1)).reshape(-1))
        neg_idx = np.flatnonzero((usable & (label == 0)).reshape(-1))

        if pos_idx.size > max_pos_per_chip:
            pos_idx = rng.choice(pos_idx, size=max_pos_per_chip, replace=False)

        target_neg = max(min_neg_per_chip, int(pos_idx.size * neg_pos_ratio))
        if neg_idx.size > target_neg:
            neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)

        chosen = np.concatenate([pos_idx, neg_idx])
        if chosen.size == 0:
            continue

        x_flat = np.moveaxis(stack, 0, -1).reshape(-1, stack.shape[0])
        y_flat = label.reshape(-1)
        xs.append(x_flat[chosen])
        ys.append(y_flat[chosen].astype(np.uint8))

    X = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.uint8)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def build_valid_tuning_cache(
    stems: list[str],
    config: ConfigName,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed strided cache of `valid` pixels for fast Optuna objective evaluation."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for stem in stems:
        s1, terrain, s2, label = load_chip_arrays(stem, config)
        stack, _, valid_mask = build_feature_stack(config, s1, terrain=terrain, s2=s2)
        usable = (valid_mask & (label != -1)).reshape(-1)
        idx = np.flatnonzero(usable)[::stride]
        if idx.size == 0:
            continue
        x_flat = np.moveaxis(stack, 0, -1).reshape(-1, stack.shape[0])
        xs.append(x_flat[idx])
        ys.append(label.reshape(-1)[idx].astype(np.uint8))

    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.uint8)


def create_estimator(family: str, params: dict[str, Any], seed: int = 42):
    if family == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
    if family == "rf":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    raise ValueError(f"Unsupported family: {family}")


def suggest_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_int("max_depth", 5, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    if family == "rf":
        # Regularized bounds keep compressed joblib strictly < 50 MB for HF Space CPU
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 160, step=20),
            "max_depth": trial.suggest_int("max_depth", 10, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 60),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5]),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(f"Unsupported family: {family}")


def calibrate_threshold_full_valid(
    flood_model: FloodModel,
    valid_stems: list[str],
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate 100% of valid pixels in streaming mode to find optimal Flood IoU threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.85, 27)
    stats_by_thr = {float(t): ConfusionStats() for t in thresholds}

    for stem in valid_stems:
        s1, terrain, s2, label = load_chip_arrays(stem, flood_model.metadata.config)
        proba, _, valid_mask = flood_model.predict(s1, terrain=terrain, s2=s2)
        for t in thresholds:
            stats_by_thr[float(t)].update(label, proba >= t, valid_mask=valid_mask)

    best_thr = 0.5
    best_stats = ConfusionStats()
    for t, st in stats_by_thr.items():
        if st.iou > best_stats.iou:
            best_thr = t
            best_stats = st

    return round(best_thr, 4), best_stats.to_dict()


def evaluate_split_streaming(
    flood_model: FloodModel,
    stems: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate 100% of valid pixels on a split, globally and per event."""
    global_stats = ConfusionStats()
    per_event: dict[str, ConfusionStats] = defaultdict(ConfusionStats)

    for stem in stems:
        event = stem.split("_")[0]
        s1, terrain, s2, label = load_chip_arrays(stem, flood_model.metadata.config)
        _, mask, valid_mask = flood_model.predict(s1, terrain=terrain, s2=s2)
        global_stats.update(label, mask, valid_mask=valid_mask)
        per_event[event].update(label, mask, valid_mask=valid_mask)

    event_dicts = {ev: st.to_dict() for ev, st in sorted(per_event.items())}
    return global_stats.to_dict(), event_dicts


def train_and_evaluate(
    family: str,
    config: ConfigName,
    n_trials: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.time()
    model_name = f"{family}_{config}"
    print(f"\n{'=' * 70}")
    print(f"Entrenando {model_name} | trials={n_trials} | features={len(FEATURE_NAMES_BY_CONFIG[config])}")
    print(f"{'=' * 70}")

    train_stems = load_split_stems("train")
    valid_stems = load_split_stems("valid")
    test_stems = load_split_stems("test")
    bolivia_stems = load_split_stems("bolivia")

    print("Construyendo muestra estratificada de train y caché de valid...")
    X_train, y_train = build_train_sample(train_stems, config, seed=seed)
    X_val_tune, y_val_tune = build_valid_tuning_cache(valid_stems, config, stride=4)
    print(
        f"  X_train={X_train.shape} (agua={y_train.mean()*100:.1f}%) | "
        f"X_val_tune={X_val_tune.shape} (agua={y_val_tune.mean()*100:.1f}%)"
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, family)
        clf = create_estimator(family, params, seed=seed)
        clf.fit(X_train, y_train)
        val_prob = clf.predict_proba(X_val_tune)[:, 1]
        _, best_m = find_best_threshold(
            y_val_tune, val_prob, thresholds=np.linspace(0.25, 0.80, 12)
        )
        return float(best_m["iou"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    print(f"Mejor trial Optuna (val strided IoU={study.best_value:.4f}): {best_params}")

    final_clf = create_estimator(family, best_params, seed=seed)
    final_clf.fit(X_train, y_train)

    meta = ModelMetadata(
        name=model_name,
        family=family,
        config=config,
        feature_names=FEATURE_NAMES_BY_CONFIG[config],
        threshold=0.5,
    )
    flood_model = FloodModel(metadata=meta, estimator=final_clf)

    print("Calibrando threshold en 100% de valid...")
    best_thr, val_metrics = calibrate_threshold_full_valid(flood_model, valid_stems)
    flood_model.metadata.threshold = best_thr
    print(
        f"  VALID  | thr={best_thr:.3f} | IoU={val_metrics['iou']:.4f} | "
        f"F1={val_metrics['f1']:.4f} | Prec={val_metrics['precision']:.4f} | Rec={val_metrics['recall']:.4f}"
    )

    print("Evaluando en 100% de test y bolivia (sin tunear)...")
    test_metrics, test_by_event = evaluate_split_streaming(flood_model, test_stems)
    bolivia_metrics, _ = evaluate_split_streaming(flood_model, bolivia_stems)

    print(
        f"  TEST   | IoU={test_metrics['iou']:.4f} | F1={test_metrics['f1']:.4f} | "
        f"Prec={test_metrics['precision']:.4f} | Rec={test_metrics['recall']:.4f}"
    )
    print(
        f"  BOLIVIA| IoU={bolivia_metrics['iou']:.4f} | F1={bolivia_metrics['f1']:.4f} | "
        f"Prec={bolivia_metrics['precision']:.4f} | Rec={bolivia_metrics['recall']:.4f}"
    )

    artifact_path = MODELS_DIR / f"{model_name}.joblib"
    flood_model.save(artifact_path, compress=3)
    size_mb = artifact_path.stat().st_size / (1024 * 1024)
    print(f"Artefacto guardado: {artifact_path} ({size_mb:.2f} MB) en {time.time() - t0:.1f}s")

    register_model(
        name=model_name,
        family=family,
        config=config,
        artifact_path=artifact_path,
        feature_names=meta.feature_names,
        threshold=best_thr,
        hyperparams=best_params,
        val_iou=val_metrics["iou"],
        test_iou=test_metrics["iou"],
        test_f1=test_metrics["f1"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        bolivia_iou=bolivia_metrics["iou"],
        bolivia_f1=bolivia_metrics["f1"],
    )

    report = {
        "name": model_name,
        "family": family,
        "config": config,
        "size_mb": round(size_mb, 3),
        "threshold": best_thr,
        "hyperparams": best_params,
        "valid": val_metrics,
        "test": test_metrics,
        "test_by_event": test_by_event,
        "bolivia": bolivia_metrics,
    }
    report_path = PROJECT_ROOT / "data" / f"metrics_{model_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular flood detection models")
    parser.add_argument("--family", choices=["lightgbm", "rf", "xgboost"], default="lightgbm")
    parser.add_argument("--config", choices=["S1_only", "A", "B"], default="A")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-mvp",
        action="store_true",
        help="Train full MVP tree matrix: LGBM (S1_only, A, B) + RF (A, B)",
    )
    args = parser.parse_args()

    if args.all_mvp:
        runs: list[tuple[str, ConfigName, int]] = [
            ("lightgbm", "S1_only", args.trials),
            ("lightgbm", "A", args.trials),
            ("lightgbm", "B", args.trials),
            ("rf", "A", min(args.trials, 12)),
            ("rf", "B", min(args.trials, 12)),
        ]
        for fam, cfg, n_tr in runs:
            train_and_evaluate(fam, cfg, n_trials=n_tr, seed=args.seed)
    else:
        train_and_evaluate(args.family, args.config, n_trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
