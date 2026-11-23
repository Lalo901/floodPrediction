"""FloodModel artifact and unified inference engine (trees by pixel batches, DeepLab ONNX by tiles).

Bundles the trained model with its exact feature order, preprocessing parameters,
normalization stats (for DL), and validation-calibrated probability threshold to
eliminate training-serving skew.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.features import ConfigName, build_feature_stack


@dataclass
class ModelMetadata:
    """Contract stored alongside every trained FloodModel."""

    name: str
    family: str  # lightgbm | rf | xgboost | deeplabv3plus
    config: ConfigName  # S1_only | A | B
    feature_names: tuple[str, ...]
    threshold: float = 0.5
    speckle_median_size: int = 3
    context_window: int = 5
    norm_mean: tuple[float, ...] | None = None
    norm_std: tuple[float, ...] | None = None
    tile_size: int = 256
    tile_overlap: int = 32
    data_version: str = "sen1floods11_handlabeled"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        if self.norm_mean is not None:
            d["norm_mean"] = list(self.norm_mean)
        if self.norm_std is not None:
            d["norm_std"] = list(self.norm_std)
        return d


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Compute tile start offsets covering `[0, length)` cleanly with overlap."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _blend_weight_2d(tile: int, overlap: int) -> np.ndarray:
    """Raised-cosine feathering weights to eliminate tile seam artifacts."""
    if overlap <= 0:
        return np.ones((tile, tile), dtype=np.float32)
    ramp = np.ones(tile, dtype=np.float32)
    edge = min(overlap, tile // 2)
    t = np.linspace(0.0, np.pi, edge + 2, dtype=np.float32)[1:-1]
    fade = 0.5 * (1.0 - np.cos(t))
    ramp[:edge] = fade
    ramp[-edge:] = fade[::-1]
    return np.outer(ramp, ramp).astype(np.float32)


class FloodModel:
    """Self-contained inference wrapper for tabular tree models and ONNX DeepLabV3+."""

    def __init__(
        self,
        metadata: ModelMetadata,
        estimator: Any = None,
        onnx_path: Path | str | None = None,
    ) -> None:
        self.metadata = metadata
        self.estimator = estimator
        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        self._ort_session = None

    def save(self, path: Path | str, compress: int = 3) -> Path:
        """Persist a tabular FloodModel to a compressed joblib artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "metadata": self.metadata.to_dict(),
            "estimator": self.estimator,
        }
        joblib.dump(payload, path, compress=compress)
        return path

    @classmethod
    def load(cls, path: Path | str) -> FloodModel:
        """Load either a `.joblib` tabular FloodModel or an `.onnx` model with companion JSON."""
        path = Path(path)
        if path.suffix == ".onnx":
            import json

            meta_path = path.with_suffix(".meta.json")
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing ONNX metadata file: {meta_path}")
            raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = ModelMetadata(
                name=raw_meta["name"],
                family=raw_meta["family"],
                config=raw_meta["config"],
                feature_names=tuple(raw_meta["feature_names"]),
                threshold=float(raw_meta.get("threshold", 0.5)),
                speckle_median_size=int(raw_meta.get("speckle_median_size", 0)),
                context_window=int(raw_meta.get("context_window", 5)),
                norm_mean=tuple(raw_meta["norm_mean"]) if raw_meta.get("norm_mean") else None,
                norm_std=tuple(raw_meta["norm_std"]) if raw_meta.get("norm_std") else None,
                tile_size=int(raw_meta.get("tile_size", 256)),
                tile_overlap=int(raw_meta.get("tile_overlap", 32)),
                data_version=raw_meta.get("data_version", "sen1floods11_handlabeled"),
            )
            return cls(metadata=meta, onnx_path=path)

        payload = joblib.load(path)
        raw_meta = payload["metadata"]
        meta = ModelMetadata(
            name=raw_meta["name"],
            family=raw_meta["family"],
            config=raw_meta["config"],
            feature_names=tuple(raw_meta["feature_names"]),
            threshold=float(raw_meta.get("threshold", 0.5)),
            speckle_median_size=int(raw_meta.get("speckle_median_size", 3)),
            context_window=int(raw_meta.get("context_window", 5)),
            norm_mean=tuple(raw_meta["norm_mean"]) if raw_meta.get("norm_mean") else None,
            norm_std=tuple(raw_meta["norm_std"]) if raw_meta.get("norm_std") else None,
            tile_size=int(raw_meta.get("tile_size", 256)),
            tile_overlap=int(raw_meta.get("tile_overlap", 32)),
            data_version=raw_meta.get("data_version", "sen1floods11_handlabeled"),
        )
        return cls(metadata=meta, estimator=payload["estimator"])

    def _get_ort_session(self):
        if self._ort_session is None:
            import onnxruntime as ort

            if self.onnx_path is None or not self.onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")
            self._ort_session = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )
        return self._ort_session

    def predict_proba_from_stack(
        self,
        stack: np.ndarray,
        feature_names: tuple[str, ...],
        valid_mask: np.ndarray,
        batch_size: int = 131072,
    ) -> np.ndarray:
        """Predict water probability map `(H, W)` from a pre-built feature stack."""
        if tuple(feature_names) != self.metadata.feature_names:
            raise ValueError(
                f"Feature contract mismatch for {self.metadata.name}: "
                f"expected {self.metadata.feature_names}, got {feature_names}"
            )

        _, h, w = stack.shape
        if self.onnx_path is not None:
            return self._predict_onnx_tiled(stack, valid_mask)

        proba_flat = np.zeros(h * w, dtype=np.float32)
        valid_flat = valid_mask.reshape(-1)
        valid_idx = np.flatnonzero(valid_flat)
        if valid_idx.size == 0:
            return proba_flat.reshape(h, w)

        x_all = np.moveaxis(stack, 0, -1).reshape(-1, stack.shape[0])
        for start in range(0, valid_idx.size, batch_size):
            idx_chunk = valid_idx[start : start + batch_size]
            x_chunk = x_all[idx_chunk]
            probs = self.estimator.predict_proba(x_chunk)
            proba_flat[idx_chunk] = probs[:, 1].astype(np.float32)

        return proba_flat.reshape(h, w)

    def _predict_onnx_tiled(self, stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Sliding-window ONNX inference with overlap blending to prevent tile seams."""
        session = self._get_ort_session()
        input_name = session.get_inputs()[0].name
        tile = self.metadata.tile_size
        overlap = max(0, min(self.metadata.tile_overlap, tile // 2))
        stride = max(1, tile - overlap)

        x = stack.astype(np.float32, copy=True)
        if self.metadata.norm_mean is not None and self.metadata.norm_std is not None:
            mean = np.asarray(self.metadata.norm_mean, dtype=np.float32)[:, None, None]
            std = np.asarray(self.metadata.norm_std, dtype=np.float32)[:, None, None]
            x = (x - mean) / np.maximum(std, 1e-6)

        _, h, w = x.shape
        pad_h = max(0, tile - h)
        pad_w = max(0, tile - w)
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        _, hp, wp = x.shape
        acc = np.zeros((hp, wp), dtype=np.float32)
        wsum = np.zeros((hp, wp), dtype=np.float32)
        blend = _blend_weight_2d(tile, overlap)

        y_starts = _tile_starts(hp, tile, stride)
        x_starts = _tile_starts(wp, tile, stride)

        for y0 in y_starts:
            for x0 in x_starts:
                patch = x[:, y0 : y0 + tile, x0 : x0 + tile][None, ...]
                logits = session.run(None, {input_name: patch})[0]
                if logits.shape[1] == 1:
                    prob = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -30.0, 30.0)))
                else:
                    shifted = logits[0] - np.max(logits[0], axis=0, keepdims=True)
                    exp_l = np.exp(shifted)
                    prob = exp_l[1] / np.sum(exp_l, axis=0)
                acc[y0 : y0 + tile, x0 : x0 + tile] += prob * blend
                wsum[y0 : y0 + tile, x0 : x0 + tile] += blend

        proba = acc[:h, :w] / np.maximum(wsum[:h, :w], 1e-6)
        proba[~valid_mask] = 0.0
        return proba.astype(np.float32)

    def predict(
        self,
        s1: np.ndarray,
        terrain: np.ndarray | None = None,
        s2: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run end-to-end feature extraction and prediction on raw aligned arrays.

        Returns `(proba_map, binary_mask, valid_mask)`.
        """
        stack, feat_names, valid_mask = build_feature_stack(
            config=self.metadata.config,
            s1=s1,
            terrain=terrain,
            s2=s2,
            speckle_median_size=self.metadata.speckle_median_size,
            context_window=self.metadata.context_window,
        )
        proba = self.predict_proba_from_stack(stack, feat_names, valid_mask)
        thr = self.metadata.threshold if threshold is None else float(threshold)
        mask = ((proba >= thr) & valid_mask).astype(np.uint8)
        return proba, mask, valid_mask
