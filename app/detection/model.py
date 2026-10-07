"""Load the trained detector bundle and score feature windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.detection.features import FEATURE_NAMES


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model is present (run ``make train``)."""


@dataclass(frozen=True)
class Detection:
    """Result of scoring one window."""

    score: float
    is_alert: bool
    top_features: list[dict[str, float | str]] = field(default_factory=list)


class Detector:
    """Wraps the joblib bundle written by ``ml/train.py``."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        if list(bundle["feature_names"]) != list(FEATURE_NAMES):
            # A bundle trained on a different feature set would silently
            # mis-score; fail loudly instead.
            raise ModelNotAvailableError(
                "Model feature set does not match app.detection.features; retrain the model."
            )
        self._bundle = bundle
        self.model = bundle["model"]
        self.model_name: str = bundle["model_name"]
        self.threshold: float = float(bundle["threshold"])
        self._mean = np.asarray(bundle["benign_mean"], dtype=float)
        # Floor the std so constant-in-benign features don't explode z-scores.
        self._std = np.maximum(np.asarray(bundle["benign_std"], dtype=float), 1e-3)
        self._importance = np.array([bundle["feature_importances"][n] for n in FEATURE_NAMES])

    @classmethod
    def load(cls, path: Path) -> Detector:
        """Load a bundle from disk or raise ModelNotAvailableError.

        Security: joblib uses pickle, so loading runs code from the file. Only
        load models you trained yourself (``python -m ml.train``) - never a
        model file downloaded from someone else.
        """
        if not path.is_file():
            raise ModelNotAvailableError(f"No model at {path}. Run `make train` first.")
        return cls(joblib.load(path))

    def score_many(self, feature_rows: list[dict[str, float]]) -> list[Detection]:
        """Score several windows in one model call."""
        if not feature_rows:
            return []
        X = np.array([[row[n] for n in FEATURE_NAMES] for row in feature_rows], dtype=float)
        if self._bundle.get("score_kind") == "anomaly":
            scores = -self.model.score_samples(X)
        else:
            scores = self.model.predict_proba(X)[:, 1]
        return [
            Detection(
                score=round(float(s), 4),
                is_alert=bool(s >= self.threshold),
                top_features=self.explain(x),
            )
            for s, x in zip(scores, X, strict=True)
        ]

    def score(self, features: dict[str, float]) -> Detection:
        """Score a single window."""
        return self.score_many([features])[0]

    def explain(self, x: np.ndarray, k: int = 3) -> list[dict[str, float | str]]:
        """Top-k features driving this window, by ``global importance x z-score``.

        A cheap, dependency-free stand-in for SHAP: "which important features
        are furthest above their benign baseline". Only positive deviations
        count — ransomware pushes rates/entropy *up*.
        """
        z = (x - self._mean) / self._std
        contrib = self._importance * np.clip(z, 0, None)
        order = np.argsort(-contrib)[:k]
        return [
            {
                "feature": FEATURE_NAMES[i],
                "value": round(float(x[i]), 4),
                "benign_mean": round(float(self._mean[i]), 4),
                "z_score": round(float(z[i]), 2),
            }
            for i in order
            if contrib[i] > 0
        ]

    def info(self) -> dict[str, Any]:
        """Metadata for /detection/status."""
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "trained_at": self._bundle.get("trained_at"),
            "test_metrics": self._bundle.get("test_metrics"),
        }
