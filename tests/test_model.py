"""Dataset generation, threshold selection, and Detector behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from app.detection.features import FEATURE_NAMES, compute_window_features
from app.detection.model import Detector, ModelNotAvailableError
from ml.generate_dataset import WINDOW, fast_rename_encrypt, generate, idle, photo_import
from ml.train import pick_threshold


def test_generate_is_deterministic_and_labeled() -> None:
    a = generate(200, 60, seed=3)
    b = generate(200, 60, seed=3)
    assert a.equals(b)
    assert list(a.columns) == [*FEATURE_NAMES, "label", "scenario"]
    assert a["label"].sum() == 60
    assert not a[list(FEATURE_NAMES)].isna().any().any()


def test_pick_threshold_respects_fpr_budget() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 200)
    scores = np.concatenate([rng.uniform(0, 0.6, 1000), rng.uniform(0.4, 1.0, 200)])
    t = pick_threshold(y, scores, max_fpr=0.01)
    assert np.mean(scores[y == 0] >= t) <= 0.01


def test_pick_threshold_on_separable_data_sits_in_gap() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < pick_threshold(y, scores) < 0.8


def test_detector_flags_attack_not_benign(detector: Detector) -> None:
    rng = np.random.default_rng(99)
    attack = detector.score(compute_window_features(fast_rename_encrypt(rng), WINDOW))
    quiet = detector.score(compute_window_features(idle(rng), WINDOW))
    photos = detector.score(compute_window_features(photo_import(rng), WINDOW))
    assert attack.is_alert and attack.score > 0.9
    assert not quiet.is_alert
    assert not photos.is_alert  # high entropy, but expected format
    names = {f["feature"] for f in attack.top_features}
    assert names and names <= set(FEATURE_NAMES)


def test_detector_rejects_feature_mismatch(detector: Detector) -> None:
    bundle = dict(detector._bundle, feature_names=["something_else"])
    with pytest.raises(ModelNotAvailableError):
        Detector(bundle)


def test_detector_load_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ModelNotAvailableError):
        Detector.load(tmp_path / "nope.joblib")
