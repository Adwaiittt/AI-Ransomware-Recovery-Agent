"""Train, evaluate, and save the ransomware detector.

Pipeline:
  1. Stratified train/test split (stratified on *scenario*, so every scenario —
     including the hard negatives — appears in both splits).
  2. Baseline: IsolationForest fit on benign training windows only (no labels
     needed — what you'd ship on day one with no attack data). Its threshold is
     the 99th percentile of benign scores, i.e. a ~1% FPR budget.
  3. Supervised: RandomForest and GradientBoosting. For each, 5-fold
     out-of-fold probabilities on the *training* set pick the decision threshold
     (max F1 subject to FPR <= 1%). The test set is never used for any choice.
  4. Select the supervised model with the best out-of-fold F1, refit on all
     training data, evaluate once on the test set.
  5. Generalisation check: retrain the selected model type with one attack
     family held out entirely and report recall on that unseen family.

Outputs:  ml/artifacts/model.joblib  and  ml/artifacts/metrics.json

Usage:  python -m ml.train --data ml/data/dataset.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from app.detection.features import FEATURE_NAMES

SEED = 42
MAX_FPR = 0.01
HOLDOUT_FAMILIES = ("partial_encryption", "inplace_stealth", "random_extension")


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Standard binary metrics at ``threshold`` plus threshold-free ROC-AUC."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    """Highest-F1 threshold whose FPR stays within budget.

    Candidates come from ``roc_curve`` (exact score values, no rounding). The
    returned cut sits halfway between the chosen score and the next lower one,
    so a model with a clean gap is not cut right at the edge of the positives.
    """
    fpr, _, thresholds = roc_curve(y_true, scores)
    best_t, best_f1 = float(np.max(scores)), -1.0
    for t, f in zip(thresholds, fpr, strict=True):
        if f > max_fpr or not np.isfinite(t):
            continue
        f1 = f1_score(y_true, scores >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    lower = scores[scores < best_t]
    return float((best_t + lower.max()) / 2) if lower.size else best_t


def supervised_candidates() -> dict[str, ClassifierMixin]:
    """Models compared by out-of-fold F1."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }


def per_scenario(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Alert rate per scenario: detection rate for attacks, false-alarm rate for benign."""
    out: dict[str, Any] = {}
    flagged = scores >= threshold
    # Mixed benign windows are named "a+b"; group by the primary activity.
    primary = df["scenario"].str.split("+").str[0].to_numpy()
    for name in sorted(set(primary)):
        mask = primary == name
        label = int(df["label"].to_numpy()[mask][0])
        out[name] = {
            "label": "attack" if label else "benign",
            "n": int(mask.sum()),
            "alert_rate": round(float(flagged[mask].mean()), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ransomware detector.")
    parser.add_argument("--data", type=Path, default=Path("ml/data/dataset.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    feats = list(FEATURE_NAMES)
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df["scenario"].str.split("+").str[0]
    )
    X_train, y_train = train_df[feats].to_numpy(), train_df["label"].to_numpy()
    X_test, y_test = test_df[feats].to_numpy(), test_df["label"].to_numpy()

    results: dict[str, Any] = {}

    # 1) Unsupervised baseline.
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED)
    iso.fit(X_train[y_train == 0])
    iso_train_scores = -iso.score_samples(X_train[y_train == 0])  # higher = more anomalous
    iso_threshold = float(np.quantile(iso_train_scores, 1 - MAX_FPR))
    results["isolation_forest"] = evaluate(y_test, -iso.score_samples(X_test), iso_threshold)

    # 2) Supervised candidates, threshold from out-of-fold predictions on train only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof: dict[str, dict[str, float]] = {}
    for name, model in supervised_candidates().items():
        oof_scores = cross_val_predict(
            clone(model), X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        threshold = pick_threshold(y_train, oof_scores)
        oof_f1 = f1_score(y_train, oof_scores >= threshold)
        oof[name] = {"threshold": threshold, "oof_f1": round(float(oof_f1), 4)}
        fitted = clone(model).fit(X_train, y_train)
        results[name] = evaluate(y_test, fitted.predict_proba(X_test)[:, 1], threshold)
        results[name]["cv_oof_f1"] = oof[name]["oof_f1"]

    selected = max(oof, key=lambda n: oof[n]["oof_f1"])
    threshold = oof[selected]["threshold"]
    final = clone(supervised_candidates()[selected]).fit(X_train, y_train)
    test_scores = final.predict_proba(X_test)[:, 1]

    # 3) Generalisation to attack families never seen in training.
    generalisation: dict[str, Any] = {}
    train_primary = train_df["scenario"].str.split("+").str[0]
    test_primary = test_df["scenario"].str.split("+").str[0]
    for family in HOLDOUT_FAMILIES:
        keep = (train_primary != family).to_numpy()
        m = clone(supervised_candidates()[selected]).fit(X_train[keep], y_train[keep])
        fam_mask = (test_primary == family).to_numpy()
        fam_scores = m.predict_proba(X_test[fam_mask])[:, 1]
        generalisation[family] = {
            "n_test": int(fam_mask.sum()),
            "recall_when_unseen": round(float(np.mean(fam_scores >= threshold)), 4),
            "recall_when_seen": round(float(np.mean(test_scores[fam_mask] >= threshold)), 4),
        }

    # 4) Stats the runtime explainer needs (benign baseline for z-scores).
    benign = X_train[y_train == 0]
    importances = dict(zip(feats, map(float, final.feature_importances_), strict=True))

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "path": args.data.as_posix(),
            "rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "attack_fraction": round(float(df["label"].mean()), 4),
        },
        "fpr_budget": MAX_FPR,
        "selected_model": selected,
        "models": results,
        "per_scenario_test": per_scenario(test_df, test_scores, threshold),
        "generalisation_unseen_family": generalisation,
        "feature_importances": dict(sorted(importances.items(), key=lambda kv: -kv[1])),
    }

    bundle = {
        "model": final,
        "model_name": selected,
        "score_kind": "proba",
        "threshold": threshold,
        "feature_names": feats,
        "benign_mean": benign.mean(axis=0).tolist(),
        "benign_std": benign.std(axis=0).tolist(),
        "feature_importances": importances,
        "trained_at": metrics["generated_at"],
        "sklearn_version": sklearn.__version__,
        "test_metrics": results[selected],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out_dir / "model.joblib")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"selected: {selected} (threshold={threshold:.4f})")
    header = f"{'model':<20}{'prec':>8}{'recall':>8}{'f1':>8}{'auc':>8}{'fpr':>8}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:<20}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}"
            f"{r['roc_auc']:>8.4f}{r['false_positive_rate']:>8.4f}"
        )
    print("unseen-family recall:", {k: v["recall_when_unseen"] for k, v in generalisation.items()})


if __name__ == "__main__":
    main()
