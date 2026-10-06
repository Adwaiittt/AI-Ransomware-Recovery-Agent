"""Shared fixtures: moto-mocked S3, temp SQLite DB, temp watch dir, API client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.session import init_db, make_engine, make_session_factory
from app.detection.features import FEATURE_NAMES
from app.detection.model import Detector
from app.main import create_app
from app.storage.s3_client import S3Storage

BUCKET = "test-backups"


@pytest.fixture(autouse=True)
def _fake_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make sure nothing in tests can ever reach real AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def storage() -> Iterator[S3Storage]:
    with mock_aws():
        s = S3Storage(boto3.client("s3", region_name="us-east-1"), BUCKET)
        s.ensure_bucket()
        yield s


@pytest.fixture
def watch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "watched"
    (d / "docs").mkdir(parents=True)
    (d / "docs" / "report.docx").write_text("Quarterly report. " * 200)
    (d / "docs" / "notes.txt").write_text("meeting notes\n" * 50)
    (d / "code.py").write_text("print('hello')\n" * 30)
    return d


@pytest.fixture
def settings(tmp_path: Path, watch_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        watch_dir=watch_dir,
        s3_bucket=BUCKET,
        exclude_patterns=["*.tmp"],
        # Never pick up a locally trained ml/artifacts model: tests must be hermetic.
        model_path=tmp_path / "no-model.joblib",
        restore_dir=tmp_path / "restore_out",
        embedding_model="hash",  # offline embedder: no model download in tests/CI
    )


@pytest.fixture
def db(settings: Settings) -> Iterator[Session]:
    engine = make_engine(settings.database_url)
    init_db(engine)
    session = make_session_factory(engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def client(settings: Settings, storage: S3Storage) -> Iterator[TestClient]:
    app = create_app(settings=settings, storage=storage)
    with TestClient(app) as c:  # context manager runs the lifespan
        yield c


@pytest.fixture(scope="session")
def detector() -> Detector:
    """Small RandomForest trained on a fresh synthetic dataset (fast, deterministic).

    Bundle layout mirrors ml/train.py so Detector is exercised exactly as in prod.
    """
    from sklearn.ensemble import RandomForestClassifier

    from ml.generate_dataset import generate

    df = generate(n_benign=1500, n_attack=500, seed=1)
    X, y = df[list(FEATURE_NAMES)].to_numpy(), df["label"].to_numpy()
    model = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=1).fit(X, y)
    benign = X[y == 0]
    return Detector(
        {
            "model": model,
            "model_name": "random_forest",
            "score_kind": "proba",
            "threshold": 0.5,
            "feature_names": list(FEATURE_NAMES),
            "benign_mean": benign.mean(axis=0).tolist(),
            "benign_std": benign.std(axis=0).tolist(),
            "feature_importances": dict(
                zip(FEATURE_NAMES, model.feature_importances_.tolist(), strict=True)
            ),
            "trained_at": "test",
            "test_metrics": {},
        }
    )


@pytest.fixture
def client_with_model(
    settings: Settings, storage: S3Storage, detector: Detector
) -> Iterator[TestClient]:
    app = create_app(settings=settings, storage=storage, detector=detector)
    with TestClient(app) as c:
        yield c


def write_attackable_tree(root: Path, n: int = 40) -> list[Path]:
    """Create ``n`` low-entropy text files under ``root`` (helper for attack tests)."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = root / f"dir{i % 4}" / f"doc_{i}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        words = rng.choice(["alpha", "beta", "gamma", "delta", "report"], size=800)
        p.write_text(" ".join(words))
        paths.append(p)
    return paths


@pytest.fixture
def make_tree():  # type: ignore[no-untyped-def]
    """Expose write_attackable_tree to tests without importing conftest."""
    return write_attackable_tree
