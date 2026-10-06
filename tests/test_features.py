"""Unit tests for window feature extraction."""

from __future__ import annotations

import pytest

from app.detection.features import (
    FEATURE_NAMES,
    EventType,
    FileEvent,
    compute_window_features,
    split_into_windows,
    to_vector,
)

W = 10.0


def _m(t: float, p: str, before: float | None, after: float | None) -> FileEvent:
    return FileEvent(t, EventType.modified, p, entropy_before=before, entropy_after=after)


def _mv(t: float, src: str, dst: str) -> FileEvent:
    return FileEvent(t, EventType.moved, src, dest_path=dst)


def test_all_features_present_and_ordered() -> None:
    f = compute_window_features([_m(0, "a.txt", 4.0, 4.1)], W)
    assert list(f) == list(FEATURE_NAMES)
    assert len(to_vector(f)) == len(FEATURE_NAMES)


def test_ransomware_burst() -> None:
    # 4 files: overwrite with ciphertext then rename to .locked
    events: list[FileEvent] = []
    for i in range(4):
        p = f"docs/d{i % 2}/f{i}.txt"
        events += [_m(i, p, 4.0, 8.0), _mv(i + 0.1, p, p + ".locked")]
    f = compute_window_features(events, W)
    assert f["files_written_per_sec"] == pytest.approx(0.4)
    assert f["renames_per_sec"] == pytest.approx(0.4)
    assert f["pct_extension_changed"] == pytest.approx(4 / 8)  # 4 changes / 8 touched names
    assert f["mean_entropy_delta"] == pytest.approx(4.0)
    assert f["frac_high_entropy"] == 1.0
    # Measured at the final name (.locked), so not excused as a compressed format.
    assert f["frac_high_entropy_unexpected"] == 1.0
    assert f["unknown_ext_count"] == 4
    assert f["unique_dirs"] == 2
    assert f["events_per_sec"] == pytest.approx(0.8)


def test_repeated_modified_events_are_deduplicated() -> None:
    events = [_m(t, "a.txt", 4.0, 4.1) for t in range(5)]
    assert compute_window_features(events, W)["files_written_per_sec"] == pytest.approx(0.1)


def test_compressed_formats_are_expected_high_entropy() -> None:
    events = [
        FileEvent(0, EventType.created, "p/a.jpg", entropy_after=7.95),
        FileEvent(1, EventType.created, "p/b.zip", entropy_after=7.99),
        FileEvent(2, EventType.created, "p/c.bin", entropy_after=7.99),
    ]
    f = compute_window_features(events, W)
    assert f["frac_high_entropy"] == 1.0
    assert f["frac_high_entropy_unexpected"] == pytest.approx(1 / 3)  # only .bin


def test_delete_create_ratio_and_unknown_entropy() -> None:
    events = [
        FileEvent(0, EventType.created, "x.enc", entropy_after=None),
        FileEvent(1, EventType.deleted, "x.txt"),
        FileEvent(2, EventType.deleted, "y.txt"),
        FileEvent(3, EventType.deleted, "z.txt"),
    ]
    f = compute_window_features(events, W)
    assert f["delete_create_ratio"] == pytest.approx(3 / 2)
    assert f["frac_high_entropy"] == 0.0  # no measured entropy -> 0, not NaN
    assert f["mean_entropy_delta"] == 0.0


def test_empty_window_is_all_zero() -> None:
    assert set(compute_window_features([], W).values()) == {0.0}


def test_invalid_window() -> None:
    with pytest.raises(ValueError):
        compute_window_features([], 0)


def test_split_into_windows() -> None:
    events = [_m(t, f"{t}.txt", 4, 4) for t in (100.0, 101.0, 109.9, 110.0, 135.0)]
    windows = split_into_windows(events, W)
    assert [len(w) for w in windows] == [3, 1, 1]
    assert split_into_windows([], W) == []
