"""Unit tests for entropy, hashing, and directory scanning."""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import pytest

from app.backup.manifest import (
    file_extension,
    hash_and_entropy,
    is_excluded,
    scan_directory,
    shannon_entropy,
)


def test_entropy_of_empty_is_zero() -> None:
    assert shannon_entropy(b"") == 0.0


def test_entropy_of_constant_bytes_is_zero() -> None:
    assert shannon_entropy(b"A" * 1000) == 0.0


def test_entropy_of_two_equiprobable_symbols_is_one_bit() -> None:
    assert shannon_entropy(b"AB" * 500) == pytest.approx(1.0)


def test_entropy_of_all_byte_values_is_eight_bits() -> None:
    assert shannon_entropy(bytes(range(256)) * 10) == pytest.approx(8.0)


def test_random_bytes_are_high_entropy_and_text_is_not() -> None:
    assert shannon_entropy(os.urandom(64 * 1024)) > 7.9
    assert shannon_entropy(b"the quick brown fox jumps over the lazy dog " * 200) < 5.0


def test_streaming_matches_in_memory(tmp_path: Path) -> None:
    # Larger than one chunk so the streaming histogram accumulation is exercised.
    data = os.urandom(1024 * 1024 + 123) + b"x" * 5000
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    sha, entropy, size = hash_and_entropy(p)
    assert sha == hashlib.sha256(data).hexdigest()
    assert entropy == pytest.approx(shannon_entropy(data))
    assert size == len(data)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a/report.docx", ".docx"),
        ("report.docx.locked", ".locked"),
        ("Makefile", ""),
        ("X.JPG", ".jpg"),
    ],
)
def test_file_extension(path: str, expected: str) -> None:
    assert file_extension(path) == expected


def test_exclusion_patterns() -> None:
    assert is_excluded("docs/~$report.docx", ["~$*"])
    assert is_excluded("a/b/c.tmp", ["*.tmp"])
    assert not is_excluded("a/b/c.txt", ["*.tmp"])


def test_scan_directory_skips_excluded_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("keep")
    (tmp_path / "skip.tmp").write_text("skip")
    # Symlinks need privileges on Windows; the rest of the test still applies.
    with contextlib.suppress(OSError, NotImplementedError):
        (tmp_path / "link.txt").symlink_to(tmp_path / "keep.txt")
    entries = scan_directory(tmp_path, ["*.tmp"])
    assert [e.path for e in entries] == ["keep.txt"]
