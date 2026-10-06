"""Text embedders.

* ``SentenceTransformerEmbedder`` — all-MiniLM-L6-v2 (384-d), the production default.
* ``HashingEmbedder`` — deterministic feature-hashing bag-of-words.
  No model download, no torch: used in tests/CI and as an offline fallback
  (``EMBEDDING_MODEL=hash``). 1024-d to keep hash collisions rare. Much weaker
  semantically than MiniLM, but our chunks are
  keyword-heavy ("incident", "extension_changed", snapshot ids), so retrieval
  still works acceptably.

Both return L2-normalised float32 rows, so inner product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Protocol

import numpy as np

# Split on anything non-alphanumeric so "doc_20.txt.locked" -> doc, 20, txt, locked
# and "ransomware-like" -> ransomware, like (otherwise those never match a query).
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that maps texts to normalised vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of unigrams + bigrams (the "hashing trick")."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vec(t) for t in texts]))


class SentenceTransformerEmbedder:
    """Lazy-loaded sentence-transformers model (loaded on first use, not at import)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.name = model_name
        self.dim = 384
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # heavy import

                self._model = SentenceTransformer(self.name)
                # Renamed in sentence-transformers 6; support both spellings.
                get_dim = (
                    getattr(self._model, "get_embedding_dimension", None)
                    or self._model.get_sentence_embedding_dimension
                )
                self.dim = int(get_dim())
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    """``"hash"`` -> HashingEmbedder, anything else -> sentence-transformers model name."""
    return HashingEmbedder() if name == "hash" else SentenceTransformerEmbedder(name)
