"""Nearest-neighbor search over normalized embeddings.

For unit-length embeddings, cosine similarity is equivalent to inner product.
Both index implementations search for the watchlist embedding with the highest score.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class NumpyIndex:
    def __init__(self, embeddings: np.ndarray):
        self._m = np.ascontiguousarray(embeddings, dtype=np.float32)

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if len(self._m) == 0:
            return np.full(len(q), -1, dtype=np.int64), np.zeros(len(q), dtype=np.float32)
        sims = q @ self._m.T
        ids = sims.argmax(axis=1)
        return ids, sims[np.arange(len(q)), ids]


class FaissIndex:
    def __init__(self, embeddings: np.ndarray):
        import faiss

        m = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._empty = len(m) == 0
        self._index = faiss.IndexFlatIP(m.shape[1])
        if not self._empty:
            self._index.add(m)

    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self._empty:
            return np.full(len(q), -1, dtype=np.int64), np.zeros(len(q), dtype=np.float32)
        scores, ids = self._index.search(q, 1)
        return ids[:, 0], scores[:, 0]


def build_index(kind: str, embeddings: np.ndarray):
    if kind in ("auto", "faiss"):
        try:
            return FaissIndex(embeddings)
        except ImportError:
            if kind == "faiss":
                raise
            log.info("faiss not installed, using NumPy index")
    return NumpyIndex(embeddings)
