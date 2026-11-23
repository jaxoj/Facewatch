"""Thread-safe, hot-reloadable matching against the enrolled face embeddings.

The repository is reloaded periodically to build a replacement vector index.
Camera threads receive an atomic entry/index pair so no one can observe a
partially updated watchlist.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np

from facewatch.domain import EMBEDDING_DIM, Match
from facewatch.index import build_index
from facewatch.interfaces import VectorIndex, WatchlistRepository

log = logging.getLogger(__name__)


class WatchlistMatcher:
    def __init__(self, repository: WatchlistRepository, threshold: float, index_kind: str = "auto"):
        self._repo = repository
        self._threshold = threshold
        self._kind = index_kind
        self._lock = threading.RLock()
        self._entries: list = []
        self._index: VectorIndex | None = None
        self.reload()

    def reload(self) -> None:
        entries = self._repo.list_all()
        matrix = (
            np.stack([e.embedding for e in entries])
            if entries
            else np.empty((0, EMBEDDING_DIM), np.float32)
        )
        index = build_index(self._kind, matrix)
        with self._lock:  # build outside the lock, swap atomically
            self._entries, self._index = entries, index
        log.info("Watchlist loaded: %d entries", len(entries))

    def match(self, embeddings: Sequence[np.ndarray]) -> list[Match | None]:
        if not len(embeddings):
            return []
        with self._lock:
            entries, index = self._entries, self._index
        if index is None:
            # reload() normally initializes this before use.  Guard it so a caller
            # cannot dereference an uninitialized index during construction/shutdown.
            return [None] * len(embeddings)
        ids, scores = index.search(np.stack(embeddings))
        out: list[Match | None] = []
        for i, s in zip(ids, scores):
            if i >= 0 and s >= self._threshold:
                e = entries[int(i)]
                out.append(Match(e.id, e.name, float(s)))
            else:
                out.append(None)
        return out
