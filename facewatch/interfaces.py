"""Structural interfaces between Facewatch subsystems.

Protocols document replaceable boundaries: a repository, vector index, camera
source, face analyzer, or alert implementation can be swapped without changing
the pipeline's orchestration.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from facewatch.domain import Alert, Detection, Face, Frame, Match, WatchlistEntry


class FrameSource(Protocol):
    camera_id: str

    def start(self) -> None: ...
    def read(self, timeout: float = 1.0) -> Frame | None: ...
    def stop(self) -> None: ...


class FaceAnalyzer(Protocol):
    def analyze(self, image: np.ndarray) -> list[Face]: ...


class VectorIndex(Protocol):
    def search(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (best_row_ids, best_scores) per query; id -1 if index is empty."""
        ...


class WatchlistRepository(Protocol):
    def add(self, entry: WatchlistEntry) -> None: ...
    def remove(self, entry_id: str) -> bool: ...
    def list_all(self) -> list[WatchlistEntry]: ...


class Matcher(Protocol):
    def match(self, embeddings: Sequence[np.ndarray]) -> list[Match | None]: ...
    def reload(self) -> None: ...


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class AlertPublisher(Protocol):
    def submit(self, alert: Alert) -> None: ...


class Renderer(Protocol):
    def draw(self, image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray: ...
