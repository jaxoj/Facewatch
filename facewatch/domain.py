"""Plain immutable data types shared across every application layer.

This module has no I/O or heavy runtime dependencies.  It is the vocabulary
used by sources, analyzers, trackers, matchers, renderers, and alert sinks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EMBEDDING_DIM = 512


@dataclass(frozen=True, slots=True)
class BBox:
    """Clipped, axis-aligned face location with geometry and overlap helpers."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    @classmethod
    def from_xyxy(cls, xyxy, width: int, height: int) -> "BBox":
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        return cls(max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2))


@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    image: np.ndarray
    timestamp: float


@dataclass(frozen=True, slots=True)
class Face:
    bbox: BBox
    embedding: np.ndarray  # L2-normalised, float32
    det_score: float


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    id: str
    name: str
    embedding: np.ndarray  # L2-normalised, float32
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Match:
    entry_id: str
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class Detection:
    """A face seen in one frame. `match` is set only once confirmed over time."""

    face: Face
    track_id: int
    candidate: Match | None  # raw single-frame match
    match: Match | None      # temporally confirmed match


@dataclass(frozen=True, slots=True)
class Alert:
    camera_id: str
    match: Match
    bbox: BBox
    timestamp: float
    image: np.ndarray  # annotated frame
