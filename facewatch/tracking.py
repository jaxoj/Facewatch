"""Lightweight bounding-box tracking for preserving identity across frames."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from facewatch.domain import BBox


@dataclass(eq=False)
class Track:
    id: int
    bbox: BBox
    last_frame: int
    history: deque  # recent per-frame Match | None


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, history: int = 5):
        self._thr, self._max_age, self._history = iou_threshold, max_age, history
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame = 0

    def update(self, boxes: Sequence[BBox]) -> list[Track]:
        """Return one Track per input box, in the same order."""
        self._frame += 1
        pairs = sorted(
            ((t.bbox.iou(b), tid, bi) for tid, t in self._tracks.items() for bi, b in enumerate(boxes)),
            reverse=True,
        )
        assigned: dict[int, Track] = {}
        used: set[int] = set()
        for iou, tid, bi in pairs:
            if iou < self._thr:
                break
            if tid in used or bi in assigned:
                continue
            track = self._tracks[tid]
            track.bbox, track.last_frame = boxes[bi], self._frame
            assigned[bi], used = track, used | {tid}
        for bi, box in enumerate(boxes):
            if bi not in assigned:
                track = Track(self._next_id, box, self._frame, deque(maxlen=self._history))
                self._tracks[track.id] = track
                assigned[bi] = track
                self._next_id += 1
        self._tracks = {
            tid: t for tid, t in self._tracks.items() if self._frame - t.last_frame <= self._max_age
        }
        return [assigned[i] for i in range(len(boxes))]
