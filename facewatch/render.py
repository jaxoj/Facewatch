"""Render confirmed matches and, optionally, unmatched detections on video frames."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from facewatch.domain import Detection

RED = (0, 0, 255)  # BGR
GREEN = (0, 200, 0)


class RedBoxRenderer:
    """Red square on confirmed watchlist matches; optionally thin green on other faces."""

    def __init__(self, show_labels: bool = True, thickness: int = 2, show_unmatched: bool = False):
        self._labels, self._thick, self._unmatched = show_labels, thickness, show_unmatched

    def draw(self, image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        out = image.copy()
        for d in detections:
            b = d.face.bbox
            if d.match:
                cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), RED, self._thick)
                if self._labels:
                    text = f"{d.match.name} {d.match.score:.2f}"
                    cv2.putText(out, text, (b.x1, max(15, b.y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
            elif self._unmatched:
                cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), GREEN, 1)
        return out
