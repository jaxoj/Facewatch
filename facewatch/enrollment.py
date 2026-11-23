"""Create watchlist entries by averaging face embeddings from reference photos.

Each photo must contain at least one usable face.  The largest detected face is
used for that photo, and all selected embeddings are normalized as a group.
"""
from __future__ import annotations

import cv2
import numpy as np

from facewatch.domain import WatchlistEntry
from facewatch.interfaces import FaceAnalyzer, WatchlistRepository


class EnrollmentError(RuntimeError):
    pass


class Enroller:
    def __init__(self, analyzer: FaceAnalyzer, repository: WatchlistRepository):
        self._analyzer, self._repo = analyzer, repository

    def enroll(self, entry_id: str, name: str, image_paths: list[str], meta: dict | None = None) -> WatchlistEntry:
        embeddings = []
        for path in image_paths:
            image = cv2.imread(path)
            if image is None:
                raise EnrollmentError(f"Cannot read image: {path}")
            faces = self._analyzer.analyze(image)
            if not faces:
                raise EnrollmentError(f"No usable face found in: {path}")
            largest = max(faces, key=lambda f: f.bbox.area)
            embeddings.append(largest.embedding)
        mean = np.mean(embeddings, axis=0)
        mean = (mean / np.linalg.norm(mean)).astype(np.float32)
        entry = WatchlistEntry(entry_id, name, mean, meta or {})
        self._repo.add(entry)
        return entry
