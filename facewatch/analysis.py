"""InsightFace implementation of face detection and embedding extraction.

This module owns the expensive model instance and serializes model calls because
InsightFace is shared by every camera worker.  It filters detector output and
converts it into the plain `Face` types used by the rest of the pipeline.
"""
from __future__ import annotations

import threading
from typing import Any, Iterable, cast

import numpy as np

from facewatch.config import AnalyzerConfig
from facewatch.domain import BBox, Face


class InsightFaceAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        from insightface.app import FaceAnalysis  # heavy import kept local

        providers = ["CPUExecutionProvider"]
        if cfg.ctx_id >= 0:
            providers.insert(0, "CUDAExecutionProvider")
        self._app = FaceAnalysis(name=cfg.model_name, providers=providers)
        self._app.prepare(ctx_id=cfg.ctx_id, det_size=cfg.det_size)
        self._cfg = cfg
        self._lock = threading.Lock()  # one model shared by all camera threads

    def analyze(self, image: np.ndarray) -> list[Face]:
        with self._lock:
            # InsightFace is dynamically typed, so establish a small interface for
            # Pylance and keep only the fields the pipeline actually uses.
            raw = cast(Iterable[Any], self._app.get(image))
        h, w = image.shape[:2]
        faces: list[Face] = []
        for r in raw:
            if r.det_score < self._cfg.min_det_score:
                continue
            bbox = BBox.from_xyxy(r.bbox, w, h)
            if min(bbox.width, bbox.height) < self._cfg.min_face_px:
                continue
            faces.append(Face(bbox, np.asarray(r.normed_embedding, np.float32), float(r.det_score)))
        return faces
