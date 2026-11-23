"""Per-camera processing: analyze -> match -> track -> confirm -> render -> alert.

This module is the orchestration layer for one camera.  Expensive inference
runs only every N frames; the last detections are reused for display in between.
"""
from __future__ import annotations

import numpy as np

from facewatch.domain import Alert, Detection, Frame
from facewatch.interfaces import AlertPublisher, FaceAnalyzer, Matcher, Renderer
from facewatch.policy import ConfirmationPolicy, Cooldown
from facewatch.tracking import IoUTracker


class CameraPipeline:
    def __init__(
        self,
        camera_id: str,
        analyzer: FaceAnalyzer,
        matcher: Matcher,
        tracker: IoUTracker,
        policy: ConfirmationPolicy,
        cooldown: Cooldown,
        publisher: AlertPublisher,
        renderer: Renderer,
        process_every_n: int = 1,
    ):
        self.camera_id = camera_id
        self._analyzer, self._matcher, self._tracker = analyzer, matcher, tracker
        self._policy, self._cooldown = policy, cooldown
        self._publisher, self._renderer = publisher, renderer
        self._every_n = process_every_n
        self._count = 0
        self._last: list[Detection] | None = None

    def process(self, frame: Frame) -> np.ndarray:
        """Return the annotated frame. Skipped frames reuse the last detections."""
        run = self._last is None or self._count % self._every_n == 0
        self._count += 1
        new_alerts: list[Detection] = []
        if run:
            self._last, new_alerts = self._analyze(frame.image)
        # Rendering produces a copy, so it neither alters the source frame nor
        # mutates detections while the image is published to the display board.
        annotated = self._renderer.draw(frame.image, self._last or ())
        for d in new_alerts:
            if d.match is not None:
                self._publisher.submit(Alert(self.camera_id, d.match, d.face.bbox, frame.timestamp, annotated))
        return annotated

    def _analyze(self, image: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        faces = self._analyzer.analyze(image)
        candidates = self._matcher.match([f.embedding for f in faces])
        tracks = self._tracker.update([f.bbox for f in faces])
        detections: list[Detection] = []
        to_alert: list[Detection] = []
        for face, cand, track in zip(faces, candidates, tracks):
            track.history.append(cand)
            confirmed = self._policy.evaluate(track)
            det = Detection(face, track.id, cand, confirmed)
            detections.append(det)
            if confirmed and self._cooldown.ready(confirmed.entry_id):
                to_alert.append(det)
        return detections, to_alert
