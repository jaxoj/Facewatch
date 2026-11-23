"""Camera frame sources with reconnect and latest-frame semantics.

A dedicated reader thread keeps only the newest frame, preventing inference
workers from accumulating a backlog of stale video frames.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2

from facewatch.config import CameraConfig
from facewatch.domain import Frame

log = logging.getLogger(__name__)


class OpenCVStreamSource:
    """RTSP/HTTP/file/webcam source with automatic reconnect and exponential backoff."""

    def __init__(self, camera_id: str, uri: str, reconnect_min_s: float = 1.0, reconnect_max_s: float = 30.0):
        self.camera_id = camera_id
        self._uri = int(uri) if str(uri).isdigit() else uri
        self._min, self._max = reconnect_min_s, reconnect_max_s
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = self._read_seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"src-{camera_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5)

    def read(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._read_seq or self._stop.is_set(), timeout)
            if self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._latest

    def _loop(self) -> None:
        delay = self._min
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._uri)
            if not cap.isOpened():
                cap.release()
                log.warning("[%s] cannot open stream, retrying in %.0fs", self.camera_id, delay)
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, self._max)
                continue
            log.info("[%s] stream connected", self.camera_id)
            delay = self._min
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    log.warning("[%s] stream lost", self.camera_id)
                    break
                with self._cond:
                    self._latest = Frame(self.camera_id, image, time.time())
                    self._seq += 1
                    self._cond.notify_all()
            cap.release()


def create_source(cfg: CameraConfig) -> OpenCVStreamSource:
    return OpenCVStreamSource(cfg.id, cfg.uri)
