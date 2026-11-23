"""Thread orchestration for camera pipelines, display, reload, and shutdown.

Inference is isolated per camera, watchlist reload runs separately, and OpenCV
display calls remain on the main thread through a snapshot board.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from facewatch.alerts import AlertDispatcher
from facewatch.interfaces import FrameSource, Matcher
from facewatch.pipeline import CameraPipeline

log = logging.getLogger(__name__)


@dataclass
class CameraRuntime:
    source: FrameSource
    pipeline: CameraPipeline


class CameraManager:
    def __init__(self, runtimes: list[CameraRuntime], dispatcher: AlertDispatcher,
                 matcher: Matcher, display: bool = True, reload_interval_s: float = 60.0):
        if not runtimes:
            raise ValueError("At least one camera is required")
        self._runtimes, self._dispatcher, self._matcher = runtimes, dispatcher, matcher
        self._display, self._reload_s = display, reload_interval_s
        self._stop = threading.Event()
        self._board: dict[str, np.ndarray] = {}
        self._board_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def run(self) -> None:
        self._dispatcher.start()
        for rt in self._runtimes:
            rt.source.start()
            t = threading.Thread(target=self._camera_loop, args=(rt,), name=f"cam-{rt.source.camera_id}", daemon=True)
            t.start()
            self._threads.append(t)
        if self._reload_s > 0:
            threading.Thread(target=self._reload_loop, name="reload", daemon=True).start()
        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        for rt in self._runtimes:
            rt.source.stop()
        self._dispatcher.stop()
        cv2.destroyAllWindows()
        log.info("Stopped")

    def _camera_loop(self, rt: CameraRuntime) -> None:
        cam = rt.source.camera_id
        while not self._stop.is_set():
            frame = rt.source.read(timeout=1.0)
            if frame is None:
                continue
            try:
                out = rt.pipeline.process(frame)
            except Exception:
                log.exception("[%s] processing failed", cam)
                continue
            with self._board_lock:
                self._board[cam] = out

    def _reload_loop(self) -> None:
        while not self._stop.wait(self._reload_s):
            try:
                self._matcher.reload()
            except Exception:
                log.exception("Watchlist reload failed")

    def _main_loop(self) -> None:
        # cv2.imshow must stay on the main thread, so workers publish to a board.
        while not self._stop.is_set():
            if self._display:
                with self._board_lock:
                    snapshot = dict(self._board)
                for cam, img in snapshot.items():
                    cv2.imshow(cam, img)
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    return
            elif self._stop.wait(0.5):
                return
