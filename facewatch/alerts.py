"""Composable alert delivery through a non-blocking dispatcher.

Sinks perform the final side effects (logging, snapshots, webhooks).  The queue
and worker thread isolate them from camera inference and display loops.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
from pathlib import Path

import cv2

from facewatch.config import AlertsConfig
from facewatch.domain import Alert
from facewatch.interfaces import AlertSink

log = logging.getLogger(__name__)


class LogSink:
    def send(self, alert: Alert) -> None:
        log.warning("MATCH camera=%s person=%s(%s) score=%.3f",
                    alert.camera_id, alert.match.name, alert.match.entry_id, alert.match.score)


class SnapshotSink:
    def __init__(self, directory: str):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        name = f"{int(alert.timestamp * 1000)}_{alert.camera_id}_{alert.match.entry_id}.jpg"
        cv2.imwrite(str(self._dir / name), alert.image)


class WebhookSink:
    def __init__(self, url: str, timeout: float = 5.0):
        self._url, self._timeout = url, timeout

    def send(self, alert: Alert) -> None:
        body = json.dumps({
            "camera_id": alert.camera_id, "entry_id": alert.match.entry_id,
            "name": alert.match.name, "score": alert.match.score, "timestamp": alert.timestamp,
        }).encode()
        req = urllib.request.Request(self._url, body, {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=self._timeout).close()


class CompositeSink:
    def __init__(self, sinks: list[AlertSink]):
        self._sinks = sinks

    def send(self, alert: Alert) -> None:
        for sink in self._sinks:
            try:
                sink.send(alert)
            except Exception:
                log.exception("Alert sink %s failed", type(sink).__name__)


class AlertDispatcher:
    """Queue + worker thread: slow sinks (webhooks, disk) can't stall inference."""

    def __init__(self, sink: AlertSink, max_queue: int = 100):
        self._sink = sink
        self._q: queue.Queue[Alert | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, name="alerts", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, alert: Alert) -> None:
        try:
            self._q.put_nowait(alert)
        except queue.Full:
            log.error("Alert queue full, dropping alert for %s", alert.match.entry_id)

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while (alert := self._q.get()) is not None:
            self._sink.send(alert)


def build_sink(cfg: AlertsConfig) -> AlertSink:
    sinks: list[AlertSink] = []
    if cfg.log:
        sinks.append(LogSink())
    if cfg.snapshot_dir:
        sinks.append(SnapshotSink(cfg.snapshot_dir))
    if cfg.webhook_url:
        sinks.append(WebhookSink(cfg.webhook_url))
    return CompositeSink(sinks)
