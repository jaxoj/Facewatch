"""Temporal confirmation and alert-rate limiting policies.

The confirmation policy turns a series of possibly noisy frame matches into a
single trusted match.  The cooldown policy prevents that match from flooding
alert sinks.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from facewatch.domain import Match
from facewatch.tracking import Track


class ConfirmationPolicy:
    """Confirm only if the same person matched in >= min_hits of the track's recent frames."""

    def __init__(self, min_hits: int):
        self._min_hits = min_hits

    def evaluate(self, track: Track) -> Match | None:
        by_entry: dict[str, list[Match]] = {}
        for m in track.history:
            if m is not None:
                by_entry.setdefault(m.entry_id, []).append(m)
        if not by_entry:
            return None
        entry_id, hits = max(by_entry.items(), key=lambda kv: len(kv[1]))
        if len(hits) < self._min_hits:
            return None
        return Match(entry_id, hits[-1].name, sum(h.score for h in hits) / len(hits))


class Cooldown:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self._seconds, self._clock = seconds, clock
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def ready(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < self._seconds:
                return False
            self._last[key] = now
            return True
