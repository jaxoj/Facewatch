"""SQLite watchlist persistence.

A repository implementation is deliberately small: callers only add, remove,
and list entries.  Embeddings are stored as binary float32 arrays.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from facewatch.domain import WatchlistEntry


class SQLiteWatchlistRepository:
    def __init__(self, path: str):
        self._path = path
        self._memory_conn: sqlite3.Connection | None = None
        self._memory_lock = threading.RLock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_conn = sqlite3.connect(path, check_same_thread=False)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS watchlist (
                       id TEXT PRIMARY KEY, name TEXT NOT NULL,
                       embedding BLOB NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
                       created_at REAL NOT NULL)"""
            )

    @contextmanager
    def _conn(self):
        if self._memory_conn is not None:
            with self._memory_lock:
                try:
                    with self._memory_conn:
                        yield self._memory_conn
                except BaseException:
                    raise
            return

        conn = sqlite3.connect(self._path)  # connection per call: thread-safe
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(self, entry: WatchlistEntry) -> None:
        emb = np.asarray(entry.embedding, np.float32).tobytes()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO watchlist VALUES (?,?,?,?,?)",
                (entry.id, entry.name, emb, json.dumps(entry.meta), time.time()),
            )

    def remove(self, entry_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM watchlist WHERE id=?", (entry_id,)).rowcount > 0

    def list_all(self) -> list[WatchlistEntry]:
        with self._conn() as c:
            rows = c.execute("SELECT id, name, embedding, meta FROM watchlist ORDER BY id").fetchall()
        return [
            WatchlistEntry(i, n, np.frombuffer(e, dtype=np.float32).copy(), json.loads(m))
            for i, n, e, m in rows
        ]
