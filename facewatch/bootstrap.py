"""Composition root: the only module that wires concrete implementations together.

The rest of the application depends mainly on protocols.  Keeping construction
here makes those protocols real boundaries rather than distributed imports.
"""
from __future__ import annotations

from facewatch.alerts import AlertDispatcher, build_sink
from facewatch.analysis import InsightFaceAnalyzer
from facewatch.config import AppConfig
from facewatch.manager import CameraManager, CameraRuntime
from facewatch.matcher import WatchlistMatcher
from facewatch.pipeline import CameraPipeline
from facewatch.policy import ConfirmationPolicy, Cooldown
from facewatch.render import RedBoxRenderer
from facewatch.repository import SQLiteWatchlistRepository
from facewatch.sources import create_source
from facewatch.tracking import IoUTracker


def build_repository(cfg: AppConfig) -> SQLiteWatchlistRepository:
    return SQLiteWatchlistRepository(cfg.watchlist.db_path)


def build_manager(cfg: AppConfig) -> CameraManager:
    cameras = [c for c in cfg.cameras if c.enabled]
    repo = build_repository(cfg)
    matcher = WatchlistMatcher(repo, cfg.matching.threshold, cfg.matching.index)
    analyzer = InsightFaceAnalyzer(cfg.analyzer)
    dispatcher = AlertDispatcher(build_sink(cfg.alerts))
    renderer = RedBoxRenderer()
    policy = ConfirmationPolicy(cfg.confirmation.min_hits)

    runtimes = []
    for cam in cameras:
        pipeline = CameraPipeline(
            camera_id=cam.id,
            analyzer=analyzer,
            matcher=matcher,
            tracker=IoUTracker(cfg.tracking.iou_threshold, cfg.tracking.max_age, cfg.confirmation.window),
            policy=policy,
            cooldown=Cooldown(cfg.confirmation.cooldown_s),
            publisher=dispatcher,
            renderer=renderer,
            process_every_n=cam.process_every_n,
        )
        runtimes.append(CameraRuntime(create_source(cam), pipeline))
    return CameraManager(runtimes, dispatcher, matcher, cfg.display, cfg.reload_interval_s)
