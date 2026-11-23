"""Typed configuration loaded from YAML, validated once at startup."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _build(cls, data: dict[str, Any] | None):
    data = data or {}
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


@dataclass
class AnalyzerConfig:
    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = (640, 640)
    ctx_id: int = 0
    min_det_score: float = 0.6
    min_face_px: int = 40

    def __post_init__(self):
        self.det_size = (int(self.det_size[0]), int(self.det_size[1]))


@dataclass
class MatchingConfig:
    threshold: float = 0.45
    index: str = "auto"  # auto | faiss | numpy


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 15


@dataclass
class ConfirmationConfig:
    window: int = 5
    min_hits: int = 3
    cooldown_s: float = 30.0


@dataclass
class WatchlistConfig:
    db_path: str = "data/watchlist.db"


@dataclass
class AlertsConfig:
    log: bool = True
    snapshot_dir: str | None = "data/alerts"
    webhook_url: str | None = None


@dataclass
class CameraConfig:
    id: str
    uri: str
    process_every_n: int = 1
    enabled: bool = True


_SECTIONS = {
    "analyzer": AnalyzerConfig,
    "matching": MatchingConfig,
    "tracking": TrackingConfig,
    "confirmation": ConfirmationConfig,
    "watchlist": WatchlistConfig,
    "alerts": AlertsConfig,
}


@dataclass
class AppConfig:
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    watchlist: WatchlistConfig = field(default_factory=WatchlistConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    display: bool = True
    reload_interval_s: float = 60.0
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AppConfig":
        raw = dict(raw or {})
        allowed_top_level = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed_top_level
        if unknown:
            raise ConfigError(f"Unknown keys for AppConfig: {sorted(unknown)}")
        sections = {n: _build(k, raw.pop(n, None)) for n, k in _SECTIONS.items()}
        cameras = [_build(CameraConfig, c) for c in (raw.pop("cameras", None) or [])]
        try:
            cfg = cls(cameras=cameras, **sections, **raw)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ConfigError("Camera ids must be unique")
        if self.confirmation.min_hits > self.confirmation.window:
            raise ConfigError("confirmation.min_hits cannot exceed confirmation.window")
        if not -1.0 <= self.matching.threshold <= 1.0:
            raise ConfigError("matching.threshold must be within [-1, 1]")
        if any(c.process_every_n < 1 for c in self.cameras):
            raise ConfigError("camera.process_every_n must be >= 1")


def load_config(path: str | Path | None = None) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    return AppConfig.from_dict(raw)
