"""Unit tests for the core domain, configuration, storage, matching, and policy."""

import pytest
import numpy as np
import yaml

from facewatch.config import (
    AppConfig,
    ConfigError,
    load_config,
)
from facewatch.domain import BBox, Face, Match, WatchlistEntry
from facewatch.enrollment import EnrollmentError, Enroller
from facewatch.index import NumpyIndex, build_index
from facewatch.matcher import WatchlistMatcher
from facewatch.policy import ConfirmationPolicy, Cooldown
from facewatch.repository import SQLiteWatchlistRepository
from facewatch.tracking import IoUTracker


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class FakeRepo:
    def __init__(self, entries):
        self.entries = entries

    def add(self, entry):
        self.entries.append(entry)

    def remove(self, entry_id):
        original_count = len(self.entries)
        self.entries = [entry for entry in self.entries if entry.id != entry_id]
        return len(self.entries) != original_count

    def list_all(self):
        return self.entries


def test_iou():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0


def test_tracker_keeps_identity_and_splits_people():
    t = IoUTracker()
    first = t.update([BBox(0, 0, 50, 50), BBox(200, 200, 250, 250)])
    second = t.update([BBox(3, 3, 53, 53), BBox(203, 203, 253, 253)])
    assert [x.id for x in first] == [x.id for x in second]
    assert first[0].id != first[1].id


def test_matcher_threshold():
    e = np.zeros(512); e[0] = 1
    m = WatchlistMatcher(FakeRepo([WatchlistEntry("1", "A", unit(e))]), 0.5, "numpy")
    exact_match = m.match([unit(e)])[0]
    assert exact_match is not None
    assert exact_match.entry_id == "1"
    other = np.zeros(512); other[1] = 1
    assert m.match([unit(other)])[0] is None


def test_confirmation_needs_repeated_hits():
    t = IoUTracker(history=5)
    track = t.update([BBox(0, 0, 50, 50)])[0]
    policy = ConfirmationPolicy(min_hits=3)
    hit = Match("1", "A", 0.8)
    for _ in range(2):
        track.history.append(hit)
    assert policy.evaluate(track) is None
    track.history.append(hit)
    confirmed = policy.evaluate(track)
    assert confirmed is not None
    assert confirmed.entry_id == "1"


def test_cooldown():
    now = [0.0]
    c = Cooldown(10, clock=lambda: now[0])
    assert c.ready("x") and not c.ready("x")
    now[0] = 11
    assert c.ready("x")


def test_bbox_dimensions_clipping_and_partial_overlap():
    bbox = BBox(2, 3, 12, 13)
    assert (bbox.width, bbox.height, bbox.area) == (10, 10, 100)
    assert BBox(20, 20, 10, 10).area == 0
    assert bbox.iou(BBox(7, 7, 17, 17)) == pytest.approx(30 / 170)
    assert BBox.from_xyxy((-2.2, 1.6, 10.4, 20.7), 10, 20).x1 == 0


def test_default_config_and_yaml_loading(tmp_path):
    cfg = AppConfig()
    assert cfg.analyzer.model_name == "buffalo_l"
    assert cfg.matching.threshold == 0.45

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "log_level": "DEBUG",
        "matching": {"threshold": 0.7, "index": "numpy"},
        "cameras": [{"id": "front", "uri": "0"}],
    }))
    loaded = load_config(path)
    assert loaded.log_level == "DEBUG"
    assert loaded.matching.threshold == 0.7
    assert loaded.cameras[0].id == "front"


def test_config_rejects_invalid_values():
    with pytest.raises(ConfigError, match="Unknown keys"):
        AppConfig.from_dict({"unknown": True})
    with pytest.raises(ConfigError, match="unique"):
        AppConfig.from_dict({"cameras": [
            {"id": "same", "uri": "0"},
            {"id": "same", "uri": "1"},
        ]})
    with pytest.raises(ConfigError, match="min_hits"):
        AppConfig.from_dict({"confirmation": {"window": 2, "min_hits": 3}})
    with pytest.raises(ConfigError, match="threshold"):
        AppConfig.from_dict({"matching": {"threshold": 1.5}})
    with pytest.raises(ConfigError, match="process_every_n"):
        AppConfig.from_dict({"cameras": [{"id": "bad", "uri": "0", "process_every_n": 0}]})


def test_repository_add_list_replace_remove_memory_and_metadata():
    repo = SQLiteWatchlistRepository(":memory:")
    entry = WatchlistEntry("p1", "Alice", unit(np.array([1.0] + [0.0] * 511)), {"role": "staff"})
    repo.add(entry)

    listed = repo.list_all()
    assert len(listed) == 1
    assert listed[0].id == "p1"
    assert listed[0].name == "Alice"
    assert listed[0].meta == {"role": "staff"}
    np.testing.assert_array_equal(listed[0].embedding, entry.embedding)

    repo.add(WatchlistEntry("p1", "Alice 2", unit(np.array([0.0] + [1.0] + [0.0] * 510))))
    assert repo.list_all()[0].name == "Alice 2"
    assert repo.remove("p1")
    assert not repo.remove("missing")
    assert repo.list_all() == []


def test_repository_persists_to_disk(tmp_path):
    path = tmp_path / "watchlist.db"
    repo = SQLiteWatchlistRepository(str(path))
    repo.add(WatchlistEntry("p1", "Alice", unit(np.ones(512))))
    reopened = SQLiteWatchlistRepository(str(path))
    assert reopened.list_all()[0].name == "Alice"


def test_numpy_index_empty_and_nearest_match():
    empty = NumpyIndex(np.empty((0, 512), np.float32))
    ids, scores = empty.search(np.zeros((2, 512), np.float32))
    assert ids.tolist() == [-1, -1]
    np.testing.assert_array_equal(scores, np.zeros(2, np.float32))

    matrix = np.vstack([unit(np.array([1.0] + [0.0] * 511)),
                        unit(np.array([0.0, 1.0] + [0.0] * 510))])
    ids, scores = NumpyIndex(matrix).search(np.array([[1.0] + [0.0] * 511], np.float32))
    assert ids.tolist() == [0]
    assert scores[0] == pytest.approx(1.0)
    assert build_index("numpy", matrix).__class__ is NumpyIndex


def test_matcher_reload_updates_entries_and_empty_watchlist():
    repo = FakeRepo([WatchlistEntry("1", "A", unit(np.array([1.0] + [0.0] * 511)))])
    matcher = WatchlistMatcher(repo, 0.5, "numpy")
    assert matcher.match([unit(np.array([1.0] + [0.0] * 511))])[0].entry_id == "1"

    repo.entries = []
    matcher.reload()
    assert matcher.match([unit(np.ones(512))])[0] is None
    assert matcher.match([]) == []


def test_confirmation_average_and_most_common_entry():
    tracker = IoUTracker(history=5)
    track = tracker.update([BBox(0, 0, 10, 10)])[0]
    track.history.extend([
        Match("a", "A", 0.8),
        Match("b", "B", 0.9),
        Match("a", "A", 0.6),
        Match("a", "A", 0.7),
    ])
    confirmed = ConfirmationPolicy(min_hits=3).evaluate(track)
    assert confirmed is not None
    assert confirmed.entry_id == "a"
    assert confirmed.score == pytest.approx(0.7)


def test_cooldown_is_per_key():
    now = [100.0]
    cooldown = Cooldown(5.0, clock=lambda: now[0])
    assert cooldown.ready("camera-a")
    assert cooldown.ready("camera-b")
    assert not cooldown.ready("camera-a")
    now[0] = 105.0
    assert cooldown.ready("camera-a")


class FakeEnrollmentAnalyzer:
    def __init__(self, faces):
        self.faces = faces

    def analyze(self, image):
        return self.faces


class FakeEnrollmentRepo:
    def __init__(self):
        self.entry = None

    def add(self, entry):
        self.entry = entry


def test_enroller_uses_largest_face_and_normalized_average(tmp_path, monkeypatch):
    image_path = tmp_path / "person.jpg"
    image_path.write_bytes(b"fake")
    small = Face(BBox(0, 0, 10, 10), unit(np.array([1.0] + [0.0] * 511)), 0.9)
    large = Face(BBox(0, 0, 30, 30), unit(np.array([0.0, 1.0] + [0.0] * 510)), 0.95)
    analyzer = FakeEnrollmentAnalyzer([small, large])
    repo = FakeEnrollmentRepo()

    monkeypatch.setattr("facewatch.enrollment.cv2.imread",
                        lambda path: np.zeros((50, 50, 3), dtype=np.uint8))
    entry = Enroller(analyzer, repo).enroll("p1", "Alice", [str(image_path)], {"team": "x"})

    assert entry.id == "p1"
    assert entry.name == "Alice"
    assert entry.meta == {"team": "x"}
    np.testing.assert_allclose(np.linalg.norm(entry.embedding), 1.0)
    assert repo.entry is entry


def test_enroller_rejects_missing_image_and_no_face(tmp_path, monkeypatch):
    missing = tmp_path / "missing.jpg"
    with pytest.raises(EnrollmentError, match="Cannot read image"):
        Enroller(FakeEnrollmentAnalyzer([]), FakeEnrollmentRepo()).enroll(
            "p1", "Alice", [str(missing)])

    monkeypatch.setattr("facewatch.enrollment.cv2.imread",
                        lambda path: np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(EnrollmentError, match="No usable face"):
        Enroller(FakeEnrollmentAnalyzer([]), FakeEnrollmentRepo()).enroll(
            "p1", "Alice", [str(tmp_path / "none.jpg")])
