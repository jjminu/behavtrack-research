"""Approach ② extension: train a grooming classifier on one session and apply it to another."""

import csv
import json

import pytest

from behavtrack import synth
from behavtrack.classifier import predict, train
from behavtrack.pipeline import track_video

from conftest import FPS

pytest.importorskip("sklearn")


def test_grooming_classifier(synthetic, tmp_path):
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "start_frame", "end_frame", "track"])
        w.writerow(["grooming", 48 * FPS, 52 * FPS - 1, "A"])
        w.writerow(["reviewed_v2", 0, 60 * FPS - 1, "A"])
    cv = train([(synthetic["session"], labels)], "grooming", tmp_path / "grooming.model")
    assert cv == {}  # a single bout cannot be cross-validated

    # a different animal: grooming at 20-25 s, freezing at 40-44 s
    path = synth.simulate_path(seconds=60, seed=11, freezes=((40, 44),), grooms=((20, 25),))
    video = synth.render(path, tmp_path / "test.mp4", seed=5)
    session = track_video(synth.open_field_config(), video, tmp_path / "res", name="test", analyze=False)
    predict(session, tmp_path / "grooming.model")
    m = json.loads((session / "metrics_A.json").read_text())["metrics"]
    assert m["grooming_time_s"] == pytest.approx(5, abs=1.0)
    assert m["grooming_bouts"] == 1
    assert m["grooming_latency_s"] == pytest.approx(20, abs=1.0)


def test_classifier_requires_reviewed_frames_and_groups_by_animal(tmp_path, monkeypatch):
    import numpy as np
    import behavtrack.classifier as c
    from behavtrack.tracks import Tracks
    from sklearn.model_selection import GroupKFold

    labels = tmp_path / "labels.csv"
    labels.write_text("behavior,start_frame,end_frame,track\ngrooming,10,24,A\nreviewed_v2,0,39,A\n")
    tr = Tracks(np.arange(40), np.arange(40) / 10, np.zeros(40), np.zeros(40))
    def fake_features(session, track):
        yield "A", tr, ["feature"], np.arange(40).reshape(-1, 1)
    monkeypatch.setattr(c, "_session_features", fake_features)
    monkeypatch.setattr(c, "load_session", lambda s: {"animal_id": {"a1": "A", "a2": "A", "b": "B", "c": "C"}[s], "tracks": {"A": {}}})
    class CheckedGroupKFold(GroupKFold):
        def split(self, X, y, groups):
            for train_idx, test_idx in super().split(X, y, groups):
                assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
                yield train_idx, test_idx
    monkeypatch.setattr("sklearn.model_selection.GroupKFold", CheckedGroupKFold)
    cv = c.train([(s, labels) for s in ("a1", "a2", "b", "c")], "grooming", tmp_path / "model", n_estimators=5)
    assert cv["animals"] == 3 and cv["folds"] == 3 and cv["frames"] == 160
    labels.write_text("behavior,start_frame,end_frame,track\ngrooming,10,24,A\nreviewed,0,39,A\n")
    with pytest.raises(SystemExit, match="reviewed_v2"):
        c.train([("a1", labels)], "grooming", tmp_path / "model", n_estimators=5)
