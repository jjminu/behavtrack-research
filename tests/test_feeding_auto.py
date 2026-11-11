import csv
import json

import joblib
import cv2
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from behavtrack.classifier import features
from behavtrack.feeding_auto import analyze_pose_session, summarize_auto
from behavtrack.config import complete
from behavtrack.pose_import import import_pose
from behavtrack.tracks import Tracks


def make_session(tmp_path):
    session = tmp_path / "pose_session"
    session.mkdir()
    frames = np.arange(60)
    t = frames / 10
    tracks = {}
    for name, x in [("A-one", 10), ("A-two", 70)]:
        xy = np.column_stack([np.full(len(t), x, float), np.full(len(t), 50., float), np.ones(len(t))])
        if name == "A-two":
            xy[10] = [np.nan, np.nan, 0]
            xy[30, 0] = 10  # duplicate detection, no valid cage count
        tr = Tracks(frames, t, xy[:, 0], xy[:, 1], {}, {"nose": xy})
        tr.save(session / f"tracks_{name}.csv")
        tracks[name] = {"arena": "A", "individual": name}
    config = {"paradigm": "custom", "arenas": [{"name": "A", "shape": {"type": "rect", "rect": [0, 0, 100, 100]},
              "zones": [{"name": "food", "shape": {"type": "rect", "rect": [0, 0, 30, 100]}}]}]}
    (session / "session.json").write_text(json.dumps({"source": {"type": "pose", "tracker": "test"}, "tracks": tracks, "config": config}))
    return session


def test_auto_pose_and_missing_behavior_model(tmp_path):
    session = make_session(tmp_path)
    out = tmp_path / "out"
    analyze_pose_session(session, out, "G3-A", "G3", 2, "food")
    summary = json.loads((out / "auto_summary.json").read_text())
    assert summary["method"] == "multi_animal_pose_only"
    assert summary["feeding_prediction"]["known_s"] == 0
    assert summary["feeding_prediction"]["mean_mice_known"] is None
    with (out / "auto_frames.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["approach_count"] == "1"
    assert rows[10]["approach_count"] == ""  # missing nose is not zero
    assert rows[30]["approach_count"] == ""  # duplicate detections
    assert all(r["feeding_predicted_count"] == "" for r in rows)
    with pytest.raises(SystemExit, match="트랙"):
        analyze_pose_session(session, tmp_path / "bad", "G3-A", "G3", 3, "food")
    with pytest.raises(FileExistsError):
        analyze_pose_session(session, out, "G3-A", "G3", 2, "food")
    summarize_auto([out], tmp_path / "all.csv")
    with (tmp_path / "all.csv").open(encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["cage_id"] == "G3-A" and row["feeding_prediction_mean_mice_known"] == ""


def test_classifier_and_missing_context_abstain(tmp_path):
    session = make_session(tmp_path)
    tr = Tracks.load(session / "tracks_A-one.csv")
    names, x = features(tr, 1., 0.6)
    model = DummyClassifier(strategy="constant", constant=1).fit(x[:2], [0, 1])
    model_path = tmp_path / "feeding.model"
    joblib.dump({"model": model, "behavior": "feeding", "features": names}, model_path)
    out = tmp_path / "prediction"
    analyze_pose_session(session, out, "C", "blind", 2, "food", model_path)
    with (out / "auto_frames.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["feeding_predicted_count"] == ""  # insufficient context at start
    assert rows[15]["feeding_predicted_count"] == ""  # context includes missing keypoint
    assert rows[40]["feeding_predicted_count"] == "2"
    assert rows[30]["feeding_predicted_count"] == ""  # duplicated nose
    with pytest.raises(SystemExit, match="구역"):
        analyze_pose_session(session, tmp_path / "bad-zone", "C", "blind", 2, "missing", model_path)

    uncertain = DummyClassifier(strategy="prior").fit(x[:2], [0, 1])
    joblib.dump({"model": uncertain, "behavior": "feeding", "features": names}, model_path)
    out2 = tmp_path / "uncertain"
    analyze_pose_session(session, out2, "C", "blind", 2, "food", model_path)
    report = json.loads((out2 / "auto_summary.json").read_text())
    assert report["feeding_prediction"]["known_s"] == 0
    assert report["feeding_prediction"]["any_positive_s"] is None


def test_imported_pose_video_alignment(tmp_path):
    video = tmp_path / "movie.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.zeros((100, 100, 3), np.uint8))
    writer.release()
    pose = tmp_path / "poses.csv"
    with pose.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer", "model", "model", "model", "model", "model", "model"])
        w.writerow(["individuals", "one", "one", "one", "two", "two", "two"])
        w.writerow(["bodyparts", "nose", "nose", "nose", "nose", "nose", "nose"])
        w.writerow(["coords", "x", "y", "likelihood", "x", "y", "likelihood"])
        for i in range(30):
            w.writerow([i, 10, 50, 1, 70, 50, 1])
    config = complete({"paradigm": "custom", "arenas": [{"name": "A", "shape": {"type": "rect", "rect": [0, 0, 100, 100]},
                      "zones": [{"name": "food", "shape": {"type": "rect", "rect": [0, 0, 30, 100]}}]}]})
    session = import_pose(pose, config, tmp_path / "sessions", video=video, name="cage", analyze=False)
    analyze_pose_session(session, tmp_path / "auto", "C", "G3", 2, "food")
    summary = json.loads((tmp_path / "auto" / "auto_summary.json").read_text())
    assert summary["approach"]["mean_mice_known"] == 1
    assert set(summary["source_file_sha256"]) == {"path", "video"}
