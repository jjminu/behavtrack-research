"""Approach ②: pose import (DeepLabCut CSV, SLEAP HDF5), keypoint metrics, social pairs, DLC-Live adapter."""

import csv
import json
import sys
import types
import shutil

import numpy as np
import pytest

from behavtrack import synth
from behavtrack.pipeline import track_video
from behavtrack.pose_import import import_pose, read_dlc_csv
from behavtrack.tracks import Tracks
from behavtrack.sources import camera_timestamps

from conftest import FPS


def _metrics(session, track="A"):
    return json.loads((session / f"metrics_{track}.json").read_text())["metrics"]


def test_dlc_import_matches_ground_truth(synthetic, tmp_path):
    dlc = synth.write_dlc_csv(synthetic["path"], tmp_path / "pose.csv")
    session = import_pose(dlc, synth.open_field_config(), tmp_path / "res", video=synthetic["video"])
    m = _metrics(session)
    assert m["freezing_method"] == "keypoints"
    assert m["freezing_bouts"] == 2  # body parts keep moving while grooming
    assert abs(m["freezing_time_s"] - 10.0) < 1.0
    assert abs(m["distance_cm"] - 12 * 46) / (12 * 46) < 0.05
    tr = Tracks.load(session / "tracks_A.csv")
    assert set(tr.keypoints) == {"nose", "left_ear", "right_ear", "center", "tail_base"}


def test_dlc_multi_animal_header(tmp_path):
    path = tmp_path / "ma.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer"] + ["S"] * 6)
        w.writerow(["individuals"] + ["mouse1"] * 3 + ["mouse2"] * 3)
        w.writerow(["bodyparts"] + ["nose"] * 3 + ["nose"] * 3)
        w.writerow(["coords"] + ["x", "y", "likelihood"] * 2)
        w.writerow([0, 1, 2, 0.9, 10, 20, 0.8])
        w.writerow([1, 3, 4, 0.9, "", "", ""])
    frames, animals = read_dlc_csv(path)
    assert frames.tolist() == [0, 1]
    assert set(animals) == {"mouse1", "mouse2"}
    assert animals["mouse1"]["nose"][1].tolist() == [3, 4, 0.9]
    assert np.isnan(animals["mouse2"]["nose"][1, 0])


def test_camera_video_reuses_original_irregular_capture_times(synthetic, tmp_path):
    video = tmp_path / "raw.mp4"
    shutil.copyfile(synthetic["video"], video)
    n = len(synthetic["path"]["t"])
    times = np.cumsum(np.r_[0., np.tile([0.02, 0.08], n // 2)[:n - 1]])
    (tmp_path / "session.json").write_text(json.dumps({"source": {"type": "camera", "video": "raw.mp4", "frames": n}}))
    with (tmp_path / "frame_times.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"])
        w.writerows(enumerate(times))
    pose = synth.write_dlc_csv(synthetic["path"], tmp_path / "pose.csv")
    imported = import_pose(pose, synth.open_field_config(), tmp_path / "pose_results", video=video, analyze=False)
    imported_tracks = Tracks.load(imported / "tracks_A.csv")
    assert imported_tracks.time == pytest.approx(times, abs=0.0001)
    assert json.loads((imported / "session.json").read_text())["source"]["timebase"] == "camera_timestamps"
    tracked = track_video(synth.open_field_config(), video, tmp_path / "retracked", start_s=1, end_s=3, analyze=False)
    retracked = Tracks.load(tracked / "tracks_A.csv")
    a, b = np.searchsorted(times, [1, 3])
    assert retracked.frame.tolist() == list(range(a, b))
    assert retracked.time == pytest.approx(times[a:b] - times[a], abs=0.0001)
    with pytest.raises(SystemExit, match="일치하지"):
        camera_timestamps(video, n - 1)


def test_sleap_two_animals_social(tmp_path):
    h5py = pytest.importorskip("h5py")
    n = 20 * FPS
    t = np.arange(n) / FPS
    nodes = ["nose", "center", "tail_base"]
    # animal 1 stands still; animal 2 approaches nose-to-nose during 5-10 s
    a1 = {"center": (200, 200), "nose": (240, 200), "tail_base": (166, 200)}
    pos2 = np.where(((t >= 5) & (t < 10))[:, None], [[300, 200]], [[350, 300]])
    tracks = np.zeros((2, 2, len(nodes), n))
    for j, node in enumerate(nodes):
        tracks[0, :, j, :] = np.array(a1[node])[:, None]
    tracks[1, :, 1, :] = pos2.T
    tracks[1, :, 0, :] = (pos2 + [-55, 0]).T  # nose of animal 2 points left, at x=245 when close
    tracks[1, :, 2, :] = (pos2 + [34, 0]).T
    path = tmp_path / "two.analysis.h5"
    with h5py.File(path, "w") as f:
        f["tracks"] = tracks
        f["node_names"] = np.array([s.encode() for s in nodes])
        f["track_names"] = np.array([b"m1", b"m2"])
        f["point_scores"] = np.ones((2, len(nodes), n))
    session = import_pose(path, synth.open_field_config(), tmp_path / "res", fps=FPS)
    pairs = json.loads((session / "social.json").read_text())
    (pair, m), = pairs.items()
    assert pair == "A-m1~A-m2"
    assert m["nose_nose_time_s"] == pytest.approx(5, abs=0.2)
    assert m["nose_nose_bouts"] == 1


def test_dlclive_adapter_with_stub_model(synthetic, tmp_path, monkeypatch):
    """The real-time pose tracker plugs into the same pipeline (stubbed model, no TensorFlow needed)."""
    parts = synth.body_points(synthetic["path"])
    names = list(parts)
    poses = np.stack([np.column_stack([synth.OFFSET + parts[b] * synth.PX_PER_CM, np.full(len(parts[b]), 0.99)])
                      for b in names], axis=1)

    class FakeDLCLive:
        def __init__(self, model_path, **kw):
            self.i = 0

        def init_inference(self, frame):
            pass

        def get_pose(self, frame):
            self.i += 1
            return poses[self.i - 1]

    monkeypatch.setitem(sys.modules, "dlclive", types.SimpleNamespace(DLCLive=FakeDLCLive))
    session = track_video(synth.open_field_config(), synthetic["video"], tmp_path / "res", name="live_pose",
                          tracker="dlclive", model=str(tmp_path), bodyparts=names)
    tr = Tracks.load(session / "tracks_A.csv")
    assert set(tr.keypoints) == set(names)
    m = _metrics(session)
    assert m["freezing_method"] == "keypoints" and m["freezing_bouts"] == 2
