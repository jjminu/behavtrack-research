"""Approach ①: tracking and common metrics against the synthetic ground truth."""

import json

import cv2
import numpy as np

from behavtrack import synth
from behavtrack.config import new_config
from behavtrack.geometry import Calibration, Polygon
from behavtrack.pipeline import analyze_session, track_video
from behavtrack.setup_gui import build_arena
from behavtrack.templates import paradigm
from behavtrack.tracks import Tracks

from conftest import FPS


def _metrics(session, track="A"):
    return json.loads((session / f"metrics_{track}.json").read_text())["metrics"]


def test_position_accuracy(synthetic):
    tr = Tracks.load(synthetic["session"] / "tracks_A.csv")
    gt = synthetic["path"]["pos"]
    err = np.hypot(*((tr.xy - synth.OFFSET) / synth.PX_PER_CM - gt).T)
    assert np.isfinite(err).mean() > 0.99
    assert np.nanmedian(err) < 1.0  # cm; the head pulls the blob centroid slightly forward


def test_distance_and_freezing(synthetic):
    m = _metrics(synthetic["session"])
    moving_s = 60 - 4 - 6 - 4
    assert abs(m["distance_cm"] - 12.0 * moving_s) / (12.0 * moving_s) < 0.05
    assert m["freezing_method"] == "motion"
    assert m["freezing_bouts"] == 2  # grooming (48-52 s) must NOT count as freezing
    assert abs(m["freezing_time_s"] - 10.0) < 0.6
    assert abs(m["freezing_latency_s"] - 12.0) < 0.3


def test_center_time(synthetic):
    gt = synthetic["path"]["pos"]
    inside = ((gt >= 10) & (gt <= 30)).all(axis=1)
    m = _metrics(synthetic["session"])
    assert abs(m["center_time_s"] - inside.sum() / FPS) < 1.0


def test_outputs_exist(synthetic):
    s = synthetic["session"]
    for name in ["session.json", "tracks_A.csv", "frames_A.csv", "bins_A.csv", "report.html", "background.png",
                 "trajectory_A.png", "heatmap_A.png", "timeline_A.png", "tracked.mp4"]:
        assert (s / name).exists(), name
    assert "중앙 체류시간" in (s / "report.html").read_text(encoding="utf-8")


def test_reanalysis_with_overrides(synthetic):
    base = analyze_session(synthetic["session"], make_report=False, verbose=False)["A"].metrics
    res = analyze_session(synthetic["session"], overrides=["analysis.end_s=30"], make_report=False, verbose=False)
    assert abs(res["A"].metrics["duration_s"] - 30.0) < 0.1
    wide = analyze_session(synthetic["session"], overrides=["params.center_fraction=0.7"],
                           make_report=False, verbose=False)["A"].metrics
    assert wide["center_time_s"] > base["center_time_s"] + 3  # template zone rebuilt from the new param
    analyze_session(synthetic["session"], make_report=False, verbose=False)  # restore full-length outputs


def test_multi_arena(tmp_path):
    """Two arenas side by side in one frame, one animal each."""
    v1 = synth.render(synth.simulate_path(seconds=20, seed=1, freezes=((5, 8),), grooms=()), tmp_path / "a.mp4")
    v2 = synth.render(synth.simulate_path(seconds=20, seed=7, freezes=(), grooms=()), tmp_path / "b.mp4")
    c1, c2 = cv2.VideoCapture(str(v1)), cv2.VideoCapture(str(v2))
    out = cv2.VideoWriter(str(tmp_path / "two.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 480))
    while True:
        ok1, f1 = c1.read()
        ok2, f2 = c2.read()
        if not (ok1 and ok2):
            break
        out.write(np.hstack([f1, f2]))
    out.release()
    cfg = new_config("open_field")
    arenas = []
    for name, dx in (("A1", 0), ("A2", 480)):
        corners = synth.corners_px() + [dx, 0]
        arenas.append(build_arena(paradigm("open_field"), name, Polygon(corners), Calibration.from_rect(corners, 40, 40),
                                  [], cfg["params"]).to_dict())
    cfg["arenas"] = arenas
    session = track_video(cfg, tmp_path / "two.mp4", tmp_path / "res", name="two")
    m1, m2 = _metrics(session, "A1"), _metrics(session, "A2")
    assert m1["tracked_pct"] > 99 and m2["tracked_pct"] > 99
    assert m1["freezing_bouts"] == 1 and m2["freezing_bouts"] == 0
    assert abs(m2["distance_cm"] - 12 * 20) / 240 < 0.06
