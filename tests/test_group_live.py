import csv
import json

import cv2
import numpy as np
import pytest

from behavtrack.arena import Arena, Zone
from behavtrack.config import new_config
from behavtrack.cumulative import build_group_summary
from behavtrack.geometry import Calibration, Polygon
from behavtrack.group_live import group_candidates, interval_summary, run_group_live
from behavtrack.trackers.blob import BlobTracker


def group_fixture():
    cfg = new_config("open_field")
    cfg["tracking"].update({"min_area_cm2": 2, "max_area_cm2": 80, "blur": 1, "tail_removal_cm": 0.1})
    arena = Arena("A", Polygon([(0, 0), (199, 0), (199, 199), (0, 199)]), Calibration(2),
                  [Zone("food", Polygon([(0, 0), (95, 0), (95, 199), (0, 199)]))])
    cfg["arenas"] = [arena.to_dict()]
    background = np.full((200, 200, 3), 220, np.uint8)
    return cfg, arena, background


def mouse_frame(background, centers):
    frame = background.copy()
    for center in centers:
        cv2.circle(frame, center, 10, (20, 20, 20), -1)
    return frame


def test_group_detection_separated_and_overlap():
    cfg, arena, background = group_fixture()
    tracker = BlobTracker([arena], cfg, background.shape[:2])
    tracker.set_background(background)
    separate = mouse_frame(background, [(40, 60), (140, 80)])
    centers, count, _ = group_candidates(tracker, tracker.preprocess(separate), 2, arena.zone("food").shape)
    assert len(centers) == 2 and count == 1
    overlapping = mouse_frame(background, [(40, 60), (55, 60)])
    centers, count, _ = group_candidates(tracker, tracker.preprocess(overlapping), 2, arena.zone("food").shape)
    assert len(centers) < 2 and count is None
    excess = mouse_frame(background, [(40, 60), (100, 60), (150, 60)])
    assert group_candidates(tracker, tracker.preprocess(excess), 2, arena.zone("food").shape)[1] is None


def test_intervals_do_not_turn_unknown_into_zero():
    result = interval_summary([{"time_s": 0, "approach": 1}, {"time_s": .5, "approach": None},
                               {"time_s": 1, "approach": 0}, {"time_s": 3, "approach": 2}])
    assert result["known_s"] == .5
    assert result["unknown_s"] == 2.5
    assert result["mean_approach_count"] == 1
    assert result["positive_pct"] == 100


def test_group_live_saves_unknown_and_cumulative_report(tmp_path, monkeypatch):
    cfg, arena, background = group_fixture()
    separate = mouse_frame(background, [(40, 60), (140, 80)])
    overlap = mouse_frame(background, [(40, 60), (55, 60)])
    frames = [background] * 31 + [separate, overlap, separate]
    class Camera:
        def read(self):
            return (True, frames.pop(0).copy()) if frames else (False, None)
        def release(self):
            pass
    script = [ord("b")] + [-1] * 29 + [ord("s"), -1, -1, ord("q")]
    clock = iter(x / 10 for x in range(1000))
    monkeypatch.setattr("behavtrack.group_live.open_camera", lambda index: Camera())
    monkeypatch.setattr("behavtrack.group_live.time.perf_counter", lambda: next(clock))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: script.pop(0))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    output_root = tmp_path / "한글 결과"
    output_root.mkdir()
    result = run_group_live(cfg, 0, output_root, 2, cage_id="C1", group="G1")
    video = cv2.VideoCapture(str(result / "raw.mp4"))
    assert video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    video.release()
    marked = cv2.VideoCapture(str(result / "tracked.mp4"))
    assert marked.isOpened() and int(marked.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    marked.release()
    data = json.loads((result / "group_session.json").read_text(encoding="utf-8"))
    with (result / "group_frames.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["approach_count_estimate"] for row in rows] == ["1", "unknown", "1"]
    assert data["summary"]["known_s"] == pytest.approx(.1)
    assert data["summary"]["unknown_s"] == pytest.approx(.1)
    from behavtrack.sources import camera_timestamps
    timestamps = camera_timestamps(result / "raw.mp4", 3)
    assert timestamps == pytest.approx([0, .1, .2])
    report, output, count, skipped = build_group_summary(output_root)
    assert count == 1 and not skipped
    assert "C1" in report.read_text(encoding="utf-8")
    assert "판정 불가 시간" in output.read_text(encoding="utf-8-sig")
