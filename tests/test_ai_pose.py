import numpy as np
import csv
import json
import queue
from pathlib import Path
import cv2

from behavtrack.ai_pose import approach_from_noses
from behavtrack.geometry import Polygon


def test_detector_coco_boxes_reach_region_counting_as_corners():
    from behavtrack.ai_pose import GroupDetectorCPU
    from behavtrack.group_approach import evaluate

    raw_boxes = np.array([[300., 200., 100., 80.]])
    class Detector:
        def inference(self, frames):
            return [{"bboxes": raw_boxes, "bbox_scores": np.array([.99])}]
    detector = GroupDetectorCPU.__new__(GroupDetectorCPU)
    detector.detector = Detector()
    prediction = detector.predict(np.zeros((576, 928, 3), dtype=np.uint8))
    assert prediction["boxes"] == [[300., 200., 400., 280.]]
    assert prediction["box_format"] == "xyxy"
    assert np.array_equal(raw_boxes, [[300., 200., 100., 80.]])
    assert evaluate(prediction, (.35, .30, .65, .70), (928, 576), 1)["status"] == "approach"
    assert evaluate(prediction, (0, 0, .2, 1), (928, 576), 1)["status"] == "away"
    detector.detector.inference = lambda frames: [{"bboxes": [], "bbox_scores": []}]
    empty = detector.predict(np.zeros((576, 928, 3), dtype=np.uint8))
    assert empty["boxes"] == []
    assert evaluate(empty, (0, 0, 1, 1), (928, 576), 1)["status"] == "unknown"


def test_count_five_noses_in_food_and_unknown_cases():
    arena = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    food = Polygon([(0, 0), (100, 0), (100, 200), (0, 200)])
    noses = np.array([[20, 20, .9], [40, 50, .9], [130, 50, .9], [160, 90, .9], [180, 120, .9]])
    assert approach_from_noses(noses, arena, food) == (2, "estimated")
    assert approach_from_noses(noses[:4], arena, food) == (None, "detected_count_mismatch")
    low = noses.copy(); low[0, 2] = .1
    assert approach_from_noses(low, arena, food) == (None, "low_nose_confidence")
    overlap = noses.copy(); overlap[1, :2] = overlap[0, :2]
    assert approach_from_noses(overlap, arena, food) == (None, "duplicate_or_overlapping_noses")
    outside = noses.copy(); outside[0, 0] = 220
    assert approach_from_noses(outside, arena, food) == (None, "outside_cage")


def test_ai_webcam_saves_sampled_result_and_timestamps(tmp_path, monkeypatch):
    from behavtrack import ai_live
    from behavtrack.arena import Arena, Zone
    from behavtrack.config import new_config
    from behavtrack.geometry import Calibration
    from behavtrack.sources import camera_timestamps

    arena = Arena("A", Polygon([(0, 0), (199, 0), (199, 199), (0, 199)]), Calibration(2),
                  [Zone("food", Polygon([(0, 0), (95, 0), (95, 199), (0, 199)]))])
    cfg = new_config("open_field")
    cfg["arenas"] = [arena.to_dict()]
    ai_python = tmp_path / ".ai_venv" / "Scripts" / "python.exe"
    ai_python.parent.mkdir(parents=True)
    ai_python.touch()
    class Camera:
        def read(self):
            return True, np.full((200, 200, 3), 180, np.uint8)
        def release(self):
            pass
    class Requests:
        def __init__(self, events):
            self.events = events
        def put_nowait(self, item):
            frame, captured, image = item
            noses = [[20, 20, .9], [40, 50, .9], [130, 50, .9], [160, 90, .9], [180, 120, .9]]
            self.events.put(("result", (frame, captured, {"detected": 5, "noses": noses, "reason": "candidate"})))
    class Worker:
        def __init__(self, executable):
            self.events = queue.Queue()
            self.events.put(("ready", None))
            self.requests = Requests(self.events)
        def close(self):
            pass
    keys = iter([ord("s"), -1, ord("q")])
    tick = iter(x / 10 for x in range(1000))
    monkeypatch.setattr(ai_live, "ROOT", tmp_path)
    monkeypatch.setattr(ai_live, "AIWorker", Worker)
    monkeypatch.setattr(ai_live, "open_camera", lambda index: Camera())
    monkeypatch.setattr(ai_live, "_sha", lambda path: "test")
    monkeypatch.setattr(ai_live.time, "perf_counter", lambda: next(tick))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    session = ai_live.run_ai_live(cfg, 0, tmp_path / "result", cage_id="A", interval=5)
    info = json.loads((session / "ai_group_session.json").read_text(encoding="utf-8"))
    with (session / "ai_samples.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["approach_count"] == "2"
    assert info["summary"]["known_samples"] == 1
    assert len(camera_timestamps(session / "raw.mp4", info["frames"])) == info["frames"]
    assert "표본 분석" in (session / "report.html").read_text(encoding="utf-8")
    from behavtrack.cumulative import build_ai_group_summary
    report, csv_path, count, skipped = build_ai_group_summary(session.parent)
    assert count == 1 and not skipped
    assert "AI 공동 사육" in report.read_text(encoding="utf-8")
    assert "판정 가능 표본" in csv_path.read_text(encoding="utf-8-sig")
