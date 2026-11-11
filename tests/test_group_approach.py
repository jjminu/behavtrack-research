import csv
import json

import cv2
import numpy as np
import pytest

from behavtrack.group_approach import Observation, build_summary, evaluate
from behavtrack.simple_app import image_point
from behavtrack.sources import camera_timestamps


def prediction():
    return {"boxes": [[10, 10, 30, 30], [60, 10, 80, 30], [110, 10, 130, 30],
                      [160, 10, 180, 30], [160, 70, 180, 90]], "scores": [.9] * 5}


def test_whole_frame_approach_and_unknown_are_distinct():
    p = prediction()
    assert evaluate(p, (0, 0, .5, 1), (200, 100), 5)["count"] == 2
    assert evaluate(p, (0, .5, .5, 1), (200, 100), 5)["status"] == "away"
    p["scores"] = p["scores"][:4]
    assert evaluate(p, (0, 0, .5, 1), (200, 100), 5)["count"] is None
    p = prediction()
    p["boxes"][1] = p["boxes"][0]
    assert evaluate(p, (0, 0, .5, 1), (200, 100), 5)["reason"] == "overlapping_detections"


def test_letterboxed_mouse_mapping_uses_entire_image():
    assert image_point(100, 50, (100, 50, 600, 300)) == (0, 0)
    assert image_point(400, 200, (100, 50, 600, 300)) == (.5, .5)
    assert image_point(99, 200, (100, 50, 600, 300)) is None


def test_recording_samples_reports_and_original_timestamps(tmp_path):
    session = Observation(tmp_path, "=테스트 <group>", 5, (0, 0, .5, 1), (200, 100), 10)
    frame = np.full((100, 200, 3), 120, np.uint8)
    for i in range(4):
        session.record(frame, i * .1)
    session.accept(0, 0, prediction())
    far = prediction()
    far["boxes"] = [[110, i*18, 120, i*18+10] for i in range(5)]
    session.accept(1, .1, far)
    session.accept(2, .2, {"boxes": [], "scores": []})
    session.accept(3, .3, {}, reason="inference_incomplete")
    session.close()
    session.close()  # close is safe when called by both stop and window close
    data = json.loads((session.folder / "group_approach_session.json").read_text(encoding="utf-8"))
    assert data["observation_area"] == "entire_frame"
    assert data["summary"]["approach_pct"] == 50
    assert data["summary"]["unknown_samples"] == 2
    assert data["summary"]["mean_count"] == 1
    assert np.allclose(camera_timestamps(session.folder / "raw.mp4", 4), [0, .1, .2, .3])
    report = build_summary(tmp_path)
    assert "&lt;group&gt;" in report.read_text(encoding="utf-8")
    assert "판정 불가" in report.read_text(encoding="utf-8")
    with report.with_name("전체_결과.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1][0].startswith("'=")
    cap = cv2.VideoCapture(str(session.folder / "raw.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    cap.release()


def test_partial_recording_preserves_error(tmp_path):
    session = Observation(tmp_path, "그룹", 5, (0, 0, .5, 1), (200, 100), 10)
    session.record(np.zeros((100, 200, 3), np.uint8), 0)
    session.close("카메라 연결 끊김")
    data = json.loads((session.folder / "group_approach_session.json").read_text(encoding="utf-8"))
    assert data["state"] == "partial" and data["error"] == "카메라 연결 끊김"
    assert data["summary"]["approach_pct"] is None


def test_simple_screen_start_stop_records_without_keypress_or_cage_setup(tmp_path, monkeypatch):
    import queue
    import tkinter as tk
    from types import SimpleNamespace
    from behavtrack import simple_app

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk runtime not available in this test interpreter: {exc}")
    root.withdraw()
    executable = tmp_path / ".ai_venv/Scripts/python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    class Camera:
        released = False
        def read(self):
            return True, np.full((100, 200, 3), 100, np.uint8)
        def release(self):
            self.released = True
    camera = Camera()
    class Requests:
        def __init__(self, events):
            self.events = events
        def put_nowait(self, item):
            self.events.put(("result", (item[0], item[1], prediction())))
    class Worker:
        def __init__(self, executable, args):
            assert args == ["--detect-only", "--animals", "5"]
            self.events = queue.Queue()
            self.events.put(("ready", None))
            self.requests = Requests(self.events)
        def close(self):
            pass
    monkeypatch.setattr(simple_app, "ROOT", tmp_path)
    monkeypatch.setattr(simple_app, "AIWorker", Worker)
    monkeypatch.setattr(simple_app, "open_camera", lambda *args: camera)
    ui = simple_app.SimpleApp(root)
    try:
        ui.open()
        ui.viewport = (0, 0, 200, 100)
        ui.begin_mark(SimpleNamespace(x=0, y=0))
        ui.end_mark(SimpleNamespace(x=100, y=100))
        assert ui.food == (0, 0, .5, 1)
        ui.start()
        assert ui.preparing
        ui.tick()
        assert ui.session is not None and not ui.preparing
        ui.tick()
        assert ui.current.get() == "2마리 접근"
        folder = ui.session.folder
        ui.stop()
        assert ui.session is None
        assert (folder / "report.html").is_file()
        assert json.loads((folder / "group_approach_session.json").read_text(encoding="utf-8"))["summary"]["approach_pct"] == 100
    finally:
        ui.close()
    assert camera.released
