"""The OpenCV windows (setup drawing, live camera loop) driven by scripted clicks and key presses."""

import json

import cv2
import numpy as np
import pytest

from behavtrack import pipeline, setup_gui, synth
from behavtrack.config import load_config


class ScriptedWindow:
    """Stands in for cv2 window functions: replays clicks/keys, one action per waitKey call."""

    def __init__(self, script):
        self.script = list(script)
        self.callback = None
        self.frames_shown = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "setMouseCallback", lambda win, cb: setattr(self, "callback", cb))
        monkeypatch.setattr(cv2, "imshow", lambda *a: setattr(self, "frames_shown", self.frames_shown + 1))
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(cv2, "waitKey", self.wait_key)

    def wait_key(self, delay=0):
        if not self.script:
            return ord("q")
        action = self.script.pop(0)
        if action is None:
            return -1
        if isinstance(action, tuple):
            self.callback(cv2.EVENT_LBUTTONDOWN, int(action[0]), int(action[1]), 0, None)
            return -1
        return ord(action) if isinstance(action, str) else action


def test_setup_open_field_by_clicks(tmp_path, monkeypatch):
    frame = np.full((480, 480, 3), 200, np.uint8)
    corners = [(440, 440), (40, 40), (440, 40), (40, 440)]  # any click order works
    win = ScriptedWindow([*corners, None, 13])  # 4 corners, then Enter to accept the layout
    win.install(monkeypatch)
    out = tmp_path / "of.json"
    setup_gui.run_setup(frame, "open_field", out, arena_size=(40, 40))
    cfg = load_config(out)
    arena = cfg["arenas"][0]
    assert arena["calibration"]["px_per_cm"] == 10.0
    assert [z["name"] for z in arena["zones"]] == ["center"]
    assert np.allclose(arena["zones"][0]["shape"]["points"], [[140, 140], [340, 140], [340, 340], [140, 340]])
    assert out.with_suffix(".preview.png").exists()


def test_setup_food_zone_rejects_outside_and_saves_inside(tmp_path, monkeypatch):
    frame = np.full((480, 480, 3), 200, np.uint8)
    arena = [(40, 40), (440, 40), (440, 440), (40, 440)]
    outside = [(10, 70), (80, 70), (80, 120), 13]
    food = [(70, 70), (130, 70), (130, 120), (70, 120), 13]
    win = ScriptedWindow([*arena, *outside, *food, None, 13])
    win.install(monkeypatch)
    out = tmp_path / "food.json"
    setup_gui.run_setup(frame, "open_field", out, arena_size=(40, 40), food_zone=True)
    zones = load_config(out)["arenas"][0]["zones"]
    assert [z["name"] for z in zones] == ["food", "center"]
    assert np.allclose(zones[0]["shape"]["points"], food[:-1])


def test_setup_live_camera_updates_through_drawing_and_confirmation(tmp_path, monkeypatch):
    initial = np.zeros((200, 200, 3), np.uint8)
    arena = [(20, 20), (180, 20), (180, 180), (20, 180)]
    food = [(30, 30), (80, 30), (80, 80), 13]
    win = ScriptedWindow([*arena, *food, None, 13])
    win.install(monkeypatch)
    seen = []
    monkeypatch.setattr(cv2, "imshow", lambda name, img: seen.append(img.copy()))
    count = 0

    def next_frame():
        nonlocal count
        count += 1
        return True, np.full_like(initial, count)

    out = tmp_path / "live.json"
    setup_gui.run_setup(initial, "open_field", out, arena_size=(40, 40), food_zone=True,
                        frame_provider=next_frame)
    assert len(seen) >= 10
    # Bottom-left lies outside text and shapes; the camera changed on every shown frame.
    assert [int(img[190, 5, 0]) for img in seen] == list(range(1, len(seen) + 1))
    assert np.all(cv2.imread(str(out.with_suffix(".reference.png"))) == count)
    assert load_config(out)["arenas"][0]["zones"][0]["name"] == "food"


def test_setup_camera_disconnect_does_not_save(tmp_path, monkeypatch):
    win = ScriptedWindow([None])
    win.install(monkeypatch)
    out = tmp_path / "live.json"
    with pytest.raises(SystemExit, match="카메라 영상이 끊어졌습니다"):
        setup_gui.run_setup(np.zeros((100, 100, 3), np.uint8), "open_field", out,
                            arena_size=(40, 40), frame_provider=lambda: (False, None))
    assert not out.exists()


def test_cli_releases_camera_when_setup_fails(monkeypatch):
    from argparse import Namespace
    from behavtrack import cli, sources

    class Camera:
        released = False

        def read(self):
            return True, np.zeros((100, 100, 3), np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(sources, "open_camera", lambda *args: camera)

    def fail(*args, **kwargs):
        assert kwargs["frame_provider"].__self__ is camera
        raise SystemExit("interrupted")

    monkeypatch.setattr(setup_gui, "run_setup", fail)
    args = Namespace(video=None, camera=0, resolution=None, paradigm="open_field", output="test.json",
                     arenas=1, arena_size=(40, 40), arena_diameter=None, scale_length=None,
                     species="mouse", color="white", arena_shape="rect", names=None, food_zone=True)
    with pytest.raises(SystemExit, match="interrupted"):
        cli.cmd_setup(args)
    assert camera.released


def test_setup_y_maze_with_scale_line(tmp_path, monkeypatch):
    frame = np.full((400, 600, 3), 200, np.uint8)
    arm = lambda x: [(x, 10), (x + 80, 10), (x + 80, 150), (x, 150), 13]  # polygon + Enter
    script = [*arm(20), *arm(260), *arm(500), (100, 200), (300, 200), (200, 300), 13,  # arms A-C, center
              (20, 10), (20, 150), None, 13]  # 30 cm scale line (140 px), accept
    win = ScriptedWindow(script)
    win.install(monkeypatch)
    out = tmp_path / "y.json"
    setup_gui.run_setup(frame, "y_maze", out, scale_length=30)
    arena = json.loads(out.read_text())["arenas"][0]
    assert [z["name"] for z in arena["zones"]] == ["arm_A", "arm_B", "arm_C", "center"]
    assert abs(arena["calibration"]["px_per_cm"] - 140 / 30) < 1e-3
    assert arena["shape"]["type"] == "polygon"  # hull around the maze


def test_annotate_writes_intervals(synthetic, tmp_path, monkeypatch):
    from behavtrack.annotate import annotate
    from behavtrack.classifier import read_labels

    script = ["]", "1"] + ["."] * 10 + ["1", "}", "2", "]", "2", "q"]  # 1 s, grooming 30-40, +10 s, rearing
    win = ScriptedWindow(script)
    win.install(monkeypatch)
    out = annotate(synthetic["video"], ["grooming", "rearing"], tmp_path / "labels.csv", track="A")
    assert read_labels(out, "grooming", "A") == [(30, 40)]
    assert read_labels(out, "rearing", "A") == [(340, 370)]
    assert read_labels(out, "reviewed_v2", "A") == [(0, 0), (30, 40), (340, 340), (370, 370)]


def test_live_loop_records_and_analyzes(synthetic, tmp_path, monkeypatch):
    """Fake camera = the synthetic video. Keys: wait, s (start), then run until duration ends."""
    empty = np.full(synth.FRAME_HW, 110, np.uint8)
    cv2.fillPoly(empty, [synth.corners_px().astype(np.int32)], 205)
    cv2.imwrite(str(tmp_path / "empty.png"), empty)
    monkeypatch.setattr(pipeline, "open_camera", lambda index, resolution=None: cv2.VideoCapture(str(synthetic["video"])))
    win = ScriptedWindow([None, None, "s"] + [None] * 5000)
    win.install(monkeypatch)
    cfg = synth.open_field_config()
    cfg["arenas"][0]["zones"].append({"name": "food", "shape": {"type": "rect", "rect": [150, 150, 40, 40]}})
    session = pipeline.track_live(cfg, 0, tmp_path / "res", name="live", duration_s=1.0,
                                  background=tmp_path / "empty.png", meta={"animal_id": "M1"})
    assert session is not None
    assert (session / "raw.mp4").exists() and (session / "report.html").exists()
    m = json.loads((session / "metrics_A.json").read_text())["metrics"]
    assert m["tracked_pct"] > 95
    assert "food" in json.loads((session / "metrics_A.json").read_text())["zones"]
    assert "food" in (session / "report.html").read_text(encoding="utf-8")
    from behavtrack.cumulative import build_camera_summary
    summary, csv_file, count, skipped = build_camera_summary(session.parent)
    assert count == 1 and not skipped
    assert "live" in summary.read_text(encoding="utf-8")
    assert "사료 구역 시간 (초)" in csv_file.read_text(encoding="utf-8-sig")
    sess = json.loads((session / "session.json").read_text())
    assert sess["animal_id"] == "M1" and sess["source"]["frames"] > 5
