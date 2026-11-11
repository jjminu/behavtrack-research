import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from behavtrack import research_web as web
from test_research import recording, analyze


def test_web_auth_origin_and_path_validation(tmp_path):
    app = web.Application(tmp_path)
    server, token = web.make_server(app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:
            assert "그룹" in response.read().decode()
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/state")
        assert exc.value.code == 403
        req = urllib.request.Request(base + "/api/state", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req) as response:
            assert json.load(response)["capture"]["recording"] is None
        req = urllib.request.Request(base + "/api/start", data=b"{}", headers={"Authorization": "Bearer " + token, "Origin": "https://example.com"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 403
        with pytest.raises(ValueError):
            app.session("../private")
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_capture_is_independent_of_browser_and_freezes_conditions(tmp_path):
    class Camera:
        def read(self):
            time.sleep(.015)
            return True, np.zeros((100, 200, 3), np.uint8)
        def release(self):
            pass
    cap = web.Capture(tmp_path, opener=lambda _: Camera())
    try:
        cap.open(0)
        for _ in range(100):
            if cap.state()["connected"]:
                break
            time.sleep(.02)
        state = cap.state()
        values = {"group": "G", "cage": "C", "animals": 2, "food_rect": [0, 0, .5, 1], "interval": 1,
                  "generation": state["generation"], "size": state["size"]}
        session = cap.start(values)
        with pytest.raises(ValueError):
            cap.open(1)
        # No browser poll or preview request is needed to continue recording.
        time.sleep(.13)
        assert cap.state()["frames"] >= 3
        assert cap.stop_recording() == session
        assert json.loads((tmp_path / session / "recording.json").read_text())["state"] == "complete"
    finally:
        cap.close()


def test_review_endpoint_hides_ai_and_detects_source_modification(tmp_path):
    app = web.Application(tmp_path)
    rec = recording(app.results)
    run, _ = analyze(tmp_path, rec)
    review = app.review(rec.folder.name, run.name, "A")
    assert review["plan"] and "status" not in review["plan"][0] and "count" not in review["plan"][0]
    image = app.review_image(rec.folder.name, run.name, 0)
    assert image[:2] == b"\xff\xd8"
    with (rec.folder / "raw.mp4").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="변경"):
        app.review(rec.folder.name, run.name, "A")


def test_preview_reuses_frame_and_does_not_block_recording_lock(tmp_path, monkeypatch):
    cap = web.Capture(tmp_path)
    cap.frame = np.zeros((100, 200, 3), np.uint8)
    cap.generation, cap.frame_number = 1, 1
    entered, release = threading.Event(), threading.Event()
    real_encode = web.cv2.imencode
    calls = []

    def encode(*args):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return real_encode(*args)

    monkeypatch.setattr(web.cv2, "imencode", encode)
    results = []
    thread = threading.Thread(target=lambda: results.append(cap.preview(with_metadata=True)))
    thread.start()
    try:
        assert entered.wait(3)
        # Capture can acquire its lock while JPEG encoding is in progress.
        assert cap.lock.acquire(timeout=.5)
        cap.lock.release()
    finally:
        release.set()
        thread.join(timeout=3)
    assert results[0][1] == "1:1"
    assert cap.preview(with_metadata=True) == results[0]
    assert len(calls) == 1
    # Camera reconnection must invalidate an otherwise identical frame index.
    cap.generation = 2
    assert cap.preview(with_metadata=True)[1] == "2:1"
    assert len(calls) == 2
    cap.frame = None
    assert cap.preview() is None
