from pathlib import Path

import numpy as np
import pytest

from behavtrack import synth
from behavtrack.config import new_config
from behavtrack.geometry import Calibration, Polygon
from behavtrack.pipeline import track_video
from behavtrack.setup_gui import build_arena
from behavtrack.templates import paradigm
from behavtrack.tracks import Tracks

FPS = 30


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """60 s open-field video (freezing 12-16 s and 34-40 s, grooming 48-52 s) tracked with approach ①."""
    root = tmp_path_factory.mktemp("synthetic")
    path = synth.simulate_path(seconds=60, fps=FPS)
    video = synth.render(path, root / "of.mp4")
    session = track_video(synth.open_field_config(), video, root / "results", name="of", annotated=True)
    return {"path": path, "video": video, "session": session, "root": root}


def make_tracks(segments, fps=FPS, keypoints=None, extra=None) -> Tracks:
    """segments: [(seconds, (x, y)), ...] in px; (x, y) may be None for 'not detected'."""
    xs = []
    for dur, xy in segments:
        xs += [(np.nan, np.nan) if xy is None else xy] * int(round(dur * fps))
    xy = np.array(xs, float)
    n = len(xy)
    return Tracks(np.arange(n), np.arange(n) / fps, xy[:, 0], xy[:, 1], extra or {}, keypoints or {})


def rect(x0, y0, x1, y1) -> Polygon:
    return Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def make_arena(key, shape, zones, px_per_cm=10.0, **params):
    cfg = new_config(key)
    cfg["params"].update(params)
    arena = build_arena(paradigm(key), "A", shape, Calibration(px_per_cm), zones, cfg["params"])
    cfg["arenas"] = [arena.to_dict()]
    return cfg, arena


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
