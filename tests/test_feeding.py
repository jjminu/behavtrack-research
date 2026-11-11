import csv
import json

import cv2
import numpy as np
import pytest

from behavtrack.feeding import _schedule, score_video, summarize_sessions


def test_sampled_cage_counts_unknown_and_resume(tmp_path, monkeypatch):
    video = tmp_path / "cage.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (80, 60))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((60, 80, 3), i * 5, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    keys = iter(map(ord, ["1", "1", "u", "0", "q"]))
    monkeypatch.setattr(cv2, "waitKey", lambda *a: next(keys))
    out = tmp_path / "session"
    args = (video, out, "C1", "blind-A", 5, 1.0)
    score_video(*args)
    with (out / "samples.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["approach"], r["feeding"]) for r in rows] == [("1", "1"), ("unknown", "0")]
    assert [int(r["frame"]) for r in rows] == [2, 7]
    partial = tmp_path / "partial.csv"
    summarize_sessions([out], partial)
    with partial.open(encoding="utf-8-sig") as f:
        partial_row = next(csv.DictReader(f))
    assert partial_row["unscored_samples"] == "2"
    assert partial_row["approach_unknown_samples"] == "1"
    assert partial_row["approach_known_samples"] == "1"
    keys = iter(map(ord, ["0", "u", "2", "1"]))
    score_video(*args, resume=True)
    result = tmp_path / "summary.csv"
    summarize_sessions([out], result)
    with result.open(encoding="utf-8-sig") as f:
        summary = next(csv.DictReader(f))
    assert summary["planned_samples"] == summary["scored_samples"] == "4"
    assert summary["unscored_samples"] == "0"
    assert summary["approach_known_samples"] == "3"
    assert summary["approach_unknown_samples"] == "1"
    assert float(summary["approach_mean_mice"]) == 1
    assert summary["feeding_positive_samples"] == "2"
    assert summary["feeding_unknown_samples"] == "1"
    assert float(summary["feeding_positive_pct"]) == pytest.approx(200 / 3)
    with pytest.raises(FileExistsError):
        summarize_sessions([out], result)
    with pytest.raises(SystemExit, match="채점 조건"):
        score_video(video, out, "C1", "other", 5, 1.0, resume=True)
    assert json.loads((out / "feeding_session.json").read_text())["group"] == "blind-A"


def test_schedule_does_not_count_partial_window_or_treat_missing_as_zero():
    schedule = _schedule(5.7, 2, 0, 5.7, 10)
    assert [(t, f) for _, t, f, _ in schedule] == [(1, 10), (3, 30)]
    with pytest.raises(ValueError):
        _schedule(2, .01, 0, 2, 10)
