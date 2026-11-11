import csv
import json

import pytest

from behavtrack.cumulative import build_camera_summary


def session(root, name, camera=True, food=None, track="A"):
    folder = root / name
    folder.mkdir()
    (folder / "session.json").write_text(json.dumps({"source": {"type": "camera" if camera else "video"},
        "created": "2026-10-28T12:00:00", "animal_id": "<M1>", "group": "=test"}), encoding="utf-8")
    metrics = {"track": track, "metrics": {"duration_s": 60, "tracked_pct": 80, "distance_cm": 123},
               "zones": {} if food is None else {"food": food}}
    (folder / f"metrics_{track}.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "report.html").write_text("individual", encoding="utf-8")
    return folder


def test_camera_summary_refreshes_and_marks_missing_food(tmp_path):
    session(tmp_path, "live_1", food={"time_s": 10, "time_pct": 20.833, "entries": 2})
    session(tmp_path, "file_session", camera=False, food={"time_s": 999})
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "session.json").write_text('{"source":{"type":"camera"}}', encoding="utf-8")
    report, output, count, skipped = build_camera_summary(tmp_path)
    assert count == 1 and len(skipped) == 1
    assert "&lt;M1&gt;" in report.read_text(encoding="utf-8")
    session(tmp_path, "live_2")
    report, output, count, skipped = build_camera_summary(tmp_path)
    with output.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert count == len(rows) == 2
    assert rows[0]["사료 구역 시간 (초)"] == "10"
    assert rows[1]["사료 구역 시간 (초)"] == ""
    assert rows[0]["군"] == "'=test"
    assert "file_session" not in report.read_text(encoding="utf-8")


def test_camera_summary_reports_no_completed_sessions(tmp_path):
    with pytest.raises(ValueError, match="완료된 웹캠 분석 결과가 없습니다"):
        build_camera_summary(tmp_path)
