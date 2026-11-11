import csv
import json
import threading

import numpy as np
import pytest

from behavtrack import research as r


def recording(tmp_path, times=(0.02, .52, 1.02, 1.52, 2.02)):
    spec = r.protocol({"group": "대조군", "cage": "C01", "animals": 2, "food_rect": [0, 0, .5, 1], "interval": .5}, (200, 100))
    rec = r.ResearchRecording(tmp_path, spec, 0, 2)
    for i, elapsed in enumerate(times):
        rec.record(np.full((100, 200, 3), i*20, np.uint8), rec.start_monotonic+elapsed)
    rec.finish()
    return rec


class Predictor:
    def __init__(self, executable, animals, cancel):
        self.runtime = {"test": True}
    def predict(self, frame, fid):
        if fid == 2:
            return {"boxes": [], "scores": []}
        return {"boxes": [[10 if fid % 2 == 0 else 110, 10, 30 if fid % 2 == 0 else 130, 30], [160, 60, 180, 80]], "scores": [.9, .9]}
    def close(self):
        pass


def analyze(tmp_path, rec, predictor=Predictor, **kwargs):
    model = tmp_path / "model.pt"
    model.write_bytes(b"test model")
    return r.analyze(rec.folder, "unused", model, predictor_factory=predictor, **kwargs)


def test_time_grid_handles_capture_gaps_without_changing_schedule():
    rows = r.sample_plan(np.array([.01, .99, 3.01, 4.01]), 1, .25)
    assert [row["target_s"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["frame"] for row in rows] == [0, 1, None, 2, 3]
    assert rows[2]["reason"] == "capture_gap"


@pytest.mark.parametrize("interval,total,unknown", [(.1, 5, 3), (.2, 3, 1)])
def test_short_interval_analysis_preserves_gaps_and_original_timestamps(tmp_path, interval, total, unknown):
    spec = r.protocol({"group": "G", "cage": "C", "animals": 2,
                       "food_rect": [0, 0, .5, 1], "interval": interval}, (200, 100))
    rec = r.ResearchRecording(tmp_path, spec, 0, 5)
    for elapsed in (.01, .21, .41):
        rec.record(np.zeros((100, 200, 3), np.uint8), rec.start_monotonic+elapsed)
    rec.finish()
    times_hash = r.digest(rec.folder / "frame_times.csv")
    run, result = analyze(tmp_path, rec)
    _, rows = r.analysis_rows(run)
    assert result["state"] == "complete"
    assert [float(row["target_s"]) for row in rows] == pytest.approx([i*interval for i in range(total)])
    assert result["summary"]["unknown_samples"] == unknown  # gaps plus undetected frame 2
    selected = [row["frame"] for row in rows if row["frame"] is not None]
    assert selected == [0, 1, 2]
    assert r.digest(rec.folder / "frame_times.csv") == times_hash


def test_fixed_grid_and_provenance_with_new_analysis_each_time(tmp_path):
    rec = recording(tmp_path)
    hashes = r.read_json(rec.folder / "recording.json")["sha256"].copy()
    run, result = analyze(tmp_path, rec)
    _, rows = r.analysis_rows(run)
    assert result["state"] == "complete"
    assert result["summary"]["known_samples"] == 4
    assert result["summary"]["unknown_samples"] == 1
    assert result["summary"]["approach_pct"] == 50
    assert [float(row["target_s"]) for row in rows] == [0, .5, 1, 1.5, 2]
    assert result["source_sha256"] == hashes
    assert result["detector_sha256"] and result["code_sha256"]
    second, _ = analyze(tmp_path, rec)
    assert second != run and (run / "samples.csv").is_file()
    assert all(r.digest(rec.folder / name) == sha for name, sha in hashes.items())


def test_changed_source_is_rejected_before_analysis(tmp_path):
    rec = recording(tmp_path)
    with (rec.folder / "frame_times.csv").open("a") as f:
        f.write("extra")
    with pytest.raises(ValueError, match="변경"):
        analyze(tmp_path, rec)


def test_detector_failure_preserves_originals_and_marks_unprocessed_unknown(tmp_path):
    class Broken(Predictor):
        def predict(self, frame, fid):
            raise RuntimeError("detector failed")
    rec = recording(tmp_path)
    original = r.digest(rec.folder / "raw.mp4")
    run, result = analyze(tmp_path, rec, Broken)
    assert result["state"] == "failed"
    assert result["summary"]["unknown_samples"] == 5
    assert result["summary"]["approach_pct"] is None
    assert r.digest(rec.folder / "raw.mp4") == original
    _, rows = r.analysis_rows(run)
    assert all(row["status"] == "unknown" for row in rows)


def test_analysis_uses_frozen_score_threshold(tmp_path):
    rec = recording(tmp_path)
    spec = r.read_json(rec.folder / "protocol.json")
    spec["criteria"]["min_bbox_score"] = .95
    spec["criteria_sha256"] = r.json_digest(spec["criteria"])
    r.atomic_json(rec.folder / "protocol.json", spec)
    info = r.read_json(rec.folder / "recording.json")
    info["sha256"]["protocol.json"] = r.digest(rec.folder / "protocol.json")
    r.atomic_json(rec.folder / "recording.json", info)
    _, result = analyze(tmp_path, rec)
    assert result["summary"]["known_samples"] == 0


def test_cancelled_samples_remain_unknown_and_raw_survives(tmp_path):
    rec = recording(tmp_path)
    event = threading.Event()
    def progress(done, *args):
        if done == 1:
            event.set()
    run, info = analyze(tmp_path, rec, cancel=event, progress=progress)
    assert info["state"] == "cancelled"
    assert info["summary"]["known_samples"] == 1
    assert info["summary"]["unknown_samples"] == 4
    assert (rec.folder / "raw.mp4").exists()


def test_blind_review_revisions_preserve_ai_and_unknown_denominators(tmp_path):
    rec = recording(tmp_path)
    run, _ = analyze(tmp_path, rec)
    original = r.digest(run / "samples.csv")
    for sample, label in [(0, "approach"), (1, "approach"), (2, "away"), (3, "unknown")]:
        r.save_review(run, {"reviewer": "검토자", "sample": sample, "label": label})
    first = r.validation(run)["reviewers"][0]
    assert first["reviewed"] == 4 and first["paired_known"] == 2
    assert first["agreement_pct"] == 50 and first["ai_unknown"] == 1 and first["human_unknown"] == 1
    r.save_review(run, {"reviewer": "검토자", "sample": 1, "label": "away"})
    assert len(r.review_events(run)) == 5
    assert r.validation(run)["reviewers"][0]["agreement_pct"] == 100
    assert not r.validation(run)["validated"]
    assert r.digest(run / "samples.csv") == original
    with pytest.raises(ValueError):
        r.save_review(run, {"reviewer": "검토자", "sample": 0, "label": "approach", "count": 0})


def test_protocol_rejects_missing_cage_invalid_region_and_nonfinite_interval():
    for values in ({"group": "G", "cage": "", "animals": 2},
                   {"group": "G", "cage": "C", "animals": 2, "food_rect": [0, 0, 1, 1], "interval": float("nan")},
                   {"group": "G", "cage": "C", "animals": 2, "food_rect": [0, 0, 2, 1]}):
        with pytest.raises(ValueError):
            r.protocol(values, (200, 100))
