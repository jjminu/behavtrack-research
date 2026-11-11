import json

import numpy as np
import pytest

from behavtrack.interpretation import describe, write_interpretation


def row(t, status, count, index=0):
    return {"sample": index, "target_s": t, "status": status, "count": count,
            "frame": index, "reason": "count_mismatch" if status == "unknown" else "estimated"}


def test_unknown_denominators_half_comparison_and_short_last_bin():
    rows = [row(0,"unknown",None),row(5,"approach",1),row(10,"away",0),
            row(60,"approach",1),row(65,"unknown",None),row(70,"away",0),row(125,"approach",1)]
    report = describe(rows,125,5)
    assert report["overall"]["approach_pct"] == 60
    assert report["overall"]["unknown_samples"] == 2
    assert [b["known_samples"] for b in report["bins"]] == [2,2,1]
    assert report["bins"][-1]["end_s"] == 125
    assert not report["bins"][-1]["full_interval"]
    assert report["highest_full_bin"]["approach_pct"] == 50
    assert report["halves"]["first"]["approach_pct"] == pytest.approx(200/3)
    assert report["halves"]["difference_pp"] == pytest.approx(50-200/3)
    assert report["unknown_reasons"] == {"count_mismatch":2}


def test_all_unknown_remains_unquantifiable():
    report=describe([row(0,"unknown",None),row(5,"unknown",None)],5,5)
    assert report["overall"]["approach_pct"] is None
    assert report["overall"]["known_rate_pct"] == 0
    assert report["halves"]["difference_pp"] is None
    assert report["highest_full_bin"] is None
    assert report["review_samples_total"] == 2


def test_completed_analysis_automatically_writes_bound_report(tmp_path):
    from behavtrack import research as r
    from test_research import Predictor

    spec=r.protocol({"group":"<script>test</script>","cage":"C1","animals":2,
                     "food_rect":[0,0,.5,1],"interval":.5},(200,100))
    rec=r.ResearchRecording(tmp_path,spec,0,2)
    for t in (.02,.52,1.02,1.52,2.02):
        rec.record(np.zeros((100,200,3),np.uint8),rec.start_monotonic+t)
    rec.finish()
    source_hashes=r.read_json(rec.folder/"recording.json")["sha256"]
    model=tmp_path/"model.pt";model.write_bytes(b"test model")
    run,meta=r.analyze(rec.folder,"unused",model,predictor_factory=Predictor)
    assert meta["state"]=="complete" and meta["interpretation"]["state"]=="complete"
    report=json.loads((run/"interpretation.json").read_text(encoding="utf-8"))
    assert report["input_samples_sha256"]==r.digest(run/"samples.csv")
    assert report["overall"]["approach_pct"]==50
    assert report["source_sha256"]==source_hashes
    page=(run/"interpretation.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;test&lt;/script&gt;" in page
    assert "<script>test</script>" not in page
    assert (run/"interpretation_chart.png").stat().st_size>1000
    assert all(r.digest(rec.folder/name)==value for name,value in source_hashes.items())
    with (run/"samples.csv").open("a",encoding="utf-8") as f:f.write("changed")
    with pytest.raises(ValueError,match="변경"):
        write_interpretation(run)


def test_report_error_does_not_discard_completed_inference(tmp_path,monkeypatch):
    from behavtrack import interpretation
    from test_research import recording, analyze

    def fail(*args,**kwargs):raise OSError("report disk error")
    monkeypatch.setattr(interpretation,"write_interpretation",fail)
    run,meta=analyze(tmp_path,recording(tmp_path))
    assert meta["state"]=="complete" and meta["error"] is None
    assert meta["interpretation"]["state"]=="failed"
    assert (run/"samples.csv").is_file()
