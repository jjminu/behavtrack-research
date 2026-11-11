from copy import deepcopy
import csv
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from behavtrack import cage_activity as a
from behavtrack import research as r


def config():
    return a.settings({"roi":[0,0,1,1]})


def test_stationary_movement_and_illumination_are_distinct():
    before=np.zeros((80,100),np.float32)
    same,_=a.pair_measure(before,before.copy(),config())
    assert same["status"]=="valid" and same["changed_area_pct"]==0
    after=before.copy();after[10:20,10:30]=100
    moved,_=a.pair_measure(before,after,config())
    assert moved["changed_area_pct"]==pytest.approx(2.5)
    lit,_=a.pair_measure(before,before+25,config())
    assert lit["status"]=="unknown" and lit["reason"]=="illumination_change"
    assert lit["changed_area_pct"] is None


def test_reference_translation_is_not_animal_activity():
    rng=np.random.default_rng(0)
    ref=rng.uniform(0,255,(128,128)).astype(np.float32)
    moved=np.roll(ref,6,axis=1)
    neutral=np.zeros((80,100),np.float32)
    result,_=a.pair_measure(neutral,neutral.copy(),config(),ref,moved)
    assert result["status"]=="unknown" and result["reason"]=="camera_shift"
    assert result["reference_shift_px"]>5


def test_interval_bins_preserve_denominators_and_partial_start():
    pairs=[dict(start_s=59.9,end_s=60.1,duration_s=.2,status="valid",reason="valid",changed_area_pct=4.,mean_abs_difference=2.),
           dict(start_s=60.1,end_s=60.3,duration_s=.2,status="unknown",reason="time_gap",changed_area_pct=None,mean_abs_difference=None)]
    bins=a.time_bins(pairs,60)
    assert bins[0]["start_s"]==59.9
    assert sum(b["observed_s"] for b in bins)==pytest.approx(.4)
    assert sum(b["valid_s"] for b in bins)==pytest.approx(.2)
    assert a.summarize(pairs)["mean_changed_area_pct"]==4
    assert a.summarize(pairs)["coverage_pct"]==50


def fake(cage,water,value,source,valid=10,group=None,admin="bottle_only",signature="same"):
    return {"id":"activity_000000000000","state":"complete","created_utc":"2026-01-01T00:00:00+00:00",
            "source_session":source,"metadata":{"cage":cage,"group":group or water,"water_type":water,
            "administration":admin,"setup_id":"camera1","comparison_block":"day1","animals":5},
            "summary":{"valid_s":valid,"observed_s":valid,"mean_changed_area_pct":value},
            "comparison_signature":signature,"source_sha256":{"raw.mp4":"hash_"+source},
            "window":{"start_s":0,"end_s":10}}


def test_repeated_videos_do_not_inflate_cage_n_or_weight_large_cages_more():
    first=fake("C1","control",1,"one",5)
    repeat=fake("C1","control",3,"two",10)
    second=fake("C2","control",5,"three",10)
    treated=fake("H1","hydrogen",7,"four",10)
    report=a.comparison([first,repeat,second,treated])
    control=next(g for g in report["strata"][0]["groups"] if g["label"]=="control")
    hydrogen=next(g for g in report["strata"][0]["groups"] if g["label"]=="hydrogen")
    assert control["n_cages"]==2
    assert control["mean"]==pytest.approx(((1*5+3*10)/15+5)/2)
    assert control["points"][0]["recordings"]==2
    assert hydrogen["difference_from_control_pp"]==pytest.approx(7-control["mean"])
    assert hydrogen["sd"] is None
    assert report["p_value"] is None


def test_duplicates_conditions_and_administration_are_separate():
    first=fake("C1","control",1,"one")
    duplicate=deepcopy(first);duplicate["source_session"]="copied";duplicate["metadata"]["cage"]="copyCage"
    oral=fake("H1","hydrogen",3,"oral",admin="oral_plus_bottle")
    different=fake("O1","oxygen",4,"different",signature="other_camera")
    report=a.comparison([first,duplicate,oral,different])
    assert len(report["strata"])==3
    assert sum(g["n_cages"] for s in report["strata"] for g in s["groups"])==3
    assert any("중복" in e["reason"] for e in report["excluded"])
    missing=fake("X","unspecified",1,"missing")
    assert not a.comparison([missing])["strata"]


def test_crossover_and_overlapping_windows_need_separate_design():
    first=fake("C1","control",1,"one")
    cross=fake("C1","hydrogen",2,"two")
    with pytest.raises(ValueError,match="교차설계"):a.comparison([first,cross])
    overlap=deepcopy(first);overlap["source_session"]="part2";overlap["window"]={"start_s":5,"end_s":15}
    with pytest.raises(ValueError,match="겹칩니다"):a.comparison([first,overlap])


def test_nonoverlapping_windows_in_same_video_remain_one_cage():
    first=fake("C1","control",1,"same_video")
    later=deepcopy(first);later["window"]={"start_s":10,"end_s":20}
    result=a.comparison([first,later])
    point=result["strata"][0]["groups"][0]["points"][0]
    assert point["recordings"]==2
    assert result["strata"][0]["groups"][0]["n_cages"]==1


def make_video(root):
    spec=r.protocol({"group":"G","cage":"C1","animals":2,"food_rect":[0,0,.5,1],"water_type":"control","administration":"bottle_only"},(160,120))
    rec=r.ResearchRecording(root,spec,0,10)
    for i in range(21):
        image=np.full((120,160,3),130,np.uint8)
        cv2.rectangle(image,(10+i*3,50),(20+i*3,60),(20,20,20),-1)
        rec.record(image,rec.start_monotonic+i*.1)
    rec.finish()
    return rec


def test_video_pipeline_uses_original_times_and_keeps_raw_immutable(tmp_path):
    rec=make_video(tmp_path)
    hashes=r.read_json(rec.folder/"recording.json")["sha256"].copy()
    values={"roi":[0,0,1,1],"setup_id":"fixed","comparison_block":"test","start_s":0,"end_s":2}
    result=a.analyze(rec.folder,values)
    assert result["state"]=="complete" and not result["error"]
    report=a.read_report(result["path"])
    assert report["summary"]["valid_pairs"]==10
    assert report["summary"]["mean_changed_area_pct"]>0
    assert report["summary"]["observed_s"]==pytest.approx(2)
    assert report["metadata"]["water_type"]=="control"
    assert all(r.digest(rec.folder/name)==value for name,value in hashes.items())
    with (Path(result["path"])/"pairs.csv").open("a") as f:f.write("changed")
    with pytest.raises(ValueError,match="변경"):a.read_report(result["path"])


def test_cancelled_pairs_are_unknown_not_stationary(tmp_path):
    rec=make_video(tmp_path);cancel=threading.Event();cancel.set()
    result=a.analyze(rec.folder,{"roi":[0,0,1,1],"setup_id":"fixed","comparison_block":"test"},cancel)
    assert result["state"]=="cancelled"
    assert result["report"]["summary"]["mean_changed_area_pct"] is None
    assert result["report"]["summary"]["valid_s"]==0


def test_comparison_exports_real_cage_points_and_empty_comparisons(tmp_path):
    result=a.write_comparison(tmp_path/"compare",[fake("C1","control",1,"one"),fake("H1","hydrogen",2,"two")])
    assert result["strata"][0]["groups"][0]["n_cages"]==1
    assert (tmp_path/"compare/chart.png").is_file()
    assert (tmp_path/"compare/report.html").is_file()
    empty=a.write_comparison(tmp_path/"empty",[fake("X","unspecified",1,"unknown")])
    assert empty["strata"]==[] and len(empty["excluded"])==1


def test_activity_source_api_is_bounded_and_can_preview(tmp_path):
    from behavtrack.research_web import Application
    app=Application(tmp_path)
    rec=make_video(app.results)
    assert app.activity_sources()["sessions"][0]["id"]==rec.folder.name
    image=app.activity_frame({"session":rec.folder.name,"time_s":".4"})
    assert cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR).shape[:2]==(120,160)
    with pytest.raises(ValueError):app.activity_source("../program")
    with pytest.raises(ValueError):app.activity_file({"comparison":"compare_000000000000","file":"../outside.txt"})


def test_invalid_water_metadata_is_rejected():
    with pytest.raises(ValueError,match="물 종류"):
        r.protocol({"group":"G","cage":"C","animals":5,"food_rect":[0,0,1,1],"water_type":"invented"},(100,100))
