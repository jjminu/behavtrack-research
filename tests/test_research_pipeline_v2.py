"""Software integrity tests only. Synthetic inputs do NOT validate mouse accuracy."""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from behavtrack import research as r
from behavtrack import research_data as rd
from behavtrack import ml_workflow as ml
from behavtrack.recording_io import RecordingVideo,frame_reference,recover
from behavtrack.integrity import atomic_json,read_json,digest
from behavtrack.sources import camera_timestamps,Recorder
from behavtrack.pose_import import import_pose
from behavtrack.config import complete
from behavtrack.tracks import Tracks
from behavtrack.session import tracks_path
from behavtrack.evaluation import binary_metrics,count_metrics
from behavtrack.research_web import Application,Capture


def make_recording(root,cage="A",times=None,seconds=600,date="2026-11-01",animals=1):
    spec=r.protocol({"cage":cage,"group":"수소","treatment":"hydrogen","administration":"oral_plus_bottle",
                     "animals":animals,"food_rect":[0,0,.5,1],"interval":.2},(96,64))
    rec=r.ResearchRecording(root,spec,0,10,segment_seconds=seconds)
    rec.info["started_utc"]=date+"T00:00:00+00:00"
    for i,t in enumerate(times if times is not None else np.arange(160)/10+.01):
        img=np.full((64,96,3),(sum(map(ord,cage+date))+i)%240,np.uint8)
        rec.record(img,rec.start_monotonic+float(t))
    rec.finish()
    return rec


def make_pose(root,cage="A",date="2026-11-01",purpose="development",animals=1,scale=None):
    rec=make_recording(root,cage,date=date,animals=animals)
    pose=root/("pose_"+cage+date+".csv")
    with pose.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["scorer"]+["model"]*3*animals)
        w.writerow(["individuals"]+sum(([str(a)]*3 for a in range(animals)),[]))
        w.writerow(["bodyparts"]+["nose"]*3*animals)
        w.writerow(["coords"]+["x","y","likelihood"]*animals)
        for i in range(160):
            w.writerow([i]+sum(([10+a*40+2*np.sin(i/(2 if i>=80 else 30)),30,1] for a in range(animals)),[]))
    arena={"name":"A","shape":{"type":"rect","rect":[0,0,96,64]},"zones":[{"name":"food","shape":{"type":"rect","rect":[0,0,48,64]}}]}
    if scale:
        arena["calibration"]={"px_per_cm":scale}
    session=import_pose(pose,complete({"paradigm":"custom","arenas":[arena]}),root/"poses",video=rec.folder,
                        name="pose_"+cage+date,analyze=False)
    sess=read_json(session/"session.json")
    for tid in sess["tracks"]:
        for a,b,label in [(0,79,"negative"),(80,159,"positive")]:
            rd.save_behavior(session,{"track":tid,"reviewer":"human","purpose":purpose,"start_frame":a,"end_frame":b,"label":label,"observable":True})
    out=session/"reference.csv"
    rd.export_behavior(session,"human",out)
    return rec,session,out


def entry(pose,labels,role="train"):
    return {"session":str(pose),"labels":str(labels),"role":role,"identity_checked":True}


def test_irregular_import_segment_mapping_and_no_silent_fps(tmp_path):
    rec=make_recording(tmp_path,times=[.01,.14,.25,.8,.91,1.04],seconds=.5)
    info,times=r.load_times(rec.folder)
    assert len(info["segments"])==2
    assert camera_timestamps(rec.folder,6)==pytest.approx(times)
    second=info["segments"][1]
    assert camera_timestamps(rec.folder/second["file"],3)==pytest.approx(times[3:])
    assert frame_reference(rec.folder,3)["segment_frame"]==0
    pose=tmp_path/"partial.csv"
    pose.write_text("scorer,m,m,m\nbodyparts,nose,nose,nose\ncoords,x,y,likelihood\n0,10,20,1\n1,20,20,1\n2,25,20,1\n")
    cfg=complete({"paradigm":"custom","arenas":[{"name":"A","shape":{"type":"rect","rect":[0,0,96,64]}}]})
    out=import_pose(pose,cfg,tmp_path/"poses",video=rec.folder/second["file"],analyze=False)
    tr=Tracks.load(out/"tracks_A.csv")
    assert tr.frame.tolist()==[3,4,5]
    assert tr.time==pytest.approx(times[3:],abs=1e-8)
    assert read_json(out/"session.json")["cage_id"]=="A"
    (rec.folder/"frame_times.csv").unlink()
    with pytest.raises(ValueError,match="변경|손상"):
        camera_timestamps(rec.folder/"raw.mp4",3)


def test_segment_boundaries_have_no_duplicate_frames_or_samples(tmp_path):
    rec=make_recording(tmp_path,times=np.arange(30)/10,seconds=.5)
    info,times=r.load_times(rec.folder)
    cap=RecordingVideo(rec.folder)
    decoded=[]
    while True:
        ok,image=cap.read()
        if not ok:break
        decoded.append(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
    cap.release()
    assert decoded==list(range(30))
    rows=r.sample_plan(times,.1,.05)
    assert [x["frame"] for x in rows]==list(range(30))
    for s in info["segments"]:
        assert s["end_frame"]-s["start_frame"]+1==s["frames"]
        assert s["state"]=="closed" and s["end_utc"]>=s["start_utc"]


def test_hard_process_exit_preserves_closed_segments_and_recovery_prefix(tmp_path):
    script=tmp_path/"crash.py"
    script.write_text('''import os,sys,numpy as np
from behavtrack.research import ResearchRecording,protocol
p=protocol(dict(group="G",cage="C",animals=1,food_rect=[0,0,.5,1],interval=.1),(96,64))
r=ResearchRecording(sys.argv[1],p,0,10,segment_seconds=.5)
for i in range(18):r.record(np.full((64,96,3),i,np.uint8),r.start_monotonic+i*.1)
r.file.flush();os.fsync(r.file.fileno())
os._exit(17)
''',encoding="utf-8")
    env=os.environ.copy();env["PYTHONPATH"]=str(Path(__file__).resolve().parents[1])
    p=subprocess.run([sys.executable,str(script),str(tmp_path)],env=env,capture_output=True,timeout=30)
    assert p.returncode==17,p.stderr
    folder=next(tmp_path.glob("research_*"))
    originals={p.name:digest(p) for p in folder.glob("raw*.mp4")}
    with pytest.raises(ValueError,match="복구"):
        r.load_times(folder)
    info=recover(folder)
    assert 10<=info["frames"]<=18 and info["state"]=="recovered"
    assert {p.name:digest(p) for p in folder.glob("raw*.mp4")}==originals
    assert len(r.load_times(folder)[1])==info["frames"]
    assert recover(folder)==info


def test_disk_resolution_disconnect_and_overwrite_safety(tmp_path):
    rec=make_recording(tmp_path,times=[.01,.1])
    with pytest.raises(FileExistsError):Recorder(rec.folder/"raw.mp4",10)
    spec=r.read_json(rec.folder/"protocol.json")
    low=r.ResearchRecording(tmp_path,spec,0,10,disk_usage=lambda _:SimpleNamespace(free=0))
    with pytest.raises(OSError,match="저장공간"):
        low.record(np.zeros((64,96,3),np.uint8),low.start_monotonic+.1)
    low.finish("저장공간 부족")
    assert low.info["state"]=="partial" and low.info["frames"]==0
    changed=r.ResearchRecording(tmp_path,spec,0,10)
    changed.record(np.zeros((64,96,3),np.uint8),changed.start_monotonic+.1)
    with pytest.raises(ValueError,match="해상도"):
        changed.record(np.zeros((32,96,3),np.uint8),changed.start_monotonic+.2)
    changed.finish("해상도 변경")
    assert len(r.load_times(changed.folder)[1])==1
    class Camera:
        def __init__(self):self.n=0
        def read(self):
            self.n+=1;time.sleep(.01)
            return (True,np.zeros((64,96,3),np.uint8)) if self.n<30 else (False,None)
        def release(self):pass
    capture=Capture(tmp_path,opener=lambda _:Camera())
    try:
        capture.open(0)
        for _ in range(100):
            state=capture.state()
            if state["connected"]:break
            time.sleep(.01)
        name=capture.start(dict(group="G",cage="C",animals=1,food_rect=[0,0,.5,1],interval=.1,size=state["size"],generation=state["generation"]))
        for _ in range(100):
            if capture.last_session:break
            time.sleep(.01)
        assert capture.last_session==name and "연결" in capture.error
        assert read_json(tmp_path/name/"recording.json")["state"]=="partial"
    finally:capture.close()


def test_final_plan_fixed_blind_and_source_bound_labels(tmp_path):
    rec=make_recording(tmp_path/"results")
    plan=rd.make_plan(rec.folder,count=20,seed=31)
    assert rd.make_plan(rec.folder,count=5,seed=98)==plan
    with pytest.raises(ValueError,match="최종"):
        rd.make_plan(rec.folder,frames=[1,3])
    path=rec.folder/"references"/plan["id"]
    sample=plan["samples"][0]
    e=rd.save_approach(path,{"sample":0,"label":"away","count":0,"reviewer":"R"})
    assert e["session_frame"]==sample["session_frame"] and e["task"]=="cage_approach"
    with pytest.raises(ValueError,match="사유"):
        rd.save_approach(path,{"sample":1,"label":"unknown","reviewer":"R"})
    app=Application(tmp_path)
    result=app.workflow_get("reference",{"session":rec.folder.name,"plan":plan["id"],"reviewer":"R"})
    assert not any(k in json.dumps(result) for k in ('"ai_count"','"detections"','"validation"'))
    assert app.raw_frame({"session":rec.folder.name,"frame":str(sample["session_frame"])})[:2]==b"\xff\xd8"
    with pytest.raises(ValueError):app.reference_path(rec.folder.name,"../escape")
    app.close()


def test_events_revision_and_individual_dose_no_assumed_shared_time(tmp_path):
    rec=make_recording(tmp_path)
    values={"type":"oral_dose","actor":"R","occurred_at":"2026-11-01T09:00:00+09:00"}
    with pytest.raises(ValueError,match="마우스"):
        rd.save_event(rec.folder,values)
    event=rd.save_event(rec.folder,dict(values,animal_id="M1"))
    updated=rd.save_event(rec.folder,dict(values,id=event["id"],animal_id="M1",occurred_at="2026-11-01T09:01:00+09:00"))
    assert event["recorded_at"]!=event["occurred_at"] and updated["revision"]==2
    assert len(rd.read_events(rec.folder/"events.jsonl"))==2


def test_pose_labels_preflight_and_same_cage_split_rejected(tmp_path):
    _,a,la=make_pose(tmp_path,"C","2026-11-01")
    _,b,lb=make_pose(tmp_path,"C","2026-11-02",purpose="final")
    spec={"entries":[entry(a,la),entry(b,lb,"final")]}
    report=ml.inspect_dataset(spec)
    assert not report["ok"] and any("케이지" in e for e in report["errors"])
    spec["scope"]="same_cage_other_day"
    report=ml.inspect_dataset(spec)
    assert report["ok"],report["errors"]
    assert report["generalization"]=="unevaluated" and report["counts"]["train"]["independent_cages"]==1
    with pytest.raises(ValueError,match="겹칩니다"):
        rd.save_behavior(a,{"track":"A","reviewer":"human","start_frame":20,"end_frame":40,"label":"positive","observable":True})


def test_duplicates_units_and_corrupt_time_fail_preflight(tmp_path):
    _,a,la=make_pose(tmp_path,"A")
    assert not ml.inspect_dataset({"entries":[entry(a,la),entry(a,la)]})["ok"]
    _,b,lb=make_pose(tmp_path,"B",scale=10)
    report=ml.inspect_dataset({"entries":[entry(a,la),entry(b,lb)]})
    assert not report["ok"] and any("단위" in e for e in report["errors"])
    tr=Tracks.load(a/"tracks_A.csv");tr.time[3]=tr.time[2];tr.save(a/"tracks_A.csv")
    assert not ml.inspect_dataset({"entries":[entry(a,la)]})["ok"]


def test_features_do_not_cross_missing_time_or_split_boundaries():
    from behavtrack.classifier import features,feature_valid
    frames=np.arange(120);t=np.r_[np.arange(60)/10,20+np.arange(60)/10]
    x=np.r_[np.arange(60),np.arange(60)]
    tr=Tracks(frames,t,x.astype(float),np.zeros(120))
    names,before=features(tr,1)
    tr.x[60:]+=100000
    _,after=features(tr,1)
    assert np.array_equal(before[:60],after[:60])
    assert not feature_valid(tr)[59] and not feature_valid(tr)[60]
    tr.x[30]=np.nan
    assert not feature_valid(tr)[30] and not feature_valid(tr)[29]


def test_metrics_abstentions_no_positive_and_real_denominators():
    m=binary_metrics([1,0,1,-1],[-1,0,-1,1],[1,1,1,1])
    assert m["human_known"]==3 and m["paired_known"]==1 and m["human_known_ai_unknown_rate"]==pytest.approx(2/3)
    assert m["recall"] is None and m["f1"] is None and not m["validated"]
    assert binary_metrics([0,0],[0,0])["recall"] is None
    c=count_metrics([0,2,-1],[0,1,5])
    assert len(c["confusion_matrix"])==6 and c["paired_known"]==2 and c["mae"]==.5


def test_candidate_training_eval_gate_and_failure_cancel_preserve_active(tmp_path,monkeypatch):
    _,a,la=make_pose(tmp_path,"A")
    _,b,lb=make_pose(tmp_path,"B",purpose="final")
    dataset=tmp_path/"datasets/dataset_one"
    spec={"scope":"independent_cages","entries":[entry(a,la),entry(b,lb,"final")]}
    manifest=ml.freeze_dataset(spec,dataset)
    models=tmp_path/"models"
    status=ml.train_candidate(dataset,models,trees=5)
    assert status["state"]=="candidate",status
    candidate=models/status["id"]
    with pytest.raises(ValueError,match="독립"):
        ml.promote(models,status["id"],"R")
    ml.promote(models,status["id"],"R",experimental=True)
    active=digest(models/"active.json")
    report=ml.evaluate_candidate(dataset,candidate,"final")
    assert set(report["scores"]["overall"])=={"predict","feeding_auto"} and not report["validated"]
    with pytest.raises(FileExistsError):ml.evaluate_candidate(dataset,candidate,"final")
    cancel=threading.Event();cancel.set()
    cancelled=ml.train_candidate(dataset,models,cancel=cancel,trees=5)
    assert cancelled["state"]=="cancelled" and digest(models/"active.json")==active
    def fail(*args,**kwargs):raise RuntimeError("test failure")
    monkeypatch.setattr(ml.c,"train",fail)
    failed=ml.train_candidate(dataset,models,trees=5)
    assert failed["state"]=="failed" and digest(models/"active.json")==active
    changed=read_json(dataset/"dataset.json");changed["seed"]=99;atomic_json(dataset/"dataset.json",changed)
    with pytest.raises(ValueError,match="고정"):
        ml.read_dataset(dataset)


def test_recording_heavy_work_mutual_exclusion_and_web_job_failures(tmp_path):
    from behavtrack.activity import Lease
    app=Application(tmp_path)
    with Lease(app.results,"recording"):
        with pytest.raises(ValueError,match="동시에"):
            app.start_job("test",lambda c,p:{})
    def fail(cancel,progress):raise RuntimeError("intentional")
    app.start_job("test",fail)
    app.job_thread.join(3)
    assert app.job["state"]=="failed"
    with Lease(app.results,"recording"):pass
    assert app.state()["workspace"]["readiness"]["capture"]["ready"]
    app.close()


def test_legacy_single_recording_and_external_fps_are_supported(tmp_path):
    rec=make_recording(tmp_path,times=[.02,.17,.4])
    info=read_json(rec.folder/"recording.json")
    info.pop("segments");info["schema"]=1
    atomic_json(rec.folder/"recording.json",info)
    assert camera_timestamps(rec.folder/"raw.mp4",3)==pytest.approx([.02,.17,.4])
    external=tmp_path/"external.mp4"
    import shutil
    shutil.copyfile(rec.folder/"raw.mp4",external)
    assert camera_timestamps(external,3) is None


def test_detector_export_only_real_boxes_and_cage_separation(tmp_path):
    from behavtrack.dlc_bridge import export_detector
    a,b=make_recording(tmp_path,"A"),make_recording(tmp_path,"B")
    for rec in (a,b):
        rd.save_boxes(rec.folder,{"frame":1,"reviewer":"R","all_animals_visible":True,"boxes":[[2,3,40,50]]})
    exported=export_detector([{"session":str(a.folder),"role":"train"},{"session":str(b.folder),"role":"development"}],tmp_path/"coco","R")
    assert exported["final_excluded"]
    annotations=read_json(tmp_path/"coco/annotations/train.json")
    assert annotations["annotations"][0]["bbox"]==[2,3,38,47]
    assert annotations["annotations"][0]["keypoints"]==[]
    with pytest.raises(ValueError,match="최종"):
        export_detector([{"session":str(a.folder),"role":"final"}],tmp_path/"forbidden","R")


def test_model_semantics_and_postprocessing_mismatch(tmp_path):
    from behavtrack.classifier import check_schema,feature_schema
    _,session,label=make_pose(tmp_path,"S")
    sess=read_json(session/"session.json")
    names=["speed"]
    bundle={"features":names,"feature_schema":feature_schema(sess,names)}
    check_schema(bundle,sess,names)
    for key,value in [("windows_s",[2]),("position_units",["cm"]),("min_pose_p",.99),("view","side")]:
        changed=dict(bundle,feature_schema=dict(bundle["feature_schema"],**{key:value}))
        with pytest.raises(SystemExit,match="호환"):
            check_schema(changed,sess,names)


def test_feeding_eval_includes_explicit_unknown_and_interrater(tmp_path):
    from behavtrack.evaluation import behavior_agreement
    _,session,label=make_pose(tmp_path,"R")
    rd.save_behavior(session,{"reviewer":"second","track":"A","start_frame":0,"end_frame":30,
                              "label":"unknown","reason":"mouth_occluded"})
    rd.save_behavior(session,{"reviewer":"second","track":"A","start_frame":31,"end_frame":79,
                              "label":"negative","observable":True})
    agreement=behavior_agreement(session)[0]
    assert agreement["common_annotated_frames"]==80
    assert agreement["paired_known_frames"]==49 and agreement["agreement"]==1


def test_measurement_intervals_are_not_silently_combined(tmp_path):
    from behavtrack.research_reports import import_measurements,result_tables
    from test_research import analyze
    rec=make_recording(tmp_path)
    csv_path=tmp_path/"food.csv"
    csv_path.write_text("cage_id,start_at,end_at,administration,food_offered_g,food_remaining_g,spillage_recovered_g,water_ml\nA,2026-11-01T00:00:00+00:00,2026-11-02T00:00:00+00:00,oral_plus_bottle,20,15,1,10\n",encoding="utf-8")
    import_measurements(rec.folder,csv_path)
    run,_=analyze(tmp_path,rec)
    report=result_tables(rec.folder,run)
    assert not report["measurements"][0]["combined_with_video"]
    assert report["measurements"][0]["apparent_food_disappearance_g"]==4


def test_final_references_cannot_be_selected_after_ai_results(tmp_path):
    from test_research import analyze
    rec=make_recording(tmp_path)
    analyze(tmp_path,rec)
    with pytest.raises(ValueError,match="이미 AI"):
        rd.make_plan(rec.folder)
    assert rd.make_plan(rec.folder,purpose="development")["purpose"]=="development"


def test_http_workflow_assets_and_auth(tmp_path):
    import urllib.request
    import urllib.error
    from behavtrack.research_web import make_server
    app=Application(tmp_path)
    server,token=make_server(app)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base+"/workflow.js") as response:
            assert "dataset-inspect" in response.read().decode("utf-8-sig")
        request=urllib.request.Request(base+"/api/work/inspect",data=json.dumps({"spec":{"entries":[]}}).encode(),headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
        with urllib.request.urlopen(request) as response:assert json.load(response)["started"]
        app.job_thread.join(5)
        assert app.job["result"]["ok"] is False
        request=urllib.request.Request(base+"/api/work/event",data=b"{}",headers={"Origin":"https://untrusted.example","Authorization":"Bearer "+token})
        with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(request)
        assert exc.value.code==403
    finally:
        server.shutdown();server.server_close();app.close()


def test_default_ten_minute_segments_with_accelerated_clock(tmp_path):
    rec=make_recording(tmp_path,times=np.arange(1201,dtype=float))
    info,times=r.load_times(rec.folder)
    assert info["segment_seconds"]==600
    assert [s["frames"] for s in info["segments"]]==[600,600,1]
    assert [s["start_frame"] for s in info["segments"]]==[0,600,1200]
    assert len(r.sample_plan(times,60,.01))==21


def test_segment_pose_merge_requires_checked_identity_and_maps_time(tmp_path):
    rec=make_recording(tmp_path,times=[.1,.2,.3,.8,.9,1.0],seconds=.5,animals=2)
    entries=[]
    info,times=r.load_times(rec.folder)
    for segment in info["segments"]:
        pose=tmp_path/(segment["file"]+".csv")
        pose.write_text("scorer,m,m,m,m,m,m\nindividuals,one,one,one,two,two,two\nbodyparts,nose,nose,nose,nose,nose,nose\ncoords,x,y,likelihood,x,y,likelihood\n0,10,20,1,70,20,1\n1,11,20,1,71,20,1\n2,12,20,1,72,20,1\n",encoding="utf-8")
        entries.append({"segment":segment["file"],"pose":pose.name})
    manifest=tmp_path/"segments.json"
    atomic_json(manifest,{"segments":entries})
    cfg=complete({"paradigm":"custom","arenas":[{"name":"A","shape":{"type":"rect","rect":[0,0,96,64]}}]})
    with pytest.raises(ValueError,match="identity_mapping"):
        import_pose(manifest,cfg,tmp_path/"poses",video=rec.folder,analyze=False)
    for e in entries:e.update(identity_checked=True,identity_mapping={"one":"mouse1","two":"mouse2"})
    atomic_json(manifest,{"segments":entries})
    session=import_pose(manifest,cfg,tmp_path/"poses",video=rec.folder,analyze=False)
    track=Tracks.load(tracks_path(session,"A-mouse1"))
    assert track.frame.tolist()==list(range(6))
    assert track.time==pytest.approx(times,abs=1e-8)
    assert len(read_json(session/"session.json")["source"]["pose_segments"])==2
    entries.reverse();atomic_json(manifest,{"segments":entries})
    with pytest.raises(ValueError,match="순서대로"):
        import_pose(manifest,cfg,tmp_path/"poses",video=rec.folder,analyze=False)


def test_atomic_metadata_retries_windows_sharing_conflict(tmp_path,monkeypatch):
    from behavtrack import integrity
    path=tmp_path/"recording.json"
    atomic_json(path,{"frames":30})
    original=integrity.os.replace
    attempts=[]
    def transient(source,destination):
        attempts.append(1)
        assert read_json(path)=={"frames":30}
        if len(attempts)<3:raise PermissionError("temporary Windows sharing conflict")
        return original(source,destination)
    monkeypatch.setattr(integrity.os,"replace",transient)
    atomic_json(path,{"frames":60})
    assert read_json(path)=={"frames":60} and len(attempts)==3


def test_final_behavior_requires_complete_reference_and_blocks_exposure(tmp_path):
    _,train,train_labels=make_pose(tmp_path,"TRAIN")
    _,final,final_labels=make_pose(tmp_path,"FINAL",purpose="final")
    spec={"scope":"independent_cages","entries":[entry(train,train_labels),entry(final,final_labels,"final")]}
    assert ml.inspect_dataset(spec)["ok"]
    side=final_labels.with_suffix(".csv.json")
    original=read_json(side)
    missing=dict(original,events=original["events"][:1])
    atomic_json(side,missing)
    assert any("전체 원본" in error for error in ml.inspect_dataset(spec)["errors"])
    atomic_json(side,original)
    rd.mark_behavior_exposure(final,"test_prediction")
    assert any("행동 AI" in error for error in ml.inspect_dataset(spec)["errors"])
    with pytest.raises(ValueError,match="행동 AI"):
        rd.save_behavior(final,{"reviewer":"new","track":"A","start_frame":0,"end_frame":10,
                               "purpose":"final","label":"negative","observable":True})
