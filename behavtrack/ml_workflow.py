"""Frozen data manifests, candidate training, one-shot evaluation and explicit promotion."""
from __future__ import annotations
import csv
import json
import math
import os
import threading
import uuid
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from .integrity import atomic_json, read_json, digest, json_digest, utcnow, append_event, read_events
from . import classifier as c
from .config import complete
from .session import load_session, tracks_path
from .tracks import Tracks
from .evaluation import binary_metrics, observed_durations

ROLES = ("train", "development", "final")


def _path(value, base):
    p = Path(value)
    return (base/p).resolve() if not p.is_absolute() else p.resolve()


def inspect_dataset(spec, base=Path.cwd()):
    """Read-only preflight with file/track/frame-specific repair instructions."""
    errors, warnings, records, reference = [], [], [], None
    groups, sources, session_ids = {}, {}, set()
    scope = spec.get("scope", "independent_cages")
    if scope not in ("independent_cages","same_cage_other_day","development_only"):
        errors.append("scope는 independent_cages / same_cage_other_day / development_only 중 하나여야 합니다.")
    counts = {r:{"positive":0,"negative":0,"frames":0,"cages":set()} for r in ROLES}
    for i, entry in enumerate(spec.get("entries", [])):
        location = f"자료 {i+1}"
        try:
            session = _path(entry["session"],base)
            labels = _path(entry["labels"],base)
            location = str(session)
            role = entry["role"]
            if role not in ROLES:
                raise ValueError("train/development/final 분할을 지정하세요.")
            sess = load_session(session)
            cage = str(sess.get("cage_id","")).strip()
            if not cage:
                raise ValueError("케이지 ID가 없습니다. 웹 녹화에 연결해 포즈를 가져오거나 session.json의 cage_id를 확인하세요.")
            if not sess["tracks"] or len(sess["tracks"]) != len(set(sess["tracks"])):
                raise ValueError("동물 트랙 ID가 없거나 중복입니다.")
            source = sess.get("source",{})
            day = (source.get("recorded_at") or entry.get("recorded_at", ""))
            if not day:
                raise ValueError("촬영 날짜·시간대가 없습니다. 실제 recorded_at을 입력하세요.")
            date = datetime.fromisoformat(day)
            if date.utcoffset() is None:
                raise ValueError("촬영 시각에 시간대를 포함하세요.")
            day = source.get("study_date") or date.date().isoformat()
            if source.get("pose_confidence_provided") is False:
                raise ValueError("포즈 신뢰도 점수가 원본에 없습니다. 실제 likelihood/point_scores가 포함된 결과를 가져오세요.")
            session_hash = digest(session/"session.json")
            if session_hash in session_ids:
                raise ValueError("같은 세션이 중복 포함됐습니다. 한 번만 선택하세요.")
            session_ids.add(session_hash)
            group = (cage,day) if scope=="same_cage_other_day" else cage
            if group in groups and groups[group]!=role:
                raise ValueError("같은 케이지의 모든 날짜·트랙을 같은 분할로 옮기세요. 날짜 평가만 same_cage_other_day로 명시할 수 있습니다.")
            groups[group]=role
            sidecar = read_json(labels.with_suffix(labels.suffix+".json"))
            if sidecar.get("task") != "individual_feeding" or sidecar.get("labels_sha256") != digest(labels) or sidecar.get("session_sha256") != session_hash:
                raise ValueError("정답이 원본 세션과 일치하지 않습니다. 개체 섭식 정답 화면에서 CSV를 다시 내보내세요.")
            if role=="final" and (sidecar.get("purposes") != ["final"] or any(e.get("ai_assisted") for e in sidecar.get("events",[]))):
                raise ValueError("최종 평가는 AI 보조 없는 final 용도의 독립 정답만 사용할 수 있습니다.")
            from .research_data import behavior_was_exposed
            if role=="final" and behavior_was_exposed(session):
                raise ValueError("이미 행동 AI 결과가 있는 촬영은 새 최종 평가 목록에 포함할 수 없습니다.")
            if role!="final" and "final" in sidecar.get("purposes",[]):
                raise ValueError("최종 평가 정답을 학습/개발에 사용할 수 없습니다.")
            if len(sess["tracks"])>1 and entry.get("identity_checked") is not True:
                raise ValueError("여러 마우스의 트랙 정체성·교환 의심 구간을 확인하고 identity_checked=true로 기록하세요. 불명 구간은 정답에서 제외하세요.")
            with labels.open(encoding="utf-8-sig",newline="") as f:
                label_rows=list(csv.DictReader(f))
            if any(row.get("track") not in sess["tracks"] for row in label_rows):
                raise ValueError("라벨 트랙 ID가 누락됐거나 포즈 트랙과 다릅니다. 다중 개체에서 빈 트랙 ID는 허용되지 않습니다.")
            video = source.get("video")
            if not video:
                raise ValueError("원본 영상 연결이 없습니다. --video를 지정해 포즈를 다시 가져오세요.")
            from .sources import open_video, video_info, camera_timestamps
            cap=open_video(video)
            try:
                vi=video_info(cap)
                stamps=camera_timestamps(video,vi["frames"])
            finally:
                cap.release()
            if source.get("timebase")=="camera_timestamps" and stamps is None:
                raise ValueError("카메라 촬영 시각 파일이 없습니다. 원본 메타데이터를 복원하세요.")
            if stamps is None:
                stamps=np.arange(vi["frames"])/vi["fps"]
            original_hashes=source.get("source_sha256",{})
            video_hashes=[v for k,v in original_hashes.items() if k.endswith(".mp4")]
            if Path(video).is_file():
                current=digest(video)
                if source.get("video_sha256") and current!=source["video_sha256"]:
                    raise ValueError("원본 영상 해시가 변경됐습니다.")
                video_hashes=[current]
            if not video_hashes:
                raise ValueError("원본 영상 해시를 찾을 수 없습니다.")
            for vh in video_hashes:
                if vh in sources:
                    raise ValueError(f"같은 영상 또는 파생 자료가 중복입니다: {sources[vh]}. 하나의 포즈 세션으로 구성하세요.")
                sources[vh]=str(session)
            track_hashes={}
            for tid,tr,names,x in c._session_features(session):
                track_hashes[tid]=digest(tracks_path(session,tid))
                if sidecar.get("track_sha256",{}).get(tid)!=track_hashes[tid]:
                    raise ValueError(f"{tid}: 정답 작성 이후 포즈가 변경됐습니다. 다시 검토하세요.")
                current_schema=c.feature_schema(sess,names)
                if reference is None:
                    reference=current_schema
                elif reference!=current_schema:
                    raise ValueError(f"{tid}: 좌표 단위·신체 부위·촬영 구도·특징 시간창이 다른 자료입니다.")
                if source.get("view") not in ("top","side","oblique"):
                    raise ValueError("촬영 구도 source.view를 top/side/oblique로 명시하세요.")
                local=tr.frame-int(source.get("frame_offset",0))
                if role=="final":
                    annotated=c.frame_mask(tr.frame,[(e["start_frame"],e["end_frame"]) for e in sidecar["events"] if e["track"]==tid])
                    if not np.array_equal(local,np.arange(len(stamps))) or not annotated.all() or len(sess["tracks"])!=sess.get("animals",len(sess["tracks"])):
                        raise ValueError(f"{tid}: 최종 행동 평가는 전체 원본 프레임·모든 동물의 정답(판정 불가 포함)을 고정해야 합니다. 미검토 구간을 완료하세요.")
                if (local<0).any() or (local>=len(stamps)).any() or not np.allclose(tr.time,stamps[local],rtol=0,atol=1e-7):
                    raise ValueError(f"{tid}: 영상·포즈 프레임/시각 불일치. 올바른 원본으로 다시 가져오세요.")
                for row in label_rows:
                    if row["track"]!=tid:
                        continue
                    a,b=int(row["start_frame"]),int(row["end_frame"])
                    chosen=(tr.frame>=a)&(tr.frame<=b)
                    if a>b or chosen.sum()!=b-a+1 or not np.allclose([tr.time[chosen][0],tr.time[chosen][-1]],
                            [float(row["start_s"]),float(row["end_s"])],rtol=0,atol=1e-7):
                        raise ValueError(f"{labels}, {tid}, {a}~{b}: 정답 프레임/시각 불일치. 원본 화면에서 구간을 다시 지정하세요.")
                y=c.frame_mask(tr.frame,c.read_labels(labels,"feeding",tid))
                reviewed=c.frame_mask(tr.frame,c.read_labels(labels,c.REVIEWED,tid))
                if (y & ~reviewed).any():
                    raise ValueError(f"{tid}: 양성 행동에 미검토 프레임이 포함됐습니다.")
                usable=reviewed & c.feature_valid(tr,complete(sess["config"])["analysis"]["keypoint_min_p"])
                counts[role]["positive"]+=int((usable & y).sum())
                counts[role]["negative"]+=int((usable & ~y).sum())
                counts[role]["frames"]+=int(usable.sum())
                counts[role]["cages"].add(cage)
                jumps=np.flatnonzero(np.linalg.norm(np.diff(tr.xy,axis=0),axis=1)>float(spec.get("max_step_px",100)))
                if len(jumps):
                    warnings.append(f"{session}, {tid}: 정체성 변경/트랙 점프 의심 프레임 {tr.frame[jumps[:10]+1].tolist()}. 해당 구간의 식별을 확인하세요.")
                excluded=int((reviewed & ~usable).sum())
                if excluded:
                    warnings.append(f"{session}, {tid}: 결측·시간 공백·창 경계 {excluded}프레임은 학습에서 제외됩니다.")
            records.append({"session":str(session),"labels":str(labels),"role":role,"cage_id":cage,"date":day,
                            "recorded_at":date.isoformat(),
                            "session_sha256":session_hash,"labels_sha256":digest(labels),
                            "label_manifest_sha256":digest(labels.with_suffix(labels.suffix+".json")),
                            "tracks_sha256":track_hashes,"video_sha256":video_hashes,
                            "identity_checked":entry.get("identity_checked",False)})
        except (Exception,SystemExit) as exc:
            errors.append(f"{location}: {exc}")
    if not records:
        errors.append("학습 자료가 없습니다. 촬영 → 포즈 가져오기 → 개체별 섭식 정답 작성 → CSV 내보내기를 먼저 완료하세요.")
    if counts["train"]["positive"]<10 or counts["train"]["negative"]<10:
        errors.append(f"학습 가능한 양성 {counts['train']['positive']} / 음성 {counts['train']['negative']} 프레임. 각각 10개 이상과 여러 행동 구간을 직접 확인하세요.")
    if scope=="same_cage_other_day" and len({r["cage_id"] for r in records})!=1:
        errors.append("같은 케이지 다른 날짜 평가는 케이지 하나만 포함해야 합니다.")
    if scope=="development_only" and any(r["role"]=="final" for r in records):
        errors.append("개발 전용 자료에는 최종 평가 분할을 두지 마세요.")
    for values in counts.values():
        values["cages"]=sorted(values["cages"])
        values["independent_cages"]=len(values["cages"])
    return {"ok":not errors,"errors":errors,"warnings":warnings,"counts":counts,"records":records,
            "feature_schema":reference,"scope":scope,"generalization":"unevaluated"}


def freeze_dataset(spec, destination, base=Path.cwd()):
    report=inspect_dataset(spec,base)
    if not report["ok"]:
        raise ValueError("\n".join(report["errors"]))
    destination=Path(destination)
    manifest={"schema":1,"id":uuid.uuid4().hex,"created_utc":utcnow(),"spec":spec,"inspection":report,
              "records":report["records"],"scope":report["scope"],"feature_schema":report["feature_schema"],
              "seed":int(spec.get("seed",0)),"acceptance":spec.get("acceptance"),
              "postprocessing":{"predict":{"threshold":float(spec.get("threshold",.5)),"min_bout_s":float(spec.get("min_bout_s",.2))},
                                "feeding_auto":{"negative":float(spec.get("negative",.2)),"positive":float(spec.get("positive",.8)),
                                                "min_p":float(spec.get("min_pose_p",.8)),"min_separation_px":float(spec.get("min_separation_px",2))}}}
    p=manifest["postprocessing"]
    acceptance=manifest["acceptance"] or {}
    for key,value in acceptance.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError(f"사전 허용 기준 {key}는 유한한 0 이상 수여야 합니다.")
        if key in ("min_precision","min_recall","min_f1","max_unknown_rate") and value>1:
            raise ValueError(f"{key}는 0~1 비율입니다.")
        if key in ("min_eval_cages","min_known") and (value<1 or int(value)!=value):
            raise ValueError(f"{key}는 1 이상 정수입니다.")
    if not 0<=p["predict"]["threshold"]<=1 or not 0<=p["feeding_auto"]["negative"]<p["feeding_auto"]["positive"]<=1 or p["predict"]["min_bout_s"]<0:
        raise ValueError("운영 임계값과 최소 행동시간을 확인하세요.")
    manifest["sha256"]=json_digest(manifest)
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/"dataset.json",manifest)
    return manifest


def read_dataset(folder):
    manifest=read_json(Path(folder)/"dataset.json")
    if json_digest({k:v for k,v in manifest.items() if k!="sha256"})!=manifest["sha256"]:
        raise ValueError("고정 데이터 목록/기준이 변경됐습니다. 새 데이터 버전을 만드세요.")
    for r in manifest["records"]:
        session,labels=Path(r["session"]),Path(r["labels"])
        checks={session/"session.json":r["session_sha256"],labels:r["labels_sha256"],
                labels.with_suffix(labels.suffix+".json"):r["label_manifest_sha256"]}
        checks.update({tracks_path(session,t):h for t,h in r["tracks_sha256"].items()})
        for path,sha in checks.items():
            if not path.is_file() or digest(path)!=sha:
                raise ValueError(f"고정 자료가 변경됐습니다: {path}")
        sess=load_session(session)
        source=sess.get("source",{})
        video=Path(source["video"])
        if video.is_dir():
            from .recording_io import load_times
            info,_=load_times(video)
            if sorted(v for k,v in info["sha256"].items() if k.endswith(".mp4"))!=sorted(r["video_sha256"]):
                raise ValueError("원본 촬영 영상이 변경됐습니다.")
        elif digest(video) not in r["video_sha256"]:
            raise ValueError("원본 영상이 변경됐습니다.")
    return manifest


def train_candidate(dataset, models, cancel=None, progress=None, trees=300):
    import joblib
    manifest=read_dataset(dataset)
    root=Path(models)/("candidate_"+uuid.uuid4().hex)
    root.mkdir(parents=True,exist_ok=False)
    status={"id":root.name,"state":"training","created_utc":utcnow(),"dataset_sha256":manifest["sha256"],
            "dataset":str(Path(dataset).resolve()),"validated":False}
    atomic_json(root/"candidate.json",status)
    try:
        records=[r for r in manifest["records"] if r["role"]=="train"]
        c.train([(r["session"],r["labels"]) for r in records],"feeding",root/"model.joblib",
                min_bout_s=manifest["postprocessing"]["predict"]["min_bout_s"],seed=manifest["seed"],
                n_estimators=trees,cancel=cancel,progress=progress)
        bundle=joblib.load(root/"model.joblib")
        bundle.update(dataset_sha256=manifest["sha256"],split=manifest["records"],
                      threshold=manifest["postprocessing"]["predict"]["threshold"],
                      postprocessing=manifest["postprocessing"])
        joblib.dump(bundle,root/"model.final.tmp")
        os.replace(root/"model.final.tmp",root/"model.joblib")
        if cancel and cancel.is_set():
            raise RuntimeError("학습 취소")
        status.update(state="candidate",model_sha256=digest(root/"model.joblib"))
    except (Exception,SystemExit) as exc:
        status.update(state="cancelled" if cancel and cancel.is_set() else "failed",error=str(exc))
    status["finished_utc"]=utcnow()
    atomic_json(root/"candidate.json",status)
    return status


def load_candidate(folder):
    import joblib
    folder=Path(folder)
    status=read_json(folder/"candidate.json")
    if status["state"]!="candidate" or digest(folder/"model.joblib")!=status["model_sha256"]:
        raise ValueError("완성되고 해시가 일치하는 후보 모델만 사용하세요.")
    return joblib.load(folder/"model.joblib"),status


def _score(bundle, manifest, role):
    reports=[]; combined={mode:{"h":[],"p":[],"d":[]} for mode in ("predict","feeding_auto")}
    strata={}
    for r in manifest["records"]:
        if r["role"]!=role:
            continue
        sess=load_session(r["session"])
        from .research_data import mark_behavior_exposure
        mark_behavior_exposure(r["session"],"evaluation_"+role)
        annotations=read_json(Path(r["labels"]).with_suffix(Path(r["labels"]).suffix+".json"))["events"]
        automatic=None
        for tid,tr,names,x in c._session_features(r["session"]):
            c.check_schema(bundle,sess,names)
            y=c.frame_mask(tr.frame,c.read_labels(r["labels"],"feeding",tid)).astype(int)
            reviewed=c.frame_mask(tr.frame,c.read_labels(r["labels"],c.REVIEWED,tid))
            y[~reviewed]=-1
            reference=reviewed | c.frame_mask(tr.frame,[(e["start_frame"],e["end_frame"]) for e in annotations if e["track"]==tid and e["label"]=="unknown"])
            d,gaps=observed_durations(tr.time)
            for mode in combined:
                if mode=="predict":
                    pred,_=c.predict_frames(
                        dict(bundle,min_bout_s=manifest["postprocessing"][mode]["min_bout_s"]),tr,x,
                        complete(sess["config"])["analysis"]["keypoint_min_p"],threshold=manifest["postprocessing"][mode]["threshold"])
                else:
                    from .feeding_auto import feeding_predictions
                    if automatic is None:
                        automatic=feeding_predictions(sess, {t:Tracks.load(tracks_path(r["session"],t)) for t in sess["tracks"]},bundle,
                                                      **manifest["postprocessing"][mode])
                    pred=automatic["labels"][tid]
                # Metrics operate only on reference-reviewed frames; unreviewed frames are not evaluation samples.
                metric=binary_metrics(y[reference],pred[reference],d[reference])
                reports.append(dict(metric,cage_id=r["cage_id"],date=r["date"],session=r["session"],track=tid,mode=mode,
                                    unreviewed_frames=int((~reference).sum()),capture_gap_seconds=gaps))
                for key,value in (("h",y[reference]),("p",pred[reference]),("d",d[reference])):
                    combined[mode][key].extend(value.tolist())
                masks={"cage:"+r["cage_id"]:reference,"date:"+r["date"]:reference}
                source=sess["source"]
                start=datetime.fromisoformat(source.get("recorded_at") or r["recorded_at"]).astimezone(timezone(timedelta(minutes=source.get("utc_offset_minutes",540))))
                hours=np.array([(start+timedelta(seconds=float(t))).hour for t in tr.time])
                for hour in np.unique(hours[reference]):masks[f"local_hour:{hour:02d}"]=reference & (hours==hour)
                for occlusion in {e.get("occlusion","unspecified") for e in annotations}:
                    masks["occlusion:"+occlusion]=reference & c.frame_mask(tr.frame,[(e["start_frame"],e["end_frame"]) for e in annotations if e["track"]==tid and e.get("occlusion","unspecified")==occlusion])
                for condition,mask in masks.items():
                    group=strata.setdefault(mode+"/"+condition,{"h":[],"p":[],"d":[]})
                    for key,value in (("h",y[mask]),("p",pred[mask]),("d",d[mask])):group[key].extend(value.tolist())
    return {"overall":{mode:binary_metrics(v["h"],v["p"],v["d"]) for mode,v in combined.items()},"by_session_track":reports,
            "by_condition":{key:binary_metrics(v["h"],v["p"],v["d"]) for key,v in strata.items()}}


def evaluate_candidate(dataset, candidate, role="development", baseline=None):
    if role not in ("development","final"):
        raise ValueError("학습 자료에서 평가 성능을 보고하지 않습니다.")
    manifest=read_dataset(dataset)
    bundle,status=load_candidate(candidate)
    if status["dataset_sha256"]!=manifest["sha256"]:
        raise ValueError("이 후보의 고정 학습/평가 목록과 다른 데이터입니다.")
    if not any(r["role"]==role for r in manifest["records"]):
        raise ValueError("평가용 영상·독립 정답이 없습니다. 검증 완료로 표시할 수 없습니다.")
    baseline_bundle,baseline_status=load_candidate(baseline) if baseline else (None,None)
    if baseline_bundle:
        if baseline_bundle.get("postprocessing")!=manifest["postprocessing"]:
            raise ValueError("기준 모델과 후보의 운영 후처리 조건이 다릅니다. 동일한 사전 조건을 고정해 비교하세요.")
        eval_hashes={h for r in manifest["records"] if r["role"]==role for h in r["video_sha256"]}
        train_hashes={h for r in baseline_bundle.get("split",[]) if r["role"]=="train" for h in r.get("video_sha256",[])}
        eval_cages={r["cage_id"] for r in manifest["records"] if r["role"]==role}
        train_cages={r["cage_id"] for r in baseline_bundle.get("split",[]) if r["role"]=="train"}
        if not baseline_bundle.get("split") or eval_hashes & train_hashes or manifest["scope"]=="independent_cages" and eval_cages & train_cages:
            raise ValueError("기준 모델의 학습 이력이 없거나 평가 자료/케이지와 겹칩니다.")
    if role=="final":
        # Exclusive persistent lock is deliberately retained on failure too. No repeated model selection.
        with (Path(dataset)/"final_evaluation_lock.json").open("x",encoding="utf-8") as f:
            json.dump({"candidate":status["model_sha256"],"baseline":baseline_status,"utc":utcnow()},f)
        # A copied/re-frozen dataset must not unlock the same final recordings.
        registry=Path(dataset).parent/".final_evaluation_registry"
        registry.mkdir(exist_ok=True)
        for r in manifest["records"]:
            if r["role"]=="final":
                for video_hash in r["video_sha256"]:
                    with (registry/(video_hash+".json")).open("x",encoding="utf-8") as f:
                        json.dump({"dataset_sha256":manifest["sha256"],"candidate":status["model_sha256"],"utc":utcnow()},f)
    scored=_score(bundle,manifest,role)
    compared=_score(baseline_bundle,manifest,role) if baseline_bundle else None
    criteria=manifest.get("acceptance") or {}
    eval_cages={r["cage_id"] for r in manifest["records"] if r["role"]==role}
    required={"min_precision","min_recall","min_f1","max_unknown_rate","max_duration_error_s","min_eval_cages","min_known"}
    checks={"predeclared_criteria":required.issubset(criteria),"independent_cages":manifest["scope"]=="independent_cages",
            "final_evaluation":role=="final","enough_cages":len(eval_cages)>=max(1,criteria.get("min_eval_cages",1))}
    for mode,m in scored["overall"].items():
        for metric in ("precision","recall","f1"):
            checks[f"{mode}_{metric}"]=m[metric] is not None and m[metric]>=criteria.get("min_"+metric,math.inf)
        checks[mode+"_coverage"]=m["ai_unknown_rate"] is not None and m["ai_unknown_rate"]<=criteria.get("max_unknown_rate",-1)
        checks[mode+"_both_classes"]=m["human_positive"]>0 and m["human_negative"]>0
        checks[mode+"_known"]=m["paired_known"]>=criteria.get("min_known",math.inf)
        differences=[abs(row["duration_difference_seconds_paired"]) for row in scored["by_session_track"] if row["mode"]==mode]
        checks[mode+"_duration"]=bool(differences) and max(differences)<=criteria.get("max_duration_error_s",-1)
    comparison={}
    if compared:
        for mode,m in scored["overall"].items():
            old=compared["overall"][mode]
            comparison[mode]={"f1_change":m["f1"]-old["f1"] if m["f1"] is not None and old["f1"] is not None else None,
                              "unknown_rate_change":m["ai_unknown_rate"]-old["ai_unknown_rate"] if m["total"] else None,
                              "automatic_winner":None}
    report={"schema":1,"utc":utcnow(),"role":role,"dataset_sha256":manifest["sha256"],
            "candidate_sha256":status["model_sha256"],"scope":manifest["scope"],"scores":scored,
            "baseline":compared,"comparison":comparison,"acceptance_checks":checks,"validated":all(checks.values()),
            "software_tests":"separate_see_test_report","same_cage_other_day":manifest["scope"]=="same_cage_other_day",
            "independent_cage_evaluation":manifest["scope"]=="independent_cages","evaluation_cages":sorted(eval_cages),
            "postprocessing":manifest["postprocessing"],"probabilities":"uncalibrated_model_scores"}
    destination=Path(candidate)/(f"evaluation_{role}_{uuid.uuid4().hex}.json")
    atomic_json(destination,report)
    return report


def promote(models, candidate_id, actor, experimental=False):
    models=Path(models)
    candidate=(models/candidate_id).resolve()
    if candidate.parent!=models.resolve():
        raise ValueError("후보 모델 경로를 확인하세요.")
    _,status=load_candidate(candidate)
    evaluations=[read_json(p) for p in candidate.glob("evaluation_*.json")]
    passed=any(e["validated"] and e["candidate_sha256"]==status["model_sha256"] for e in evaluations)
    if not passed and not experimental:
        raise ValueError("독립 평가/사전 기준을 통과하지 않았습니다. 개발 시험 적용은 experimental=true로 명시하세요.")
    if not actor.strip():
        raise ValueError("적용 연구자 이름을 입력하세요.")
    previous=read_json(models/"active.json") if (models/"active.json").exists() else None
    entry={"candidate":candidate_id,"model_sha256":status["model_sha256"],"actor":actor,"utc":utcnow(),
           "validated":passed,"experimental":not passed,"previous":previous}
    append_event(models/"promotion_history.jsonl",entry)
    atomic_json(models/"active.json",entry)
    return entry


def rollback(models, actor):
    models=Path(models)
    active=read_json(models/"active.json")
    previous=active.get("previous")
    if not previous:
        raise ValueError("되돌릴 이전 운영 모델이 없습니다.")
    return promote(models,previous["candidate"],actor,experimental=previous.get("experimental",False))

