"""Workflow additions to the existing loopback web application."""
from __future__ import annotations
import importlib.util
import json
import re
import shutil
import threading
import uuid
from pathlib import Path
import cv2
import numpy as np
from .integrity import read_json,atomic_json,read_events,utcnow,digest
from . import research_data as labels
from . import ml_workflow as ml
from .recording_io import load_times,recover
from .cage_activity_api import CageActivityWorkflow
from .locomotion_api import LocomotionWorkflow


class ResearchWorkflow(CageActivityWorkflow,LocomotionWorkflow):
    def workspace_state(self):
        result={"poses":[],"datasets":[],"models":[],"active":None,
                "readiness":{"capture":{"ready":True,"detail":"OpenCV 원본 녹화; AI 가중치 불필요"},
                             "classifier":{"ready":importlib.util.find_spec("sklearn") is not None,"detail":".venv에 scikit-learn 필요; 설치하기.cmd"},
                             "detector":{"ready":(self.root/"models/topviewmouse/detector.pt").is_file() and (self.root/".ai_venv/Scripts/python.exe").is_file(),
                                         "detail":"접근 분석에는 .ai_venv와 models/topviewmouse/detector.pt 필요; ai_setup.cmd"}},
                "free_bytes":shutil.disk_usage(self.results).free}
        for path in (self.results/"poses").glob("*/session.json"):
            s=read_json(path)
            result["poses"].append({"id":path.parent.name,"path":str(path.parent),"cage_id":s.get("cage_id"),
                                     "source":s["source"],"tracks":list(s["tracks"]),
                                     "labels":[str(p) for p in path.parent.glob("reference_*.csv")]})
        for path in (self.results/"datasets").glob("*/dataset.json"):
            s=read_json(path)
            result["datasets"].append({"id":path.parent.name,"scope":s["scope"],"counts":s["inspection"]["counts"],
                                       "final_used":(path.parent/"final_evaluation_lock.json").exists()})
        for path in (self.results/"models").glob("candidate_*/candidate.json"):
            s=read_json(path)
            s["evaluations"]=[read_json(p) for p in path.parent.glob("evaluation_*.json")]
            result["models"].append(s)
        active=self.results/"models/active.json"
        if active.exists():
            result["active"]=read_json(active)
        return result

    def child(self,kind,name):
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9_가-힣.-]+",name):
            raise ValueError("목록에서 올바른 항목을 선택하세요.")
        parent=(self.results/kind).resolve()
        path=(parent/name).resolve()
        if path.parent!=parent or not path.is_dir():
            raise ValueError("자료 폴더가 없습니다.")
        return path

    def reference_path(self,session,plan):
        if not isinstance(plan,str) or not re.fullmatch(r"[a-f0-9]{32}",plan):
            raise ValueError("고정 표본 목록을 선택하세요.")
        path=self.session(session)/"references"/plan
        if path.resolve().parent!=(self.session(session)/"references").resolve():
            raise ValueError("잘못된 표본 경로입니다.")
        return path

    def idle_required(self):
        if self.capture.state()["recording"] or self.capture.saving:
            raise ValueError("촬영을 보호하기 위해 녹화 종료 후 자료 점검·학습·평가를 실행하세요.")
        if self.job_thread and self.job_thread.is_alive():
            raise ValueError("진행 중인 작업이 끝나거나 취소된 뒤 실행하세요.")

    def start_job(self,title,operation):
        self.idle_required()
        from .activity import Lease
        lease=Lease(self.results,"heavy")
        self.cancel=threading.Event()
        self.job={"state":"running","kind":title,"message":title,"done":0,"total":0}
        def progress(done,total,message):
            with self.lock:
                self.job.update(done=done,total=total,message=message)
        def worker():
            try:
                result=operation(self.cancel,progress)
                with self.lock:
                    self.job.update(state=result.get("state","complete") if isinstance(result,dict) else "complete",
                                    result=result,message=title+" 완료")
                    if self.job["state"]=="candidate":
                        self.job["state"]="complete"
                    if self.job["state"] in ("failed","cancelled"):
                        self.job["message"]=result.get("error",title+" 중단")
            except (Exception,SystemExit) as exc:
                with self.lock:
                    self.job.update(state="cancelled" if self.cancel.is_set() else "failed",message=str(exc))
            finally:
                lease.close()
        self.job_thread=threading.Thread(target=worker,daemon=True)
        self.job_thread.start()
        return {"started":True}

    def workflow_get(self,action,q):
        if action.startswith("locomotion-"):
            return self.locomotion_get(action,q)
        if action.startswith("activity-"):
            return self.activity_get(action,q)
        if action=="workspace":
            return self.workspace_state()
        if action=="events":
            return read_events(self.session(q["session"])/"events.jsonl")
        if action=="plans":
            return [labels.read_plan(p.parent) for p in (self.session(q["session"])/"references").glob("*/plan.json")]
        if action=="reference":
            path=self.reference_path(q["session"],q["plan"])
            return {"plan":labels.read_plan(path),"labels":[e for e in labels.latest(read_events(path/"labels.jsonl")) if e["reviewer"]==q.get("reviewer")]}
        if action=="behavior":
            from .evaluation import behavior_agreement
            pose=self.child("poses",q["pose"])
            return {"events":read_events(pose/"behavior_labels.jsonl"),"inter_rater":behavior_agreement(pose)}
        if action=="context":
            if q.get("pose"):
                from .session import tracks_path
                from .tracks import Tracks
                pose=self.child("poses",q["pose"])
                sess=read_json(pose/"session.json")
                tid=q.get("track")
                if tid not in sess["tracks"]:
                    raise ValueError("동물 트랙을 선택하세요.")
                tr=Tracks.load(tracks_path(pose,tid))
                fr=int(q["frame"])
                where=np.flatnonzero(tr.frame==fr)
                if not len(where):
                    raise ValueError("포즈에 없는 프레임입니다.")
                t=tr.time[where[0]]
                selected=np.flatnonzero((tr.time>=t-2)&(tr.time<=t+2))
                return [{"frame":int(tr.frame[i]),"time_s":float(tr.time[i])} for i in selected]
            folder=self.session(q["session"])
            _,times=load_times(folder,verify=False)
            fr=int(q["frame"])
            if not 0<=fr<len(times):
                raise ValueError("프레임 범위를 확인하세요.")
            selected=np.flatnonzero((times>=times[fr]-2)&(times<=times[fr]+2))
            return [{"frame":int(i),"time_s":float(times[i])} for i in selected]
        if action=="report":
            from .research_reports import result_tables
            return result_tables(self.session(q["session"]),self.run_path(q["session"],q["run"]))
        raise ValueError("지원하지 않는 연구 자료 조회입니다.")

    def raw_frame(self,q):
        from .sources import open_video
        frame_number=int(q["frame"])
        if "pose" in q:
            from .session import load_session,tracks_path
            from .tracks import Tracks
            pose=self.child("poses",q["pose"])
            sess=load_session(pose)
            source=sess["source"]["video"]
            local=frame_number-int(sess["source"].get("frame_offset",0))
        else:
            source=self.session(q["session"]); local=frame_number
        if Path(source).is_dir():
            from .recording_io import RecordingVideo
            info,_=load_times(source,verify=False)
            self.check_review_source(Path(source).name,{"source_sha256":info["sha256"]})
            cap=RecordingVideo(source,verify=False)
        else:
            cap=open_video(source)
        try:
            if not 0<=local<int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                raise ValueError("원본 프레임 범위를 확인하세요.")
            cap.set(cv2.CAP_PROP_POS_FRAMES,local)
            ok,image=cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=local+1:
                raise ValueError("원본 프레임을 읽지 못했습니다.")
        finally:
            cap.release()
        if "pose" in q and q.get("track") in sess["tracks"]:
            tr=Tracks.load(tracks_path(pose,q["track"]))
            where=np.flatnonzero(tr.frame==frame_number)
            if len(where) and np.isfinite(tr.xy[where[0]]).all():
                x,y=tr.xy[where[0]].astype(int)
                cv2.circle(image,(x,y),10,(0,200,255),2)
                cv2.putText(image,q["track"],(max(0,x),max(20,y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),1)
        ok,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:
            raise ValueError("검토 영상을 만들 수 없습니다.")
        return encoded.tobytes()

    def workflow(self,action,v):
        if action=="event":
            return labels.save_event(self.session(v["session"]),v)
        if action=="approach-label":
            path=self.reference_path(v["session"],v["plan"])
            if (path/"evaluation_lock.json").exists():
                raise ValueError("평가를 실행한 최종 정답은 수정할 수 없습니다. 정정 이력은 별도 개발 자료로 보관하세요.")
            return labels.save_approach(path,v)
        if action=="behavior-label":
            return labels.save_behavior(self.child("poses",v["pose"]),v)
        if action=="boxes":
            return labels.save_boxes(self.session(v["session"]),v)
        self.idle_required()
        if action.startswith("locomotion-"):
            return self.locomotion_workflow(action,v)
        if action.startswith("activity-"):
            return self.activity_workflow(action,v)
        if action=="recover":
            return recover(self.session(v["session"]))
        if action=="plan":
            return labels.make_plan(self.session(v["session"]),v.get("purpose","final"),int(v.get("count",60)),
                                    int(v.get("seed",0)),v.get("frames"),v.get("rationale",""))
        if action=="import-pose":
            from .pose_import import import_pose
            from .config import complete
            folder=self.session(v["session"])
            p=read_json(folder/"protocol.json");w,h=p["criteria"]["frame_size"]
            x1,y1,x2,y2=p["criteria"]["food_rect"]
            config=complete({"paradigm":"custom","arenas":[{"name":"A","shape":{"type":"rect","rect":[0,0,w,h]},
                           "zones":[{"name":"food","shape":{"type":"rect","rect":[x1*w,y1*h,(x2-x1)*w,(y2-y1)*h]}}]}]})
            scale=v.get("px_per_cm")
            if scale is not None:
                if not np.isfinite(scale) or scale<=0:
                    raise ValueError("실측한 px/cm을 양수로 입력하세요.")
                config["arenas"][0]["calibration"]={"px_per_cm":scale}
            return self.start_job("포즈 가져오기",lambda cancel,progress:{"path":str(import_pose(v["path"],config,
                            self.results/"poses",video=folder,name="pose_"+uuid.uuid4().hex[:10],analyze=False))})
        if action=="export-labels":
            pose=self.child("poses",v["pose"])
            out=pose/("reference_"+uuid.uuid4().hex[:10]+".csv")
            labels.export_behavior(pose,v["reviewer"],out)
            return {"path":str(out)}
        if action=="inspect":
            return self.start_job("학습 자료 점검",lambda cancel,progress:ml.inspect_dataset(v["spec"],self.root))
        if action=="freeze":
            return self.start_job("자료 목록 고정",lambda cancel,progress:ml.freeze_dataset(v["spec"],
                               self.results/"datasets"/("dataset_"+uuid.uuid4().hex[:12]),self.root))
        if action=="train":
            dataset=self.child("datasets",v["dataset"])
            return self.start_job("랜덤 포레스트 학습",lambda cancel,progress:ml.train_candidate(dataset,self.results/"models",cancel,progress))
        if action=="evaluate":
            dataset=self.child("datasets",v["dataset"])
            candidate=self.child("models",v["candidate"])
            baseline=self.child("models",v["baseline"]) if v.get("baseline") else None
            return self.start_job("운영 방식별 성능 평가",lambda cancel,progress:ml.evaluate_candidate(dataset,candidate,v.get("role","development"),baseline))
        if action=="promote":
            return ml.promote(self.results/"models",v["candidate"],v.get("actor",""),v.get("experimental") is True)
        if action=="rollback":
            return ml.rollback(self.results/"models",v.get("actor",""))
        if action=="predict":
            from .feeding_auto import analyze_pose_session
            from .session import load_session
            pose=self.child("poses",v["pose"]);sess=load_session(pose)
            active=read_json(self.results/"models/active.json")
            candidate=self.child("models",active["candidate"])
            bundle,status=ml.load_candidate(candidate)
            output=self.results/("feeding_"+uuid.uuid4().hex[:12])
            settings=bundle["postprocessing"]["feeding_auto"]
            def execute(cancel,progress):
                analyze_pose_session(pose,output,sess["cage_id"],sess["group"],sess.get("animals",len(sess["tracks"])),"food",candidate/"model.joblib",recorded_at=sess["source"].get("recorded_at",""),**settings)
                report=read_json(output/"auto_summary.json")
                report.update(applied_model=active,model_validated=active["validated"],application_independently_evaluated=False)
                atomic_json(output/"auto_summary.json",report)
                return {"path":str(output),"summary":report}
            return self.start_job("운영 모델 적용",execute)
        if action=="reference-evaluate":
            from . import research
            from .research_reports import reference_evaluation
            path=self.reference_path(v["session"],v["plan"])
            detector=self.root/"models/topviewmouse/detector.pt"
            if v.get("detector"):
                candidate=self.child("detectors",v["detector"])
                metadata=read_json(candidate/"detector.json")
                detector=candidate/"detector.pt"
                if digest(detector)!=metadata["sha256"]:
                    raise ValueError("후보 검출기 파일이 변경됐습니다.")
                cage=read_json(self.session(v["session"])/"protocol.json")["cage"]
                if any(s["cage"]==cage for s in metadata["export"]["sources"]):
                    raise ValueError("추가 학습·개발에 사용한 케이지로 독립 검출기 평가를 할 수 없습니다.")
            plan=labels.read_plan(path)
            if not detector.is_file() or not (self.root/".ai_venv/Scripts/python.exe").is_file():
                raise ValueError("접근 평가에는 .ai_venv와 검출 가중치가 필요합니다. ai_setup.cmd로 준비한 뒤 실행하세요. 최종 평가 정답은 아직 잠기지 않았습니다.")
            reviewed=labels.latest(read_events(path/"labels.jsonl"))
            reviewers={e["reviewer"] for e in reviewed}
            if not reviewers or any(sum(e["reviewer"]==r for e in reviewed)!=len(plan["samples"]) for r in reviewers):
                raise ValueError("평가 전에 각 작성자가 고정 표본 전체의 정답(판정 불가 포함)을 완료해야 합니다.")
            if plan["purpose"]=="final":
                with (path/"evaluation_lock.json").open("x",encoding="utf-8") as f:
                    json.dump({"utc":utcnow(),"labels_sha256":digest(path/"labels.jsonl")},f)
            def evaluate(cancel,progress):
                run,result=research.analyze(self.session(v["session"]),self.root/".ai_venv/Scripts/python.exe",
                            detector,cancel,
                            lambda done,total,run,message:progress(done,total,message),self.predictor_factory,path)
                report=reference_evaluation(path,run)
                return {"state":result["state"],"report":report,"path":str(run)}
            return self.start_job("고정 표본 접근 평가",evaluate)
        if action=="measurements":
            from .research_reports import import_measurements
            return import_measurements(self.session(v["session"]),v["path"])
        if action=="export-results":
            from .research_reports import result_tables,tables_csv
            folder=self.session(v["session"]);run=self.run_path(v["session"],v["run"])
            report=result_tables(folder,run)
            name="research_tables_"+uuid.uuid4().hex[:10]
            atomic_json(run/(name+".json"),report)
            (run/(name+".csv")).write_bytes(tables_csv(report))
            return {"path":str(run/(name+".csv")),"report":report}
        if action=="detector-export":
            from .dlc_bridge import export_detector
            entries=[dict(e,session=str(self.session(e["session"]))) for e in v["entries"]]
            output=self.results/"detector_exports"/("export_"+uuid.uuid4().hex[:12])
            return self.start_job("검출 상자 내보내기",lambda cancel,progress:dict(export_detector(entries,output,v["reviewer"]),path=str(output)))
        if action=="detector-train":
            from .dlc_bridge import launch
            return self.start_job("기존 DLC 검출기 추가 학습",lambda cancel,progress:launch(self.root,v["export"],v["config"],cancel=cancel,progress=progress,epochs=int(v.get("epochs",5))))
        if action=="detector-import":
            from .dlc_bridge import import_detector
            return import_detector(v["snapshot"],self.results/"detectors"/("detector_"+uuid.uuid4().hex[:12]))
        raise ValueError("지원하지 않는 연구 작업입니다.")
