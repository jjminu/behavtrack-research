"""Bound local sources and files for the cage activity screen."""
from __future__ import annotations
import re
import uuid
from pathlib import Path
import cv2
import numpy as np
from .integrity import read_json
from .recording_io import load_times, RecordingVideo


class CageActivityWorkflow:
    def activity_source(self,name):
        if not isinstance(name,str) or not re.fullmatch(r"(?:research_|public_mouse_demo_)[A-Za-z0-9_]+",name):
            raise ValueError("활동성 분석에 사용할 촬영을 선택하세요.")
        folder=(self.results/name).resolve()
        if folder.parent!=self.results.resolve() or not folder.is_dir():raise ValueError("촬영 폴더를 찾을 수 없습니다.")
        return folder

    def activity_run(self,session,name):
        if not isinstance(name,str) or not re.fullmatch(r"activity_[a-f0-9]{12}",name):raise ValueError("활동성 결과를 선택하세요.")
        source=self.activity_source(session);parent=(source/"cage_activity").resolve();run=(parent/name).resolve()
        if parent.parent!=source:raise ValueError("활동성 결과 경로를 확인하세요.")
        if run.parent!=parent or not run.is_dir():raise ValueError("활동성 결과가 없습니다.")
        return run

    def activity_sources(self):
        result=[]
        for path in sorted(self.results.glob("*/recording.json"),reverse=True):
            try:
                folder=self.activity_source(path.parent.name)
                rec=read_json(folder/("recovery.json" if (folder/"recovery.json").exists() else "recording.json"))
                if rec["state"] not in ("complete","partial","recovered"):continue
                spec=read_json(folder/"protocol.json");analyses=[]
                for file in sorted((folder/"cage_activity").glob("activity_*/activity.json"),key=lambda p:p.stat().st_mtime,reverse=True):
                    r=read_json(file);analyses.append({"id":file.parent.name,"state":r["state"],"summary":r.get("summary"),"metadata":r["metadata"]})
                result.append({"id":folder.name,"protocol":spec,"state":rec["state"],"frames":rec["frames"],
                               "duration_s":rec.get("last_capture_s",(rec["frames"]-1)/rec["video_fps"]),"started_utc":rec.get("started_utc",""),"analyses":analyses})
            except (OSError,ValueError,KeyError):continue
        return {"sessions":result}

    def activity_frame(self,q):
        folder=self.activity_source(q["session"]);info,times=load_times(folder)
        elapsed=float(q.get("time_s",0))
        if not np.isfinite(elapsed) or not 0<=elapsed<=times[-1]+1e-6:raise ValueError("영상 시각 범위를 확인하세요.")
        right=int(np.searchsorted(times,elapsed));choices=[i for i in (right-1,right) if 0<=i<len(times)]
        fid=min(choices,key=lambda i:abs(times[i]-elapsed));cap=RecordingVideo(folder,verify=False)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,frame=cap.read()
            if not ok:raise ValueError("미리보기 프레임을 읽지 못했습니다.")
        finally:cap.release()
        if frame.shape[1]>960:frame=cv2.resize(frame,(960,round(frame.shape[0]*960/frame.shape[1])))
        ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok:raise ValueError("미리보기를 만들지 못했습니다.")
        return encoded.tobytes()

    def activity_file(self,q):
        name=q.get("file","")
        if q.get("comparison"):
            ident=q["comparison"]
            if not re.fullmatch(r"compare_[a-f0-9]{12}",ident):raise ValueError("비교 결과를 선택하세요.")
            parent=(self.results/"activity_comparisons").resolve();folder=(parent/ident).resolve()
            if folder.parent!=parent or name not in ("comparison.json","cages.csv","chart.png","report.html"):
                raise ValueError("지원하지 않는 비교 파일입니다.")
        else:
            folder=self.activity_run(q["session"],q["run"])
            if name not in ("activity.json","pairs.csv","chart.png","report.html") and not re.fullmatch(r"example_\d{6}\.jpg",name):
                raise ValueError("지원하지 않는 활동성 파일입니다.")
        path=(folder/name).resolve()
        if path.parent!=folder.resolve():raise ValueError("허용된 결과 폴더 안의 파일을 선택하세요.")
        if not path.is_file():raise ValueError("아직 생성되지 않은 파일입니다.")
        return path

    def activity_get(self,action,q):
        from .cage_activity import read_report
        if action=="activity-sources":return self.activity_sources()
        if action=="activity-report":return read_report(self.activity_run(q["session"],q["run"]))
        if action=="activity-comparison":return read_json(self.activity_file(dict(q,file="comparison.json")))
        raise ValueError("지원하지 않는 활동성 조회입니다.")

    def activity_workflow(self,action,v):
        from .cage_activity import analyze,read_report,write_comparison
        if action=="activity-analyze":
            folder=self.activity_source(v.get("session"))
            return self.start_job("케이지 활동성",lambda cancel,progress:analyze(folder,v,cancel,progress))
        if action=="activity-compare":
            entries=v.get("entries",[])
            if not isinstance(entries,list) or not 1<=len(entries)<=300:raise ValueError("비교할 활동성 결과를 선택하세요.")
            def execute(cancel,progress):
                reports=[]
                for i,item in enumerate(entries):
                    if cancel.is_set():return {"state":"cancelled","error":"비교를 취소했습니다."}
                    reports.append(read_report(self.activity_run(item["session"],item["run"])))
                    progress(i+1,len(entries),"활동성 원본과 표본 확인 중")
                folder=self.results/"activity_comparisons"/("compare_"+uuid.uuid4().hex[:12])
                result=write_comparison(folder,reports,v.get("mode","water"))
                return {"kind":"cage_activity_comparison","state":"complete","comparison":folder.name,"path":str(folder),"report":result}
            return self.start_job("케이지 활동성 비교",execute)
        raise ValueError("지원하지 않는 활동성 작업입니다.")
