"""Cage-level image movement, quality accounting, and independent-cage summaries.

The endpoint is changed image area at a fixed interval, not locomotor distance.
"""
from __future__ import annotations

import base64
import csv
import html
import math
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .activity import heavy
from .integrity import atomic_json, csv_safe, digest, json_digest, read_json, utcnow
from .recording_io import load_times, RecordingVideo

VERSION = "cage_image_movement_v1"
WATER = {"control":"일반수(대조)","hydrogen":"수소수","oxygen":"산소수","unspecified":"미지정"}
ADMIN = {"bottle_only":"급수병만","oral_plus_bottle":"경구투여＋급수병","unspecified":"미지정"}
FIELDS = ["pair","start_s","end_s","duration_s","start_frame","end_frame","status","reason",
          "changed_area_pct","mean_abs_difference","brightness_shift","reference_shift_px"]
DEFAULTS = {"interval_s":.2,"time_tolerance_s":.04,"pixel_threshold":12.,"process_width":320,
            "blur":3,"max_brightness_shift":12.,"max_changed_fraction":.35,
            "max_reference_shift_px":2.,"min_phase_response":.2,"bin_seconds":60.}
REASONS = {"valid":"유효", "not_analyzed":"미분석", "missing_frame":"표본 프레임 부족",
           "time_gap":"프레임 시각 공백", "illumination_change":"조명 변화 의심",
           "camera_shift":"고정 배경 이동 의심", "large_scene_change":"큰 장면 변화 의심"}


def rectangle(value, optional=False):
    if optional and value is None:return None
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError("화면에서 케이지 영역을 표시하세요.")
    r=[float(x) for x in value]
    if not all(math.isfinite(x) and 0<=x<=1 for x in r) or min(r[2]-r[0],r[3]-r[1])<.02:
        raise ValueError("표시 영역은 화면 안의 유효한 사각형이어야 합니다.")
    return r


def settings(values):
    out=dict(DEFAULTS)
    for key in DEFAULTS:
        if key in values:out[key]=float(values[key])
    if not all(math.isfinite(v) for v in out.values()):raise ValueError("활동성 설정은 유한한 수여야 합니다.")
    if (not .1<=out["interval_s"]<=1 or not 0<out["time_tolerance_s"]<out["interval_s"]/2 or
        not 1<=out["pixel_threshold"]<=100 or not 64<=out["process_width"]<=640 or
        out["blur"] not in (1,3,5,7) or not 1<=out["max_brightness_shift"]<=100 or
        not .05<=out["max_changed_fraction"]<=1 or not 0<out["max_reference_shift_px"]<=30 or
        not 0<out["min_phase_response"]<=1 or not 1<=out["bin_seconds"]<=3600):
        raise ValueError("활동성 설정 범위를 확인하세요.")
    out["process_width"],out["blur"]=int(out["process_width"]),int(out["blur"])
    out.update(roi=rectangle(values.get("roi")),reference_roi=rectangle(values.get("reference_roi"),True),method=VERSION)
    return out


def metadata(values, protocol):
    out={key:str(values.get(key,protocol.get(key,""))).strip() for key in ("cage","group","water_type","administration")}
    out["water_type"]=out["water_type"] or "unspecified"
    out["administration"]=out["administration"] or "unspecified"
    for key in ("setup_id","comparison_block"):
        out[key]=str(values.get(key,"")).strip()
    if any(not out[k] or len(out[k])>100 for k in ("cage","group","setup_id","comparison_block")):
        raise ValueError("케이지·그룹·촬영 설정 ID·비교 블록을 각각 1~100자로 입력하세요.")
    if out["water_type"] not in WATER or out["administration"] not in ADMIN:
        raise ValueError("물 종류와 투여 방식을 확인하세요.")
    out["animals"]=protocol["criteria"]["animals"]
    return out


def gray_roi(frame, roi, config):
    h,w=frame.shape[:2];x1,y1,x2,y2=roi
    crop=frame[round(y1*h):round(y2*h),round(x1*w):round(x2*w)]
    if min(crop.shape[:2])<8:raise ValueError("표시한 영역이 너무 작습니다.")
    width=config["process_width"];height=max(8,round(crop.shape[0]*width/crop.shape[1]))
    # A very tall, narrow reference must not allocate unbounded images.
    if height>1024:
        width=max(8,round(width*1024/height));height=1024
    gray=cv2.cvtColor(cv2.resize(crop,(width,height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    if config["blur"]>1:gray=cv2.GaussianBlur(gray,(config["blur"],config["blur"]),0)
    return gray.astype(np.float32)


def pair_measure(previous,current,config,previous_reference=None,current_reference=None):
    delta=current-previous
    brightness=float(np.median(delta))
    shift=None
    reason="valid"
    if previous_reference is not None:
        window=cv2.createHanningWindow((previous_reference.shape[1],previous_reference.shape[0]),cv2.CV_32F)
        (dx,dy),response=cv2.phaseCorrelate(previous_reference.copy(),current_reference.copy(),window)
        if math.isfinite(response) and response>=config["min_phase_response"]:
            shift=math.hypot(dx,dy)
            if not math.isfinite(shift):shift=None
        if shift is not None and shift>config["max_reference_shift_px"]:reason="camera_shift"
    difference=np.abs(delta-brightness)
    mask=difference>=config["pixel_threshold"]
    if abs(brightness)>config["max_brightness_shift"]:reason="illumination_change"
    elif float(mask.mean())>config["max_changed_fraction"]:reason="large_scene_change"
    return {"status":"valid" if reason=="valid" else "unknown","reason":reason,
            "changed_area_pct":float(mask.mean()*100) if reason=="valid" else None,
            "mean_abs_difference":float(difference.mean()) if reason=="valid" else None,
            "brightness_shift":brightness,"reference_shift_px":shift},mask


def summarize(pairs):
    valid=[r for r in pairs if r["status"]=="valid"]
    known=sum(r["duration_s"] for r in valid);total=sum(r["duration_s"] for r in pairs)
    return {"pairs":len(pairs),"valid_pairs":len(valid),"unknown_pairs":len(pairs)-len(valid),
            "observed_s":total,"valid_s":known,"unknown_s":total-known,
            "coverage_pct":100*known/total if total else None,
            "mean_changed_area_pct":sum(r["changed_area_pct"]*r["duration_s"] for r in valid)/known if known else None,
            "mean_abs_difference":sum(r["mean_abs_difference"]*r["duration_s"] for r in valid)/known if known else None,
            "unknown_reasons":dict(Counter(r["reason"] for r in pairs if r["status"]!="valid"))}


def time_bins(pairs, seconds):
    """Split intervals at bin boundaries; each second contributes at most once."""
    bins=defaultdict(list)
    for row in pairs:
        start=row["start_s"];end=row["end_s"]
        while start<end-1e-9:
            index=math.floor((start+1e-9)/seconds);stop=min(end,(index+1)*seconds)
            bins[index].append(dict(row,start_s=start,end_s=stop,duration_s=stop-start));start=stop
    return [dict(summarize(rows),start_s=min(r["start_s"] for r in rows),end_s=max(r["end_s"] for r in rows))
            for i,rows in sorted(bins.items())]


def _plan(times,start,end,config):
    targets=start+np.arange(math.floor((end-start)/config["interval_s"]+1e-8)+1)*config["interval_s"]
    selected=[];used=set()
    for target in targets:
        pos=int(np.searchsorted(times,target));choices=[i for i in (pos-1,pos) if 0<=i<len(times)]
        fid=min(choices,key=lambda i:(abs(times[i]-target),i))
        good=abs(times[fid]-target)<=config["time_tolerance_s"]+1e-9 and fid not in used
        if good:used.add(fid)
        selected.append({"target":float(target),"frame":fid if good else None})
    return selected


def _example(frame,mask,config,path,elapsed,score):
    view=cv2.resize(frame,(720,round(frame.shape[0]*720/frame.shape[1])))
    h,w=view.shape[:2];overlay=view.copy();x1,y1,x2,y2=[round(v*d) for v,d in zip(config["roi"],(w,h,w,h))]
    if mask is not None:
        m=cv2.resize(mask.astype(np.uint8),(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST).astype(bool)
        part=overlay[y1:y2,x1:x2];part[m]=(part[m]*.35+np.array([190,30,230])*.65).astype(np.uint8)
    cv2.rectangle(overlay,(x1,y1),(x2,y2),(70,210,140),2)
    if config["reference_roi"]:
        a,b,c,d=[round(v*q) for v,q in zip(config["reference_roi"],(w,h,w,h))]
        cv2.rectangle(overlay,(a,b),(c,d),(0,170,255),2)
    cv2.putText(overlay,f"{elapsed:.1f}s | changed area: {score:.3f}%" if score is not None else f"{elapsed:.1f}s | UNKNOWN",
                (12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(str(path))


@heavy
def analyze(folder,values,cancel=None,progress=None):
    folder=Path(folder);cancel=cancel or threading.Event();progress=progress or (lambda *args:None)
    info,times=load_times(folder);protocol=read_json(folder/"protocol.json")
    config=settings(values);meta=metadata(values,protocol)
    start=float(values.get("start_s",0));end=float(values.get("end_s") if values.get("end_s") is not None else times[-1])
    if not math.isfinite(start) or not math.isfinite(end) or start<0 or end>times[-1]+1e-6 or end-start<config["interval_s"]:
        raise ValueError("영상 범위 안에서 시작·종료 시각을 지정하세요.")
    plan=_plan(times,start,end,config)
    pairs=[{"pair":i-1,"start_s":plan[i-1]["target"],"end_s":plan[i]["target"],
            "duration_s":plan[i]["target"]-plan[i-1]["target"],"start_frame":plan[i-1]["frame"],"end_frame":plan[i]["frame"],
            "status":"unknown","reason":"not_analyzed","changed_area_pct":None,"mean_abs_difference":None,
            "brightness_shift":None,"reference_shift_px":None} for i in range(1,len(plan))]
    run=folder/"cage_activity"/("activity_"+uuid.uuid4().hex[:12]);run.mkdir(parents=True)
    compatible=dict(config,animals=meta["animals"],window_duration_s=round(plan[-1]["target"]-plan[0]["target"],6),
                    source_size=protocol["criteria"]["frame_size"])
    manifest={"schema":1,"id":run.name,"method":VERSION,"state":"running","created_utc":utcnow(),"source_session":folder.name,
              "metadata":meta,"settings":config,"window":{"start_s":start,"requested_end_s":end,"end_s":plan[-1]["target"]},
              "comparison_signature":json_digest(compatible),"source_sha256":info["sha256"],
              "recorded_start_utc":info.get("started_utc"),
              "code_sha256":digest(Path(__file__)),"timestamp_source":info.get("timestamp_source"),"examples":[],
              "scope":"cage_image_movement_not_individual_distance","validated":False,"error":None}
    atomic_json(run/"activity.json",manifest)
    cap=RecordingVideo(folder,verify=False);decoded=-1;previous=None;previous_ref=None;previous_time=None
    try:
        for index,point in enumerate(plan):
            if cancel.is_set():manifest["state"]="cancelled";break
            fid=point["frame"]
            if fid is None:
                if index:pairs[index-1]["reason"]="missing_frame"
                previous=previous_ref=previous_time=None
            else:
                while decoded<fid:
                    ok,frame=cap.read()
                    if not ok:raise ValueError("선택한 관측 구간의 영상 프레임을 읽지 못했습니다.")
                    decoded+=1
                current=gray_roi(frame,config["roi"],config)
                current_ref=gray_roi(frame,config["reference_roi"],config) if config["reference_roi"] else None
                mask=None
                if index:
                    row=pairs[index-1]
                    dt=float(times[fid])-previous_time if previous_time is not None else None
                    if previous is None:row["reason"]="missing_frame"
                    elif dt<=0 or abs(dt-config["interval_s"])>2*config["time_tolerance_s"]+1e-6:row["reason"]="time_gap"
                    else:
                        measurement,mask=pair_measure(previous,current,config,previous_ref,current_ref);row.update(measurement)
                    if (index-1)%max(1,len(pairs)//12)==0 and len(manifest["examples"])<13:
                        name=f"example_{index-1:06d}.jpg";_example(frame,mask,config,run/name,point["target"],row["changed_area_pct"])
                        manifest["examples"].append({"file":name,"time_s":point["target"],"status":row["status"],"reason":row["reason"]})
                previous,previous_ref,previous_time=current,current_ref,float(times[fid])
            if index%10==0 or index==len(plan)-1:progress(index+1,len(plan),"케이지 전체 영상 움직임 계산 중")
        else:
            last=int(np.searchsorted(times,end,side="right")-1)
            while decoded<last:
                if not cap.read()[0]:raise ValueError("선택한 구간의 마지막 프레임을 읽지 못했습니다.")
                decoded+=1
            manifest["state"]="complete"
    except (Exception,SystemExit) as exc:
        manifest.update(state="failed",error=str(exc))
    finally:
        cap.release()
        with (run/"pairs.csv").open("x",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(pairs)
        manifest.update(finished_utc=utcnow(),summary=summarize(pairs),bins=time_bins(pairs,config["bin_seconds"]),
                        pairs_sha256=digest(run/"pairs.csv"))
        atomic_json(run/"activity.json",manifest)
    try:
        write_report(run,manifest)
    except Exception as exc:
        manifest["report_error"]=str(exc);atomic_json(run/"activity.json",manifest)
    return {"kind":"cage_activity","state":manifest["state"],"session":folder.name,"run":run.name,
            "path":str(run),"report":manifest,"error":manifest["error"]}


def read_report(run,verify_source=True):
    run=Path(run);report=read_json(run/"activity.json")
    if digest(run/"pairs.csv")!=report["pairs_sha256"]:raise ValueError("활동성 표본 CSV가 변경됐습니다.")
    with (run/"pairs.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    for row in rows:
        for key in ("start_s","end_s","duration_s","changed_area_pct","mean_abs_difference","brightness_shift","reference_shift_px"):
            row[key]=float(row[key]) if row[key]!="" else None
        if row["duration_s"]<=0 or not math.isclose(row["duration_s"],row["end_s"]-row["start_s"],abs_tol=1e-8):
            raise ValueError("활동성 표본의 시간 대응이 잘못됐습니다.")
        if row["status"] not in ("valid","unknown") or row["status"]=="valid" and (row["changed_area_pct"] is None or not 0<=row["changed_area_pct"]<=100):
            raise ValueError("활동성 표본의 상태·값을 확인하세요.")
    if json_digest(summarize(rows))!=json_digest(report["summary"]) or json_digest(time_bins(rows,report["settings"]["bin_seconds"]))!=json_digest(report["bins"]):
        raise ValueError("활동성 요약이 원본 표본 CSV와 다릅니다.")
    if verify_source:
        info,_=load_times(run.parent.parent)
        if info["sha256"]!=report["source_sha256"]:raise ValueError("활동성 원본이 분석 당시와 다릅니다.")
    return report


def comparison(reports,mode="water",*,metric="mean_changed_area_pct",unit="%",kind="cage_activity_comparison"):
    """One cage is one descriptive point; repeated videos cannot inflate n."""
    if mode not in ("water","group"):raise ValueError("물 종류별 또는 그룹별 비교를 선택하세요.")
    excluded=[];latest={}
    for report in reports:
        m=report["metadata"];reason=None
        if report["state"]!="complete" or report["summary"]["valid_s"]<=0:reason="완료된 유효 구간 없음"
        elif m["administration"]=="unspecified":reason="투여 방식 미지정"
        elif mode=="water" and m["water_type"]=="unspecified":reason="물 종류 미지정"
        if reason:excluded.append({"session":report["source_session"],"reason":reason});continue
        key=(report["source_session"],report["comparison_signature"],m["setup_id"],m["comparison_block"],
             report["window"]["start_s"],report["window"]["end_s"])
        if key not in latest or report["created_utc"]>latest[key]["created_utc"]:latest[key]=report
    strata=defaultdict(list)
    for r in latest.values():
        m=r["metadata"];strata[(r["comparison_signature"],m["setup_id"],m["comparison_block"],m["administration"])].append(r)
    outputs=[]
    for (signature,setup,block,admin),items in sorted(strata.items()):
        labels=defaultdict(set)
        for r in items:
            m=r["metadata"];labels[m["cage"]].add((m["group"],m["water_type"]))
        if any(len(v)>1 for v in labels.values()):
            raise ValueError("같은 비교 블록에서 같은 케이지에 다른 그룹·물 종류가 있습니다. 교차설계의 기간을 별도 블록으로 지정하세요.")
        cages=defaultdict(list);seen_sources=set()
        for r in items:
            source=(tuple(sorted((k,v) for k,v in r["source_sha256"].items() if k.startswith("raw") and k.endswith(".mp4"))),
                    r["window"]["start_s"],r["window"]["end_s"])
            # Hash duplicates are repeated input, including copies with new session names.
            if source in seen_sources:
                excluded.append({"session":r["source_session"],"reason":"동일한 원본 영상 중복"});continue
            seen_sources.add(source);cages[r["metadata"]["cage"]].append(r)
        points=[]
        for cage,repeat in sorted(cages.items()):
            # For native recordings, reject overlapping capture windows for one cage.
            intervals=[];source_intervals=[]
            from datetime import datetime,timedelta
            for r in repeat:
                identity=tuple(sorted((k,v) for k,v in r["source_sha256"].items() if k.startswith("raw") and k.endswith(".mp4")))
                aa,bb=r["window"]["start_s"],r["window"]["end_s"]
                if any(identity==old and aa<y and bb>x for old,x,y in source_intervals):
                    raise ValueError("같은 케이지에서 동일 원본의 관측 구간이 겹칩니다. 중복 구간을 제외하세요.")
                source_intervals.append((identity,aa,bb))
                capture=r.get("recorded_start_utc")
                if capture:
                    a=datetime.fromisoformat(capture)+timedelta(seconds=r["window"]["start_s"])
                    b=datetime.fromisoformat(capture)+timedelta(seconds=r["window"]["end_s"])
                    if any(a<y and b>x for x,y in intervals):raise ValueError("같은 케이지의 관측 구간이 겹칩니다. 중복 영상을 제외하세요.")
                    intervals.append((a,b))
            valid=sum(r["summary"]["valid_s"] for r in repeat);total=sum(r["summary"]["observed_s"] for r in repeat)
            m=repeat[0]["metadata"]
            points.append({"cage":cage,"group":m["group"],"water_type":m["water_type"],"recordings":len(repeat),
                           "value":sum(r["summary"][metric]*r["summary"]["valid_s"] for r in repeat)/valid,
                           "valid_s":valid,"observed_s":total,"coverage_pct":100*valid/total})
        grouped=defaultdict(list)
        for p in points:grouped[p["water_type"] if mode=="water" else p["group"]].append(p)
        groups=[]
        for label,pts in sorted(grouped.items()):
            vals=[p["value"] for p in pts]
            groups.append({"label":label,"name":WATER.get(label,label) if mode=="water" else label,"n_cages":len(pts),
                           "mean":float(np.mean(vals)),"sd":float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                           "points":pts,"difference_from_control_pp":None,"difference_from_control":None,"relative_to_control_pct":None})
        baseline=next((g for g in groups if g["label"]=="control"),None) if mode=="water" else None
        if baseline:
            for g in groups:
                g["difference_from_control"]=g["mean"]-baseline["mean"]
                if unit=="%":g["difference_from_control_pp"]=g["mean"]-baseline["mean"]
                g["relative_to_control_pct"]=100*(g["mean"]-baseline["mean"])/baseline["mean"] if baseline["mean"]>0 else None
        outputs.append({"comparison_signature":signature,"setup_id":setup,"comparison_block":block,
                        "administration":admin,"groups":groups,"comparable_groups":len(groups),
                        "interpretation":"케이지별 영상 움직임의 기술적 비교이며 처리 효과·통계적 유의성을 확정하지 않습니다."+
                        (" 경구투여＋급수병 조건은 두 투여 경로가 함께 적용된 처치이며 급수병 물만의 효과를 분리하지 못합니다." if admin=="oral_plus_bottle" else "")})
    return {"kind":kind,"schema":1,"mode":mode,"metric":metric,"unit":unit,"strata":outputs,"excluded":excluded,
            "independent_unit":"cage","p_value":None,"inferential_status":"descriptive_only",
            "criteria":"같은 촬영 설정·비교 블록·분석 조건·관찰 길이·마릿수에서 투여 방식을 나눠 비교"}


def write_report(run,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run=Path(run);s=report["summary"];m=report["metadata"]
    with plt.rc_context({"font.family":"Malgun Gothic","axes.unicode_minus":False}):
        fig,(ax,quality)=plt.subplots(2,1,figsize=(10,5),layout="constrained",gridspec_kw={"height_ratios":[3,1]})
        bins=report["bins"];x=[(b["start_s"]+b["end_s"])/120 for b in bins]
        ax.bar(x,[b["mean_changed_area_pct"] or 0 for b in bins],width=[max(.002,(b["end_s"]-b["start_s"])/60*.85) for b in bins],color="#20846c")
        ax.set(ylabel="변화 픽셀 비율 (%)",title="케이지 전체 영상 움직임 · 0.2초 프레임 차이".replace("0.2",f"{report['settings']['interval_s']:g}"))
        quality.bar(x,[b["coverage_pct"] or 0 for b in bins],width=[max(.002,(b["end_s"]-b["start_s"])/60*.85) for b in bins],color="#a4bac4")
        quality.set(ylabel="유효 (%)",xlabel="영상 경과시간 (분)",ylim=(0,105))
        for a in (ax,quality):a.spines[["top","right"]].set_visible(False);a.grid(axis="y",alpha=.15)
        fig.savefig(run/"chart.png",dpi=140);plt.close(fig)
    report["notes"]=["케이지 영역에서 프레임 사이 변한 픽셀의 비율입니다. 이동·그루밍·물체 움직임 등이 함께 반영됩니다.",
                     "개체별 이동거리·속도·활동시간으로 환산하지 않습니다. 판정 불가 구간은 정지로 세지 않습니다.",
                     "물 효과를 비교하려면 같은 촬영 조건·시간대·관찰 길이·마릿수를 맞추고 독립 케이지를 확보해야 합니다.",
                     "가림·국소 그림자·카메라 회전 등을 완전히 검출하지 못합니다. 실제 공동 사육 영상의 기준 채점과 비교하기 전까지 연구 지표로 검증된 상태가 아닙니다."]
    if not report["settings"]["reference_roi"]:report["notes"].append("고정 배경 영역을 지정하지 않아 카메라 이동 자동 점검을 사용하지 않았습니다.")
    atomic_json(run/"activity.json",report)
    f=lambda v:"계산 불가" if v is None else f"{v:.3f}"
    esc=html.escape;image=base64.b64encode((run/"chart.png").read_bytes()).decode()
    examples="".join(f"<figure><img alt='움직임 픽셀 표시 장면' src='data:image/jpeg;base64,{base64.b64encode((run/e['file']).read_bytes()).decode()}'><figcaption>{e['time_s']:.1f}초 · {esc(REASONS.get(e['reason'],e['reason']))}</figcaption></figure>" for e in report["examples"])
    table="".join(f"<tr><td>{b['start_s']:.1f}–{b['end_s']:.1f}초</td><td>{f(b['mean_changed_area_pct'])}</td><td>{b['valid_s']:.1f}</td><td>{b['unknown_s']:.1f}</td></tr>" for b in report["bins"])
    page=f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>케이지 전체 활동성</title><style>body{{font:15px 'Malgun Gothic',sans-serif;line-height:1.7;background:#f4f7f1;color:#183b2d;max-width:1100px;margin:30px auto;padding:20px}}section{{background:white;padding:24px;border:1px solid #dce5db;border-radius:12px;margin:18px 0}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}.examples{{display:grid;grid-template-columns:1fr 1fr}}figure{{margin:10px}}strong{{color:#16765b}}</style><h1>케이지 전체 활동성</h1><p>{esc(m['group'])} · {esc(m['cage'])} · {m['animals']}마리 · {WATER[m['water_type']]} · {ADMIN[m['administration']]}</p><section><h2>영상 움직임 지수 <strong>{f(s['mean_changed_area_pct'])}%</strong></h2><p>유효 {s['valid_s']:.1f}초 / {s['observed_s']:.1f}초 · 판정 불가 {s['unknown_s']:.1f}초 · 분석 상태 {esc(report['state'])}</p><p>설정 {esc(m['setup_id'])} · 비교 블록 {esc(m['comparison_block'])}</p></section><section><h2>시간대별 변화</h2><img alt='활동성 지수와 유효 관측 비율' src='data:image/png;base64,{image}'><table><tr><th>영상 구간</th><th>평균 변화 픽셀 (%)</th><th>유효 초</th><th>판정 불가 초</th></tr>{table}</table></section><section><h2>원본의 움직임 픽셀 확인</h2><p>자홍색은 프레임 사이 변한 픽셀, 초록색은 케이지 영역, 주황색은 고정 배경 영역입니다.</p><div class='examples'>{examples}</div></section><section><h2>해석 범위</h2><ul>{''.join('<li>'+esc(n)+'</li>' for n in report['notes'])}</ul><p>방법 참고: <a href='https://link.springer.com/article/10.1186/s13036-017-0079-y'>영상 픽셀 변화와 활동성 연구</a> · <a href='https://eda.nc3rs.org.uk/experimental-design-unit'>공유 급수의 실험 단위</a></p></section></html>"""
    (run/"report.html").write_text(page,encoding="utf-8")


def write_comparison(output,reports,mode="water"):
    result=comparison(reports,mode);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result.update(created_utc=utcnow(),state="complete",id=output.name)
    result["inputs"]=[{"session":r["source_session"],"run":r.get("id"),"pairs_sha256":r.get("pairs_sha256"),
                       "source_sha256":r["source_sha256"],"metadata":r["metadata"]} for r in reports]
    result["code_sha256"]=digest(Path(__file__))
    with (output/"cages.csv").open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f);writer.writerow(["setup_id","comparison_block","administration","group","water_type","cage","value_changed_area_pct","recordings","valid_s","coverage_pct"])
        for st in result["strata"]:
            for g in st["groups"]:
                for p in g["points"]:writer.writerow([csv_safe(str(x)) for x in (st["setup_id"],st["comparison_block"],st["administration"],p["group"],p["water_type"],p["cage"],p["value"],p["recordings"],p["valid_s"],p["coverage_pct"])])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family":"Malgun Gothic","axes.unicode_minus":False}):
        shown=result["strata"][:8]
        result["chart_strata_count"]=len(shown)
        fig,axes=plt.subplots(max(1,len(shown)),1,figsize=(10,4*max(1,len(shown))),squeeze=False,layout="constrained")
        if not result["strata"]:
            axes[0,0].text(.5,.5,"비교 가능한 물·투여 조건과 독립 케이지 자료가 없습니다.",ha="center",va="center");axes[0,0].axis("off")
        for ax,st in zip(axes.flat,shown):
            groups=st["groups"];positions=np.arange(len(groups))
            ax.bar(positions,[g["mean"] for g in groups],color="#a8cfc0",width=.6)
            for i,g in enumerate(groups):
                if g["sd"] is not None:ax.errorbar(i,g["mean"],yerr=g["sd"],color="#1a5946",capsize=5)
                offsets=np.linspace(-.12,.12,g["n_cages"]) if g["n_cages"]>1 else [0]
                ax.scatter(i+np.asarray(offsets),[p["value"] for p in g["points"]],color="#155d48",s=45,zorder=4)
            ax.set_xticks(positions,[g["name"]+f"\n독립 케이지 n={g['n_cages']}" for g in groups])
            ax.set(ylabel="평균 변화 픽셀 비율 (%)",title=f"{st['comparison_block']} · {ADMIN[st['administration']]}\n점 = 케이지 한 개 · 오차막대 = 케이지 간 표준편차")
            ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15)
        fig.savefig(output/"chart.png",dpi=140);plt.close(fig)
    esc=html.escape;image=base64.b64encode((output/"chart.png").read_bytes()).decode();sections=[]
    fmt=lambda v:"계산 불가" if v is None else f"{v:.3f}"
    for st in result["strata"]:
        table="".join(f"<tr><td>{esc(g['name'])}</td><td>{g['n_cages']}</td><td>{fmt(g['mean'])}</td><td>{fmt(g['sd'])}</td><td>{fmt(g['difference_from_control_pp'])}</td><td>{fmt(g['relative_to_control_pct'])}</td></tr>" for g in st["groups"])
        points="".join(f"<li>{esc(g['name'])} · {esc(p['cage'])}: {p['value']:.3f}% · 반복 촬영 {p['recordings']}개 · 유효 {p['coverage_pct']:.1f}%</li>" for g in st["groups"] for p in g["points"])
        sections.append(f"<section><h2>{esc(st['comparison_block'])} · {ADMIN[st['administration']]}</h2><p>촬영 설정: {esc(st['setup_id'])}</p><table><tr><th>조건</th><th>케이지 n</th><th>평균 (%)</th><th>SD</th><th>대조수 대비 %p</th><th>대조수 대비 변화 %</th></tr>{table}</table><ul>{points}</ul><p>{esc(st['interpretation'])}</p></section>")
    excluded="".join(f"<li>{esc(x['session'])}: {esc(x['reason'])}</li>" for x in result["excluded"])
    page=f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>그룹·물 종류별 케이지 활동성 비교</title><style>body{{font:15px 'Malgun Gothic',sans-serif;line-height:1.7;background:#f4f7f1;color:#193b2e;max-width:1100px;margin:30px auto;padding:20px}}section{{background:white;border:1px solid #ddd;padding:24px;border-radius:12px;margin:18px 0}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>그룹·물 종류별 케이지 활동성 비교</h1><p>{esc(result['criteria'])}. 같은 케이지 반복 촬영은 한 케이지 값으로 합칩니다.</p><section><img alt='독립 케이지별 움직임 값과 그룹 평균' src='data:image/png;base64,{image}'></section>{''.join(sections)}<section><h2>비교 제외 기록</h2><ul>{excluded}</ul><p>이동거리·속도·개체별 활동시간 지표가 아닙니다. 비교는 기술적 요약이며 물의 인과 효과나 유의성을 확정하지 않습니다.</p><a href='https://eda.nc3rs.org.uk/experimental-design-unit'>공유 급수 처치의 실험 단위: NC3Rs</a></section></html>"""
    (output/"report.html").write_text(page,encoding="utf-8")
    result["cages_csv_sha256"]=digest(output/"cages.csv")
    atomic_json(output/"comparison.json",result)
    return result
