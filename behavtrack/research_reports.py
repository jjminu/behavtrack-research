"""Source-bound reference evaluation and descriptive research exports."""
from __future__ import annotations
import csv
import io
import math
import uuid
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
from .integrity import read_json,read_events,digest,atomic_json,csv_safe,utcnow
from .research_data import read_plan,latest,date_time
from .evaluation import binary_metrics,count_metrics,agreement
from .recording_io import load_times


def reference_evaluation(reference, run):
    from .research import analysis_rows
    reference=Path(reference)
    plan=read_plan(reference)
    manifest,rows=analysis_rows(Path(run))
    if manifest.get("reference_plan_sha256")!=plan["sha256"]:
        raise ValueError("동일한 고정 표본으로 분석한 결과가 아닙니다.")
    events=latest(read_events(reference/"labels.jsonl"))
    counts=manifest["protocol"]["criteria"]["animals"]
    spec=manifest["protocol"]
    info,_=load_times(reference.parent.parent)
    start=datetime.fromisoformat(info["started_utc"])
    result={"plan_sha256":plan["sha256"],"purpose":plan["purpose"],"sampling":plan["rule"],
            "planned":len(plan["samples"]),"empty_time_bins":len(plan["empty_time_bins"]),"reviewers":[],
            "inter_rater":agreement(events),"validated":False,"independent_cage_generalization":"unevaluated",
            "scope":"single_cage_reference_evaluation","cage_id":spec["cage"],"date":start.date().isoformat(),
            "detector_sha256":manifest["detector_sha256"],"labels_sha256":digest(reference/"labels.jsonl")}
    for reviewer in sorted({e["reviewer"] for e in events}):
        labels=[e for e in events if e["reviewer"]==reviewer]
        def score(selected):
            as_int=lambda value:{"approach":1,"away":0,"unknown":-1}[value]
            out=binary_metrics([as_int(e["label"]) for e in selected],[as_int(rows[e["sample"]]["status"]) for e in selected])
            out["counts"]=count_metrics([e["count"] if e["count"] is not None else -1 for e in selected],
                                       [rows[e["sample"]]["count"] if rows[e["sample"]]["count"] is not None else -1 for e in selected],counts)
            return out
        strata=defaultdict(list)
        for e in labels:
            hour=(start+timedelta(seconds=e["time_s"])).hour
            strata[f"utc_hour_{hour:02d}"].append(e)
            strata["occlusion_"+e.get("occlusion","unspecified")].append(e)
        result["reviewers"].append(dict(score(labels),reviewer=reviewer,unreviewed=len(plan["samples"])-len(labels),
                                         by_condition={k:score(v) for k,v in strata.items()}))
    atomic_json(Path(run)/"reference_evaluation.json",result)
    return result


def import_measurements(folder, path):
    folder,path=Path(folder),Path(path)
    spec=read_json(folder/"protocol.json")
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError("측정 CSV가 비어 있습니다.")
    for n,row in enumerate(rows,2):
        if row.get("cage_id")!=spec["cage"]:
            raise ValueError(f"{n}행: 케이지 ID가 촬영과 다릅니다.")
        a,b=date_time(row.get("start_at")),date_time(row.get("end_at"))
        if a>=b:
            raise ValueError(f"{n}행: 측정 시작·종료 시각을 확인하세요.")
        if row.get("administration")!=spec.get("administration"):
            raise ValueError(f"{n}행: 급수병/경구투여 조건이 촬영과 다릅니다.")
        for key in ("food_offered_g","food_remaining_g","spillage_recovered_g","water_ml"):
            value=float(row[key])
            if not math.isfinite(value) or value<0:
                raise ValueError(f"{n}행 {key}: 0 이상 측정값을 입력하세요.")
            row[key]=value
    output={"id":uuid.uuid4().hex,"imported_utc":utcnow(),"source_file":str(path.resolve()),"sha256":digest(path),"rows":rows}
    atomic_json(folder/("measurements_"+output["id"]+".json"),output)
    return output


def result_tables(folder,run,bin_seconds=3600):
    from .research import analysis_rows
    from .group_approach import summary
    folder=Path(folder)
    manifest,rows=analysis_rows(Path(run))
    info,times=load_times(folder)
    spec=read_json(folder/"protocol.json")
    if not math.isfinite(bin_seconds) or bin_seconds<=0:
        raise ValueError("시간대 구간은 양수 초여야 합니다.")
    bins=defaultdict(list)
    for row in rows:
        bins[int(float(row["target_s"])//bin_seconds)].append(row)
    reports=[]
    for b,samples in sorted(bins.items()):
        s=summary(samples)
        reports.append(dict(s,start_s=b*bin_seconds,end_s=(b+1)*bin_seconds,
                            unknown_rate=s["unknown_samples"]/s["samples"] if s["samples"] else None))
    started=datetime.fromisoformat(info["started_utc"])
    events=[]
    for e in latest(read_events(folder/"events.jsonl")):
        elapsed=(date_time(e["occurred_at"])-started).total_seconds()
        before=[r for r in rows if elapsed-bin_seconds<=float(r["target_s"])<elapsed]
        after=[r for r in rows if elapsed<=float(r["target_s"])<elapsed+bin_seconds]
        events.append(dict(event=e,event_s=elapsed,before=summary(before),after=summary(after),
                           endpoint="cage_approach_samples_not_individual_intake",window_s=bin_seconds))
    measurements=[]
    observed_start,observed_end=started+timedelta(seconds=float(times[0])),started+timedelta(seconds=float(times[-1]))
    for file in folder.glob("measurements_*.json"):
        for row in read_json(file)["rows"]:
            a,b=date_time(row["start_at"]),date_time(row["end_at"])
            match=abs((a-observed_start).total_seconds())<=1e-6 and abs((b-observed_end).total_seconds())<=1e-6
            measurements.append(dict(row,exact_observation_interval_match=match,
                                     apparent_food_disappearance_g=row["food_offered_g"]-row["food_remaining_g"]-row["spillage_recovered_g"],
                                     combined_with_video=match))
    return {"cage_id":spec["cage"],"treatment":spec.get("treatment",spec["group"]),"administration":spec.get("administration","unspecified"),
            "bins":reports,"events":events,"measurements":measurements,"observed_start":observed_start.isoformat(),
            "observed_end":observed_end.isoformat(),"endpoint":"sparse_approach_samples", "validated":False,
            "interpretation":"방문 횟수·연속 섭식시간·실제 섭취량·식욕 억제·GLP-1 인과관계를 추정하지 않습니다."}


def tables_csv(report):
    out=io.StringIO(newline="")
    writer=csv.writer(out)
    keys=["start_s","end_s","samples","known_samples","unknown_samples","approach_pct","mean_count","unknown_rate"]
    writer.writerow(["cage_id","treatment","administration",*keys])
    for row in report["bins"]:
        writer.writerow([csv_safe(report[k]) for k in ("cage_id","treatment","administration")]+[row[k] for k in keys])
    return out.getvalue().encode("utf-8-sig")
