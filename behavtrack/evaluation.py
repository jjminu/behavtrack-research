"""Denominator-explicit metrics; abstentions never become negatives."""
from __future__ import annotations
import itertools
import numpy as np


def ratio(a,b):
    return float(a/b) if b else None


def binary_metrics(human, ai, durations=None):
    h, p = np.asarray(human, int), np.asarray(ai, int)
    if h.shape != p.shape or h.ndim != 1 or not np.isin(h,[-1,0,1]).all() or not np.isin(p,[-1,0,1]).all():
        raise ValueError("평가 정답과 예측은 -1(불명), 0, 1의 같은 길이 배열이어야 합니다.")
    human_known, ai_known = h >= 0, p >= 0
    pair = human_known & ai_known
    tp, tn = int(np.sum(pair & (h==1) & (p==1))), int(np.sum(pair & (h==0) & (p==0)))
    fp, fn = int(np.sum(pair & (h==0) & (p==1))), int(np.sum(pair & (h==1) & (p==0)))
    result = {"total":len(h), "human_known":int(human_known.sum()), "paired_known":int(pair.sum()),
              "human_unknown":int((~human_known).sum()), "ai_known":int(ai_known.sum()), "ai_unknown":int((~ai_known).sum()),
              "ai_answerable_rate":ratio(ai_known.sum(),len(h)), "ai_unknown_rate":ratio((~ai_known).sum(),len(h)),
              "human_known_ai_unknown":int(np.sum(human_known & ~ai_known)),
              "human_known_ai_unknown_rate":ratio(np.sum(human_known & ~ai_known),human_known.sum()),
              "tp":tp,"tn":tn,"fp":fp,"fn":fn,"confusion_matrix":[[tn,fp],[fn,tp]],
              "precision":ratio(tp,tp+fp),"recall":ratio(tp,tp+fn),"f1":ratio(2*tp,2*tp+fp+fn),
              "precision_denominator":tp+fp,"recall_denominator":tp+fn,"f1_denominator":2*tp+fp+fn,
              "human_positive":int(np.sum(h==1)),"human_negative":int(np.sum(h==0)),
              "accuracy_paired":ratio(tp+tn,pair.sum()),"validated":False}
    if durations is not None:
        d = np.asarray(durations,float)
        if d.shape != h.shape or not np.isfinite(d).all() or (d<0).any():
            raise ValueError("평가 시간 간격이 올바르지 않습니다.")
        result.update(paired_seconds=float(d[pair].sum()),
                      human_positive_seconds_paired=float(d[pair & (h==1)].sum()),
                      ai_positive_seconds_paired=float(d[pair & (p==1)].sum()),
                      duration_difference_seconds_paired=float(d[pair & (p==1)].sum()-d[pair & (h==1)].sum()),
                      human_positive_seconds_all_reviewed=float(d[h==1].sum()),
                      human_known_ai_unknown_seconds=float(d[human_known & ~ai_known].sum()))
    return result


def count_metrics(human, ai, animals=5):
    h, p = np.asarray(human,int), np.asarray(ai,int)
    if h.shape != p.shape or not np.isin(h,range(-1,animals+1)).all() or not np.isin(p,range(-1,animals+1)).all():
        raise ValueError("마릿수 정답과 예측의 범위를 확인하세요.")
    known = (h>=0) & (p>=0)
    matrix = np.zeros((animals+1,animals+1),int)
    for a,b in zip(h[known],p[known]):
        matrix[a,b] += 1
    return {"labels":list(range(animals+1)),"confusion_matrix":matrix.tolist(),"total":len(h),
            "paired_known":int(known.sum()),"mae":float(np.abs(h[known]-p[known]).mean()) if known.any() else None,
            "exact_match_rate":ratio(np.sum(h[known]==p[known]),known.sum())}


def agreement(events):
    latest = {(e["reviewer"],e["sample"]):e for e in events}
    reports = []
    for a,b in itertools.combinations(sorted({k[0] for k in latest}),2):
        shared = sorted({k[1] for k in latest if k[0]==a} & {k[1] for k in latest if k[0]==b})
        h = [latest[a,i] for i in shared]; p = [latest[b,i] for i in shared]
        pairs = [(x["label"],y["label"]) for x,y in zip(h,p) if x["label"]!="unknown" and y["label"]!="unknown"]
        observed = ratio(sum(x==y for x,y in pairs),len(pairs))
        pe = sum(ratio(sum(x==label for x,_ in pairs),len(pairs))*ratio(sum(y==label for _,y in pairs),len(pairs))
                 for label in ("approach","away")) if pairs else None
        reports.append({"reviewers":[a,b],"common_samples":len(shared),"paired_known":len(pairs),
                        "agreement":observed,"cohen_kappa":(observed-pe)/(1-pe) if pe is not None and pe<1 else None,
                        "counts":count_metrics([x["count"] if x["count"] is not None else -1 for x in h],
                                                [y["count"] if y["count"] is not None else -1 for y in p],
                                                max([5]+[x["count"] or 0 for x in h+p]))})
    return reports


def observed_durations(t):
    t = np.asarray(t,float)
    if len(t)<2:
        return np.zeros(len(t)),0.
    delta = np.diff(t)
    if not np.isfinite(t).all() or (delta<=0).any():
        raise ValueError("프레임 시각이 증가하지 않습니다.")
    gap = delta > max(.5,2*np.median(delta))
    return np.r_[np.where(gap,0,delta),0], float(delta[gap].sum())


def behavior_agreement(session_dir):
    from .research_data import latest
    from .integrity import read_events
    from .session import tracks_path
    from .tracks import Tracks
    from .classifier import frame_mask
    from pathlib import Path
    events=latest(read_events(Path(session_dir)/"behavior_labels.jsonl"))
    reports=[]
    for tid in sorted({e["track"] for e in events}):
        tr=Tracks.load(tracks_path(session_dir,tid));dur,_=observed_durations(tr.time)
        reviewers=sorted({e["reviewer"] for e in events if e["track"]==tid})
        annotated={};truth={}
        for reviewer in reviewers:
            own=[e for e in events if e["track"]==tid and e["reviewer"]==reviewer]
            annotated[reviewer]=frame_mask(tr.frame,[(e["start_frame"],e["end_frame"]) for e in own])
            y=np.full(len(tr),-1,int)
            for e in own:
                y[frame_mask(tr.frame,[(e["start_frame"],e["end_frame"])])]={"positive":1,"negative":0,"unknown":-1}[e["label"]]
            truth[reviewer]=y
        for a,b in itertools.combinations(reviewers,2):
            shared=annotated[a]&annotated[b]
            m=binary_metrics(truth[a][shared],truth[b][shared],dur[shared])
            n=m["paired_known"]
            pe=((m["tn"]+m["fp"])*(m["tn"]+m["fn"])+(m["tp"]+m["fn"])*(m["tp"]+m["fp"]))/n**2 if n else None
            reports.append({"track":tid,"reviewers":[a,b],"common_annotated_frames":int(shared.sum()),
                            "paired_known_frames":n,"agreement":m["accuracy_paired"],
                            "cohen_kappa":(m["accuracy_paired"]-pe)/(1-pe) if pe is not None and pe<1 else None,
                            "duration_difference_seconds_paired":m["duration_difference_seconds_paired"]})
    return reports
