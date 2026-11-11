"""Research events and independent, source-bound reference labels.

Approach counts, individual feeding intervals and detector boxes are deliberately
different records. Nothing here turns unreviewed time into a negative label.
"""
from __future__ import annotations
import csv
import math
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from .integrity import atomic_json, read_json, digest, json_digest, utcnow, append_event, read_events
from .recording_io import load_times, frame_reference

EVENT_TYPES = ("oral_dose", "bottle_change", "food_refill", "cleaning", "cage_move", "other")


def named(value, what):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError(f"{what}: 1~160자 값을 입력하세요.")
    return value.strip()


def date_time(value):
    try:
        result = datetime.fromisoformat(value)
        if result.utcoffset() is None:
            raise ValueError
        return result
    except (ValueError, TypeError):
        raise ValueError("시각은 시간대를 포함해야 합니다. 예: 2026-11-03T09:00:00+09:00") from None


def latest(events, key="id"):
    return list({e[key]: e for e in events}.values())


def save_event(folder, values):
    folder = Path(folder)
    spec = read_json(folder/"protocol.json")
    kind = values.get("type")
    if kind not in EVENT_TYPES:
        raise ValueError("연구 사건 종류를 선택하세요.")
    actor = named(values.get("actor"), "기록자")
    occurred = date_time(values.get("occurred_at")).isoformat()
    subject = str(values.get("animal_id", "")).strip()
    if kind == "oral_dose" and not subject:
        raise ValueError("경구투여는 해당 마우스의 ID를 따로 입력하세요. 다섯 마리의 시각을 일괄 추정하지 않습니다.")
    events = read_events(folder/"events.jsonl")
    event_id = values.get("id") or uuid.uuid4().hex
    previous = next((e for e in reversed(events) if e["id"] == event_id), None)
    if values.get("id") and previous is None:
        raise ValueError("수정할 사건 ID가 없습니다.")
    entry = {"id": event_id, "revision": 1+(previous["revision"] if previous else 0),
             "type": kind, "occurred_at": occurred, "recorded_at": utcnow(), "actor":actor,
             "animal_id":subject, "cage_id":spec["cage"], "note":str(values.get("note", ""))[:2000],
             "supersedes_sha256": json_digest(previous) if previous else None}
    append_event(folder/"events.jsonl", entry)
    return entry


def make_plan(folder, purpose="final", count=60, seed=0, frames=None, rationale=""):
    folder = Path(folder)
    info, times = load_times(folder)
    if purpose not in ("final", "development", "assisted") or isinstance(count, bool) or not 1 <= count <= 5000:
        raise ValueError("표본 용도와 개수(1~5000)를 확인하세요.")
    if purpose == "final" and frames is not None:
        raise ValueError("최종 평가 표본은 AI 결과나 수동 지정 프레임으로 선택할 수 없습니다.")
    plans = folder/"references"
    plans.mkdir(exist_ok=True)
    if purpose == "final":
        existing = list(plans.glob("*/plan.json"))
        old = [p for p in existing if read_json(p)["purpose"] == "final"]
        if old:
            return read_plan(old[0].parent)  # once fixed, even a changed seed cannot resample
        if any((folder/"analyses").glob("*/analysis.json")):
            raise ValueError("이미 AI 분석한 세션은 새 최종 평가 표본으로 지정하지 않습니다. 개발용으로 검토하거나 새 독립 촬영에서 먼저 표본을 고정하세요.")
    rng = random.Random(seed)
    if frames is None:
        # Stratify by elapsed time, not frame density or model error. Empty bins stay gaps.
        edges = np.linspace(times[0], times[-1]+1e-9, min(count, len(times))+1)
        chosen, missing = [], []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            start, end = np.searchsorted(times, [a, b], side="left")
            if end > start:
                chosen.append(rng.randrange(int(start), int(end)))
            else:
                missing.append(i)
        rule = "elapsed_time_stratified_random_one_per_nonempty_bin"
    else:
        chosen, missing = sorted(set(frames)), []
        if not rationale.strip():
            raise ValueError("어려운 장면 선택 이유를 기록하세요.")
        rule = "development_purposive_do_not_generalize_score"
    plan = {"id": uuid.uuid4().hex, "schema": 1, "purpose":purpose, "created_utc":utcnow(),
            "seed":seed, "rule":rule, "requested":count, "empty_time_bins":missing,
            "source_sha256":info["sha256"], "rationale":rationale, "samples":[]}
    for i, fr in enumerate(chosen):
        plan["samples"].append(dict(frame_reference(folder, fr, info), sample=i, time_s=float(times[fr])))
    plan["sha256"] = json_digest(plan)
    destination = plans/plan["id"]
    destination.mkdir(exist_ok=False)
    atomic_json(destination/"plan.json", plan)
    return plan


def read_plan(path):
    plan = read_json(Path(path)/"plan.json")
    if plan["sha256"] != json_digest({k:v for k,v in plan.items() if k != "sha256"}):
        raise ValueError("고정 표본 목록이 변경됐습니다.")
    info, _ = load_times(Path(path).parent.parent)
    if plan["source_sha256"] != info["sha256"]:
        raise ValueError("표본의 원본 영상·시각이 변경됐습니다.")
    return plan


def save_approach(path, values):
    path = Path(path)
    plan = read_plan(path)
    sample = values.get("sample")
    if isinstance(sample, bool) or not isinstance(sample, int) or not 0 <= sample < len(plan["samples"]):
        raise ValueError("고정 목록의 표본 번호를 선택하세요.")
    label, count = values.get("label"), values.get("count")
    reason = str(values.get("reason", "")).strip()
    if label not in ("approach", "away", "unknown"):
        raise ValueError("접근·비접근·판정 불가를 선택하세요.")
    animals = read_json(path.parent.parent/"protocol.json")["criteria"]["animals"]
    if label == "unknown":
        if not reason:
            raise ValueError("판정 불가 사유를 입력하세요.")
        count = None
    elif isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= animals or (count > 0) != (label == "approach"):
        raise ValueError(f"접근 판정에 맞는 마릿수 0~{animals}를 입력하세요.")
    reviewer = named(values.get("reviewer"), "정답 작성자")
    events = read_events(path/"labels.jsonl")
    entry = dict(plan["samples"][sample], id=f"{reviewer}:{sample}", reviewer=reviewer,
                 label=label, count=count, reason=reason, occlusion=str(values.get("occlusion", "unspecified")),
                 revision=1+sum(e["id"] == f"{reviewer}:{sample}" for e in events), recorded_at=utcnow(),
                 task="cage_approach", purpose=plan["purpose"], plan_sha256=plan["sha256"],
                 ai_assisted=plan["purpose"] == "assisted")
    append_event(path/"labels.jsonl", entry)
    return entry


def behavior_exposure_path(session_dir):
    session_dir=Path(session_dir)
    sess=read_json(session_dir/"session.json")
    recording=sess.get("source",{}).get("recording")
    return (Path(recording) if recording else session_dir)/"behavior_ai_exposure.json"


def mark_behavior_exposure(session_dir, method):
    path=behavior_exposure_path(session_dir)
    if not path.exists():
        atomic_json(path,{"first_prediction_utc":utcnow(),"method":method})


def behavior_was_exposed(session_dir):
    return behavior_exposure_path(session_dir).exists() or any(Path(session_dir).glob("behavior_feeding_*.csv"))


def save_behavior(session_dir, values):
    from .session import load_session, tracks_path
    from .tracks import Tracks
    session_dir = Path(session_dir)
    sess = load_session(session_dir)
    tid = values.get("track")
    if tid not in sess["tracks"]:
        raise ValueError("포즈 세션에 실제 존재하는 동물 트랙을 선택하세요.")
    reviewer = named(values.get("reviewer"), "정답 작성자")
    behavior = named(values.get("behavior", "feeding"), "행동 이름")
    if behavior != "feeding":
        raise ValueError("이 화면의 개체 행동 정답은 feeding입니다.")
    label = values.get("label")
    if label not in ("positive", "negative", "unknown"):
        raise ValueError("실제 섭식·비섭식·판정 불가를 선택하세요.")
    if label != "unknown" and values.get("observable") is not True:
        raise ValueError("동물 정체성과 입을 관찰할 수 있는 경우에만 섭식·비섭식 정답을 만드세요.")
    if label == "unknown" and not str(values.get("reason", "")).strip():
        raise ValueError("가림·정체성 불명 등 판정 불가 사유를 입력하세요.")
    tr = Tracks.load(tracks_path(session_dir, tid))
    a, b = values.get("start_frame"), values.get("end_frame")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (a,b)) or a > b:
        raise ValueError("시작·종료 프레임을 정수로 지정하세요 (양 끝 포함).")
    mask = (tr.frame >= a) & (tr.frame <= b)
    if mask.sum() != b-a+1:
        raise ValueError("지정 구간에 원본과 대응하지 않는 포즈 프레임이 있습니다.")
    purpose = values.get("purpose", "development")
    if purpose not in ("development", "final", "assisted"):
        raise ValueError("정답 용도를 선택하세요.")
    if purpose=="final" and behavior_was_exposed(session_dir):
        raise ValueError("이미 행동 AI 예측·평가에 사용한 촬영에는 최종 정답을 새로 작성하거나 수정할 수 없습니다. 개발용 자료 또는 새 독립 촬영을 사용하세요.")
    events = read_events(session_dir/"behavior_labels.jsonl")
    label_id = values.get("id") or uuid.uuid4().hex
    previous = next((e for e in reversed(events) if e["id"] == label_id), None)
    if values.get("id") and (previous is None or previous["reviewer"] != reviewer):
        raise ValueError("자신의 기존 정답 ID만 수정할 수 있습니다.")
    for e in latest(events):
        if e["id"] != label_id and e["reviewer"] == reviewer and e["track"] == tid and max(a,e["start_frame"]) <= min(b,e["end_frame"]):
            raise ValueError("같은 작성자의 구간이 겹칩니다. 기존 정답 ID를 지정해 수정하세요.")
    entry = {"id":label_id, "revision":1+(previous["revision"] if previous else 0), "recorded_at":utcnow(),
             "reviewer":reviewer, "behavior":behavior, "task":"individual_feeding", "track":tid,
             "start_frame":a, "end_frame":b, "start_s":float(tr.time[mask][0]), "end_s":float(tr.time[mask][-1]),
             "observable":values.get("observable") is True, "label":label, "reason":str(values.get("reason", "")),
             "occlusion":str(values.get("occlusion","unspecified")),
             "purpose":purpose, "ai_assisted":purpose == "assisted", "session_sha256":digest(session_dir/"session.json"),
             "track_sha256":digest(tracks_path(session_dir, tid)), "source":sess["source"]}
    append_event(session_dir/"behavior_labels.jsonl", entry)
    return entry


def export_behavior(session_dir, reviewer, destination):
    from .session import load_session, tracks_path
    session_dir, destination = Path(session_dir), Path(destination)
    sess = load_session(session_dir)
    events = [e for e in latest(read_events(session_dir/"behavior_labels.jsonl")) if e["reviewer"] == reviewer]
    if not events:
        raise ValueError("이 작성자의 실제 섭식 정답이 없습니다.")
    for e in events:
        if e["session_sha256"] != digest(session_dir/"session.json") or e["track_sha256"] != digest(tracks_path(session_dir,e["track"])):
            raise ValueError("정답 작성 이후 원본 포즈/세션이 변경됐습니다.")
    with destination.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "start_frame", "end_frame", "start_s", "end_s", "track"])
        for e in events:
            if e["label"] == "unknown":
                continue
            for kind in (["feeding", "reviewed_v2"] if e["label"] == "positive" else ["reviewed_v2"]):
                w.writerow([kind, e["start_frame"], e["end_frame"], e["start_s"], e["end_s"], e["track"]])
    atomic_json(destination.with_suffix(destination.suffix+".json"), {
        "schema":1, "task":"individual_feeding", "reviewer":reviewer, "labels_sha256":digest(destination),
        "session_sha256":digest(session_dir/"session.json"), "source":sess["source"],
        "track_sha256":{t:digest(tracks_path(session_dir,t)) for t in sess["tracks"]},
        "purposes":sorted({e["purpose"] for e in events}), "events":events})
    return destination


def save_boxes(folder, values):
    folder = Path(folder)
    info, _ = load_times(folder)
    ref = frame_reference(folder, values.get("frame"), info)
    reviewer = named(values.get("reviewer"), "상자 작성자")
    spec = read_json(folder/"protocol.json")
    boxes = values.get("boxes")
    if not isinstance(boxes, list) or not boxes or values.get("all_animals_visible") is not True:
        raise ValueError("검출기 정답은 모든 동물을 확인하고 직접 그린 상자 좌표가 필요합니다.")
    if len(boxes) != spec["criteria"]["animals"]:
        raise ValueError("모든 동물의 상자를 작성하세요. 마릿수 라벨로 상자를 생성하지 않습니다.")
    w, h = spec["criteria"]["frame_size"]
    for b in boxes:
        if len(b) != 4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in b) or not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
            raise ValueError("상자는 영상 픽셀 좌표 [x1,y1,x2,y2] 형식이어야 합니다.")
    event = dict(ref, id=f"{reviewer}:{ref['session_frame']}", recorded_at=utcnow(), reviewer=reviewer,
                 task="detector_boxes", boxes=boxes, source_sha256=info["sha256"], purpose="development")
    append_event(folder/"box_labels.jsonl", event)
    return event
