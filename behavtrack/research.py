"""Immutable recordings, time-grid analysis and append-only human review.

This records an operational proximity endpoint, not food consumption or animal identity.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import queue
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .group_approach import evaluate, summary
from .sources import Recorder
from .activity import heavy

VERSION = "research-2.0"
SAMPLE_FIELDS = ["sample", "target_s", "frame", "captured_s", "time_error_s", "status", "count", "detected", "reason", "detections_json"]


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol(values, size):
    animals = values.get("animals")
    if isinstance(animals, bool) or not isinstance(animals, int) or not 1 <= animals <= 20:
        raise ValueError("전체 마릿수는 1~20 사이의 정수로 입력해 주세요.")
    group, cage = str(values.get("group", "")).strip(), str(values.get("cage", "")).strip()
    if not group or not cage or max(len(group), len(cage)) > 80:
        raise ValueError("그룹 이름과 케이지 ID를 각각 1~80자로 입력해 주세요.")
    interval = float(values.get("interval", 1))
    rect = [float(v) for v in values.get("food_rect", [])]
    if not math.isfinite(interval) or not .1 <= interval <= 60:
        raise ValueError("분석 간격은 0.1~60초 사이여야 합니다.")
    if (len(rect) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in rect)
            or rect[2]-rect[0] < .01 or rect[3]-rect[1] < .01):
        raise ValueError("영상에서 사료 주변을 드래그해 표시해 주세요.")
    criteria = {"endpoint": "body_center_inside_marked_food_surroundings", "food_rect": rect,
                "frame_size": list(size), "animals": animals, "interval_s": interval,
                "sampling": "nearest_capture_to_fixed_grid_ties_earlier_no_frame_reuse",
                "max_time_error_s": min(.25, interval / 2), "min_bbox_score": .6,
                "duplicate_iou_threshold": .5, "species": "mouse", "view": "top",
                "model": "superanimal_topviewmouse_detector", "observation_area": "entire_frame"}
    treatment = str(values.get("treatment", group)).strip()
    administration = values.get("administration", "unspecified")
    if administration not in ("bottle_only", "oral_plus_bottle", "unspecified"):
        raise ValueError("투여 방식은 급수병 단독 또는 경구투여＋급수병으로 지정하세요.")
    water_type = values.get("water_type", "unspecified")
    if water_type not in ("control", "hydrogen", "oxygen", "unspecified"):
        raise ValueError("급수병 물 종류를 확인하세요.")
    return {"schema": 2, "created_utc": utcnow(), "group": group, "cage": cage,
            "treatment": treatment, "administration": administration, "water_type": water_type,
            "utc_offset_minutes":540,
            "criteria": criteria, "criteria_sha256": json_digest(criteria), "software": VERSION}


# One implementation is shared by capture, analysis, pose import and annotation.
from .integrity import atomic_json, digest, json_digest, read_json, utcnow
from .recording_io import ResearchRecording, load_times


def sample_plan(times, interval, tolerance):
    """Select frames before inference so machine throughput cannot change sampling."""
    if not len(times) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("촬영 시각과 분석 간격을 확인하세요.")
    rows, used = [], set()
    for n in range(math.floor(float(times[-1]) / interval + 1e-9) + 1):
        target = n * interval
        right = int(np.searchsorted(times, target))
        choices = [i for i in (right-1, right) if 0 <= i < len(times)]
        index = min(choices, key=lambda i: (abs(float(times[i])-target), i))
        delta = float(times[index]) - target
        missing = abs(delta) > tolerance + 1e-9 or index in used
        if not missing:
            used.add(index)
        rows.append({"sample": n, "target_s": target, "frame": None if missing else index,
                     "captured_s": None if missing else float(times[index]), "time_error_s": None if missing else delta,
                     "status": "unknown", "count": None, "detected": None,
                     "reason": "capture_gap" if missing else "not_analyzed", "detections_json": ""})
    return rows


class Cancelled(Exception):
    pass


class WorkerPredictor:
    def __init__(self, executable, animals, cancel, detector_path=None):
        from .ai_live import AIWorker
        self.cancel, self.runtime = cancel, {}
        args=["--detect-only", "--animals", str(animals)]
        if detector_path:args += ["--detector-path",str(detector_path)]
        self.worker = AIWorker(executable, args)
        try:
            kind, value = self.wait(180)
            if kind != "ready":
                raise RuntimeError(str(value))
            self.runtime = value or {}
        except BaseException:
            self.close()
            raise

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel.is_set():
                raise Cancelled()
            try:
                return self.worker.events.get(timeout=.2)
            except queue.Empty:
                pass
        raise RuntimeError("AI 응답 시간이 초과됐습니다.")

    def predict(self, frame, fid):
        ok, encoded = cv2.imencode(".png", frame)  # lossless transport to the worker
        if not ok:
            raise ValueError("분석 표본을 인코딩하지 못했습니다.")
        self.worker.requests.put((fid, 0.0, encoded.tobytes()))
        kind, value = self.wait(180)
        if kind != "result" or value[0] != fid:
            raise RuntimeError(str(value))
        return value[2]

    def close(self):
        self.worker.close()


@heavy
def analyze(folder, executable, model_path, cancel=None, progress=None, predictor_factory=WorkerPredictor, reference_plan=None):
    folder = Path(folder)
    cancel = cancel or threading.Event()
    progress = progress or (lambda *args: None)
    info, times = load_times(folder)
    spec = read_json(folder / "protocol.json")
    c = spec["criteria"]
    rows = sample_plan(times, c["interval_s"], c["max_time_error_s"])
    if reference_plan is not None:
        from .research_data import read_plan
        plan=read_plan(reference_plan)
        rows=[{"sample":s["sample"],"target_s":s["time_s"],"frame":s["session_frame"],
               "captured_s":s["time_s"],"time_error_s":0,"status":"unknown","count":None,
               "detected":None,"reason":"not_analyzed","detections_json":""} for s in plan["samples"]]
    model_sha = digest(model_path)
    run = folder / "analyses" / ("run_" + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "software": VERSION, "state": "running", "created_utc": utcnow(),
                "protocol": spec, "recording_state": info["state"], "source_sha256": info["sha256"],
                "detector_sha256": model_sha, "device": "cpu", "runtime": {}, "planned_samples": len(rows),
                "code_sha256": {name: digest(Path(__file__).parent / name) for name in ("research.py", "group_approach.py", "ai_pose.py", "ai_worker.py", "ai_live.py", "sources.py")}}
    atomic_json(run / "analysis.json", manifest)
    if reference_plan is not None:
        manifest["reference_plan_sha256"]=plan["sha256"]
    from .sources import open_video
    cap, predictor = open_video(folder), None
    error = None
    try:
        if not cap.isOpened():
            raise ValueError("녹화 영상을 열 수 없습니다.")
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(times):
            raise ValueError("영상 프레임 수와 촬영 시각의 수가 다릅니다.")
        progress(0, len(rows), run.name, "모델 준비 중")
        predictor = predictor_factory(executable, c["animals"], cancel,detector_path=model_path) if predictor_factory is WorkerPredictor else predictor_factory(executable, c["animals"], cancel)
        manifest["runtime"] = predictor.runtime
        decoded = -1
        with (run / "samples.partial.csv").open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            for index, row in enumerate(rows):
                if cancel.is_set():
                    raise Cancelled()
                if row["frame"] is not None:
                    while decoded < row["frame"]:
                        if cancel.is_set():
                            raise Cancelled()
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError("녹화 영상의 프레임을 읽지 못했습니다.")
                        decoded += 1
                    if list(frame.shape[1::-1]) != c["frame_size"]:
                        raise ValueError("영상 크기가 녹화 조건과 다릅니다.")
                    prediction = predictor.predict(frame, row["frame"])
                    row.update(evaluate(prediction, c["food_rect"], c["frame_size"], c["animals"],
                                        c["min_bbox_score"], c["duplicate_iou_threshold"]))
                    row["detections_json"] = json.dumps(prediction, ensure_ascii=True, allow_nan=False)
                writer.writerow(row)
                f.flush()
                progress(index + 1, len(rows), run.name, "일정 간격 분석 중")
        # Decode the remaining tail too; don't certify a truncated video as complete.
        while decoded < len(times)-1:
            if cancel.is_set():
                raise Cancelled()
            ok, _ = cap.read()
            if not ok:
                raise ValueError("영상의 마지막 부분이 손상됐습니다.")
            decoded += 1
        if cap.read()[0]:
            raise ValueError("타임스탬프가 없는 추가 프레임이 있습니다.")
        manifest["state"] = "complete"
    except Cancelled:
        error = "사용자가 분석을 취소했습니다. 미분석 표본은 판정 불가입니다."
        manifest["state"] = "cancelled"
    except (Exception, SystemExit) as exc:
        error, manifest["state"] = str(exc), "failed"
    finally:
        cap.release()
        if predictor:
            predictor.close()
        with (run / "samples.csv").open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest.update(error=error, finished_utc=utcnow(), summary=summary(rows),
                        samples_sha256=digest(run / "samples.csv"))
        atomic_json(run / "analysis.json", manifest)
    if manifest["state"] == "complete":
        progress(len(rows), len(rows), run.name, "자동 결과 해석 생성 중")
        try:
            from .interpretation import write_interpretation
            write_interpretation(run)
            manifest["interpretation"] = {"state": "complete", "report": "interpretation.html"}
        except (Exception, SystemExit) as exc:
            # A derived report failure must not discard completed AI results.
            manifest["interpretation"] = {"state": "failed", "error": str(exc)}
        atomic_json(run / "analysis.json", manifest)
    return run, manifest


def analysis_rows(run):
    manifest = read_json(run / "analysis.json")
    if manifest["state"] == "running":
        raise ValueError("분석이 끝난 다음 검토해 주세요.")
    if digest(run / "samples.csv") != manifest["samples_sha256"]:
        raise ValueError("분석 표본 파일이 변경됐습니다.")
    with (run / "samples.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("sample", "frame", "count"):
            row[key] = int(row[key]) if row[key] != "" else None
    return manifest, rows


def review_plan(rows, count=20):
    candidates = [r["sample"] for r in rows if r["frame"] is not None]
    return sorted(random.Random(20261030).sample(candidates, min(count, len(candidates))))


def review_events(run):
    path = run / "review_events.jsonl"
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                raise ValueError("수동 검토 기록이 손상됐습니다. 원본 파일을 보관하고 확인해 주세요.") from None
    return events


def save_review(run, values):
    manifest, rows = analysis_rows(run)
    reviewer = str(values.get("reviewer", "")).strip()
    if not reviewer or len(reviewer) > 80:
        raise ValueError("검토자 이름을 1~80자로 입력해 주세요.")
    sample = values.get("sample")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample not in review_plan(rows):
        raise ValueError("검토 대상으로 선택된 표본을 지정해 주세요.")
    label, count = values.get("label"), values.get("count")
    if label not in ("approach", "away", "unknown"):
        raise ValueError("접근·비접근·판정 불가 중 하나를 선택해 주세요.")
    if label == "unknown":
        count = None
    elif count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= manifest["protocol"]["criteria"]["animals"]:
            raise ValueError("주변 마릿수를 확인해 주세요.")
        if (label == "approach") != (count > 0):
            raise ValueError("접근 여부와 마릿수가 서로 맞지 않습니다.")
    elif label == "away":
        count = 0
    events = review_events(run)
    entry = {"revision": len(events)+1, "utc": utcnow(), "reviewer": reviewer, "sample": sample,
             "label": label, "count": count, "note": str(values.get("note", ""))[:1000],
             "samples_sha256": manifest["samples_sha256"], "ai_sample_label_hidden_in_review_ui": True}
    with (run / "review_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def validation(run):
    _, rows = analysis_rows(run)
    latest = {}
    for event in review_events(run):
        latest[(event["reviewer"], event["sample"])] = event
    reports = []
    for reviewer in sorted({key[0] for key in latest}):
        pairs = [e for (name, _), e in latest.items() if name == reviewer]
        tp = tn = fp = fn = human_unknown = ai_unknown = 0
        for event in pairs:
            ai = rows[event["sample"]]["status"]
            human = event["label"]
            human_unknown += human == "unknown"
            ai_unknown += ai == "unknown"
            if ai == "unknown" or human == "unknown":
                continue
            tp += ai == human == "approach"
            tn += ai == human == "away"
            fp += ai == "approach" and human == "away"
            fn += ai == "away" and human == "approach"
        known = tp+tn+fp+fn
        reports.append({"reviewer": reviewer, "reviewed": len(pairs), "paired_known": known,
                        "human_unknown": human_unknown, "ai_unknown": ai_unknown, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                        "agreement_pct": 100*(tp+tn)/known if known else None,
                        "sensitivity": tp/(tp+fn) if tp+fn else None, "specificity": tn/(tn+fp) if tn+fp else None})
    from .evaluation import binary_metrics, count_metrics, agreement
    for report in reports:
        pairs=[e for (name,_),e in latest.items() if name==report["reviewer"]]
        to_int=lambda label: {"approach":1,"away":0,"unknown":-1}[label]
        report.update(binary_metrics([to_int(e["label"]) for e in pairs],[to_int(rows[e["sample"]]["status"]) for e in pairs]))
        report["counts"]=count_metrics([e["count"] if e["count"] is not None else -1 for e in pairs],
                                     [rows[e["sample"]]["count"] if rows[e["sample"]]["count"] is not None else -1 for e in pairs],
                                     read_json(run/"analysis.json")["protocol"]["criteria"]["animals"])
    return {"sampling": "uniform_without_replacement_seed_20261030_max20", "planned": len(review_plan(rows)),
            "inter_rater":agreement(review_events(run)),"scope":"development_review_not_final_independent_validation",
            "reviewers": reports, "events": len(review_events(run)), "validated": False}


def export_review_csv(run):
    manifest, rows = analysis_rows(run)
    latest = {}
    for event in review_events(run):
        latest[(event["reviewer"], event["sample"])] = event
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["reviewer", "sample", "target_s", "captured_s", "AI", "human", "AI_count", "human_count", "revision", "reviewed_utc"])
    for (name, index), event in sorted(latest.items()):
        safe_name = "'" + name if name.lstrip().startswith(("=", "+", "-", "@")) else name
        row = rows[index]
        writer.writerow([safe_name, index, row["target_s"], row["captured_s"], row["status"], event["label"], row["count"], event["count"], event["revision"], event["utc"]])
    return output.getvalue().encode("utf-8-sig")
