"""Manual point sampling of feeding-related behaviour in a group-housed cage.

Counts are at sampled instants, not identified animals, bouts, intake or appetite.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .sources import camera_timestamps, open_video, video_info

FIELDS = ("sample", "target_s", "frame", "time_s", "approach", "feeding")
WINDOW = "Feeding observation"


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _schedule(duration, interval, start, end, fps, stamps=None):
    if not all(math.isfinite(v) for v in (duration, interval, start, end, fps)):
        raise ValueError("시간과 FPS는 유한한 숫자여야 합니다.")
    if fps <= 0 or interval <= 0 or start < 0 or end > duration + 1e-6 or end <= start:
        raise ValueError("관찰 범위 또는 표본 간격이 유효하지 않습니다.")
    n = int(math.floor((end - start) / interval + 1e-10))
    if n < 1:
        raise ValueError("관찰 범위는 표본 간격 이상이어야 합니다.")
    targets = start + (np.arange(n) + 0.5) * interval
    if stamps is None:
        frames = np.floor(targets * fps).astype(int)
        actual = frames / fps
    else:
        frames = np.searchsorted(stamps, targets, side="left")
        frames = np.minimum(frames, len(stamps) - 1)
        actual = stamps[frames]
    if len(np.unique(frames)) != len(frames):
        raise ValueError("표본 간격이 영상 프레임 간격보다 짧습니다.")
    return [(i, float(t), int(f), float(a)) for i, (t, f, a) in enumerate(zip(targets, frames, actual))]


def _read_rows(path, meta, schedule):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("채점 CSV 열이 올바르지 않습니다.")
        rows = list(reader)
    if len(rows) > len(schedule):
        raise ValueError("표본 행이 예상보다 많습니다.")
    for i, row in enumerate(rows):
        index, target, frame, actual = schedule[i]
        if (int(row["sample"]) != index or int(row["frame"]) != frame
                or abs(float(row["target_s"]) - target) > 1e-5
                or abs(float(row["time_s"]) - actual) > 1e-5):
            raise ValueError("채점 CSV와 영상의 표본 시각이 일치하지 않습니다.")
        for name in ("approach", "feeding"):
            value = row[name]
            if value != "unknown" and (not value.isdigit() or int(value) > meta["animals"]):
                raise ValueError(f"유효하지 않은 {name} 값: {value}")
    return rows


def _frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"프레임 {index}을 읽지 못했습니다. 채점을 중단합니다.")
    position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if math.isfinite(position) and position > 0 and abs(position - (index + 1)) > 0.5:
        raise RuntimeError(f"요청한 프레임 {index}과 실제 재생 위치가 일치하지 않습니다.")
    return img


def _show(img, text):
    h, w = img.shape[:2]
    scale = min(1.0, 1280 / w, 800 / h)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    panel = np.zeros((82, img.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(text):
        cv2.putText(panel, line, (8, 23 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, np.vstack((panel, img)))


def _replay(cap, frame, fps, nframes):
    # Replay is context only; counts refer to the target frame, not the entire clip.
    start = max(0, frame - int(fps))
    end = min(nframes, frame + int(fps) + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for index in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        _show(img, ["Context replay: assess the TARGET frame", f"frame {index} / target {frame}", "wait for replay to finish"])
        if cv2.waitKey(max(1, int(1000 / fps))) & 0xff == ord("q"):
            break


def _ask_count(cap, frame, fps, nframes, animals, sample, total, behavior):
    img = _frame(cap, frame)
    while True:
        _show(img, [f"Sample {sample + 1}/{total}  frame {frame}  {behavior}",
                    f"0-{animals}: count | u: unknown | r: replay context | q: save and quit",
                    "Count at target frame; if the count is uncertain, use unknown"])
        key = cv2.waitKey(0) & 0xff
        if ord("0") <= key <= ord(str(animals)):
            return str(key - ord("0"))
        if key == ord("u"):
            return "unknown"
        if key in (ord("q"), 27):
            return None
        if key == ord("r"):
            _replay(cap, frame, fps, nframes)


def score_video(video, out, cage_id, group, animals, interval, start=0., end=None, resume=False, recorded_at=""):
    video = Path(video).resolve()
    out = Path(out)
    if not cage_id.strip() or not group.strip() or not 1 <= animals <= 9:
        raise SystemExit("케이지 ID, 군 코드와 마우스 수 1~9를 확인하세요.")
    if recorded_at:
        try:
            dt = datetime.fromisoformat(recorded_at)
            if dt.utcoffset() is None:
                raise ValueError
        except ValueError:
            raise SystemExit("기록 일시는 시간대가 포함된 ISO 형식이어야 합니다 (예: 2026-10-28T20:00:00+09:00).") from None
    cap = open_video(video)
    try:
        info = video_info(cap)
        fps, nframes = float(cap.get(cv2.CAP_PROP_FPS)), info["frames"]
        if nframes < 2 or not math.isfinite(fps) or fps <= 0:
            raise SystemExit("영상 프레임 수 또는 FPS를 확인할 수 없습니다.")
        stamps = camera_timestamps(video, nframes)
        duration = float(stamps[-1] + np.median(np.diff(stamps))) if stamps is not None else nframes / fps
        end = duration if end is None else end
        try:
            schedule = _schedule(duration, interval, start, end, fps, stamps)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        timestamp_hash = hashlib.sha256(stamps.tobytes()).hexdigest() if stamps is not None else ""
        meta = {"version": 1, "video": str(video), "sha256": _hash(video), "cage_id": cage_id,
                "group": group, "animals": animals, "interval_s": interval, "start_s": start,
                "end_s": end, "recorded_at": recorded_at, "fps": fps, "frames": nframes,
                "timebase": "camera_timestamps" if stamps is not None else "constant_fps_assumed", "timestamp_hash": timestamp_hash,
                "planned_samples": len(schedule)}
        meta_path, csv_path = out / "feeding_session.json", out / "samples.csv"
        if resume:
            if not meta_path.is_file() or not csv_path.is_file():
                raise SystemExit("이어갈 채점 폴더를 찾지 못했습니다.")
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old != meta:
                raise SystemExit("영상 또는 채점 조건이 이전 기록과 다릅니다. 기존 기록을 변경하지 않았습니다.")
            rows = _read_rows(csv_path, meta, schedule)
        else:
            out.mkdir(parents=True, exist_ok=False)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            with csv_path.open("x", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(FIELDS)
            rows = []
        print("각 표본 시점의 마릿수를 입력하세요. 접근과 섭식은 별도로 채점하며, 가려져 전체 마릿수를 모르면 u입니다.")
        print("r: 앞뒤 1초 재생, q: 현재 표본을 저장하지 않고 종료. 이미 완료한 표본은 저장됩니다.")
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for index, target, frame, actual in schedule[len(rows):]:
                approach = _ask_count(cap, frame, fps, nframes, animals, index, len(schedule), "APPROACH")
                if approach is None:
                    break
                feeding = _ask_count(cap, frame, fps, nframes, animals, index, len(schedule), "FEEDING")
                if feeding is None:
                    break
                writer.writerow([index, f"{target:.6f}", frame, f"{actual:.6f}", approach, feeding])
                f.flush()
                rows.append({"approach": approach, "feeding": feeding})
        print(f"채점 완료 {len(rows)}/{len(schedule)} 표본: {csv_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def summarize_sessions(sessions, output):
    if len({str(Path(p).resolve()) for p in sessions}) != len(sessions):
        raise SystemExit("같은 채점 폴더가 중복되었습니다.")
    summaries = []
    for p in map(Path, sessions):
        meta = json.loads((p / "feeding_session.json").read_text(encoding="utf-8"))
        video = Path(meta["video"])
        if not video.is_file() or _hash(video) != meta["sha256"]:
            raise SystemExit(f"영상 원본이 없거나 변경되었습니다: {video}")
        cap = open_video(video)
        try:
            stamps = camera_timestamps(video, meta["frames"])
            if ("camera_timestamps" if stamps is not None else "constant_fps_assumed") != meta["timebase"]:
                raise SystemExit("영상의 시간 기준이 바뀌었습니다.")
            if stamps is not None and hashlib.sha256(stamps.tobytes()).hexdigest() != meta["timestamp_hash"]:
                raise SystemExit("카메라 타임스탬프가 채점 이후 변경되었습니다.")
            duration = float(stamps[-1] + np.median(np.diff(stamps))) if stamps is not None else meta["frames"] / meta["fps"]
            schedule = _schedule(duration, meta["interval_s"], meta["start_s"], meta["end_s"], meta["fps"], stamps)
            rows = _read_rows(p / "samples.csv", meta, schedule)
        finally:
            cap.release()
        row = {"cage_id": meta["cage_id"], "group": meta["group"], "recorded_at": meta["recorded_at"],
               "video": meta["video"], "video_sha256": meta["sha256"], "animals": meta["animals"],
               "interval_s": meta["interval_s"], "start_s": meta["start_s"], "end_s": meta["end_s"],
               "timebase": meta["timebase"], "planned_samples": len(schedule), "scored_samples": len(rows),
               "unscored_samples": len(schedule) - len(rows)}
        for behavior in ("approach", "feeding"):
            known = [int(r[behavior]) for r in rows if r[behavior] != "unknown"]
            row[f"{behavior}_known_samples"] = len(known)
            row[f"{behavior}_unknown_samples"] = len(rows) - len(known)
            row[f"{behavior}_positive_samples"] = sum(v > 0 for v in known)
            row[f"{behavior}_positive_pct"] = 100 * sum(v > 0 for v in known) / len(known) if known else ""
            row[f"{behavior}_mean_mice"] = sum(known) / len(known) if known else ""
        summaries.append(row)
    with open(output, "x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
