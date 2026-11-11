"""Experimental group-cage monitoring using non-overlapping foreground blobs.

Detections are not identities or confirmed eating; mismatched counts are unknown.
"""
from __future__ import annotations

import csv
import html
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import draw
from .config import arenas_from
from .session import create_session_dir
from .sources import Recorder, open_camera
from .trackers.blob import BlobTracker

WINDOW = "Group cage - experimental approach count"
MAX_GAP_S = 1.0


def group_candidates(tracker, gray, expected, food):
    """Return tentative centroids and a count only when all expected blobs are separate.

    A matching count is not proof: debris, lighting, and hidden mice can still cause errors.
    """
    roi = tracker.rois[0]
    binary = tracker.binarize(gray)
    mask = tracker._blobs(roi, binary)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    too_large = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > roi.max_area:
            too_large = True
        if roi.min_area <= area <= roi.max_area:
            moments = cv2.moments(contour)
            if moments["m00"]:
                centers.append(((moments["m10"] / moments["m00"] + roi.x0) / tracker.f,
                                (moments["m01"] / moments["m00"] + roi.y0) / tracker.f))
    if too_large or len(centers) != expected:
        return centers, None, mask
    return centers, int(np.count_nonzero(food.contains(centers))), mask


class IntervalAccumulator:
    """Attribute measured intervals to preceding sample with bounded memory."""
    def __init__(self):
        self.previous = None
        self.known = self.unknown = self.animal_seconds = self.positive = 0.0

    def add(self, time_s, count):
        if self.previous is not None:
            previous_time, previous_count = self.previous
            dt = time_s - previous_time
            if dt <= 0 or not math.isfinite(dt):
                raise ValueError("프레임 시간 순서가 올바르지 않습니다.")
            if dt > MAX_GAP_S or previous_count is None:
                self.unknown += dt
            else:
                self.known += dt
                self.animal_seconds += previous_count * dt
                self.positive += (previous_count > 0) * dt
        self.previous = (time_s, count)

    def summary(self):
        return {"observed_s": self.known + self.unknown, "known_s": self.known, "unknown_s": self.unknown,
                "positive_s": self.positive, "animal_seconds": self.animal_seconds,
                "mean_approach_count": self.animal_seconds / self.known if self.known else None,
                "positive_pct": 100 * self.positive / self.known if self.known else None}


def interval_summary(rows):
    accumulator = IntervalAccumulator()
    for row in rows:
        accumulator.add(row["time_s"], row["approach"])
    return accumulator.summary()


def write_group_report(folder, info):
    s = info["summary"]
    def fmt(value):
        return "–" if value is None else f"{value:.2f}"
    data = [("케이지 ID", info["cage_id"] or "–"), ("군 코드", info["group"] or "–"),
            ("마우스 수", info["animals"]), ("관찰 시간 (초)", fmt(s["observed_s"])),
            ("추정 가능 시간 (초)", fmt(s["known_s"])), ("판정 불가 시간 (초)", fmt(s["unknown_s"])),
            ("한 마리 이상 접근 시간 (초)", fmt(s["positive_s"])),
            ("접근 비율 (추정 가능 시간 기준, %)", fmt(s["positive_pct"])),
            ("평균 접근 마릿수 (추정 가능 시간 기준)", fmt(s["mean_approach_count"]))]
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in data)
    body = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>공동 사육 접근 추정</title>
<style>body{{font:15px 'Malgun Gothic',sans-serif;max-width:900px;margin:30px auto;padding:0 20px;color:#233}}
td,th{{border:1px solid #ddd;padding:8px;text-align:left}}table{{border-collapse:collapse}}.note{{background:#fff6dc;padding:14px;line-height:1.7}}</style>
<h1>공동 사육 사료 구역 접근 추정</h1><p class="note">이 결과는 빈 배경과의 차이로 추정한 실험적 결과입니다.
검출 수가 맞더라도 오인식이나 숨은 동물을 놓칠 수 있습니다. 몸 중심이 구역에 있는지를 계산하며 실제 섭식이나 개체별 이동이 아닙니다.
판정 불가 구간은 0으로 계산하지 않습니다. 이 환경에서 수동 채점과 비교하여 검증하세요.</p><table>{rows}</table>
<p>프레임별 자료: group_frames.csv · 원본: raw.mp4 · 표시 영상: tracked.mp4 · 프레임 시각: frame_times.csv · 설정 및 요약: group_session.json</p></html>"""
    (folder / "report.html").write_text(body, encoding="utf-8")


def run_group_live(cfg, camera, out_root, animals, cage_id="", group="", duration=None):
    if not isinstance(animals, int) or animals < 2:
        raise SystemExit("공동 사육 마릿수는 2 이상의 정수여야 합니다.")
    arenas = arenas_from(cfg)
    if len(arenas) != 1 or arenas[0].zone("food") is None:
        raise SystemExit("사료 구역이 포함된 단일 케이지 설정을 선택하세요.")
    if cfg["tracking"]["method"] != "background":
        raise SystemExit("공동 사육 추정에는 빈 배경 차분 설정이 필요합니다.")
    arena = arenas[0]
    food = arena.zone("food").shape
    cap = open_camera(camera)
    trial = None
    accumulator = IntervalAccumulator()
    frame_count = 0
    frames_file = times_file = raw = marked = None
    bg_frames = None
    tracker = None
    stamps = deque(maxlen=60)
    started = None
    size = None
    camera_failed = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                camera_failed = True
                print("카메라 프레임을 읽지 못해 녹화를 종료합니다.")
                break
            now = time.perf_counter()
            stamps.append(now)
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 and stamps[-1] > stamps[0] else 30.0
            if tracker is None:
                size = frame.shape[:2]
                tracker = BlobTracker([arena], cfg, size)
            elif frame.shape[:2] != size:
                raise RuntimeError("녹화 중 카메라 해상도가 바뀌었습니다.")
            gray = tracker.preprocess(frame)
            if bg_frames is not None:
                bg_frames.append(gray)
                if len(bg_frames) >= 30:
                    tracker.set_background(np.median(np.stack(bg_frames), axis=0).astype(np.uint8))
                    bg_frames = None
                    print("배경 저장 완료. 동물을 넣고 영상 창에서 s를 누르세요.")
            centers, count, _ = group_candidates(tracker, gray, animals, food) if tracker.ready else ([], None, None)
            vis = frame.copy()
            draw.layout(vis, [arena])
            for center in centers:
                cv2.circle(vis, tuple(int(v) for v in center), 5, (0, 255, 255), -1)
            status = "UNKNOWN (count mismatch or overlap)" if count is None else f"ESTIMATED food count: {count} / {animals}"
            if tracker.bg is None:
                status = "Empty cage: press b to capture background"
            elif bg_frames is not None:
                status = "Capturing empty background..."
            elif trial is None:
                status = "Ready: put animals in, press s to record"
            draw.lines(vis, [status, f"detected {len(centers)} / expected {animals} | fps {fps:.1f} | threshold {tracker.threshold}",
                             "b: background | s: start/stop | +/-: threshold | q: save and quit"])
            if trial is not None:
                elapsed = max(0.0, now - started)
                frame_index = frame_count
                raw.write(frame)
                marked.write(vis)
                accumulator.add(elapsed, count)
                frame_count += 1
                csv.writer(frames_file).writerow([frame_index, f"{elapsed:.9f}", len(centers), "unknown" if count is None else count, tracker.threshold])
                csv.writer(times_file).writerow([frame_index, f"{elapsed:.9f}"])
            cv2.imshow(WINDOW, vis)
            if trial is not None and duration is not None and elapsed >= duration:
                break
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("b") and trial is None:
                bg_frames = []
                print("빈 케이지의 배경을 촬영합니다 (30 프레임).")
            elif key in (ord("s"), ord(" ")):
                if trial is not None:
                    break
                if not tracker.ready or bg_frames is not None:
                    print("먼저 빈 케이지에서 b로 배경을 촬영하세요.")
                    continue
                trial = create_session_dir(out_root, "group_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                encoded, image = cv2.imencode(".png", tracker.bg)
                if not encoded:
                    raise RuntimeError("배경 이미지를 저장하지 못했습니다.")
                image.tofile(str(trial / "background.png"))
                raw = Recorder(trial / "raw.mp4", round(fps))
                marked = Recorder(trial / "tracked.mp4", round(fps))
                frames_file = (trial / "group_frames.csv").open("w", newline="", encoding="utf-8-sig")
                times_file = (trial / "frame_times.csv").open("w", newline="", encoding="utf-8")
                csv.writer(frames_file).writerow(["frame", "time_s", "detected_blobs", "approach_count_estimate", "threshold"])
                csv.writer(times_file).writerow(["frame", "time_s"])
                started = time.perf_counter()
                print(f"공동 사육 녹화 시작: {trial}")
            elif key in (ord("+"), ord("=")):
                tracker.threshold = min(250, tracker.threshold + 2)
            elif key == ord("-"):
                tracker.threshold = max(2, tracker.threshold - 2)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for f in (frames_file, times_file):
            if f:
                f.close()
        if raw:
            raw.close()
        if marked:
            marked.close()
    if trial is None:
        print("녹화를 시작하지 않아 저장된 결과가 없습니다.")
        return None
    summary = accumulator.summary()
    info = {"version": 1, "method": "experimental_background_blobs", "created": datetime.now().isoformat(timespec="seconds"),
            "cage_id": cage_id, "group": group, "animals": animals, "camera": camera, "frames": frame_count,
            "threshold": tracker.threshold, "config": cfg, "camera_failed": camera_failed, "summary": summary,
            "source": {"type": "camera", "video": "raw.mp4", "frames": frame_count}}
    (trial / "group_session.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    write_group_report(trial, info)
    print(f"저장 완료: {trial}\n보고서: {trial / 'report.html'}")
    if camera_failed:
        raise SystemExit("카메라 읽기 오류로 일찍 종료했습니다. 저장된 부분만 확인하세요.")
    return trial
