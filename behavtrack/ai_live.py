"""Webcam recording with asynchronous, sampled CPU pose inference."""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import draw
from .ai_pose import approach_from_noses
from .config import arenas_from
from .session import create_session_dir
from .sources import Recorder, open_camera

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "Group camera - AI sampled CPU estimation"


class AIWorker:
    """One frame in flight at most; no inference backlog or stale frame substitution."""
    def __init__(self, executable, worker_args=()):
        self.requests = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.process = subprocess.Popen([str(executable), "-u", "-m", "behavtrack.ai_worker", *worker_args],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            encoding="utf-8", bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("AI 처리기가 종료되었습니다. 실행 상태의 오류를 확인하세요.")
        return json.loads(line)

    def _run(self):
        try:
            initial = self._read()
            if not initial.get("ready"):
                raise RuntimeError(initial.get("error", "AI 모델을 열 수 없습니다."))
            self.events.put(("ready", initial.get("runtime", {})))
            while True:
                item = self.requests.get()
                if item is None:
                    break
                frame_index, captured_s, encoded = item
                payload = {"id": frame_index, "image": base64.b64encode(encoded).decode("ascii")}
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                response = self._read()
                if response.get("id") != frame_index or "error" in response:
                    raise RuntimeError(response.get("error", "AI 결과의 프레임 번호가 다릅니다."))
                self.events.put(("result", (frame_index, captured_s, response["result"])))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def close(self):
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_report(folder, info):
    s = info["summary"]
    def fmt(v):
        return "–" if v is None else f"{v:.2f}"
    rows = [("케이지", info["cage_id"] or "–"), ("군", info["group"] or "–"),
            ("분석한 표본", s["samples"]), ("추정 가능 표본", s["known_samples"]),
            ("판정 불가 표본", s["unknown_samples"]),
            ("한 마리 이상 접근한 표본 비율 (%)", fmt(s["positive_pct_known"])),
            ("표본 평균 접근 마릿수", fmt(s["mean_count_known"]))]
    table = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    report = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>AI 공동 사육 표본 분석</title>
<style>body{{font:15px 'Malgun Gothic',sans-serif;max-width:950px;margin:32px auto;padding:0 20px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}table{{border-collapse:collapse}}.note{{background:#fff6dc;padding:14px;line-height:1.7}}</style>
<h1>AI 공동 사육 사료 접근 표본 분석</h1><p class="note">위에서 촬영한 다섯 마리의 코가 사료 구역에 있는지 추정합니다.
CPU 처리 속도에 따라 불규칙하게 표본을 분석합니다. 비율은 판정 가능한 표본만 분모로 계산하며
실제 섭식, 개체별 정체성 또는 연속 행동 시간이 아닙니다. 가림·오검출은 판정 가능으로 잘못 표시될 수 있으므로 실제 영상의 수동 채점과 비교해야 합니다.</p>
<table>{table}</table><p>표본별 결과와 시각: ai_samples.csv · 원본: raw.mp4 · 촬영 시각: frame_times.csv</p></html>"""
    (folder / "report.html").write_text(report, encoding="utf-8")


def run_ai_live(cfg, camera, out_root, cage_id="", group="", duration=None, interval=5.0):
    arenas = arenas_from(cfg)
    if len(arenas) != 1 or arenas[0].zone("food") is None:
        raise SystemExit("사료 구역이 포함된 단일 케이지 설정이 필요합니다.")
    if not np.isfinite(interval) or interval <= 0:
        raise SystemExit("표본 간격은 양수여야 합니다.")
    arena, food = arenas[0], arenas[0].zone("food").shape
    ai_python = ROOT / ".ai_venv" / "Scripts" / "python.exe"
    if not ai_python.is_file():
        raise SystemExit("AI 실행 환경이 없습니다. 먼저 AI 모델 설치를 실행하세요.")
    cap = open_camera(camera)
    worker = None
    folder = raw = samples_file = times_file = None
    sample_writer = times_writer = None
    ready = inflight = False
    pending = None
    last_status = "Loading AI model..."
    last_request = -interval
    last_sample_at = None
    trial_start = None
    frame_index = samples = known = positive = total_count = 0
    stamps = deque(maxlen=60)
    failed = None
    try:
        worker = AIWorker(ai_python)
        print("AI 모델을 불러오는 중입니다. 준비가 끝나면 영상 창에서 s를 눌러 녹화를 시작하세요.", flush=True)
        while True:
            while True:
                try:
                    kind, event = worker.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "ready":
                    ready = True
                    last_status = "AI ready: press s to start recording"
                    print("AI 준비 완료. 빈 케이지 배경은 필요하지 않습니다. 영상 창에서 s를 누르세요.", flush=True)
                elif kind == "error":
                    failed = event
                    print("AI 오류: " + str(event), flush=True)
                    break
                else:
                    inflight = False
                    fid, sample_time, prediction = event
                    pending = None
                    count, reason = None, prediction.get("reason", "invalid_output")
                    noses = prediction.get("noses", [])
                    if reason == "candidate":
                        try:
                            count, reason = approach_from_noses(noses, arena.shape, food)
                        except (TypeError, ValueError):
                            reason = "invalid_pose_output"
                    samples += 1
                    if count is not None:
                        known += 1
                        positive += count > 0
                        total_count += count
                    sample_writer.writerow([fid, f"{sample_time:.9f}", prediction.get("detected", ""),
                                            "unknown" if count is None else count, reason,
                                            json.dumps(noses, ensure_ascii=True, allow_nan=False)])
                    samples_file.flush()
                    last_sample_at = trial_start + sample_time
                    last_status = "UNKNOWN" if count is None else f"ESTIMATED nose in food: {count} / 5"
            if failed:
                break
            ok, frame = cap.read()
            if not ok:
                failed = "카메라 프레임을 읽지 못했습니다."
                print(failed, flush=True)
                break
            now = time.perf_counter()
            stamps.append(now)
            fps = (len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps) > 2 and stamps[-1] > stamps[0] else 30.0
            if folder is not None:
                elapsed = max(0.0, now - trial_start)
                raw.write(frame)
                times_writer.writerow([frame_index, f"{elapsed:.9f}"])
                if ready and not inflight and elapsed - last_request >= interval:
                    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise RuntimeError("AI 표본 프레임을 만들지 못했습니다.")
                    worker.requests.put_nowait((frame_index, elapsed, enc.tobytes()))
                    pending = (frame_index, elapsed)
                    inflight = True
                    last_request = elapsed
                frame_index += 1
            vis = frame.copy()
            draw.layout(vis, [arena])
            age = f" | sample age: {max(0.0, now-last_sample_at):.1f}s" if last_sample_at is not None else ""
            status = last_status + age
            if inflight:
                status += " | AI working"
            draw.lines(vis, [status, f"CPU sampling >= {interval:g}s | s: start/stop | q: save and quit",
                             "Nose in food zone is not feeding; review video and samples"])
            cv2.imshow(WINDOW, vis)
            if folder is not None and duration is not None and elapsed >= duration:
                break
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord(" ")):
                if folder is not None:
                    break
                if not ready:
                    print("AI 모델 준비가 끝날 때까지 기다려 주세요.", flush=True)
                    continue
                folder = create_session_dir(out_root, "ai_group_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                raw = Recorder(folder / "raw.mp4", max(1, min(60, round(fps))))
                samples_file = (folder / "ai_samples.csv").open("w", newline="", encoding="utf-8-sig")
                times_file = (folder / "frame_times.csv").open("w", newline="", encoding="utf-8")
                sample_writer, times_writer = csv.writer(samples_file), csv.writer(times_file)
                sample_writer.writerow(["frame", "time_s", "detected", "approach_count", "status", "nose_xy_confidence"])
                times_writer.writerow(["frame", "time_s"])
                trial_start = time.perf_counter()
                print(f"AI 표본 녹화 시작: {folder}", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pending is not None and sample_writer is not None:
            sample_writer.writerow([pending[0], f"{pending[1]:.9f}", "", "unknown", "inference_incomplete", ""])
            samples += 1
        if worker:
            worker.close()
        for f in (samples_file, times_file):
            if f:
                f.close()
        if raw:
            raw.close()
    if folder is None:
        if failed:
            raise SystemExit("AI를 시작하지 못했습니다: " + str(failed))
        print("녹화를 시작하지 않아 저장된 결과가 없습니다.")
        return None
    summary = {"samples": samples, "known_samples": known, "unknown_samples": samples-known,
               "positive_pct_known": 100*positive/known if known else None,
               "mean_count_known": total_count/known if known else None}
    model_dir = ROOT / "models" / "topviewmouse"
    info = {"version": 1, "method": "superanimal_topviewmouse_cpu_sampled", "created": datetime.now().isoformat(timespec="seconds"),
            "cage_id": cage_id, "group": group, "animals": 5, "interval_s": interval, "frames": frame_index,
            "pose_confidence_threshold": 0.8, "detector_confidence_threshold": 0.6,
            "source": {"type": "camera", "video": "raw.mp4", "frames": frame_index},
            "config": cfg, "summary": summary, "error": failed,
            "model_sha256": {"pose": _sha(model_dir / "pose.pt"), "detector": _sha(model_dir / "detector.pt")}}
    (folder / "ai_group_session.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(folder, info)
    print(f"AI 결과 저장: {folder}\n보고서: {folder / 'report.html'}", flush=True)
    if failed:
        raise SystemExit("일부만 저장됨: " + str(failed))
    return folder
