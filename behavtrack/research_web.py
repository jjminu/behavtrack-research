"""Loopback-only browser interface. Capture continues when the browser tab closes."""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import research as data
from .sources import open_camera
from .research_api import ResearchWorkflow

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "web"


class Capture:
    def __init__(self, root, opener=open_camera):
        self.root, self.opener = Path(root), opener
        self.lock, self.command_lock = threading.RLock(), threading.RLock()
        self.preview_lock = threading.Lock()
        self.preview_cache = None
        self.preview_profile = None
        self.frame_number = 0
        self.thread = self.cap = self.frame = self.recording = None
        self.camera, self.error, self.last_session = None, None, None
        self.stop_event = threading.Event()
        self.stamps = deque(maxlen=60)
        self.generation = 0
        self.saving = False
        self.lease = None

    def open(self, camera):
        with self.command_lock:
            with self.lock:
                if self.recording or self.saving:
                    raise ValueError("녹화를 먼저 종료해 주세요.")
            self.close_camera()
            if isinstance(camera, bool) or not isinstance(camera, int) or not 0 <= camera <= 20:
                raise ValueError("카메라 번호를 확인해 주세요.")
            self.cap = self.opener(camera)
            self.camera, self.error = camera, None
            self.stop_event.clear()
            with self.lock:
                self.frame = None
                self.stamps.clear()
                self.generation += 1
            self.thread = threading.Thread(target=self._loop, name="research-camera", daemon=True)
            self.thread.start()

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                captured = time.perf_counter()
                if not ok:
                    raise ValueError("카메라 연결이 끊겼습니다. 촬영된 부분을 저장합니다.")
                with self.lock:
                    if self.frame is not None and frame.shape != self.frame.shape:
                        raise ValueError("카메라 해상도가 바뀌었습니다. 다시 연결해 주세요.")
                    self.frame = frame
                    self.frame_number += 1
                    self.stamps.append(captured)
                    if self.recording:
                        self.recording.record(frame, captured)
        except (Exception, SystemExit) as exc:
            self.error = str(exc)
            self.stop_recording(error=self.error)
        finally:
            self.cap.release()
            with self.lock:
                self.frame = None

    def state(self):
        with self.lock:
            rec = self.recording
            fps = ((len(self.stamps)-1)/(self.stamps[-1]-self.stamps[0])) if len(self.stamps) > 1 and self.stamps[-1] > self.stamps[0] else None
            return {"connected": self.frame is not None, "camera": self.camera, "generation": self.generation,
                    "capture_fps": fps if self.frame is not None else None,
                    "size": list(self.frame.shape[1::-1]) if self.frame is not None else None,
                    "recording": rec.folder.name if rec else None, "frames": rec.info["frames"] if rec else 0,
                    "elapsed": time.perf_counter()-rec.start_monotonic if rec else 0,
                    "error": self.error, "saving": self.saving, "last_session": self.last_session}

    def preview(self, with_metadata=False, max_width=960, quality=80):
        # Encode each requested camera frame once, even with multiple browser tabs.
        # Resizing/JPEG encoding never holds the capture/recording lock.
        with self.preview_lock:
            with self.lock:
                if self.frame is None:
                    return None
                key = f"{self.generation}:{self.frame_number}"
                profile=(int(max_width),int(quality))
                if self.preview_cache and self.preview_cache[1] == key and self.preview_profile==profile:
                    return self.preview_cache if with_metadata else self.preview_cache[0]
                frame = self.frame.copy()
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, round(h*max_width/w)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if not ok:
                return None
            self.preview_cache = (encoded.tobytes(), key)
            self.preview_profile=profile
            return self.preview_cache if with_metadata else self.preview_cache[0]

    def start(self, values):
        with self.command_lock, self.lock:
            if self.recording or self.saving or self.frame is None:
                raise ValueError("카메라를 먼저 켜고, 기존 녹화가 끝났는지 확인해 주세요.")
            size = list(self.frame.shape[1::-1])
            if values.get("size") != size or values.get("generation") != self.generation:
                raise ValueError("카메라가 바뀌었습니다. 사료 주변을 다시 표시해 주세요.")
            spec = data.protocol(values, size)
            fps = ((len(self.stamps)-1)/(self.stamps[-1]-self.stamps[0])) if len(self.stamps) > 2 and self.stamps[-1] > self.stamps[0] else 30
            from .activity import Lease
            self.lease=Lease(self.root,"recording")
            try:
                self.recording = data.ResearchRecording(self.root, spec, self.camera, max(1, min(60, round(fps, 2))))
            except BaseException:
                self.lease.close(); self.lease=None
                raise
            self.error = None
            return self.recording.folder.name

    def stop_recording(self, error=None):
        with self.lock:
            rec, self.recording = self.recording, None
            if rec:
                self.saving = True
        if rec:
            try:
                folder = rec.finish(error)
                self.last_session = folder.name
                return folder.name
            except (Exception, SystemExit) as exc:
                self.error = "저장 오류: " + str(exc)
                raise
            finally:
                self.saving = False
                if self.lease:
                    self.lease.close(); self.lease=None
        return None

    def close_camera(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise ValueError("카메라 응답을 기다리고 있습니다. 잠시 후 다시 시도해 주세요.")
        self.thread = None

    def close(self):
        self.stop_recording()
        self.close_camera()


class Application(ResearchWorkflow):
    def __init__(self, root=ROOT, capture=None, predictor_factory=data.WorkerPredictor):
        self.root = Path(root)
        self.results = self.root / "results"
        self.results.mkdir(exist_ok=True)
        self.capture = capture or Capture(self.results)
        self.predictor_factory = predictor_factory
        self.lock = threading.RLock()
        self.job = {"state": "idle"}
        self.job_thread = None
        self.cancel = threading.Event()
        self.verified_review_sources = {}
        from .remote_service import RemoteHub
        self.remote = RemoteHub(self)

    def session(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"research_[A-Za-z0-9_]+", name):
            raise ValueError("올바른 촬영 결과를 선택해 주세요.")
        path = (self.results / name).resolve()
        if path.parent != self.results.resolve() or not path.is_dir():
            raise ValueError("촬영 결과를 찾을 수 없습니다.")
        return path

    def run_path(self, session, name):
        if not isinstance(name, str) or not re.fullmatch(r"run_[a-f0-9]+", name):
            raise ValueError("올바른 분석 결과를 선택해 주세요.")
        parent = self.session(session) / "analyses"
        path = (parent / name).resolve()
        if path.parent != parent.resolve() or not path.is_dir():
            raise ValueError("분석 결과를 찾을 수 없습니다.")
        return path

    def entries(self):
        entries = []
        for path in sorted(self.results.glob("research_*/recording.json"), reverse=True):
            try:
                info = data.read_json(path.parent/"recovery.json" if (path.parent/"recovery.json").exists() else path)
                spec = data.read_json(path.parent / "protocol.json")
                runs = []
                for result in sorted((path.parent / "analyses").glob("run_*/analysis.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    m = data.read_json(result)
                    runs.append({"id": result.parent.name, "state": m["state"], "created_utc": m["created_utc"],
                                 "summary": m.get("summary"), "error": m.get("error")})
                entries.append({"id": path.parent.name, "recording": info, "protocol": spec, "analyses": runs})
            except (OSError, KeyError, ValueError):
                continue
        return entries

    def state(self):
        with self.lock:
            job = dict(self.job)
        return {"capture": self.capture.state(), "job": job, "software": data.VERSION,
                "output": str(self.results), "entries": self.entries(), "workspace":self.workspace_state(), "remote":self.remote.snapshot()}

    def start_analysis(self, session):
        with self.lock:
            if self.job_thread and self.job_thread.is_alive():
                raise ValueError("현재 분석이 끝난 뒤 다시 시작해 주세요.")
            if self.capture.state()["recording"] or self.capture.saving:
                raise ValueError("녹화를 종료한 뒤 분석합니다.")
            folder = self.session(session)
            self.cancel = threading.Event()
            self.job = {"state": "running", "session": session, "done": 0, "total": 0, "message": "원본 무결성 확인 중"}
            self.job_thread = threading.Thread(target=self._analyze, args=(folder, self.cancel), daemon=True)
            self.job_thread.start()

    def _analyze(self, folder, cancel):
        def progress(done, total, run, message):
            with self.lock:
                self.job.update(done=done, total=total, run=run, message=message)
        try:
            run, result = data.analyze(folder, self.root / ".ai_venv/Scripts/python.exe",
                                       self.root / "models/topviewmouse/detector.pt", cancel, progress, self.predictor_factory)
            with self.lock:
                self.job.update(state=result["state"], run=run.name, message=result.get("error") or "분석 완료")
        except (Exception, SystemExit) as exc:
            with self.lock:
                self.job.update(state="failed", message=str(exc))

    def review(self, session, run_name, reviewer):
        run = self.run_path(session, run_name)
        manifest, rows = data.analysis_rows(run)
        self.check_review_source(session, manifest)
        plan = data.review_plan(rows)
        events = {e["sample"]: e for e in data.review_events(run) if e["reviewer"] == reviewer}
        # Deliberately omit AI labels from the blind review endpoint.
        return {"plan": [{"sample": i, "target_s": rows[i]["target_s"], "captured_s": rows[i]["captured_s"],
                          "review": events.get(i)} for i in plan], "protocol": manifest["protocol"],
                "validation": {"reviewers":[],"planned":len(plan),"validated":False}}

    def check_review_source(self, session, manifest):
        folder = self.session(session)
        for name, expected in manifest["source_sha256"].items():
            if name not in ("protocol.json", "frame_times.csv", "recovered_frame_times.csv") and not re.fullmatch(r"raw(?:_\d{6})?\.mp4",name):
                raise ValueError("지원하지 않는 원본 파일 정보입니다.")
            path = folder / name
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns, expected)
            if key not in self.verified_review_sources:
                if data.digest(path) != expected:
                    raise ValueError("원본이 분석 시점 이후 변경됐습니다: " + name)
                self.verified_review_sources[key] = True

    def summary_csv(self):
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["group", "cage", "session", "started_utc", "analysis", "analysis_state", "animals", "interval_s",
                         "criteria_sha256", "approach_pct_known", "mean_count_known", "known_samples", "unknown_samples", "total_samples"])
        for item in self.entries():
            for run in item["analyses"]:
                if not run["summary"]:
                    continue
                p, s = item["protocol"], run["summary"]
                safe = lambda v: "'" + v if v.lstrip().startswith(("=", "+", "-", "@")) else v
                writer.writerow([safe(p["group"]), safe(p["cage"]), item["id"], item["recording"]["started_utc"], run["id"],
                                 run["state"], p["criteria"]["animals"], p["criteria"]["interval_s"], p["criteria_sha256"],
                                 s["approach_pct"], s["mean_count"], s["known_samples"], s["unknown_samples"], s["samples"]])
        return output.getvalue().encode("utf-8-sig")

    def review_image(self, session, run_name, sample):
        run = self.run_path(session, run_name)
        manifest, rows = data.analysis_rows(run)
        self.check_review_source(session, manifest)
        if sample not in data.review_plan(rows):
            raise ValueError("검토 대상으로 선택된 표본이 아닙니다.")
        fid = rows[sample]["frame"]
        from .sources import open_video
        cap = open_video(self.session(session))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != fid + 1:
                raise ValueError("검토 영상을 읽지 못했습니다.")
        finally:
            cap.release()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = manifest["protocol"]["criteria"]["food_rect"]
        cv2.rectangle(frame, (round(x1*w), round(y1*h)), (round(x2*w), round(y2*h)), (80, 230, 100), 2)
        ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("검토 영상을 만들지 못했습니다.")
        return image.tobytes()

    def close(self):
        self.cancel.set()
        self.remote.close()
        self.capture.close()
        if self.job_thread:
            self.job_thread.join(timeout=10)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def make_server(app, port=0, token=None):
    token = token or secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value, mime="application/json; charset=utf-8", filename=None, headers=None):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") if not isinstance(value, bytes) else value
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + urllib.parse.quote(filename))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self, api=False, mutation=False):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected:
                self.respond(403, {"error": "로컬 주소로만 접근할 수 있습니다."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + expected:
                self.respond(403, {"error": "다른 웹사이트의 요청은 허용되지 않습니다."})
                return False
            if api and not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.respond(403, {"error": "실행하기.cmd로 화면을 다시 열어 주세요."})
                return False
            return True

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            api = url.path.startswith("/api/")
            if not self.allowed(api):
                return
            q = {key: values[0] for key, values in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/app.js", "/workflow.js", "/cage-activity.js", "/calibration.js", "/locomotion.js", "/remote.js", "/remote-view.js", "/style.css"):
                    name = "index.html" if url.path == "/" else url.path[1:]
                    return self.respond(200, (ASSETS / name).read_bytes(), mimetypes.guess_type(name)[0] + "; charset=utf-8")
                if url.path == "/api/state":
                    return self.respond(200, app.state())
                if url.path == "/api/remote-preview":
                    packet=app.remote.preview_packet(q.get('device'),q.get('after'))
                    headers={'X-Preview-Sequence':str(packet['sequence']),'X-Preview-FPS':f"{packet['preview_fps']:.2f}",
                             'X-Preview-Age':f"{packet['preview_age_s']:.3f}" if packet['preview_age_s'] is not None else '',
                             'X-Preview-Mode':packet['preview_mode']}
                    return self.respond(200 if packet['content'] else 204,packet['content'] or b'',"image/jpeg",headers=headers)
                if url.path == "/api/remote-file":
                    receipt=app.remote.receipt(q["id"])
                    path=app.remote.result_file(receipt["device"],q["id"],q["stage"],q["file"])
                    # Reports embed images and styles; serve as an attachment so the
                    # administrative origin never executes imported report markup.
                    return self.respond(200,path.read_bytes(),mimetypes.guess_type(path.name)[0],path.name)
                if url.path == "/api/frame":
                    return self.respond(200, app.raw_frame(q), "image/jpeg")
                if url.path == "/api/activity-frame":
                    return self.respond(200, app.activity_frame(q), "image/jpeg")
                if url.path in ("/api/activity-file","/api/locomotion-file"):
                    path = app.locomotion_file(q) if url.path=="/api/locomotion-file" else app.activity_file(q)
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(path.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(1024*1024), b""):
                            self.wfile.write(block)
                    return
                if url.path.startswith("/api/work/"):
                    return self.respond(200, app.workflow_get(url.path.rsplit("/",1)[-1],q))
                if url.path == "/api/preview":
                    packet = app.capture.preview(with_metadata=True)
                    return self.respond(200, packet[0], "image/jpeg", headers={"X-Preview-Frame": packet[1]}) if packet else self.respond(409, {"error": "카메라 준비 중"})
                if url.path == "/api/review":
                    return self.respond(200, app.review(q["session"], q["run"], q.get("reviewer", "")))
                if url.path == "/api/review-image":
                    return self.respond(200, app.review_image(q["session"], q["run"], int(q["sample"])), "image/jpeg")
                if url.path == "/api/summary.csv":
                    return self.respond(200, app.summary_csv(), "text/csv; charset=utf-8", "group_summary.csv")
                if url.path == "/api/download":
                    session = app.session(q["session"])
                    name = q["file"]
                    if name in ("protocol.json", "recording.json", "frame_times.csv", "raw.mp4"):
                        path = session / name
                    elif name in ("analysis.json", "samples.csv", "review_events.jsonl", "validation.csv", "validation.json",
                                  "interpretation.html", "interpretation.json", "interpretation_bins.csv", "interpretation_chart.png"):
                        run = app.run_path(q["session"], q["run"])
                        if name.startswith("interpretation") and not (run / name).exists():
                            if app.job.get("state") == "running":
                                raise ValueError("자동 결과 해석 생성이 끝날 때까지 기다려 주세요.")
                            from .interpretation import write_interpretation
                            write_interpretation(run)
                        if name == "validation.csv":
                            return self.respond(200, data.export_review_csv(run), "text/csv; charset=utf-8", name)
                        if name == "validation.json":
                            return self.respond(200, data.validation(run), filename=name)
                        path = run / name
                    else:
                        raise ValueError("이 파일은 제공하지 않습니다.")
                    # Stream recordings; do not load a multi-hour video into RAM.
                    size = path.stat().st_size
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Content-Disposition", "attachment; filename=" + name)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(1024*1024), b""):
                            self.wfile.write(block)
                    return
                self.respond(404, {"error": "찾을 수 없습니다."})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (Exception, SystemExit) as exc:
                self.respond(400, {"error": str(exc)})

        def do_POST(self):
            if not self.allowed(True, True):
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 1024*1024:
                    raise ValueError("요청 크기를 확인해 주세요.")
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict):
                    raise ValueError("요청 형식이 잘못됐습니다.")
                with app.lock:
                    if self.path.startswith("/api/remote/"):
                        command=self.path.rsplit("/",1)[-1]
                        if command=="start":result=app.remote.start(values["host"],int(values.get("port",8877)))
                        elif command=="pair":result=app.remote.pair(values.get("name",""))
                        elif command=="revoke":result=app.remote.revoke(values["device"])
                        elif command=="retry":result=app.remote.retry(values["id"])
                        else:raise ValueError("지원하지 않는 원격 서버 기능입니다.")
                        return self.respond(200,result)
                    if self.path.startswith("/api/work/"):
                        result=app.workflow(self.path.rsplit("/",1)[-1],values)
                        return self.respond(200,result)
                    if self.path == "/api/camera":
                        app.capture.open(values.get("camera"))
                    elif self.path == "/api/start":
                        if app.job_thread and app.job_thread.is_alive():
                            raise ValueError("분석이 끝난 뒤 다음 녹화를 시작해 주세요.")
                        app.capture.start(values)
                    elif self.path == "/api/stop":
                        session = app.capture.stop_recording()
                        if session and values.get("analyze", True):
                            app.start_analysis(session)
                    elif self.path == "/api/analyze":
                        app.start_analysis(values["session"])
                    elif self.path == "/api/cancel":
                        app.cancel.set()
                    elif self.path == "/api/review":
                        data.save_review(app.run_path(values["session"], values["run"]), values)
                    elif self.path == "/api/open-folder":
                        os.startfile(str(app.session(values["session"])))
                    elif self.path == "/api/shutdown":
                        self.respond(200, {"ok": True})
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                        return
                    else:
                        self.respond(404, {"error": "찾을 수 없습니다."})
                        return
                self.respond(200, {"ok": True})
            except (Exception, SystemExit) as exc:
                self.respond(400, {"error": str(exc)})

    server = LocalServer(("127.0.0.1", port), Handler)
    return server, token


def open_browser(url):
    candidates = [Path(os.environ.get(name, "")) / "Google/Chrome/Application/chrome.exe" for name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    for path in candidates:
        if path.is_file():
            subprocess.Popen([str(path), "--new-window", url])
            return
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--require-receiver", action="store_true",
                        help="원격 수신 서버를 시작할 수 없으면 오류로 종료")
    parser.add_argument("--receiver-host", help="원격 촬영을 수신할 이 컴퓨터의 VPN IPv4 주소")
    parser.add_argument("--receiver-port", type=int, default=8877)
    args = parser.parse_args()
    runtime_path = ROOT / ".research_runtime.json"
    # Reopen the same backend on repeated double clicks instead of competing for the camera.
    if runtime_path.is_file():
        try:
            old = data.read_json(runtime_path)
            request = urllib.request.Request(f'http://127.0.0.1:{int(old["port"])}/api/state', headers={"Authorization": "Bearer " + old["token"]})
            with urllib.request.urlopen(request, timeout=2) as response:
                state = json.load(response)
            if state.get("output") == str(ROOT / "results"):
                if args.require_receiver and not state.get("remote", {}).get("enabled"):
                    raise RuntimeError("실행 중인 BehavTrack에서 원격 수신 서버가 꺼져 있습니다.")
                if not args.no_browser:
                    open_browser(f'http://127.0.0.1:{int(old["port"])}/#token={old["token"]}')
                print("이미 실행 중인 연구 화면을 열었습니다.", flush=True)
                return
        except (OSError, ValueError, KeyError):
            pass
    settings_path = ROOT / "remote_server/settings.json"
    receiver = None
    if args.receiver_host:
        receiver = {"host": args.receiver_host, "port": args.receiver_port}
    elif settings_path.exists():
        receiver = data.read_json(settings_path)
    if args.require_receiver and not receiver:
        parser.error("저장된 원격 수신 설정을 찾을 수 없습니다.")

    app = Application()
    if receiver:
        try:
            app.remote.start(receiver["host"], int(receiver["port"]))
        except Exception as exc:
            if args.require_receiver:
                raise
            app.remote.error = str(exc)
    server, token = make_server(app)
    url = f"http://127.0.0.1:{server.server_port}/#token={token}"
    data.atomic_json(runtime_path, {"port": server.server_port, "token": token, "url": url})
    print("BehavTrack 연구 화면: " + url, flush=True)
    print("영상은 이 컴퓨터에 저장됩니다. 브라우저를 닫아도 녹화는 계속됩니다. 종료는 화면의 프로그램 종료 버튼을 사용하세요.", flush=True)
    if not args.no_browser:
        open_browser(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()
        if runtime_path.is_file() and data.read_json(runtime_path).get("token") == token:
            runtime_path.unlink()


if __name__ == "__main__":
    main()
