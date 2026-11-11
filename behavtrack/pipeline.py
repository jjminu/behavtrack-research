"""Running a session: track from a camera (live) or video files, then analyze and report."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import analysis
from . import background as bgmod
from . import draw, report
from .config import apply_overrides, arenas_from, complete
from .session import create_session_dir, find_sessions, load_session, save_session, tracks_path
from .sources import Recorder, camera_timestamps, open_camera, open_video, video_info
from .trackers import BlobTracker, make_tracker
from .tracks import Tracks, TrackWriter

WINDOW = "behavtrack"


def _writers(session_dir, arenas, tracker) -> dict[str, TrackWriter]:
    return {a.name: TrackWriter(tracks_path(session_dir, a.name), list(tracker.extra_fields),
                                list(tracker.keypoint_names)) for a in arenas}


def _overlay(frame, arenas, dets, trails, status, binary=None):
    vis = frame.copy()
    if binary is not None:
        mask = cv2.resize(binary, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        vis[mask] = (vis[mask] // 2 + np.array([0, 0, 127], np.uint8))
    draw.layout(vis, arenas)
    draw.detections(vis, dets, trails)
    draw.lines(vis, status)
    return vis


def _reference_from(tracker, frame) -> np.ndarray:
    h, w = frame.shape[:2]
    if isinstance(tracker, BlobTracker) and tracker.bg is not None:
        return cv2.cvtColor(cv2.resize(tracker.bg, (w, h)), cv2.COLOR_GRAY2BGR)
    return frame


# ---------------------------------------------------------------- video files

def track_video(cfg: dict, video, out_root, name: str | None = None, meta: dict | None = None,
                tracker: str = "blob", model=None, bodyparts=None, model_type=None, background=None,
                start_s: float = 0.0, end_s: float | None = None, display: bool = False,
                annotated: bool = False, analyze: bool = True) -> Path:
    meta = meta or {}
    cfg = apply_overrides(cfg, meta.get("overrides"))
    cap = open_video(video)
    info = video_info(cap)
    fps = info["fps"]
    timestamps = camera_timestamps(video, info["frames"])
    start_f = int(np.searchsorted(timestamps, start_s)) if timestamps is not None else int(round(start_s * fps))
    end_f = (int(np.searchsorted(timestamps, end_s)) if end_s is not None else len(timestamps)) if timestamps is not None else (int(round(end_s * fps)) if end_s is not None else (info["frames"] if info["frames"] > 0 else None))
    if start_s < 0 or (info["frames"] > 0 and start_f >= info["frames"]) or (end_f is not None and end_f <= start_f):
        raise SystemExit("영상 분석 시작/끝 시간이 유효하지 않습니다.")
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"영상의 첫 프레임을 읽을 수 없습니다: {video}")
    h, w = first.shape[:2]
    arenas = arenas_from(cfg)
    trk = make_tracker(tracker, arenas, cfg, (h, w), model, bodyparts, model_type)
    session_dir = create_session_dir(out_root, name or Path(video).stem)
    print(f"[{Path(video).name}] → {session_dir}")
    if trk.needs_background:
        if background:
            trk.set_background(bgmod.load_image(background))
        else:
            print("  빈 아레나 배경 추정 중...")
            trk.set_background(bgmod.from_video(video, trk, cfg["tracking"]["bg_samples"], start_f / fps,
                                                end_f / fps if end_f is not None else None))
        cv2.imwrite(str(session_dir / "background.png"), trk.bg)
    cv2.imwrite(str(session_dir / "reference.png"), _reference_from(trk, first))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    writers = _writers(session_dir, arenas, trk)
    rec = Recorder(session_dir / "tracked.mp4", fps) if annotated else None
    trails = [deque(maxlen=int(2 * fps)) for _ in arenas]
    i, last_print = start_f, time.perf_counter()
    try:
        while end_f is None or i < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            dets = trk.process(frame)
            t = float(timestamps[i] - timestamps[start_f]) if timestamps is not None else (i - start_f) / fps
            for a, d in zip(arenas, dets):
                writers[a.name].write(i, t, d)
            if rec or display:
                for tr, d in zip(trails, dets):
                    tr.append((int(d.x), int(d.y)) if d.found else None)
                vis = _overlay(frame, arenas, dets, trails, [f"{Path(video).name}  t={t:7.1f}s"])
                if rec:
                    rec.write(vis)
                if display:
                    cv2.imshow(WINDOW, vis)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        print("\n  사용자가 중단했습니다 (여기까지 저장).")
                        break
            i += 1
            if end_f and time.perf_counter() - last_print > 2:
                last_print = time.perf_counter()
                print(f"\r  추적 {100 * (i - start_f) / max(1, end_f - start_f):5.1f}%", end="", flush=True)
    finally:
        for wtr in writers.values():
            wtr.close()
        if rec:
            rec.close()
        cap.release()
        if display:
            cv2.destroyAllWindows()
    print(f"\r  추적 완료: {i - start_f} 프레임          ")
    source = {"type": "video", "path": str(Path(video).resolve()), "fps": fps, "start_frame": start_f,
              "frames": i - start_f, "width": w, "height": h, "tracker": tracker,
              "timebase": "camera_timestamps" if timestamps is not None else "video_fps",
              "background": "background.png" if trk.needs_background else None}
    save_session(session_dir, cfg, source, {a.name: {"arena": a.name} for a in arenas}, meta)
    if analyze:
        analyze_session(session_dir)
    return session_dir


# ---------------------------------------------------------------- live camera

class _LiveTrial:
    def __init__(self, out_root, name, arenas, tracker, fps, annotated):
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = create_session_dir(out_root, name or f"live_{stamp}")
        self.writers = _writers(self.dir, arenas, tracker)
        self.raw = Recorder(self.dir / "raw.mp4", round(fps))
        self.times_file = open(self.dir / "frame_times.csv", "w", newline="", encoding="utf-8")
        self.times = csv.writer(self.times_file)
        self.times.writerow(["frame", "time_s"])
        self.rec = Recorder(self.dir / "tracked.mp4", round(fps)) if annotated else None
        self.t0 = time.perf_counter()
        self.frames = 0
        self.elapsed = 0.0

    def close(self):
        for w in self.writers.values():
            w.close()
        self.raw.close()
        self.times_file.close()
        if self.rec:
            self.rec.close()


LIVE_HELP = """
  [실시간 추적 키]
    b       빈 아레나 배경 찍기 (동물을 넣기 전에!)
    s/space 녹화 시작 / 정지
    m       감지 마스크 보기 (빨간색 = 동물로 인식된 픽셀)
    + / -   감지 임계값 올리기 / 내리기
    q/Esc   종료 (녹화 중이면 저장 후 종료)
"""


def track_live(cfg: dict, camera: int, out_root, name: str | None = None, meta: dict | None = None,
               duration_s: float | None = None, resolution=None, background=None, auto_start: bool = False,
               tracker: str = "blob", model=None, bodyparts=None, model_type=None,
               annotated: bool = False, analyze: bool = True) -> Path | None:
    meta = meta or {}
    cfg = apply_overrides(cfg, meta.get("overrides"))
    cap = open_camera(camera, resolution)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("카메라에서 영상을 읽을 수 없습니다.")
    h, w = frame.shape[:2]
    arenas = arenas_from(cfg)
    trk = make_tracker(tracker, arenas, cfg, (h, w), model, bodyparts, model_type)
    blob = isinstance(trk, BlobTracker)
    if blob:
        trk.keep_debug = True
        if background:
            trk.set_background(bgmod.load_image(background))
    print(LIVE_HELP)
    if duration_s:
        print(f"  녹화는 시작 후 {duration_s:g}초 뒤 자동으로 끝납니다.")
    if auto_start:
        print("  배경이 준비되고 모든 아레나에서 동물이 보이면 자동으로 녹화를 시작합니다.")

    stamps = deque(maxlen=60)
    trails = [deque(maxlen=60) for _ in arenas]
    bg_frames, show_mask, trial = None, False, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못해 중단합니다.")
                break
            now = time.perf_counter()
            stamps.append(now)
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 and stamps[-1] > stamps[0] else 30.0
            if blob:
                gray = trk.preprocess(frame)
                if bg_frames is not None:
                    bg_frames.append(gray)
                    if len(bg_frames) >= 30:
                        trk.set_background(bgmod.median_of(bg_frames))
                        bg_frames = None
                        print("  배경 저장 완료. 이제 동물을 넣고 s 를 눌러 녹화를 시작하세요.")
                dets = trk.process_gray(gray)
            else:
                dets = trk.process(frame)
            if trial is None and auto_start and trk.ready and all(d.found for d in dets):
                trial = _LiveTrial(out_root, name, arenas, trk, fps, annotated)
                print(f"  녹화 시작 → {trial.dir}")
            if trial is not None:
                trial.elapsed = max(0.0, now - trial.t0)
                trial.raw.write(frame)
                trial.times.writerow([trial.frames, f"{trial.elapsed:.9f}"])
                for a, d in zip(arenas, dets):
                    trial.writers[a.name].write(trial.frames, trial.elapsed, d)
                trial.frames += 1
                if duration_s and trial.elapsed >= duration_s:
                    print("  설정한 시간이 끝나 녹화를 마칩니다.")
                    break
            for tr, d in zip(trails, dets):
                tr.append((int(d.x), int(d.y)) if d.found else None)
            if trial is not None:
                left = f"  left {max(0.0, duration_s - trial.elapsed):6.1f}s" if duration_s else ""
                state = f"REC {trial.elapsed:7.1f}s{left}"
            elif bg_frames is not None:
                state = "capturing background..."
            elif not trk.ready:
                state = "press b (empty arena) to capture background"
            else:
                state = "ready - put the animal in, press s to record"
            status = [state, f"fps {fps:4.1f}" + (f"   threshold {trk.threshold}" if blob else "")]
            vis = _overlay(frame, arenas, dets, trails, status, trk.last_binary if (blob and show_mask) else None)
            if trial is not None and trial.rec:
                trial.rec.write(vis)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b") and trial is None and blob:
                bg_frames = []
                print("  배경 촬영 중 (약 1초)... 아레나가 비어 있어야 합니다.")
            elif key in (ord("s"), ord(" ")):
                if trial is not None:
                    break
                if not trk.ready:
                    print("  먼저 b 키로 빈 아레나 배경을 찍어주세요.")
                else:
                    trial = _LiveTrial(out_root, name, arenas, trk, fps, annotated)
                    print(f"  녹화 시작 → {trial.dir}")
            elif key == ord("m"):
                show_mask = not show_mask
            elif blob and key in (ord("+"), ord("=")):
                trk.threshold = min(250, trk.threshold + 2)
            elif blob and key == ord("-"):
                trk.threshold = max(2, trk.threshold - 2)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if trial is not None:
            trial.close()
    if trial is None:
        print("녹화를 시작하지 않아 저장된 세션이 없습니다.")
        return None

    if blob:
        cfg["tracking"]["threshold"] = trk.threshold
        cv2.imwrite(str(trial.dir / "background.png"), trk.bg)
    cv2.imwrite(str(trial.dir / "reference.png"), _reference_from(trk, frame))
    source = {"type": "camera", "camera": camera, "fps": trial.frames / trial.elapsed if trial.elapsed else 0,
              "frames": trial.frames, "width": w, "height": h, "tracker": tracker, "video": "raw.mp4",
              "background": "background.png" if blob else None}
    save_session(trial.dir, cfg, source, {a.name: {"arena": a.name} for a in arenas}, meta)
    print(f"  저장 완료: {trial.frames} 프레임, {trial.elapsed:.1f}초")
    if analyze:
        analyze_session(trial.dir)
    return trial.dir


# ---------------------------------------------------------------- analysis

def behavior_files(session_dir, track_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Classifier outputs behavior_<name>_<track>.csv (frame, time_s, probability, label)."""
    out = {}
    suffix = f"_{track_id}.csv"
    for path in sorted(Path(session_dir).glob("behavior_*.csv")):
        if not path.name.endswith(suffix):
            continue
        name = path.name[len("behavior_"):-len(suffix)]
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out[name] = (np.array([int(r["frame"]) for r in rows]), np.array([int(r["label"]) if r["label"] else -1 for r in rows]))
    return out


def analyze_session(session_dir, cfg: dict | None = None, overrides=None, make_report: bool = True,
                    verbose: bool = True) -> dict:
    session_dir = Path(session_dir)
    sess = load_session(session_dir)
    base = complete(cfg) if cfg else complete(sess["config"])
    cfg = apply_overrides(base, list(sess.get("overrides", [])) + list(overrides or []))
    arenas = {a.name: a for a in arenas_from(cfg)}
    results = {}
    for tid, info in sess["tracks"].items():
        arena = arenas.get(info["arena"])
        if arena is None:
            raise SystemExit(f"설정에 아레나 '{info['arena']}'가 없습니다.")
        tracks = Tracks.load(tracks_path(session_dir, tid))
        res = analysis.analyze(tracks, arena, cfg, tid, behavior_files(session_dir, tid))
        res.save(session_dir)
        results[tid] = res
    pairs = analysis.social(results, cfg)
    if pairs:
        with open(session_dir / "social.json", "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
    if make_report:
        report.build(session_dir, sess, cfg, results, pairs)
    if verbose:
        for tid, res in results.items():
            print(report.console_summary(tid, res))
        if make_report:
            print(f"  리포트: {session_dir / 'report.html'}")
    return results


# ---------------------------------------------------------------- batch

def read_batch_list(path) -> list[dict]:
    """CSV with a 'video' column, optional animal_id / group / notes / start_s / end_s and any
    'section.key' columns (e.g. params.novel_object) used as per-video overrides."""
    items = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not (row.get("video") or "").strip():
                continue
            video = Path(row["video"].strip())
            if not video.is_absolute():
                video = Path(path).parent / video
            items.append({
                "video": video,
                "meta": {k: (row.get(k) or "").strip() for k in ("animal_id", "group", "notes")}
                | {"overrides": [f"{k}={v.strip()}" for k, v in row.items() if k and "." in k and (v or "").strip()]},
                "start_s": float(row["start_s"]) if (row.get("start_s") or "").strip() else 0.0,
                "end_s": float(row["end_s"]) if (row.get("end_s") or "").strip() else None,
            })
    return items


def run_batch(cfg: dict, items: list[dict], out_root, **kwargs) -> list[Path]:
    done = []
    for k, item in enumerate(items, 1):
        print(f"\n=== {k}/{len(items)} ===")
        name = item["meta"].get("animal_id") or Path(item["video"]).stem
        done.append(track_video(cfg, item["video"], out_root, name=name, meta=item["meta"],
                                start_s=item.get("start_s", 0.0), end_s=item.get("end_s"), **kwargs))
    report.summarize(done, Path(out_root) / "summary.csv")
    print(f"\n요약표: {Path(out_root) / 'summary.csv'}")
    return done


def summarize(paths, out_csv) -> Path:
    sessions = find_sessions(paths)
    if not sessions:
        raise SystemExit("세션 폴더를 찾지 못했습니다.")
    return report.summarize(sessions, out_csv)
