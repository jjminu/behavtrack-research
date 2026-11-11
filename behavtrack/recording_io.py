"""Segmented camera recordings and a single, verified session frame address space."""
from __future__ import annotations
import csv
import math
import os
import platform
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
from .integrity import atomic_json, read_json, digest, utcnow, contained_file

TIME_FIELDS = ["frame", "time_s", "segment", "segment_frame", "captured_utc", "host_utc"]


class ResearchRecording:
    def __init__(self, root, spec, camera, fps, segment_seconds=600, reserve_bytes=256*1024**2,
                 disk_usage=shutil.disk_usage):
        from .sources import Recorder
        if not math.isfinite(segment_seconds) or segment_seconds <= 0:
            raise ValueError("영상 조각 길이는 양수 초여야 합니다.")
        self.folder = Path(root) / ("research_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        self.folder.mkdir(parents=True, exist_ok=False)
        self.spec, self.disk_usage, self.reserve_bytes = spec, disk_usage, reserve_bytes
        self.segment_seconds, self.recorder_type = float(segment_seconds), Recorder
        self.info = {"schema": 2, "software": "research-2.0", "started_utc": utcnow(), "state": "recording",
                     "camera": camera, "frames": 0, "timestamp_source": "host_perf_counter_after_capture_read",
                     "video_fps": float(fps), "video_codec": "mp4v", "video_is_lossless": False,
                     "python": platform.python_version(), "opencv": cv2.__version__, "error": None,
                     "segments": [], "gaps": [], "segment_seconds": self.segment_seconds,
                     "reserve_bytes": reserve_bytes, "code_sha256": {n: digest(Path(__file__).parent/n)
                     for n in ("research.py", "research_web.py", "sources.py", "recording_io.py")}}
        self.start_monotonic = time.perf_counter()
        self.last_s = self.raw = self.segment = None
        self.closed = False
        self.last_disk_check = -math.inf
        atomic_json(self.folder / "protocol.json", spec)
        self.file = (self.folder / "frame_times.csv").open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(TIME_FIELDS)
        self._checkpoint()

    def _checkpoint(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        atomic_json(self.folder / "recording.json", self.info)

    def _close_segment(self):
        if self.raw is None:
            return
        self.raw.close()
        self.file.flush()
        os.fsync(self.file.fileno())
        s = self.segment
        path = self.folder / s["file"]
        # Finalization is committed only after both video and timestamp sidecar exist.
        with path.open("r+b") as f:
            os.fsync(f.fileno())
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        s.update(state="closed" if count == s["frames"] else "damaged", closed_utc=utcnow(),
                 sha256=digest(path), container_frames=count)
        atomic_json(self.folder / (s["file"] + ".json"), s)
        self.raw = None
        self._checkpoint()
        if s["state"] != "closed":
            raise ValueError(f"영상 저장 프레임 수 불일치: {s['file']}. 복구 점검이 필요합니다.")

    def record(self, frame, captured):
        if self.closed:
            raise ValueError("종료된 녹화에는 프레임을 추가할 수 없습니다.")
        elapsed = captured - self.start_monotonic
        if elapsed < 0:
            return
        if not math.isfinite(elapsed) or (self.last_s is not None and elapsed <= self.last_s):
            raise ValueError("촬영 시각이 증가하지 않습니다.")
        if list(frame.shape[1::-1]) != self.spec["criteria"]["frame_size"]:
            raise ValueError("녹화 중 카메라 해상도가 바뀌었습니다.")
        if elapsed - self.last_disk_check >= 1:
            self.info["free_bytes"] = self.disk_usage(self.folder).free
            self.last_disk_check = elapsed
            if self.info["free_bytes"] < self.reserve_bytes:
                raise OSError("저장공간 부족: 기존 자료를 보존하며 녹화를 종료합니다. 다른 저장장치를 준비하세요.")
        if self.segment and elapsed - self.segment["start_s"] >= self.segment_seconds:
            self._close_segment()
        if self.raw is None:
            number = len(self.info["segments"])
            name = "raw.mp4" if number == 0 else f"raw_{number:06d}.mp4"
            self.segment = {"index": number, "file": name, "start_frame": self.info["frames"],
                            "end_frame": None, "frames": 0, "start_s": elapsed, "end_s": None,
                            "start_utc": self._absolute(elapsed), "state": "open"}
            self.info["segments"].append(self.segment)
            self._checkpoint()  # Register the new segment BEFORE writing any frames.
            self.raw = self.recorder_type(self.folder / name, self.info["video_fps"])
        if self.last_s is not None and elapsed-self.last_s > max(.5, 3/self.info["video_fps"]):
            self.info["gaps"].append({"after_frame": self.info["frames"]-1, "start_s": self.last_s,
                                      "end_s": elapsed, "reason": "capture_gap"})
        self.raw.write(frame)
        self.writer.writerow([self.info["frames"], f"{elapsed:.9f}", self.segment["file"],
                              self.segment["frames"], self._absolute(elapsed), utcnow()])
        self.last_s = elapsed
        self.segment.update(end_frame=self.info["frames"], end_s=elapsed, end_utc=self._absolute(elapsed))
        self.segment["frames"] += 1
        self.info["frames"] += 1
        if self.info["frames"] % 30 == 0:
            self._checkpoint()

    def _absolute(self, elapsed):
        return (datetime.fromisoformat(self.info["started_utc"]) + timedelta(seconds=elapsed)).isoformat()

    def finish(self, error=None):
        if self.closed:
            return self.folder
        try:
            self._close_segment()
        except (Exception, SystemExit) as exc:
            error = f"{error or ''} {exc}".strip()
        finally:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            self.closed = True
        self.info.update(state="partial" if error or not self.info["frames"] else "complete",
                         error=error or (None if self.info["frames"] else "녹화된 프레임이 없습니다."),
                         ended_utc=utcnow(), last_capture_s=self.last_s)
        self.info["sha256"] = {n: digest(self.folder/n) for n in ("protocol.json", "frame_times.csv")}
        self.info["sha256"].update({s["file"]: s["sha256"] for s in self.info["segments"] if "sha256" in s})
        atomic_json(self.folder/"recording.json", self.info)
        return self.folder


def load_times(folder, verify=True):
    folder = Path(folder)
    info = read_json(folder / "recovery.json" if (folder/"recovery.json").exists() else folder/"recording.json")
    if info["state"] not in ("complete", "partial", "recovered") or not info["frames"]:
        raise ValueError("종료된 녹화가 필요합니다. 비정상 종료 세션은 먼저 복구 점검하세요.")
    if info.get("segments") and any(s["state"] not in ("closed", "recovered") for s in info["segments"]):
        raise ValueError("닫히지 않았거나 손상된 조각이 있습니다. 복구 점검하세요.")
    stamp_name = info.get("timestamps_file", "frame_times.csv")
    required = ["protocol.json", stamp_name] + [s["file"] for s in info.get("segments", [])]
    if not info.get("segments"):
        required.append("raw.mp4")
    for name in required:
        p = contained_file(folder, name)
        if name not in info.get("sha256", {}) or not p.is_file() or (verify and digest(p) != info["sha256"][name]):
            raise ValueError(f"원본 파일이 변경되었거나 손상됐습니다: {name}")
    try:
        with (folder/stamp_name).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        frames = np.array([int(r["frame"]) for r in rows])
        times = np.array([float(r["time_s"]) for r in rows])
        if (len(times) != info["frames"] or not np.array_equal(frames, np.arange(len(rows))) or
                not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0)):
            raise ValueError("프레임과 시간의 수 또는 순서 불일치")
        next_frame = 0
        for s in info.get("segments", []):
            if s["start_frame"] != next_frame or s["end_frame"] != next_frame + s["frames"] - 1:
                raise ValueError("조각 경계 프레임 중복/누락")
            for local, row in enumerate(rows[next_frame:next_frame+s["frames"]]):
                if row["segment"] != s["file"] or int(row["segment_frame"]) != local:
                    raise ValueError("조각 내부 프레임 대응 불일치")
            next_frame += s["frames"]
        if info.get("segments") and next_frame != len(rows):
            raise ValueError("조각 합계 불일치")
    except (KeyError, IndexError, OSError, ValueError) as exc:
        raise ValueError(f"{folder/stamp_name}: 촬영 시각 오류 ({exc}). FPS로 대체하지 않습니다.") from exc
    return info, times


def frame_reference(folder, frame, info=None):
    info = info or load_times(folder, verify=False)[0]
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < info["frames"]:
        raise ValueError("원본 프레임 범위를 확인하세요.")
    segments = info.get("segments") or [{"file": "raw.mp4", "start_frame": 0, "frames": info["frames"]}]
    for s in segments:
        if s["start_frame"] <= frame < s["start_frame"] + s["frames"]:
            return {"session_frame": int(frame), "segment": s["file"], "segment_frame": int(frame-s["start_frame"]),
                    "video_sha256": info["sha256"][s["file"]]}
    raise ValueError("프레임의 원본 영상 조각을 찾을 수 없습니다.")


def recover(folder):
    """Read/decode a durable prefix. Never rewrite any raw video or original timing file."""
    folder = Path(folder)
    if (folder/"recovery.json").exists():
        load_times(folder)
        return read_json(folder/"recovery.json")
    original = read_json(folder/"recording.json")
    if original.get("state") == "complete":
        load_times(folder)
        return original
    if not original.get("segments"):
        raise ValueError("이전 단일 영상의 비정상 종료는 자동 복구할 수 없습니다. 원본을 보존하세요.")
    with (folder/"frame_times.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    accepted, segments, report = [], [], []
    for declared in original["segments"]:
        s = dict(declared)
        p = contained_file(folder, s["file"])
        sidecar = folder/(s["file"]+".json")
        if sidecar.exists():
            s = read_json(sidecar)
        if s.get("sha256") and (not p.exists() or digest(p) != s["sha256"]):
            raise ValueError(f"닫힌 영상 조각이 변경됐습니다: {p}")
        cap = cv2.VideoCapture(str(p))
        decoded = 0
        while cap.isOpened() and cap.read()[0]:
            decoded += 1
        cap.release()
        matched = []
        for row in rows[len(accepted):len(accepted)+decoded]:
            try:
                previous = float(matched[-1]["time_s"]) if matched else (float(accepted[-1]["time_s"]) if accepted else -1)
                if (int(row["frame"]) != len(accepted)+len(matched) or row["segment"] != s["file"] or
                        int(row["segment_frame"]) != len(matched) or not math.isfinite(float(row["time_s"])) or
                        float(row["time_s"]) <= previous):
                    break
            except (ValueError, TypeError, KeyError):
                break
            matched.append(row)
        report.append({"file": s["file"], "decoded": decoded, "accepted": len(matched),
                       "unusable_decoded": decoded-len(matched), "original_state": s["state"]})
        if not matched:
            break
        s.update(start_frame=len(accepted), frames=len(matched), end_frame=len(accepted)+len(matched)-1,
                 start_s=float(matched[0]["time_s"]), end_s=float(matched[-1]["time_s"]),
                 state="closed" if s["state"] == "closed" and len(matched) == s["frames"] else "recovered", sha256=digest(p))
        segments.append(s)
        accepted.extend(matched)
        if len(matched) != decoded or declared["state"] != "closed" and not sidecar.exists():
            break
    if not accepted:
        atomic_json(folder/"recovery_failure.json", {"checked_utc": utcnow(), "segments": report, "usable_frames": 0})
        raise ValueError("복구 가능한 영상·타임스탬프 쌍이 없습니다. 원본은 보존했습니다.")
    with (folder/"recovered_frame_times.csv").open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, TIME_FIELDS)
        w.writeheader(); w.writerows(accepted)
        f.flush(); os.fsync(f.fileno())
    info = dict(original, state="recovered", frames=len(accepted), segments=segments,
                timestamps_file="recovered_frame_times.csv", recovery_report=report, recovered_utc=utcnow(),
                error="비정상 종료: 읽을 수 있고 실제 시각과 대응하는 프레임만 사용합니다.",
                original_recording_sha256=digest(folder/"recording.json"))
    info["sha256"] = {n: digest(folder/n) for n in ("protocol.json", "recovered_frame_times.csv")}
    info["sha256"].update({s["file"]:s["sha256"] for s in segments})
    atomic_json(folder/"recovery.json", info)
    load_times(folder)
    return info


class RecordingVideo:
    """cv2-like sequential reader; preserves global frame indexes across MP4 boundaries."""
    def __init__(self, folder, verify=True):
        self.folder = Path(folder)
        self.info, self.times = load_times(folder,verify=verify)
        self.position, self.cap, self.filename = 0, None, None
        self.size = read_json(self.folder/"protocol.json")["criteria"]["frame_size"]

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: self.info["frames"], cv2.CAP_PROP_FPS:self.info["video_fps"],
                cv2.CAP_PROP_POS_FRAMES:self.position, cv2.CAP_PROP_FRAME_WIDTH:self.size[0],
                cv2.CAP_PROP_FRAME_HEIGHT:self.size[1]}.get(prop, 0)

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.release()
        self.position = int(value)
        return True

    def read(self):
        if self.position >= self.info["frames"]:
            return False, None
        ref = frame_reference(self.folder, self.position, self.info)
        if ref["segment"] != self.filename:
            self.release()
            self.filename = ref["segment"]
            self.cap = cv2.VideoCapture(str(contained_file(self.folder, self.filename)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, ref["segment_frame"])
        ok, frame = self.cap.read()
        if not ok or round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) != ref["segment_frame"]+1:
            return False, None
        self.position += 1
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.filename = None, None
