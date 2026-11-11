"""Camera and video file input, and video recording."""

from __future__ import annotations

import sys
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def open_camera(index: int, resolution: tuple[int, int] | None = None) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"카메라 {index}번을 열 수 없습니다. `behavtrack cameras`로 번호를 확인하고, "
            "macOS라면 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널 접근을 허용해주세요."
        )
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    for _ in range(10):  # let auto exposure settle
        cap.read()
    return cap


def open_video(path) -> cv2.VideoCapture:
    if Path(path).is_dir() and (Path(path) / "recording.json").exists():
        from .recording_io import RecordingVideo
        return RecordingVideo(path)
    if not Path(path).exists():
        raise SystemExit(f"영상 파일이 없습니다: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {path}")
    return cap


def video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def camera_timestamps(video, frame_count: int) -> np.ndarray | None:
    """Read original capture times when video is the raw file of a camera session."""
    video = Path(video)
    folder = video if video.is_dir() else video.parent
    if (folder / "recording.json").exists():
        from .recording_io import load_times
        info, times = load_times(folder)
        if not video.is_dir():
            segments = info.get("segments") or [{"file": "raw.mp4", "start_frame": 0, "frames": info["frames"]}]
            segment = next((s for s in segments if s["file"] == video.name), None)
            if segment is None:
                raise ValueError("촬영 세션에 등록되지 않은 영상입니다. 원본 연결을 확인하세요.")
            times = times[segment["start_frame"]:segment["start_frame"]+segment["frames"]]
        if len(times) != frame_count:
            raise ValueError("촬영 영상과 실제 시간의 프레임 수가 다릅니다. FPS로 대체하지 않습니다.")
        return times  # session-relative time; NEVER reset each segment to zero
    meta = video.parent / "session.json"
    if not meta.exists():
        meta = video.parent / "group_session.json"
    if not meta.exists():
        meta = video.parent / "ai_group_session.json"
    if not meta.exists():
        meta = video.parent / "group_approach_session.json"
    if not meta.exists():
        if (video.parent / "frame_times.csv").exists():
            raise SystemExit("촬영 시각 파일은 있으나 세션 정보가 없습니다. 원본 메타데이터를 복원하세요.")
        return None
    with meta.open(encoding="utf-8") as f:
        session = json.load(f)
    source = session.get("source", {})
    if source.get("type") != "camera" or source.get("video") != video.name:
        return None
    stamps_path = video.parent / "frame_times.csv"
    if not stamps_path.exists():
        tracks = session.get("tracks", {})
        if not tracks:
            raise SystemExit("카메라 영상의 원본 타임스탬프를 찾을 수 없습니다.")
        stamps_path = video.parent / f"tracks_{next(iter(tracks))}.csv"
    try:
        with stamps_path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        frames = np.array([int(r["frame"]) for r in rows])
        times = np.array([float(r["time_s"]) for r in rows])
    except (OSError, KeyError, ValueError) as e:
        raise SystemExit(f"카메라 타임스탬프를 읽을 수 없습니다: {e}") from e
    if (not len(times) or len(times) != frame_count or len(times) != source.get("frames")
            or not np.array_equal(frames, np.arange(len(times))) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise SystemExit("카메라 영상과 타임스탬프의 프레임 수 또는 순서가 일치하지 않습니다.")
    return times - times[0]


def frame_at(path, t_s: float = 0.0):
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{t_s}초 위치의 프레임을 읽을 수 없습니다: {path}")
    return frame


def list_cameras(max_index: int = 6) -> list[dict]:
    found = []
    for i in range(max_index):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                found.append({"index": i, "width": frame.shape[1], "height": frame.shape[0],
                              "fps": cap.get(cv2.CAP_PROP_FPS)})
        cap.release()
    return found


class Recorder:
    """mp4 writer that opens on the first frame (so the size is known)."""

    def __init__(self, path, fps: float):
        self.path, self.fps, self.writer = Path(path), float(fps), None
        self.size, self.closed = None, False
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("저장 FPS가 올바르지 않습니다.")
        if self.path.exists():
            raise FileExistsError(f"기존 원본은 덮어쓰지 않습니다: {self.path}")

    def write(self, frame) -> None:
        if self.closed:
            raise ValueError("종료된 영상입니다.")
        size = tuple(frame.shape[1::-1])
        if self.size is not None and self.size != size:
            raise ValueError("녹화 중 해상도가 바뀌었습니다.")
        if self.writer is None:
            # Exclusive reservation prevents races with another writer.
            with self.path.open("xb"):
                pass
            self.size = size
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise SystemExit(f"영상 저장 파일을 만들 수 없습니다: {self.path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.closed = True


def frame_offset(video):
    video = Path(video)
    if video.is_file() and (video.parent / "recording.json").exists():
        from .recording_io import load_times
        info, _ = load_times(video.parent, verify=False)
        return next((s["start_frame"] for s in info.get("segments", []) if s["file"] == video.name), 0)
    return 0
