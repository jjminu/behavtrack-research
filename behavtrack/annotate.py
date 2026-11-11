"""Frame-by-frame behavior annotation (OpenCV window), used to train classifiers.

Keys: space play/pause · , . one frame back/forward · [ ] one second · { } ten seconds
      1-9 start/stop the numbered behavior at the current frame · u undo last interval
      s save · q save and quit
Output CSV: behavior,start_frame,end_frame,start_s,end_s,track
plus 'reviewed_v2' rows for frames displayed (skipped frames are not used as negatives).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from . import draw
from .classifier import REVIEWED
from .sources import open_video, video_info, camera_timestamps, frame_offset

WINDOW = "behavtrack annotate"


def _load(path) -> tuple[list[tuple[str, int, int]], set[int]]:
    rows, reviewed = [], set()
    if Path(path).exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["behavior"] == "reviewed":
                    raise SystemExit("기존 reviewed 라벨은 건너뛴 프레임을 포함할 수 있습니다. 새 파일로 다시 채점하세요.")
                if r["behavior"] == REVIEWED:
                    reviewed.update(range(int(r["start_frame"]), int(r["end_frame"]) + 1))
                else:
                    rows.append((r["behavior"], int(r["start_frame"]), int(r["end_frame"])))
    return rows, reviewed


def _save(path, intervals, reviewed: set[int], fps: float, track: str, times=None, offset=0) -> None:
    stamp=lambda n: float(times[n]) if times is not None else n/fps
    from .integrity import append_event, utcnow
    append_event(str(path)+".history.jsonl", {"utc":utcnow(),"track":track,"intervals":intervals,
                                            "reviewed_frames":sorted(reviewed),"frame_offset":offset})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["behavior", "start_frame", "end_frame", "start_s", "end_s", "track"])
        for b, a, e in sorted(intervals, key=lambda r: r[1]):
            w.writerow([b, a+offset, e+offset, f"{stamp(a):.9f}", f"{stamp(e):.9f}", track])
        values = sorted(reviewed)
        if values:
            start = previous = values[0]
            for current in values[1:] + [None]:
                if current is None or current != previous + 1:
                    w.writerow([REVIEWED, start+offset, previous+offset, f"{stamp(start):.9f}", f"{stamp(previous):.9f}", track])
                    start = current
                previous = current


def annotate(video, behaviors: list[str], out_csv, track: str = "") -> Path:
    if not 1 <= len(behaviors) <= 9:
        raise SystemExit("행동은 1~9개까지 지정할 수 있습니다 (예: --behaviors grooming,rearing).")
    cap = open_video(video)
    info = video_info(cap)
    fps, n = info["fps"], max(1, info["frames"])
    times=camera_timestamps(video,n)
    offset=frame_offset(video)
    from .integrity import digest,read_json,atomic_json,append_event,utcnow
    import getpass
    source_path=Path(video)
    if source_path.is_dir():
        from .recording_io import load_times
        hashes=load_times(source_path)[0]["sha256"]
    else:
        hashes={source_path.name:digest(source_path)}
    provenance_path=Path(str(out_csv)+".source.json")
    if provenance_path.exists():
        previous=read_json(provenance_path)
        if previous["source_sha256"]!=hashes or previous["track"]!=track:
            cap.release()
            raise SystemExit("기존 라벨의 원본 영상 또는 동물 트랙과 다릅니다. 새 라벨 파일을 지정하세요.")
    atomic_json(provenance_path,{"source":str(source_path.resolve()),"source_sha256":hashes,"track":track,
                               "author":getpass.getuser(),"author_source":"windows_account","updated_utc":utcnow(),
                               "timebase":"camera_timestamps" if times is not None else "video_fps","frame_offset":offset,
                               "research_reference":"웹 개체 정답 화면으로 독립 판독 여부를 기록해야 최종 평가에 사용할 수 있습니다."})
    intervals, reviewed = _load(out_csv)
    if offset:
        intervals=[(b,a-offset,e-offset) for b,a,e in intervals]
        reviewed={r-offset for r in reviewed}
    open_marks: dict[str, int] = {}
    pos, cur, frame, playing = 0, -1, None, False
    colors = draw.ZONE_COLORS
    print(__doc__)
    while True:
        if pos != cur:
            if pos != cur + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, img = cap.read()
            if not ok:
                pos, playing = max(0, cur), False
                if frame is None:
                    raise SystemExit("영상을 읽을 수 없습니다.")
            else:
                frame, cur = img, pos
        reviewed.add(cur)
        vis = frame.copy()
        h, w = vis.shape[:2]
        now=float(times[cur]) if times is not None else cur/fps
        rows = [f"frame {cur+offset}/{n - 1+offset}  t={now:7.2f}s  {'PLAY' if playing else 'PAUSE'}"]
        for i, b in enumerate(behaviors):
            rows.append(f"{i + 1}: {b}" + (f"  <ON since {open_marks[b]}>" if b in open_marks else ""))
        draw.lines(vis, rows)
        bar_y = h - 18
        cv2.rectangle(vis, (0, bar_y), (w, h), (40, 40, 40), -1)
        for b, a, e in intervals:
            c = colors[behaviors.index(b) % len(colors)] if b in behaviors else (128, 128, 128)
            cv2.rectangle(vis, (int(a / n * w), bar_y + 2), (max(int(e / n * w), int(a / n * w) + 1), h - 2), c, -1)
        cv2.line(vis, (int(cur / n * w), bar_y - 4), (int(cur / n * w), h), (0, 0, 255), 2)
        cv2.imshow(WINDOW, vis)
        delay=(times[min(cur+1,n-1)]-times[cur])*1000 if times is not None and cur<n-1 else 1000/fps
        key = cv2.waitKey(max(1, min(2000,int(delay))) if playing else 30) & 0xFF
        step = {ord(","): -1, ord("."): 1, ord("["): -int(fps), ord("]"): int(fps),
                ord("{"): -int(10 * fps), ord("}"): int(10 * fps)}.get(key)
        if step:
            pos, playing = int(np.clip(cur + step, 0, n - 1)), False
        elif key == ord(" "):
            playing = not playing
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(behaviors):
            b = behaviors[key - ord("1")]
            if b in open_marks:
                a = open_marks.pop(b)
                intervals.append((b, min(a, cur), max(a, cur)))
                print(f"  {b}: {min(a, cur)}–{max(a, cur)}")
            else:
                open_marks[b] = cur
        elif key == ord("u") and intervals:
            print(f"  취소: {intervals.pop()}")
        elif key == ord("s"):
            _save(out_csv, intervals, reviewed, fps, track, times, offset)
            print(f"  저장: {out_csv}")
        elif key in (ord("q"), 27):
            break
        if playing:
            if cur + 1 >= n:
                playing = False
            else:
                pos = cur + 1
    cap.release()
    cv2.destroyAllWindows()
    for b, a in open_marks.items():
        intervals.append((b, min(a, cur), max(a, cur)))
    _save(out_csv, intervals, reviewed, fps, track, times, offset)
    print(f"라벨 저장: {out_csv} ({len(intervals)}개 구간)")
    return Path(out_csv)
