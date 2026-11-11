"""Whole-frame, detector-only group approach observations and auditable recording."""
from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .session import create_session_dir
from .sources import Recorder

METHOD = "whole_frame_body_center_v1"


def evaluate(prediction, food_rect, size, expected, min_bbox_score=.6, duplicate_iou_threshold=.5):
    """Rectangle coordinates are fractions of the entire image; no cage crop."""
    unknown = {"count": None, "detected": 0, "status": "unknown", "reason": "invalid_detection"}
    try:
        boxes = np.asarray(prediction.get("boxes", []), dtype=float)
        scores = np.asarray(prediction.get("scores", []), dtype=float).reshape(-1)
        unknown["detected"] = len(scores)
        if len(scores) != expected:
            return dict(unknown, reason="count_mismatch")
        if (boxes.shape != (expected, 4) or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or np.any(scores < min_bbox_score) or np.any(scores > 1)):
            return unknown
        width, height = size
        if (np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(boxes[:, :2] < 0)
                or np.any(boxes[:, 2] > width) or np.any(boxes[:, 3] > height)):
            return unknown
        # Near duplicates are not independent animals, even if the count happens to match.
        for i, box in enumerate(boxes):
            for other in boxes[i + 1:]:
                inter = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])).prod()
                union = (box[2:] - box[:2]).prod() + (other[2:] - other[:2]).prod() - inter
                if inter / union > duplicate_iou_threshold:
                    return dict(unknown, reason="overlapping_detections")
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 / [width, height]
        x1, y1, x2, y2 = food_rect
        count = int(np.sum((centers[:, 0] >= x1) & (centers[:, 0] <= x2)
                           & (centers[:, 1] >= y1) & (centers[:, 1] <= y2)))
        return {"count": count, "detected": expected,
                "status": "approach" if count else "away", "reason": "estimated"}
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return unknown


def summary(samples):
    counts = [r["count"] for r in samples if r["count"] is not None]
    positive = sum(n > 0 for n in counts)
    return {"samples": len(samples), "known_samples": len(counts), "unknown_samples": len(samples) - len(counts),
            "approach_samples": positive, "away_samples": len(counts) - positive,
            "approach_pct": 100 * positive / len(counts) if counts else None,
            "mean_count": sum(counts) / len(counts) if counts else None}


class Observation:
    def __init__(self, root, group, animals, food_rect, size, fps, interval=5.0, camera=0):
        if (not isinstance(animals, int) or not 1 <= animals <= 20 or len(size) != 2 or min(size) <= 0
                or len(food_rect) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in food_rect)
                or not food_rect[0] < food_rect[2] or not food_rect[1] < food_rect[3]
                or not math.isfinite(interval) or interval <= 0):
            raise ValueError("마릿수와 사료 주변 표시를 확인하세요.")
        self.folder = create_session_dir(root, "group_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.food_rect, self.size, self.animals = list(food_rect), tuple(size), animals
        self.samples, self.frames, self.closed = [], 0, False
        self.info = {"method": METHOD, "created": datetime.now().isoformat(timespec="seconds"),
                     "group": group, "animals": animals, "camera": camera, "food_rect_normalized": self.food_rect,
                     "frame_size": self.size, "interval_s": interval, "state": "recording",
                     "criterion": "body_center_inside_food_surroundings", "observation_area": "entire_frame",
                     "model": "superanimal_topviewmouse_detector", "min_bbox_score": .6}
        self.raw = Recorder(self.folder / "raw.mp4", max(1, min(60, fps)))
        self.times_file = (self.folder / "frame_times.csv").open("w", newline="", encoding="utf-8")
        self.sample_file = (self.folder / "approach_samples.csv").open("w", newline="", encoding="utf-8-sig")
        self.times, self.writer = csv.writer(self.times_file), csv.writer(self.sample_file)
        self.times.writerow(["frame", "time_s"])
        self.writer.writerow(["frame", "time_s", "status", "approach_count", "detected", "reason", "detections_json"])
        self._save()

    def record(self, frame, elapsed):
        if tuple(frame.shape[1::-1]) != self.size:
            raise ValueError("촬영 중 영상 크기가 바뀌었습니다. 다시 시작해 주세요.")
        index = self.frames
        self.raw.write(frame)
        self.times.writerow([index, f"{elapsed:.9f}"])
        self.frames += 1
        if self.frames % 30 == 0:
            self.times_file.flush()
        return index

    def accept(self, fid, elapsed, prediction, reason=None):
        result = evaluate(prediction, self.food_rect, self.size, self.animals) if reason is None else {
            "count": None, "detected": "", "status": "unknown", "reason": reason}
        self.samples.append(result)
        self.writer.writerow([fid, f"{elapsed:.9f}", result["status"], result["count"], result["detected"],
                              result["reason"], json.dumps(prediction, ensure_ascii=True, allow_nan=False)])
        self.sample_file.flush()
        return result

    def _save(self):
        self.info.update(summary=summary(self.samples), source={"type": "camera", "video": "raw.mp4", "frames": self.frames})
        path = self.folder / "group_approach_session.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        self.raw.close()
        self.times_file.close()
        self.sample_file.close()
        self.info.update(state="partial" if error else "complete", error=error)
        self._save()
        write_report(self.folder / "report.html", [(self.folder, self.info)])


def write_report(path, entries):
    esc = lambda value: html.escape(str(value))
    number = lambda value: "—" if value is None else f"{value:.1f}"
    rows = []
    for folder, info in entries:
        s = info["summary"]
        link = esc(folder.resolve().as_uri())
        rows.append(f'<tr><td>{esc(info["created"])}</td><td>{esc(info["group"])}</td>'
                    f'<td>{info["animals"]}</td><td>{number(s["approach_pct"])}%</td>'
                    f'<td>{number(s["mean_count"])}</td><td>{s["approach_samples"]}</td><td>{s["away_samples"]}</td>'
                    f'<td>{s["unknown_samples"]} / {s["samples"]}</td><td>{esc(info.get("error") or "완료")}</td>'
                    f'<td><a href="{link}/raw.mp4">영상</a> · <a href="{link}/approach_samples.csv">표본 CSV</a></td></tr>')
    path.write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>그룹 사료 접근 결과</title>
<style>body{font:16px 'Malgun Gothic',sans-serif;color:#213547;max-width:1200px;margin:40px auto;padding:0 24px}
h1{color:#146956}p{line-height:1.8}.note{background:#eff7f4;padding:18px;border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dce5e1;text-align:left}
th{background:#eff7f4}a{color:#146956}.scroll{overflow:auto}</style><h1>그룹 사료 접근 결과</h1>
<p class="note">화면 전체에서 마우스를 찾고, 몸 중심이 표시한 사료 주변 안에 있는지 확인합니다.
접근 비율은 판정 가능한 표본 중 한 마리 이상 접근한 표본의 비율입니다. 연속 체류 시간이나 사료 섭취량은 아닙니다.
마릿수 불일치 등은 판정 불가로 남기며, 비접근으로 계산하지 않습니다. 오인식은 여전히 있을 수 있습니다.<br>
위에서 촬영한 마우스용 검출기를 사용합니다. 다른 각도·랫은 검증되지 않았습니다.
그룹 간 비교 시 촬영 조건과 사료 주변 표시 범위를 동일하게 맞추세요. 같은 케이지의 반복 관찰은 독립 표본이 아닙니다.</p>
<div class="scroll"><table><thead><tr><th>촬영 시각</th><th>그룹</th><th>전체 마릿수</th><th>접근 관측 비율</th>
<th>평균 주변 마릿수</th><th>접근 표본</th><th>비접근 표본</th><th>판정 불가 / 전체</th><th>상태</th><th>원자료</th>
</tr></thead><tbody>''' + "".join(rows) + "</tbody></table></div></html>", encoding="utf-8")


def build_summary(root):
    root = Path(root)
    entries = []
    for path in sorted(root.glob("*/group_approach_session.json")):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("method") == METHOD and info.get("state") in ("complete", "partial"):
                entries.append((path.parent, info))
        except (OSError, ValueError):
            continue
    if not entries:
        raise ValueError("아직 저장된 그룹 접근 결과가 없습니다. 관찰을 시작한 뒤 종료해 주세요.")
    out = root / "그룹_접근_결과"
    out.mkdir(exist_ok=True)
    report = out / "전체_결과.html"
    write_report(report, entries)
    with (out / "전체_결과.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["그룹", "촬영 시각", "전체 마릿수", "접근 관측 비율 (%)", "평균 주변 마릿수", "판정 불가 표본", "전체 표본"])
        for _, i in entries:
            s = i["summary"]
            writer.writerow(["'" + i["group"] if i["group"].lstrip().startswith(("=", "+", "-", "@")) else i["group"],
                             i["created"], i["animals"], s["approach_pct"], s["mean_count"], s["unknown_samples"], s["samples"]])
    return report
