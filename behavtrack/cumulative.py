"""Compare completed camera sessions without mixing their individual measurements."""
from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

SUMMARY_DIR = "누적_분석"
REPORT_NAME = "누적_보고서.html"
CSV_NAME = "누적_결과.csv"

FIELDS = ["session", "created", "animal_id", "group", "track", "duration_s", "tracked_pct",
          "distance_cm", "distance_px", "mean_speed_cm_s", "food_time_s", "food_time_pct", "food_entries"]
HEADERS = ["촬영 결과", "저장 시각", "동물 ID", "군", "추적 ID", "관찰 시간 (초)", "추적 성공률 (%)",
           "이동거리 (cm)", "이동거리 (px)", "평균 속도 (cm/초)", "사료 구역 시간 (초)",
           "사료 구역 비율 (%)", "사료 구역 진입 횟수"]


def _text(value):
    if value is None:
        return "–"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}" if math.isfinite(value) else "–"
    return str(value) if str(value).strip() else "–"


def _atomic_write(path: Path, content: str, encoding: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="", dir=path.parent,
                                         delete=False, suffix=".tmp") as out:
            tmp = Path(out.name)
            out.write(content)
        os.replace(tmp, path)
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


def build_camera_summary(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("결과 폴더를 찾을 수 없습니다: " + str(root))
    rows, skipped = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "session.json").is_file():
            continue
        try:
            session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            if session.get("source", {}).get("type") != "camera":
                continue
            files = sorted(folder.glob("metrics_*.json"))
            if not files:
                skipped.append(f"{folder.name}: 분석 결과 없음")
                continue
            session_rows = []
            for metric_file in files:
                data = json.loads(metric_file.read_text(encoding="utf-8"))
                metrics, zones = data["metrics"], data.get("zones", {})
                food = zones.get("food")
                session_rows.append({
                    "session": folder.name, "created": session.get("created", ""),
                    "animal_id": session.get("animal_id", ""), "group": session.get("group", ""),
                    "track": data["track"], "duration_s": metrics.get("duration_s"),
                    "tracked_pct": metrics.get("tracked_pct"), "distance_cm": metrics.get("distance_cm"),
                    "distance_px": metrics.get("distance_px"), "mean_speed_cm_s": metrics.get("mean_speed_cm_s"),
                    "food_time_s": food.get("time_s") if food is not None else None,
                    "food_time_pct": food.get("time_pct") if food is not None else None,
                    "food_entries": food.get("entries") if food is not None else None,
                    "report": (folder / "report.html").resolve().as_uri() if (folder / "report.html").is_file() else "",
                })
            rows.extend(session_rows)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            skipped.append(f"{folder.name}: 읽을 수 없음 ({exc})")
    if not rows:
        detail = " 제외: " + "; ".join(skipped) if skipped else ""
        raise ValueError("완료된 웹캠 분석 결과가 없습니다." + detail)
    rows.sort(key=lambda row: (row["created"], row["session"], row["track"]))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(HEADERS)
    for row in rows:
        # Prefix text that spreadsheet programs could interpret as a formula.
        values = [row.get(key) if row.get(key) is not None else "" for key in FIELDS]
        writer.writerow(["'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for v in values])

    table = []
    for row in rows:
        cells = []
        for key in FIELDS:
            value = html.escape(_text(row.get(key)))
            if key == "session" and row["report"]:
                value = f'<a href="{html.escape(row["report"], quote=True)}">{value}</a>'
            cells.append(f"<td>{value}</td>")
        table.append("<tr>" + "".join(cells) + "</tr>")
    warning = "".join(f"<li>{html.escape(item)}</li>" for item in skipped)
    skipped_html = f"<h2>제외된 결과</h2><ul>{warning}</ul>" if skipped else ""
    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    food_values = [r["food_time_pct"] for r in rows if finite_number(r["food_time_pct"])
                   and finite_number(r["tracked_pct"]) and r["tracked_pct"] > 0]
    tracked_values = [r["tracked_pct"] for r in rows if finite_number(r["tracked_pct"])]
    def mean_caption(values):
        return f"{sum(values) / len(values):.1f}% <small>({len(values)}건)</small>" if values else "–"

    title = "웹캠 누적 분석"
    report = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px 'Malgun Gothic',sans-serif;color:#1c2530;max-width:1600px;margin:32px auto;padding:0 24px}}
h1{{margin-bottom:4px}} .muted{{color:#536273;line-height:1.6}} .scroll{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #dce3e8;padding:9px;text-align:left;white-space:nowrap}}th{{background:#edf3f4}}tr:nth-child(even){{background:#fafcfc}}
.notice{{background:#f4f8f7;padding:12px 16px;border-left:4px solid #288377}} a{{color:#176357}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}} .card{{background:#edf3f4;padding:14px 18px;min-width:190px;border-radius:6px}}
.card strong{{display:block;font-size:22px;margin-top:4px}} small{{font-size:13px;font-weight:normal}}</style></head><body>
<h1>{title}</h1><p class="muted">결과 폴더: {html.escape(str(root.resolve()))}<br>갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {len(rows)}개 추적 결과</p>
<div class="notice">각 행은 개별 촬영 결과입니다. 촬영 조건과 개체가 다르면 직접 비교에 주의하세요. 추적 성공률이 낮거나 촬영 시간이 다르면 함께 확인하세요.<br>
사료 구역은 몸 중심의 위치를 나타내며 실제 섭식량이나 섭식 여부가 아닙니다. 구역이 없거나 값이 없으면 –로 표시합니다.</div>
<div class="cards"><div class="card">촬영 결과<strong>{len(rows)}건</strong></div>
<div class="card">사료 구역 비율의 촬영별 평균<strong>{mean_caption(food_values)}</strong></div>
<div class="card">추적 성공률의 촬영별 평균<strong>{mean_caption(tracked_values)}</strong></div></div>
<p class="muted">평균은 유효한 각 행을 동일한 비중으로 계산한 단순 참고값입니다. 서로 다른 조건이나 같은 개체의 반복 촬영도 함께 포함될 수 있습니다.</p>
<p><a href="{html.escape(CSV_NAME, quote=True)}">엑셀용 CSV 다운로드</a> · 촬영 결과 이름을 클릭하면 개별 보고서를 엽니다.</p>
<div class="scroll"><table><thead><tr>{''.join('<th>' + html.escape(h) + '</th>' for h in HEADERS)}</tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p class="muted">사료 구역 비율은 추적된 시간 중 해당 구역에 있었던 비율입니다. 저장 시각은 녹화가 끝난 뒤 세션이 저장된 시각입니다.</p>{skipped_html}</body></html>"""
    out = root / SUMMARY_DIR
    _atomic_write(out / CSV_NAME, buffer.getvalue(), "utf-8-sig")
    _atomic_write(out / REPORT_NAME, report, "utf-8")
    return out / REPORT_NAME, out / CSV_NAME, len(rows), skipped


def build_group_summary(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("결과 폴더를 찾을 수 없습니다: " + str(root))
    columns = [("session", "촬영 결과"), ("created", "저장 시각"), ("cage_id", "케이지 ID"),
               ("group", "군"), ("animals", "마우스 수"), ("observed_s", "관찰 시간 (초)"),
               ("known_s", "추정 가능 시간 (초)"), ("unknown_s", "판정 불가 시간 (초)"),
               ("positive_pct", "한 마리 이상 접근 비율 (%)"), ("mean_approach_count", "평균 접근 마릿수"),
               ("camera_failed", "카메라 중단")]
    rows, skipped = [], []
    for folder in sorted(root.iterdir()):
        source = folder / "group_session.json"
        if not folder.is_dir() or not source.is_file():
            continue
        try:
            info = json.loads(source.read_text(encoding="utf-8"))
            if info["method"] != "experimental_background_blobs":
                raise ValueError("지원하지 않는 분석 방식")
            summary = info["summary"]
            rows.append({"session": folder.name, "created": info["created"], "cage_id": info["cage_id"],
                         "group": info["group"], "animals": info["animals"],
                         **{key: summary[key] for key in ("observed_s", "known_s", "unknown_s", "positive_pct", "mean_approach_count")},
                         "camera_failed": "예" if info.get("camera_failed") else "아니요",
                         "report": (folder / "report.html").resolve().as_uri() if (folder / "report.html").is_file() else ""})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            skipped.append(f"{folder.name}: 읽을 수 없음 ({exc})")
    if not rows:
        raise ValueError("완료된 공동 사육 웹캠 분석 결과가 없습니다.")
    rows.sort(key=lambda row: (row["created"], row["session"]))
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow([label for _, label in columns])
    for row in rows:
        values = [row[key] if row[key] is not None else "" for key, _ in columns]
        writer.writerow(["'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for v in values])
    table = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = html.escape(_text(row[key]))
            if key == "session" and row["report"]:
                value = f'<a href="{html.escape(row["report"], quote=True)}">{value}</a>'
            cells.append(f"<td>{value}</td>")
        table.append("<tr>" + "".join(cells) + "</tr>")
    skipped_html = "".join(f"<li>{html.escape(item)}</li>" for item in skipped)
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>공동 사육 누적 분석</title>
<style>body{{font:15px 'Malgun Gothic',sans-serif;margin:30px;color:#233}}.scroll{{overflow:auto}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:8px;white-space:nowrap}}th{{background:#eef4f4}}
.note{{background:#fff6dc;padding:14px;line-height:1.7}}a{{color:#176357}}</style>
<h1>공동 사육 누적 분석</h1><p>촬영 결과 {len(rows)}건</p><p class="note">빈 배경과의 차이를 사용한 미검증 추정입니다.
겹침이나 가림, 배경 변화 등은 수치를 왜곡할 수 있으며 추정 가능 시간이 정확성을 보장하지 않습니다.
사료 구역은 몸 중심 기준이며 실제 섭식이 아닙니다. 판정 불가 시간을 함께 확인하고 수동 채점으로 검증하세요.
같은 케이지의 반복 촬영은 서로 독립된 케이지가 아닙니다.</p>
<p><a href="공동사육_누적_결과.csv">엑셀용 CSV</a> · 촬영 결과 이름을 클릭하면 개별 보고서가 열립니다.</p>
<div class="scroll"><table><thead><tr>{''.join('<th>' + html.escape(label) + '</th>' for _, label in columns)}</tr></thead>
<tbody>{''.join(table)}</tbody></table></div><p>접근 비율과 평균은 각 촬영의 추정 가능 시간만 분모로 사용합니다.</p>
{'<h2>제외된 결과</h2><ul>' + skipped_html + '</ul>' if skipped else ''}</html>"""
    output = root / "공동사육_누적_분석"
    csv_path = output / "공동사육_누적_결과.csv"
    report_path = output / "공동사육_누적_보고서.html"
    _atomic_write(csv_path, buf.getvalue(), "utf-8-sig")
    _atomic_write(report_path, document, "utf-8")
    return report_path, csv_path, len(rows), skipped


def build_ai_group_summary(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("결과 폴더를 찾을 수 없습니다: " + str(root))
    columns = [("session", "촬영 결과"), ("created", "저장 시각"), ("cage_id", "케이지 ID"),
               ("group", "군"), ("samples", "분석 표본"), ("known_samples", "판정 가능 표본"),
               ("unknown_samples", "판정 불가 표본"), ("positive_pct_known", "접근 표본 비율 (%)"),
               ("mean_count_known", "평균 접근 마릿수"), ("error", "중단 오류")]
    rows, skipped = [], []
    for folder in sorted(root.iterdir()):
        source = folder / "ai_group_session.json"
        if not folder.is_dir() or not source.is_file():
            continue
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            if data["method"] != "superanimal_topviewmouse_cpu_sampled":
                raise ValueError("지원하지 않는 분석 방식")
            summary = data["summary"]
            rows.append({"session": folder.name, "created": data["created"], "cage_id": data["cage_id"],
                         "group": data["group"], **{key: summary[key] for key in
                         ("samples", "known_samples", "unknown_samples", "positive_pct_known", "mean_count_known")},
                         "error": data.get("error") or "", "report": (folder / "report.html").resolve().as_uri()
                         if (folder / "report.html").is_file() else ""})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            skipped.append(f"{folder.name}: 읽을 수 없음 ({exc})")
    if not rows:
        raise ValueError("완료된 AI 공동 사육 결과가 없습니다.")
    rows.sort(key=lambda row: (row["created"], row["session"]))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([label for _, label in columns])
    for row in rows:
        values = [row[key] if row[key] is not None else "" for key, _ in columns]
        writer.writerow(["'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for v in values])
    table = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = html.escape(_text(row[key]))
            if key == "session" and row["report"]:
                value = f'<a href="{html.escape(row["report"], quote=True)}">{value}</a>'
            cells.append(f"<td>{value}</td>")
        table.append("<tr>" + "".join(cells) + "</tr>")
    warning = "".join(f"<li>{html.escape(item)}</li>" for item in skipped)
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>AI 공동 사육 누적 결과</title>
<style>body{{font:15px 'Malgun Gothic',sans-serif;margin:30px;color:#233}}.scroll{{overflow:auto}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:8px;white-space:nowrap}}th{{background:#eef4f4}}
.note{{background:#fff6dc;padding:14px;line-height:1.7}}a{{color:#176357}}</style>
<h1>AI 공동 사육 누적 결과</h1><p>촬영 결과 {len(rows)}건</p><p class="note">AI가 표본 시점마다 다섯 마리의 코와 사료 구역을 비교한 미검증 추정입니다.
추론 속도에 따라 표본 간격이 달라질 수 있으며 실제 섭식이나 연속 행동 시간이 아닙니다. 판정 가능 표본만 분모로 사용합니다.
겹침·가림이나 오검출이 발생할 수 있으므로 원본 영상의 수동 채점과 비교하세요. 같은 케이지의 반복 촬영은 독립된 케이지가 아닙니다.</p>
<p><a href="AI_누적_결과.csv">엑셀용 CSV</a> · 촬영 결과 이름을 클릭하면 개별 보고서가 열립니다.</p>
<div class="scroll"><table><thead><tr>{''.join('<th>' + html.escape(label) + '</th>' for _, label in columns)}</tr></thead>
<tbody>{''.join(table)}</tbody></table></div>{'<h2>제외된 결과</h2><ul>' + warning + '</ul>' if skipped else ''}</html>"""
    output = root / "AI_공동사육_누적분석"
    csv_path, report_path = output / "AI_누적_결과.csv", output / "AI_누적_보고서.html"
    _atomic_write(csv_path, buffer.getvalue(), "utf-8-sig")
    _atomic_write(report_path, document, "utf-8")
    return report_path, csv_path, len(rows), skipped
