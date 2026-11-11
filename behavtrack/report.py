"""Plots, the per-session HTML report, the console summary and the multi-session summary table."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .analysis import Result, write_rows  # noqa: E402
from .analysis.common import runs  # noqa: E402
from .session import load_session  # noqa: E402
from .templates import PARADIGMS  # noqa: E402

LABELS = {
    "duration_s": "분석 시간 (s)", "tracked_pct": "추적 성공률 (%)",
    "distance_cm": "총 이동거리 (cm)", "distance_px": "총 이동거리 (px)",
    "mean_speed_cm_s": "평균 속도 (cm/s)", "moving_speed_cm_s": "이동 중 평균 속도 (cm/s)",
    "moving_time_s": "이동 시간 (s)", "moving_pct": "이동 시간 비율 (%)",
    "freezing_time_s": "Freezing 시간 (s)", "freezing_pct": "Freezing 비율 (판정 가능 시간 대비, %)",
    "freezing_observed_s": "Freezing 판정 가능 시간 (s)", "freezing_unknown_s": "Freezing 판정 불가 시간 (s)",
    "freezing_bouts": "Freezing 횟수", "freezing_mean_bout_s": "Freezing 평균 지속 (s)",
    "freezing_latency_s": "첫 freezing까지 (s)", "freezing_method": "Freezing 판정 방식",
    "center_time_s": "중앙 체류시간 (s)", "center_time_pct": "중앙 체류 비율 (%)", "center_entries": "중앙 진입 횟수",
    "center_latency_s": "첫 중앙 진입까지 (s)", "center_distance_cm": "중앙 이동거리 (cm)",
    "center_distance_pct": "중앙 이동거리 비율 (%)", "periphery_time_s": "가장자리 체류시간 (s)",
    "periphery_time_pct": "가장자리 체류 비율 (%)",
    "open_time_s": "열린 팔 시간 (s)", "closed_time_s": "닫힌 팔 시간 (s)",
    "open_time_pct": "열린 팔 시간 % (열린/(열린+닫힌))", "open_entries": "열린 팔 진입", "closed_entries": "닫힌 팔 진입",
    "total_arm_entries": "총 팔 진입", "open_entries_pct": "열린 팔 진입 %", "open_latency_s": "첫 열린 팔 진입까지 (s)",
    "open_distance_cm": "열린 팔 이동거리 (cm)", "head_dips": "Head dip 횟수", "head_dip_time_s": "Head dip 시간 (s)",
    "arm_entries": "팔 진입 총수", "alternations": "교대 횟수", "alternation_pct": "자발적 교대율 (%)",
    "same_arm_returns": "같은 팔 재진입", "alternate_arm_returns": "다른 팔 거쳐 재진입", "arm_sequence": "팔 진입 순서",
    "exploration_method": "탐색 판정 방식", "total_explore_s": "총 물체 탐색시간 (s)",
    "discrimination_index": "변별지수 DI", "novel_preference_pct": "새 물체 선호 (%)",
    "time_to_criterion_s": "기준 탐색시간 도달 (s)", "discrimination_index_at_criterion": "기준 도달 시점 DI",
    "light_time_s": "밝은 칸 시간 (s)", "light_time_pct": "밝은 칸 비율 (%)", "dark_time_s": "어두운 칸 시간 (s)",
    "transitions": "칸 이동 횟수", "dark_entries": "어두운 칸 진입", "light_entries": "밝은 칸 진입",
    "dark_latency_s": "첫 어두운 칸 진입까지 (s)", "light_distance_cm": "밝은 칸 이동거리 (cm)",
    "social_side": "사회 자극 쪽", "social_chamber_time_s": "사회 자극 칸 시간 (s)",
    "other_chamber_time_s": "반대 칸 시간 (s)", "chamber_preference_index": "칸 선호 지수",
    "social_sniff_s": "사회 자극 컵 탐색 (s)", "other_sniff_s": "반대 컵 탐색 (s)",
    "sniff_preference_index": "탐색 선호 지수", "interaction_method": "상호작용 판정 방식",
    "chamber_index_A_vs_B": "칸 지수 (A−B)/(A+B)", "sniff_index_A_vs_B": "탐색 지수 (A−B)/(A+B)",
    "paired_chamber": "짝지은 칸", "preference_score_s": "선호 점수 (짝칸−반대칸, s)", "preference_ratio": "선호 비율",
    "found_target": "목표 도달", "latency_s": "도달 잠복기 (s)", "path_to_target_cm": "목표까지 경로 (cm)",
    "speed_to_target_cm_s": "목표까지 속도 (cm/s)", "mean_target_distance_cm": "목표와 평균 거리 (cm)",
    "target_crossings": "목표 통과 횟수", "target_quadrant": "목표 사분면", "target_quadrant_pct": "목표 사분면 체류 (%)",
    "opposite_quadrant_pct": "반대 사분면 체류 (%)", "thigmotaxis_pct": "벽 근처 수영 (%)",
    "mean_distance_cm": "두 동물 평균 거리 (cm)", "proximity_time_s": "근접 시간 (s)", "proximity_bouts": "근접 횟수",
    "nose_nose_time_s": "코-코 접촉 (s)", "nose_nose_bouts": "코-코 접촉 횟수",
    "a_sniffs_b_anogenital_s": "A가 B 항문생식기 탐색 (s)", "b_sniffs_a_anogenital_s": "B가 A 항문생식기 탐색 (s)",
}

PATTERNS = [
    (r"^(object_\w+)_explore_s$", "{} 탐색시간 (s)"), (r"^(object_\w+)_bouts$", "{} 탐색 횟수"),
    (r"^(object_\w+)_latency_s$", "{} 첫 탐색까지 (s)"), (r"^index_(\w+)$", "선호 지수 {}"),
    (r"^(object_\w+)_preference_pct$", "{} 선호 (%)"),
    (r"^chamber_(\w+)_time_s$", "{}칸 시간 (s)"), (r"^chamber_(\w+)_time_pct$", "{}칸 비율 (%)"),
    (r"^chamber_(\w+)_entries$", "{}칸 진입"), (r"^cup_(\w+)_sniff_s$", "{}컵 탐색 (s)"),
    (r"^cup_(\w+)_sniff_bouts$", "{}컵 탐색 횟수"), (r"^neutral_time_s$", "중립칸 시간 (s)"),
    (r"^freezing_pct_(\w+)$", "Freezing % ({})"), (r"^freezing_s_(\w+)$", "Freezing 시간 ({}, s)"),
    (r"^(Q_\w+)_pct$", "사분면 {} (%)"),
    (r"^(\w+)_time_s$", "{} 시간 (s)"), (r"^(\w+)_pct$", "{} (%)"), (r"^(\w+)_bouts$", "{} 횟수"),
    (r"^(\w+)_latency_s$", "첫 {}까지 (s)"),
]


def label(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    for pat, fmt in PATTERNS:
        m = re.match(pat, key)
        if m:
            return fmt.format(m.group(1))
    return key


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v):
            return "–"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


# ---------------------------------------------------------------- plots

def _box(res: Result, hw, margin: float = 0.08):
    pts = np.vstack([res.arena.shape.outline()] + [z.shape.outline() for z in res.arena.zones])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return max(0, x0 - mx), max(0, y0 - my), min(hw[1], x1 + mx), min(hw[0], y1 + my)


def _background(ax, ref, box):
    x0, y0, x1, y1 = box
    ax.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cmap="gray", vmin=0, vmax=255)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()


def _zones(ax, res: Result):
    o = res.arena.shape.outline()
    ax.plot(*np.vstack([o, o[:1]]).T, color="white", lw=1.2)
    colors = plt.get_cmap("tab10")
    for i, z in enumerate(res.arena.zones):
        o = z.shape.outline()
        ax.fill(*o.T, color=colors(i % 10), alpha=0.12)
        ax.plot(*np.vstack([o, o[:1]]).T, color=colors(i % 10), lw=0.8)
        c = z.shape.centroid()
        ax.text(c[0], c[1], z.name, color="white", fontsize=7, ha="center", va="center",
                bbox={"facecolor": colors(i % 10), "alpha": 0.7, "pad": 1, "lw": 0})


def plot_trajectory(ref, res: Result, path) -> None:
    p = res.prep
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    _background(ax, ref, _box(res, ref.shape))
    _zones(ax, res)
    if len(p.t) > 1:
        segs = np.stack([p.xy[:-1], p.xy[1:]], axis=1)
        ok = np.isfinite(segs).all(axis=(1, 2))
        lc = LineCollection(segs[ok], cmap="viridis", linewidths=1.0)
        lc.set_array((p.t[1:] - p.t[0])[ok])
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02, label="time (s)")
        fz = res.freeze & p.valid
        ax.scatter(p.xy[fz, 0], p.xy[fz, 1], s=2, color="red", alpha=0.3, label="freezing")
        if fz.any():
            ax.legend(loc="lower right", fontsize=7, markerscale=4)
    ax.set_title(f"{res.track_id} trajectory", fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(ref, res: Result, path) -> None:
    p = res.prep
    box = _box(res, ref.shape)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    _background(ax, ref, box)
    ok = p.valid
    if ok.any():
        b = max(2.0, p.scale) if p.cal.calibrated else 8.0  # ~1 cm bins
        x0, y0, x1, y1 = box
        hist, _, _ = np.histogram2d(p.xy[ok, 0], p.xy[ok, 1], bins=[np.arange(x0, x1 + b, b), np.arange(y0, y1 + b, b)],
                                    weights=p.dur[ok])
        hist = cv2.GaussianBlur(hist.T.astype(np.float32), (0, 0), 1.5)
        shown = np.ma.masked_less(hist, hist.max() * 0.02)
        im = ax.imshow(shown, extent=(x0, x0 + hist.shape[1] * b, y0 + hist.shape[0] * b, y0),
                       cmap="inferno", alpha=0.8, interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="time (s per bin)")
    o = res.arena.shape.outline()
    ax.plot(*np.vstack([o, o[:1]]).T, color="white", lw=1.0)
    for z in res.arena.zones:
        o = z.shape.outline()
        ax.plot(*np.vstack([o, o[:1]]).T, color="cyan", lw=0.6, alpha=0.7)
    ax.set_title(f"{res.track_id} occupancy heatmap", fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(res: Result, path) -> None:
    p = res.prep
    if len(p.t) == 0:
        return
    t = p.t - p.t[0]
    rows = [(f"in {k}", v) for k, v in res.states.items()] + list(res.behaviors.items())
    fig, axes = plt.subplots(2 if rows else 1, 1, figsize=(10, 4.2 if rows else 2.6), dpi=110, sharex=True,
                             gridspec_kw={"height_ratios": [2, max(1, 0.35 * len(rows))]} if rows else None)
    ax = axes[0] if rows else axes
    speed = p.speed
    finite = speed[np.isfinite(speed)]
    top = max(float(np.percentile(finite, 99.5)) * 1.1, 1.0) if len(finite) else 1.0
    ax.plot(t, speed, lw=0.5, color="0.2")
    ax.fill_between(t, 0, top, where=res.freeze, step="post", color="tab:blue", alpha=0.25, lw=0, label="freezing")
    ax.fill_between(t, 0, top, where=~res.freeze_observed, step="post", color="tab:orange", alpha=0.3, lw=0, label="unknown")
    ax.set_ylim(0, top)
    ax.set_ylabel(f"speed ({p.units}/s)")
    ax.legend(loc="upper right", fontsize=7)
    if rows:
        ax2 = axes[1]
        cmap = plt.get_cmap("tab10")
        for k, (name, mask) in enumerate(rows):
            spans = [(t[a], t[b - 1] - t[a] + p.dur[b - 1]) for a, b in runs(mask)]
            ax2.broken_barh(spans, (k - 0.4, 0.8), color=cmap(k % 10))
        ax2.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
        ax2.set_ylim(-0.6, len(rows) - 0.4)
        ax2.invert_yaxis()
        ax2.set_xlabel("time (s)")
    else:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report

CSS = """
body{font-family:-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;margin:24px;color:#222;max-width:1200px}
h1{font-size:22px} h2{font-size:18px;border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:32px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px} td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}
th{background:#f4f4f4} td:first-child,th:first-child{text-align:left}
.meta td{text-align:left} .imgs{display:flex;flex-wrap:wrap;gap:12px} .imgs img{max-width:560px;width:100%;border:1px solid #eee}
.wide img{max-width:1100px} .note{color:#666;font-size:12px}
"""


def _table(rows, header=None, cls="") -> str:
    out = [f'<table class="{cls}">']
    if header:
        out.append("<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(fmt(c) if not isinstance(c, str) else c)}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "".join(out)


def build(session_dir, sess: dict, cfg: dict, results: dict[str, Result], pairs: dict) -> Path:
    d = Path(session_dir)
    ref = cv2.imread(str(d / "reference.png"))
    par = PARADIGMS.get(cfg.get("paradigm", "custom"))
    src = sess.get("source", {})
    meta_rows = [
        ("실험 종류", par.title if par else cfg.get("paradigm")),
        ("동물 ID", sess.get("animal_id") or "–"), ("그룹", sess.get("group") or "–"),
        ("입력", src.get("path") or f"카메라 {src.get('camera')} ({src.get('video', '')})"),
        ("기록 시각", sess.get("created", "")), ("추적 방식", src.get("tracker", "")),
        ("fps", fmt(src.get("fps"))), ("설정 덮어쓰기", ", ".join(sess.get("overrides", [])) or "–"),
    ]
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(d.name)}</title>"
             f"<style>{CSS}</style></head><body><h1>행동 분석 리포트 — {html.escape(d.name)}</h1>",
             _table(meta_rows, cls="meta")]
    if par and par.description:
        parts.append(f"<p class='note'>{html.escape(par.description)}</p>")
    for tid, res in results.items():
        if ref is not None:
            plot_trajectory(ref, res, d / f"trajectory_{tid}.png")
            plot_heatmap(ref, res, d / f"heatmap_{tid}.png")
        plot_timeline(res, d / f"timeline_{tid}.png")
        parts.append(f"<h2>{html.escape(tid)} (아레나 {html.escape(res.arena.name)})</h2>")
        if not res.prep.cal.calibrated:
            parts.append("<p class='note'>⚠ 보정(cm)이 없어 거리 단위가 픽셀입니다.</p>")
        if res.metrics.get("tracked_pct", 100) < 90:
            parts.append("<p class='note'>⚠ 추적 성공률이 90% 미만입니다. 조명·대비·임계값을 확인하세요 (tracked.mp4 로 확인 가능).</p>")
        if res.metrics.get("freezing_unknown_s", 0) > 0:
            parts.append("<p class='note'>⚠ Freezing 비율은 판정 가능한 시간만 분모로 사용합니다. 판정 불가 시간을 함께 확인하세요.</p>")
        if res.metrics.get("undetected_handling") == "dark":
            parts.append("<p class='note'>⚠ 미검출 구간을 어두운 칸으로 가정했습니다. 추적 실패도 포함될 수 있습니다.</p>")
        if res.paradigm in ("nor", "three_chamber"):
            parts.append("<p class='note'>⚠ 물체 근접·방향은 실제 냄새 맡기나 탐색의 대리지표입니다. 수동 채점과 비교하세요.</p>")
        parts.append(_table([(label(k), v) for k, v in res.metrics.items()], ["지표", "값"]))
        if res.zones:
            u = res.prep.units
            parts.append("<h3>구역별</h3>")
            parts.append(_table(
                [(z, s["time_s"], s["time_pct"], s["entries"], s["latency_s"], s[f"distance_{u}"], s["mean_visit_s"])
                 for z, s in res.zones.items()],
                ["구역", "시간 (s)", "비율 (%)", "진입", "첫 진입 (s)", f"이동거리 ({u})", "평균 방문 (s)"]))
        if res.bins:
            keys = list(res.bins[0])
            parts.append("<h3>시간 구간별</h3>")
            parts.append(_table([[b.get(k) for k in keys] for b in res.bins], keys))
        imgs = [f"trajectory_{tid}.png", f"heatmap_{tid}.png"]
        parts.append("<div class='imgs'>" + "".join(f"<img src='{i}'>" for i in imgs if (d / i).exists()) + "</div>")
        parts.append(f"<div class='imgs wide'><img src='timeline_{tid}.png'></div>")
    if pairs:
        parts.append("<h2>동물 간 상호작용</h2>")
        for pair, m in pairs.items():
            parts.append(f"<h3>{html.escape(pair)}</h3>" + _table([(label(k), v) for k, v in m.items()], ["지표", "값"]))
    files = ", ".join(sorted(p.name for p in d.iterdir() if p.suffix in (".csv", ".json", ".mp4")))
    parts.append(f"<p class='note'>폴더의 파일: {html.escape(files)}</p></body></html>")
    out = d / "report.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out


def console_summary(tid: str, res: Result) -> str:
    m, u = res.metrics, res.prep.units
    keys = [f"distance_{u}", "freezing_pct"]
    common = {"duration_s", "tracked_pct", f"distance_{u}", f"mean_speed_{u}_s", f"moving_speed_{u}_s",
              "moving_time_s", "moving_pct", "freezing_time_s", "freezing_pct", "freezing_bouts",
              "freezing_mean_bout_s", "freezing_latency_s", "freezing_method"}
    keys += [k for k in m if k not in common][:5]
    body = ", ".join(f"{label(k)}={fmt(m.get(k))}" for k in keys)
    return f"  [{tid}] 추적 {fmt(m['tracked_pct'])}% | {body}"


def summarize(session_dirs, out_csv) -> Path:
    rows = []
    for sd in map(Path, session_dirs):
        sess = load_session(sd)
        for mf in sorted(sd.glob("metrics_*.json")):
            data = json.loads(mf.read_text(encoding="utf-8"))
            row = {"session": sd.name, "animal_id": sess.get("animal_id", ""), "group": sess.get("group", ""),
                   "track": data["track"], "arena": data["arena"], "paradigm": data["paradigm"],
                   "source": sess.get("source", {}).get("path") or sess.get("source", {}).get("video", "")}
            row.update(data["metrics"])
            for z, stats in data.get("zones", {}).items():
                for k, v in stats.items():
                    row[f"zone_{z}_{k}"] = v
            rows.append(row)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out_csv, rows)
    return out_csv
