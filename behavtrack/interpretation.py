"""Descriptive follow-up reports bound to the immutable analysis samples."""
from __future__ import annotations

import base64
import csv
import html
import math
from collections import Counter
from pathlib import Path

from .group_approach import summary
from .integrity import atomic_json, digest, utcnow

REASONS = {"count_mismatch": "예상 마릿수와 검출 수 불일치", "invalid_detection": "상자·신뢰도 조건 불충족",
           "overlapping_detections": "검출 상자 중복", "capture_gap": "해당 시각의 촬영 프레임 부족",
           "not_analyzed": "미분석"}


def describe(rows, end_s, interval_s, bin_seconds=60):
    """Summarize sample proportions; never interpolate visits or behavior duration."""
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("요약 구간은 양수 초여야 합니다.")
    if not math.isfinite(end_s) or end_s < 0 or interval_s <= 0:
        raise ValueError("관측 범위와 분석 간격을 확인하세요.")
    for row in rows:
        status, count = row["status"], row["count"]
        if status not in ("approach", "away", "unknown"):
            raise ValueError("알 수 없는 표본 판정입니다.")
        if (status == "unknown") != (count is None) or status == "away" and count != 0 or status == "approach" and count <= 0:
            raise ValueError("표본 판정과 마릿수가 일치하지 않습니다.")
        target = float(row["target_s"])
        if not math.isfinite(target) or not 0 <= target <= end_s + 1e-6:
            raise ValueError("표본 시각이 관측 범위를 벗어났습니다.")
    total = summary(rows)
    n = total["samples"]
    total.update(known_rate_pct=100*total["known_samples"]/n if n else None,
                 unknown_rate_pct=100*total["unknown_samples"]/n if n else None)
    grouped = {}
    for row in rows:
        grouped.setdefault(math.floor(float(row["target_s"])/bin_seconds), []).append(row)
    bins = []
    for index, selected in sorted(grouped.items()):
        start, end = index*bin_seconds, min((index+1)*bin_seconds, end_s)
        bins.append(dict(summary(selected), start_s=start, end_s=end,
                         full_interval=end-start >= bin_seconds-1e-6))
    middle = end_s/2
    first = summary([r for r in rows if float(r["target_s"]) < middle])
    second = summary([r for r in rows if float(r["target_s"]) >= middle])
    delta = (second["approach_pct"]-first["approach_pct"]
             if first["approach_pct"] is not None and second["approach_pct"] is not None else None)
    ranked = [b for b in bins if b["full_interval"] and b["known_samples"]]
    peak = max(ranked, key=lambda b: b["approach_pct"]) if ranked else None
    low = min(ranked, key=lambda b: b["approach_pct"]) if ranked else None
    unknown = [r for r in rows if r["status"] == "unknown"]
    return {"overall": total, "bins": bins, "bin_seconds": bin_seconds,
            "observed_end_s": end_s, "interval_s": interval_s,
            "halves": {"split_s": middle, "first": first, "second": second, "difference_pp": delta},
            "highest_full_bin": peak, "lowest_full_bin": low,
            "unknown_reasons": dict(Counter(r["reason"] for r in unknown)),
            "review_samples": [{"sample": r["sample"], "target_s": float(r["target_s"]),
                                "frame": r["frame"], "reason": r["reason"]} for r in unknown[:20]],
            "review_samples_total": len(unknown), "inference_scope": "descriptive_single_recording",
            "statistical_independence_unit": "cage_or_animal_not_frames",
            "validation_status": "independent_behavior_accuracy_not_established"}


def _fmt(value, suffix=""):
    return "계산 불가" if value is None else f"{value:.1f}{suffix}"


def _time(value):
    return f"{int(value)//60:02d}:{value%60:04.1f}"


def _chart(report, rows, path, region):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10}):
        fig, (ax, timeline) = plt.subplots(2, 1, figsize=(11.8, 5.3), gridspec_kw={"height_ratios": [4, 1]}, layout="constrained")
        fig.patch.set_facecolor("#f6f8f3")
        bins = report["bins"]
        for b in bins:
            width = max((b["end_s"]-b["start_s"])/60, .025)
            middle = (b["start_s"]+b["end_s"])/120
            value = b["approach_pct"]
            ax.bar(middle, value or 0, width=width*.82, color="#20846c" if b["full_interval"] else "#98b8aa",
                   hatch=None if b["full_interval"] else "///", zorder=3)
            label = "계산 불가" if value is None else f"{value:.1f}%"
            ax.text(middle, min((value or 0)+3, 97), label, ha="center", fontsize=9)
            ax.text(middle, -13, f"n={b['known_samples']}", ha="center", color="#586c60", fontsize=8)
        ax.set(xlim=(0, max(report["observed_end_s"]/60, .05)), ylim=(0, 105),
               ylabel="판정 가능한 표본 중 영역 안 (%)", title=f"시간대별 {region} 관측 비율")
        ax.set_xticks(range(math.floor(report["observed_end_s"]/60)+1))
        ax.grid(axis="y", alpha=.18, zorder=0)
        colors = {"approach": "#20846c", "away": "#8faaba", "unknown": "#cc943e"}
        for row in rows:
            t = float(row["target_s"])/60
            timeline.scatter(t, 0, s=24, marker="s", color=colors[row["status"]])
        timeline.set(xlim=ax.get_xlim(), ylim=(-1, 1), xlabel="영상 경과시간 (분)", yticks=[])
        timeline.legend(handles=[Line2D([0],[0],marker="s",linestyle="",color=color,label=label)
                        for label,color in (("영역 안",colors["approach"]),("영역 밖",colors["away"]),("판정 불가",colors["unknown"]))],
                        loc="upper center", bbox_to_anchor=(.5,1.75), ncol=3, frameon=False)
        for area in (ax,timeline):
            area.set_facecolor("white")
            area.spines[["top","right"]].set_visible(False)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def write_interpretation(run, bin_seconds=60):
    from .research import analysis_rows
    from .recording_io import load_times

    run = Path(run)
    manifest, rows = analysis_rows(run)
    if manifest["state"] != "complete":
        raise ValueError("전체 분석을 완료한 뒤 자동 해석을 생성하세요.")
    folder = run.parent.parent
    recording, times = load_times(folder)
    if any(recording["sha256"].get(k) != v for k,v in manifest["source_sha256"].items()):
        raise ValueError("분석 당시의 원본과 현재 원본이 다릅니다.")
    protocol = manifest["protocol"]
    report = describe(rows, float(times[-1]), protocol["criteria"]["interval_s"], bin_seconds)
    region = "임시 테스트 영역" if protocol.get("region_purpose") == "test_region_not_food" else "표시한 사료 주변"
    report.update(schema=1, generated_utc=utcnow(), group=protocol["group"], cage=protocol["cage"],
                  animals=protocol["criteria"]["animals"], region_name=region,
                  analysis_state=manifest["state"], input_samples_sha256=manifest["samples_sha256"],
                  source_sha256=manifest["source_sha256"], detector_sha256=manifest["detector_sha256"],
                  report_code_sha256=digest(Path(__file__)), timestamp_source=recording.get("timestamp_source"))
    overall, halves = report["overall"], report["halves"]
    facts = [f"총 {overall['samples']}개 표본 중 {overall['known_samples']}개가 판정 가능하고, {overall['unknown_samples']}개는 판정 불가입니다."]
    if overall["known_samples"]:
        facts.append(f"{region} 안 표본은 {overall['approach_samples']}개, 밖 표본은 {overall['away_samples']}개입니다. 영역 안 비율은 {overall['approach_samples']}/{overall['known_samples']} = {_fmt(overall['approach_pct'], '%')}입니다.")
    else:
        facts.append("판정 가능한 표본이 없어 영역 안 비율과 평균 마릿수를 계산할 수 없습니다.")
    if halves["difference_pp"] is not None:
        facts.append(f"같은 길이의 전반({_time(0)}–{_time(halves['split_s'])})은 {_fmt(halves['first']['approach_pct'], '%')}, 후반은 {_fmt(halves['second']['approach_pct'], '%')}입니다. 후반−전반 차이는 {halves['difference_pp']:+.1f}%p이며, 이 영상 내의 기술적 비교입니다.")
    if report["highest_full_bin"]:
        peak, low = report["highest_full_bin"], report["lowest_full_bin"]
        facts.append(f"완전한 {bin_seconds:g}초 구간 중 {_time(peak['start_s'])}–{_time(peak['end_s'])}에서 가장 높았고({_fmt(peak['approach_pct'], '%')}), {_time(low['start_s'])}–{_time(low['end_s'])}에서 가장 낮았습니다({_fmt(low['approach_pct'], '%')}).")
    report["observations"] = facts
    report["limits"] = ["판정 가능률은 모델 정확도의 검증 결과와 별개입니다. 정확도는 독립적인 정답과 비교해 확인합니다.",
                        "고정 간격의 표본 비율입니다. 방문 횟수, 연속 체류시간, 실제 섭식시간 또는 섭취량으로 환산하지 않습니다.",
                        "한 촬영의 반복 표본이며 독립 동물·케이지 수로 세지 않습니다. 처리 효과, 통계적 유의성, 식욕·GLP-1의 인과관계는 이 결과만으로 판단할 수 없습니다.",
                        "마지막 구간은 짧을 수 있습니다. 각 구간의 판정 가능 표본 수를 함께 확인하세요."]
    if region == "임시 테스트 영역":
        report["limits"].insert(0,"이번 원본은 사료 없는 오픈필드 영상입니다. 영역 점유를 섭식 지표로 해석하지 않습니다.")
    chart = run/"interpretation_chart.png"
    _chart(report, rows, chart, region)
    atomic_json(run/"interpretation.json", report)
    keys = ["start_s","end_s","full_interval","samples","known_samples","unknown_samples","approach_samples","away_samples","approach_pct","mean_count"]
    with (run/"interpretation_bins.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for item in report["bins"]:writer.writerow({k:item[k] for k in keys})
    esc=html.escape
    table="".join(f"<tr><td>{_time(b['start_s'])}–{_time(b['end_s'])}{'' if b['full_interval'] else ' · 짧은 구간'}</td><td>{b['known_samples']}/{b['samples']}</td><td>{b['approach_samples']}</td><td>{b['away_samples']}</td><td>{b['unknown_samples']}</td><td>{_fmt(b['approach_pct'],'%')}</td></tr>" for b in report["bins"])
    unknown_table="".join(f"<tr><td>{_time(r['target_s'])}</td><td>{r['sample']}</td><td>{esc(REASONS.get(r['reason'],r['reason']))}</td></tr>" for r in report["review_samples"])
    png=base64.b64encode(chart.read_bytes()).decode()
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BehavTrack · 자동 결과 해석</title><style>
    *{{box-sizing:border-box}}body{{font:15px 'Malgun Gothic',sans-serif;line-height:1.7;margin:0;background:#f4f7f1;color:#193d30}}main{{max-width:1200px;margin:24px auto;padding:0 24px}}h1{{font-size:30px;margin:5px 0}}h2{{font-size:20px;margin:0 0 12px}}p{{color:#566c5e}}.panel{{background:white;border:1px solid #dce5db;border-radius:14px;padding:24px;margin:18px 0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}}.metrics strong{{font-size:30px;display:block;color:#16765c}}.metrics span{{font-size:13px;color:#596b60}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #e0e8df;padding:10px;text-align:left}}.scroll{{overflow:auto}}li{{margin:10px 0}}details{{margin:16px 0}}summary{{font-weight:bold;cursor:pointer}}.note{{background:#eaf3e8;padding:14px;border-radius:8px}}small{{color:#647468;overflow-wrap:anywhere}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}main{{padding:0 12px}}.panel{{padding:16px}}}}</style>
    <main><div>BehavTrack · 결과를 읽고 해석하는 단계</div><h1>자동 결과 해석</h1><p>{esc(protocol['group'])} · {esc(protocol['cage'])} · {protocol['criteria']['animals']}마리 · {_time(report['observed_end_s'])} · {report['interval_s']:g}초 간격</p>
    <div class="note">분석 CSV와 원본 해시를 확인해 생성한 기술적 요약입니다. 각 수치의 분모와 판정 불가를 함께 표시합니다.</div>
    <section class="panel metrics"><div><span>판정 가능</span><strong>{overall['known_samples']}/{overall['samples']}</strong><small>{_fmt(overall['known_rate_pct'],'%')}</small></div><div><span>{region} 안</span><strong>{_fmt(overall['approach_pct'],'%')}</strong><small>{overall['approach_samples']}/{overall['known_samples']} 표본</small></div><div><span>판정 불가</span><strong>{overall['unknown_samples']}</strong><small>{_fmt(overall['unknown_rate_pct'],'%')}</small></div><div><span>후반−전반</span><strong>{_fmt(halves['difference_pp'],'%p')}</strong><small>동일 길이의 두 구간 비교</small></div></section>
    <section class="panel"><h2>시간대별 변화와 개별 표본</h2><img alt="시간대별 영역 안 관측 비율과 표본 판정 타임라인" src="data:image/png;base64,{png}"><p>막대 아래 n은 판정 가능한 표본 수입니다. 빗금은 짧은 마지막 구간을 나타냅니다.</p></section>
    <section class="panel"><h2>결과가 보여주는 것</h2><ol>{''.join('<li>'+esc(s)+'</li>' for s in facts)}</ol></section>
    <section class="panel"><h2>시간대별 수치</h2><div class="scroll"><table><tr><th>영상 구간</th><th>판정 가능/전체</th><th>영역 안</th><th>영역 밖</th><th>판정 불가</th><th>영역 안 비율</th></tr>{table}</table></div></section>
    <section class="panel"><h2>검토가 필요한 표본</h2><p>{len(report['review_samples'])}개 표시 / 총 {report['review_samples_total']}개. 판정 불가는 영역 밖으로 세지 않습니다.</p><table><tr><th>영상 시각</th><th>표본 번호</th><th>사유</th></tr>{unknown_table}</table></section>
    <section class="panel"><h2>해석 범위</h2><ul>{''.join('<li>'+esc(s)+'</li>' for s in report['limits'])}</ul><details><summary>계산과 원본 연결</summary><p>영역 안 비율 = 영역 안 표본 / 판정 가능 표본. 후반−전반은 퍼센트포인트 차이입니다. 프레임 사이 행동을 보간하지 않습니다.</p><small>분석: {esc(run.name)}<br>CSV SHA-256: {manifest['samples_sha256']}<br>검출기 SHA-256: {manifest['detector_sha256']}<br>시각 기준: {esc(str(report['timestamp_source']))}</small></details></section></main></html>'''
    (run/"interpretation.html").write_text(body,encoding="utf-8")
    return report
