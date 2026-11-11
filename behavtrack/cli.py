"""Command line interface:  behavtrack <command> --help"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _size(text: str) -> tuple[float, float]:
    parts = text.lower().replace("×", "x").split("x")
    try:
        w = float(parts[0])
        return (w, float(parts[1]) if len(parts) > 1 else w)
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("크기는 가로x세로 cm 로 적어주세요 (예: 40x40)") from None


def _res(text: str) -> tuple[int, int]:
    w, h = _size(text)
    return int(w), int(h)


def _list(text: str | None) -> list[str]:
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def _meta(a) -> dict:
    return {"animal_id": a.animal_id or "", "group": a.group or "", "notes": a.notes or "", "overrides": a.set or []}


def _tracker_kwargs(a) -> dict:
    return {"tracker": a.tracker, "model": a.model, "bodyparts": _list(a.bodyparts) or None,
            "model_type": a.model_type, "annotated": a.annotated}


# ---------------------------------------------------------------- commands

def cmd_paradigms(a):
    from .templates import PARADIGMS

    for key, p in PARADIGMS.items():
        print(f"  {key:<18} {p.title}\n  {'':<18} {p.description}")


def cmd_cameras(a):
    from .sources import list_cameras

    cams = list_cameras()
    if not cams:
        print("열 수 있는 카메라가 없습니다. macOS: 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널을 허용하세요.")
    for c in cams:
        print(f"  카메라 {c['index']}: {c['width']}x{c['height']} @ {c['fps']:.0f} fps")


def cmd_camera_summary(a):
    from .cumulative import build_camera_summary

    try:
        report, csv_file, count, skipped = build_camera_summary(a.folder)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"웹캠 누적 분석: {count}개 추적 결과")
    print(f"보고서: {report}\n엑셀용 CSV: {csv_file}")
    for item in skipped:
        print(f"제외: {item}")


def cmd_group_summary(a):
    from .cumulative import build_group_summary

    try:
        report, csv_file, count, skipped = build_group_summary(a.folder)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"공동 사육 누적 분석: {count}건\n보고서: {report}\n엑셀용 CSV: {csv_file}")
    for item in skipped:
        print(f"제외: {item}")


def cmd_group_ai_summary(a):
    from .cumulative import build_ai_group_summary

    try:
        report, csv_file, count, skipped = build_ai_group_summary(a.folder)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"AI 공동 사육 누적 분석: {count}건\n보고서: {report}\n엑셀용 CSV: {csv_file}")
    for item in skipped:
        print(f"제외: {item}")


def cmd_group_live(a):
    import math
    from .config import load_config
    from .group_live import run_group_live

    if a.duration is not None and (not math.isfinite(a.duration) or a.duration <= 0):
        raise SystemExit("녹화 시간은 양수여야 합니다.")
    run_group_live(load_config(a.config), a.camera, a.output, a.animals, a.cage_id, a.group, a.duration)


def cmd_group_ai_live(a):
    import math
    from .config import load_config
    from .ai_live import run_ai_live

    if a.duration is not None and (not math.isfinite(a.duration) or a.duration <= 0):
        raise SystemExit("녹화 시간은 양수여야 합니다.")
    run_ai_live(load_config(a.config), a.camera, a.output, a.cage_id, a.group, a.duration, a.interval)


def cmd_setup(a):
    from .setup_gui import run_setup
    from .sources import frame_at, open_camera

    if a.video:
        frame = frame_at(a.video, a.time)
        run_setup(frame, a.paradigm, a.output, a.arenas, a.arena_size, a.arena_diameter, a.scale_length,
                  a.species, a.color, a.arena_shape, _list(a.names) or None, a.food_zone)
    else:
        cap = open_camera(a.camera, a.resolution)
        try:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("카메라에서 영상을 읽을 수 없습니다.")
            run_setup(frame, a.paradigm, a.output, a.arenas, a.arena_size, a.arena_diameter, a.scale_length,
                      a.species, a.color, a.arena_shape, _list(a.names) or None, a.food_zone,
                      frame_provider=cap.read)
        finally:
            cap.release()
    out = Path(a.output)
    print(f"\n설정 저장: {out}\n미리보기: {out.with_suffix('.preview.png')}")
    source = f"--video {a.video}" if a.video else f"--camera {a.camera}"
    print(f"다음 단계 예:  behavtrack track -c {out} {source} --animal-id M01")


def cmd_preview(a):
    import cv2

    from .config import arenas_from, load_config
    from .setup_gui import render_preview
    from .sources import frame_at

    cfg = load_config(a.config)
    if a.video:
        frame = frame_at(a.video, a.time)
    else:
        img = a.image or Path(a.config).with_suffix(".reference.png")
        frame = cv2.imread(str(img))
        if frame is None:
            raise SystemExit(f"배경으로 쓸 이미지를 읽을 수 없습니다: {img} (--video 나 --image 를 지정하세요)")
    out = a.output or Path(a.config).with_suffix(".preview.png")
    cv2.imwrite(str(out), render_preview(frame, arenas_from(cfg)))
    print(f"미리보기: {out}")


def cmd_track(a):
    from .config import load_config
    from .pipeline import summarize, track_live, track_video

    cfg = load_config(a.config)
    kw = _tracker_kwargs(a)
    if a.video:
        done = [track_video(cfg, v, a.output, name=a.name if len(a.video) == 1 else None, meta=_meta(a),
                            background=a.background, start_s=a.start, end_s=a.end, display=a.display,
                            analyze=not a.no_analyze, **kw) for v in a.video]
        if len(done) > 1 and not a.no_analyze:
            print(f"\n요약표: {summarize(done, Path(a.output) / 'summary.csv')}")
    else:
        track_live(cfg, a.camera, a.output, a.name, _meta(a), a.duration, a.resolution, a.background, a.auto_start,
                   analyze=not a.no_analyze, **kw)


def cmd_batch(a):
    from .config import load_config
    from .pipeline import read_batch_list, run_batch

    cfg = load_config(a.config)
    if a.list:
        items = read_batch_list(a.list)
    else:
        items = [{"video": Path(v), "meta": {"overrides": a.set or []}} for v in a.videos]
    if not items:
        raise SystemExit("처리할 영상이 없습니다 (--list 또는 --videos).")
    run_batch(cfg, items, a.output, **_tracker_kwargs(a))


def cmd_analyze(a):
    from .config import load_config
    from .pipeline import analyze_session, summarize
    from .session import find_sessions

    sessions = find_sessions(a.sessions)
    if not sessions:
        raise SystemExit("세션 폴더를 찾지 못했습니다.")
    cfg = load_config(a.config) if a.config else None
    for s in sessions:
        print(f"[{s}]")
        analyze_session(s, cfg, a.set)
    if len(sessions) > 1:
        out = a.summary or Path(sessions[0]).parent / "summary.csv"
        print(f"\n요약표: {summarize(sessions, out)}")


def cmd_summarize(a):
    from .pipeline import summarize

    print(f"요약표: {summarize(a.sessions, a.output)}")


def cmd_import(a):
    from .config import load_config
    from .pose_import import import_pose

    import_pose(a.pose, load_config(a.config), a.output, a.fps, a.video, a.name, _meta(a))


def cmd_annotate(a):
    from .annotate import annotate

    out = a.output or Path(a.video).with_suffix(".labels.csv")
    annotate(a.video, _list(a.behaviors), out, a.track or "")


def cmd_classifier(a):
    from .classifier import predict, train
    from .session import find_sessions

    if a.action == "train":
        if len(a.sessions) != len(a.labels):
            raise SystemExit("--sessions 와 --labels 의 개수가 같아야 합니다 (세션마다 라벨 파일 하나).")
        cv = train(list(zip(a.sessions, a.labels)), a.behavior, a.output, a.track, a.min_bout)
        print(f"모델 저장: {a.output}")
        if cv:
            print(f"교차검증 (동물 단위, {cv['animals']}마리): precision {cv['precision']:.2f}, recall {cv['recall']:.2f}, F1 {cv['f1']:.2f}")
    else:
        for s in find_sessions(a.sessions):
            for p in predict(s, a.model, a.threshold):
                print(f"  {p}")


def cmd_feeding_score(a):
    from .feeding import score_video

    score_video(a.video, a.output, a.cage_id, a.group, a.animals, a.interval,
                a.start, a.end, a.resume, a.recorded_at)


def cmd_feeding_summary(a):
    from .feeding import summarize_sessions

    summarize_sessions(a.sessions, a.output)
    print(f"케이지별 요약표: {a.output}")


def cmd_feeding_auto(a):
    from .feeding_auto import analyze_pose_session

    analyze_pose_session(a.session, a.output, a.cage_id, a.group, a.animals,
                         a.zone, a.model, a.min_confidence, a.negative,
                         a.positive, a.min_separation_px, a.recorded_at)


def cmd_feeding_auto_summary(a):
    from .feeding_auto import summarize_auto

    summarize_auto(a.sessions, a.output)
    print(f"자동 분석 요약표: {a.output}")


def cmd_demo(a):
    from . import synth
    from .config import save_config
    from .pipeline import summarize, track_video
    from .pose_import import import_pose

    out = Path(a.output)
    print("1) 가상 실험 영상 만들기 (60초, 오픈필드, freezing 2회 + grooming 1회 포함)")
    path = synth.simulate_path(seconds=60, fps=30)
    video = synth.render(path, out / "demo_open_field.mp4")
    cfg = synth.open_field_config()
    save_config(cfg, out / "demo_config.json")
    print("\n2) 접근법 ①: 영상에서 배경차분으로 추적·분석")
    s1 = track_video(cfg, video, out / "results", name="demo_blob", annotated=True,
                     meta={"animal_id": "demo", "group": "approach1"})
    print("\n3) 접근법 ②: 같은 동물의 포즈 추정 결과(DeepLabCut CSV 형식)를 가져와 분석")
    dlc = synth.write_dlc_csv(path, out / "demo_open_field_DLC.csv")
    s2 = import_pose(dlc, cfg, out / "results", video=video, name="demo_pose",
                     meta={"animal_id": "demo", "group": "approach2"})
    print(f"\n요약표: {summarize([s1, s2], out / 'results' / 'summary.csv')}")
    print(f"리포트: {s1 / 'report.html'}\n        {s2 / 'report.html'}")


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="behavtrack", description="웹캠 기반 설치류 행동 추적·분석")
    sub = p.add_subparsers(dest="cmd", required=True)

    def tracker_opts(sp):
        g = sp.add_argument_group("추적 방식")
        g.add_argument("--tracker", default="blob", choices=["blob", "dlclive"], help="blob=①배경차분(기본), dlclive=②실시간 포즈")
        g.add_argument("--model", help="dlclive: 내보낸 DeepLabCut 모델 폴더")
        g.add_argument("--bodyparts", help="dlclive: 신체 부위 이름 (쉼표 구분, 모델 순서대로)")
        g.add_argument("--model-type", help="dlclive: 모델 종류 (예: pytorch)")
        g.add_argument("--annotated", action="store_true", help="추적 결과를 그린 tracked.mp4 도 저장")

    def meta_opts(sp):
        g = sp.add_argument_group("기록 정보")
        g.add_argument("--animal-id")
        g.add_argument("--group")
        g.add_argument("--notes")
        g.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="설정 덮어쓰기 (예: params.novel_object=object_2, analysis.end_s=300). 여러 번 가능")

    sp = sub.add_parser("paradigms", help="지원하는 실험 종류 보기")
    sp.set_defaults(func=cmd_paradigms)

    sp = sub.add_parser("cameras", help="연결된 카메라 번호 찾기")
    sp.set_defaults(func=cmd_cameras)

    sp = sub.add_parser("camera-summary", help="웹캠 결과의 누적 보고서 생성")
    sp.add_argument("folder", help="웹캠 세션이 저장된 결과 폴더")
    sp.set_defaults(func=cmd_camera_summary)

    sp = sub.add_parser("group-summary", help="공동 사육 웹캠 결과의 누적 보고서 생성")
    sp.add_argument("folder", help="공동 사육 세션이 저장된 결과 폴더")
    sp.set_defaults(func=cmd_group_summary)

    sp = sub.add_parser("group-ai-summary", help="AI 공동 사육 웹캠 결과의 누적 보고서 생성")
    sp.add_argument("folder", help="AI 공동 사육 세션이 저장된 결과 폴더")
    sp.set_defaults(func=cmd_group_ai_summary)

    sp = sub.add_parser("group-live", help="실험적 공동 사육 사료 구역 접근 마릿수 추정")
    sp.add_argument("--config", "-c", required=True)
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--animals", type=int, required=True)
    sp.add_argument("--cage-id", default="")
    sp.add_argument("--group", default="")
    sp.add_argument("--duration", type=float)
    sp.add_argument("--output", "-o", default="results")
    sp.set_defaults(func=cmd_group_live)

    sp = sub.add_parser("group-ai-live", help="CPU AI로 다섯 마리 사료 접근 표본 추정")
    sp.add_argument("--config", "-c", required=True)
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--cage-id", default="")
    sp.add_argument("--group", default="")
    sp.add_argument("--duration", type=float)
    sp.add_argument("--interval", type=float, default=5)
    sp.add_argument("--output", "-o", default="results")
    sp.set_defaults(func=cmd_group_ai_live)

    sp = sub.add_parser("setup", help="아레나·구역 그리기 → 설정 파일 만들기")
    sp.add_argument("--paradigm", "-p", required=True, help="실험 종류 (behavtrack paradigms 참고)")
    src = sp.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", help="카메라 대신 녹화 영상의 한 장면에서 그리기")
    sp.add_argument("--time", type=float, default=1.0, help="--video 에서 사용할 시점 (초)")
    sp.add_argument("--resolution", type=_res, help="카메라 해상도 (예: 1280x720)")
    sp.add_argument("--arena-size", type=_size, help="사각 아레나 실제 크기 cm, 화면 기준 가로x세로 (예: 40x40)")
    sp.add_argument("--arena-diameter", type=float, help="원형 아레나 지름 cm (수중미로/반스)")
    sp.add_argument("--scale-length", type=float, help="미로 등: 화면에서 클릭할 기준 길이 cm (예: 팔 길이 30)")
    sp.add_argument("--arenas", type=int, default=1, help="한 화면에 있는 아레나 수 (각 아레나에 동물 1마리)")
    sp.add_argument("--names", help="아레나 이름들 (쉼표 구분)")
    sp.add_argument("--arena-shape", default="rect", choices=["rect", "circle", "polygon"], help="custom 실험의 아레나 모양")
    sp.add_argument("--food-zone", action="store_true", help="관찰 영역을 그린 뒤 사료 접근 구역을 다각형으로 지정")
    sp.add_argument("--species", default="mouse", choices=["mouse", "rat"])
    sp.add_argument("--color", default="dark", choices=["dark", "light", "any"], help="바닥 대비 동물 색 (검은 쥐 on 흰 바닥 = dark)")
    sp.add_argument("--output", "-o", default="config.json")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("preview", help="설정의 아레나·구역을 이미지 위에 그려 확인")
    sp.add_argument("config")
    sp.add_argument("--video")
    sp.add_argument("--time", type=float, default=1.0)
    sp.add_argument("--image")
    sp.add_argument("--output", "-o")
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("track", help="추적 + 분석 (카메라 실시간 또는 영상 파일)")
    sp.add_argument("--config", "-c", required=True)
    src = sp.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", nargs="+", help="영상 파일 (여러 개 가능)")
    sp.add_argument("--output", "-o", default="results")
    sp.add_argument("--name", help="세션 폴더 이름")
    sp.add_argument("--duration", type=float, help="실시간: 녹화 길이 (초), 끝나면 자동 정지")
    sp.add_argument("--auto-start", action="store_true", help="실시간: 동물이 감지되면 자동 녹화 시작")
    sp.add_argument("--resolution", type=_res, help="실시간: 카메라 해상도 (예: 1280x720)")
    sp.add_argument("--start", type=float, default=0.0, help="영상: 분석 시작 시점 (초)")
    sp.add_argument("--end", type=float, help="영상: 분석 끝 시점 (초)")
    sp.add_argument("--background", help="빈 아레나 이미지 (없으면 자동 추정/실시간은 b 키)")
    sp.add_argument("--display", action="store_true", help="영상: 추적 화면 보기")
    sp.add_argument("--no-analyze", action="store_true", help="추적만 하고 분석은 나중에")
    tracker_opts(sp)
    meta_opts(sp)
    sp.set_defaults(func=cmd_track)

    sp = sub.add_parser("batch", help="여러 영상을 한 번에 추적·분석 + 요약표")
    sp.add_argument("--config", "-c", required=True)
    src = sp.add_mutually_exclusive_group(required=True)
    src.add_argument("--list", help="CSV: video, animal_id, group, (start_s, end_s, params.xxx ...)")
    src.add_argument("--videos", nargs="+")
    sp.add_argument("--output", "-o", default="results")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE")
    tracker_opts(sp)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser("analyze", help="저장된 세션 다시 분석 (구역·기준을 바꿨을 때)")
    sp.add_argument("sessions", nargs="+", help="세션 폴더 (또는 세션들이 든 폴더)")
    sp.add_argument("--config", "-c", help="새 설정 파일 (구역/파라미터를 이걸로 교체)")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE")
    sp.add_argument("--summary", help="요약표 경로")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("summarize", help="여러 세션 결과를 한 표(summary.csv)로")
    sp.add_argument("sessions", nargs="+")
    sp.add_argument("--output", "-o", default="summary.csv")
    sp.set_defaults(func=cmd_summarize)

    sp = sub.add_parser("import-pose", help="② DeepLabCut/SLEAP 결과 가져와 분석")
    sp.add_argument("pose", help="DeepLabCut .csv 또는 SLEAP analysis .h5")
    sp.add_argument("--config", "-c", required=True)
    sp.add_argument("--video", help="원본 영상 (fps와 배경 그림용)")
    sp.add_argument("--fps", type=float)
    sp.add_argument("--output", "-o", default="results")
    sp.add_argument("--name")
    meta_opts(sp)
    sp.set_defaults(func=cmd_import)

    sp = sub.add_parser("annotate", help="② 행동 라벨링 도구 (분류기 학습용)")
    sp.add_argument("video")
    sp.add_argument("--behaviors", required=True, help="행동 이름들, 쉼표 구분 (예: grooming,rearing)")
    sp.add_argument("--output", "-o", help="라벨 CSV (기본: 영상이름.labels.csv)")
    sp.add_argument("--track", help="여러 아레나 영상이면 트랙 이름 (예: A1)")
    sp.set_defaults(func=cmd_annotate)

    sp = sub.add_parser("classifier", help="② 행동 분류기 학습/적용")
    csub = sp.add_subparsers(dest="action", required=True)
    tp = csub.add_parser("train")
    tp.add_argument("--sessions", nargs="+", required=True)
    tp.add_argument("--labels", nargs="+", required=True)
    tp.add_argument("--behavior", required=True)
    tp.add_argument("--track")
    tp.add_argument("--min-bout", type=float, default=0.2, help="이보다 짧은 행동은 무시 (초)")
    tp.add_argument("--output", "-o", required=True, help="모델 파일 (예: grooming.model)")
    tp.set_defaults(func=cmd_classifier)
    pp = csub.add_parser("predict")
    pp.add_argument("sessions", nargs="+")
    pp.add_argument("--model", required=True)
    pp.add_argument("--threshold", type=float)
    pp.set_defaults(func=cmd_classifier)

    sp = sub.add_parser("feeding", help="공동 사육 케이지의 표본 장면을 수동 채점하고 요약")
    fsub = sp.add_subparsers(dest="action", required=True)
    fs = fsub.add_parser("score", help="영상의 정해진 표본 장면에서 접근·섭식 마릿수를 채점")
    fs.add_argument("video")
    fs.add_argument("--output", "-o", required=True, help="이 영상의 새 채점 폴더")
    fs.add_argument("--cage-id", required=True)
    fs.add_argument("--group", required=True, help="G1/G2/G3 등 실제 군 또는 눈가림 코드")
    fs.add_argument("--animals", type=int, required=True, help="케이지 내 마우스 수 (1~9)")
    fs.add_argument("--interval", type=float, required=True, help="표본 간격 초; 분석 전에 결정")
    fs.add_argument("--start", type=float, default=0.0, help="관찰 시작 시각 (초)")
    fs.add_argument("--end", type=float, help="관찰 종료 시각 (초); 기본은 영상 끝")
    fs.add_argument("--recorded-at", default="", help="영상 시작 일시와 시간대 (예: 2026-10-28T20:00:00+09:00)")
    fs.add_argument("--resume", action="store_true", help="기존 채점을 같은 조건에서 이어가기")
    fs.set_defaults(func=cmd_feeding_score)
    fs = fsub.add_parser("auto", help="다중 개체 포즈와 섭식 분류 모델을 이용한 자동 분석")
    fs.add_argument("session", help="다중 개체 포즈 가져오기 세션 폴더")
    fs.add_argument("--output", "-o", required=True, help="새 결과 폴더; 기존 폴더는 덮어쓰지 않음")
    fs.add_argument("--cage-id", required=True)
    fs.add_argument("--group", required=True)
    fs.add_argument("--animals", type=int, required=True)
    fs.add_argument("--zone", required=True, help="설정에 그린 먹이 접근 구역 이름")
    fs.add_argument("--model", help="신뢰할 수 있는 출처의 섭식 행동 분류기 파일")
    fs.add_argument("--min-confidence", type=float, default=0.8, help="포즈 신뢰도 기준 (검증 필요)")
    fs.add_argument("--negative", type=float, default=0.2, help="섭식 아님 확률 상한 (검증 필요)")
    fs.add_argument("--positive", type=float, default=0.8, help="섭식 추정 확률 하한 (검증 필요)")
    fs.add_argument("--min-separation-px", type=float, default=2.0, help="코 좌표가 이보다 가까운 개체는 중복 의심")
    fs.add_argument("--recorded-at", default="", help="영상 시작 일시와 시간대")
    fs.set_defaults(func=cmd_feeding_auto)
    fs = fsub.add_parser("auto-summary", help="자동 분석 결과를 케이지 영상별 CSV로 정리")
    fs.add_argument("sessions", nargs="+", help="자동 분석 결과 폴더들")
    fs.add_argument("--output", "-o", required=True, help="새 요약 CSV")
    fs.set_defaults(func=cmd_feeding_auto_summary)
    fs = fsub.add_parser("summarize", help="각 케이지 영상의 표본 결과를 CSV로 정리")
    fs.add_argument("sessions", nargs="+", help="채점 폴더들")
    fs.add_argument("--output", "-o", required=True, help="새 요약 CSV; 기존 파일은 덮어쓰지 않음")
    fs.set_defaults(func=cmd_feeding_summary)

    sp = sub.add_parser("demo", help="가상 영상으로 전체 과정 체험 (카메라 불필요)")
    sp.add_argument("--output", "-o", default="demo")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
