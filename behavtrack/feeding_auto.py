"""Experimental multi-animal feeding inference from an existing pose session.

Pose landmarks alone establish proximity, not eating. A separate trained feeding
classifier is required for predicted feeding. Unknown is never treated as zero.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .analysis.common import frame_durations
from .analysis.keypoints import find_part
from .classifier import features, check_schema, predict_frames
from .config import arenas_from, complete
from .session import load_session, tracks_path
from .sources import camera_timestamps, open_video, video_info
from .tracks import Tracks
from .activity import heavy


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _context_valid(valid, t, radius_s=1.1):
    """Reject model windows touching missing landmarks or video edges."""
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    left = np.searchsorted(t, t - radius_s, side="left")
    right = np.searchsorted(t, t + radius_s, side="right")
    return ((invalid_prefix[right] - invalid_prefix[left]) == 0) & (t - t[0] >= radius_s) & (t[-1] - t >= radius_s)


def _stats(count, duration):
    known = count >= 0
    observed = float(duration[known].sum())
    positive = float(duration[known & (count > 0)].sum())
    return {"known_s": observed, "unknown_s": float(duration[~known].sum()),
            "any_positive_s": positive if observed else None, "positive_pct_known": 100 * positive / observed if observed else None,
            "mouse_seconds_known": float((duration[known] * count[known]).sum()) if observed else None,
            "mean_mice_known": float((duration[known] * count[known]).sum() / observed) if observed else None}


def feeding_predictions(session, track_map, bundle, min_p=.8, negative=.2, positive=.8, min_separation_px=2.):
    """The identical operational path is used for evaluation and feeding_auto."""
    if bundle.get("behavior") != "feeding":
        raise SystemExit("feeding 행동 모델이 필요합니다.")
    if not 0<=negative<positive<=1 or not 0<min_p<=1 or not math.isfinite(min_separation_px) or min_separation_px<0:
        raise ValueError("섭식 후처리 기준을 확인하세요.")
    cfg=complete(session["config"])
    arenas={a.name:a for a in arenas_from(cfg)}
    ids=list(track_map)
    t=track_map[ids[0]].time
    frame=track_map[ids[0]].frame
    labels,probabilities,noses,known=[],[],[],np.ones(len(t),bool)
    for tid,tr in track_map.items():
        if not np.array_equal(frame,tr.frame) or not np.array_equal(t,tr.time):
            raise ValueError("개체별 프레임/시각이 다릅니다.")
        arena=arenas[session["tracks"][tid]["arena"]]
        nose=find_part(tr.keypoints,"nose","snout")
        if not nose:
            raise ValueError("섭식 적용/평가에는 각 동물의 nose 또는 snout 포즈가 필요합니다.")
        xy=tr.keypoints[nose]
        noses.append(xy[:,:2])
        known &= np.isfinite(xy).all(axis=1) & (xy[:,2]>=min_p) & (xy[:,2]<=1) & arena.shape.contains(xy[:,:2])
        all_good=np.isfinite(tr.xy).all(axis=1)
        for points in tr.keypoints.values():
            all_good &= np.isfinite(points).all(axis=1) & (points[:,2]>=min_p) & (points[:,2]<=1)
        delta=np.r_[np.diff(t),0]
        all_good &= delta<=max(.5,2*np.median(np.diff(t)))
        known &= _context_valid(all_good,t)
        names,x=features(tr,arena.calibration.scale,cfg["analysis"]["keypoint_min_p"])
        check_schema(bundle,session,names)
        lab,prob=predict_frames(bundle,tr,x,cfg["analysis"]["keypoint_min_p"],mode="feeding_auto",negative=negative,positive=positive)
        labels.append(lab); probabilities.append(prob)
        known &= lab>=0
    for i in range(len(noses)):
        for j in range(i+1,len(noses)):
            distance=np.linalg.norm(noses[i]-noses[j],axis=1)
            known &= np.isfinite(distance) & (distance>min_separation_px)
    labels=np.array(labels)
    labels[:,~known]=-1
    return {"labels":{tid:labels[i] for i,tid in enumerate(ids)},"probabilities":np.array(probabilities),
            "count":np.where(known,(labels==1).sum(axis=0),-1)}


@heavy
def analyze_pose_session(session_dir, output, cage_id, group, animals, zone_name, model_path=None,
                         min_p=0.8, negative=0.2, positive=0.8, min_separation_px=2., recorded_at=""):
    session_dir, output = Path(session_dir), Path(output)
    if not cage_id.strip() or not group.strip() or animals < 1:
        raise SystemExit("케이지 ID, 군 코드 및 마리 수를 확인하세요.")
    if not (0 < min_p <= 1 and 0 <= negative < positive <= 1 and math.isfinite(min_separation_px) and min_separation_px >= 0):
        raise SystemExit("신뢰도·확률 또는 거리 기준이 유효하지 않습니다.")
    if recorded_at:
        try:
            if datetime.fromisoformat(recorded_at).utcoffset() is None:
                raise ValueError
        except ValueError:
            raise SystemExit("촬영 시작 일시에 시간대를 포함하세요.") from None
    session = load_session(session_dir)
    if model_path:
        from .research_data import mark_behavior_exposure
        mark_behavior_exposure(session_dir,"feeding_auto")
    if session.get("source",{}).get("pose_confidence_provided") is False:
        raise SystemExit("원본 포즈에 실제 신뢰도 점수가 없습니다. likelihood/point_scores를 포함해 다시 가져오세요.")
    if session.get("source", {}).get("type") != "pose":
        raise SystemExit("다중 마우스 포즈 가져오기 세션이 필요합니다. 기본 blob 추적은 지원하지 않습니다.")
    track_ids = list(session["tracks"])
    if len(track_ids) != animals:
        raise SystemExit(f"포즈 트랙 {len(track_ids)}개와 케이지 마리 수 {animals}가 다릅니다. 누락이나 추가 트랙을 확인하세요.")
    arena_names = {session["tracks"][tid]["arena"] for tid in track_ids}
    if len(arena_names) != 1:
        raise SystemExit("한 케이지에 속한 포즈 트랙만 분석할 수 있습니다.")
    cfg = complete(session["config"])
    arena = next((a for a in arenas_from(cfg) if a.name in arena_names), None)
    zone = arena.zone(zone_name) if arena else None
    if zone is None:
        raise SystemExit(f"먹이 접근 구역 '{zone_name}'을 설정에서 찾을 수 없습니다.")
    tracks = [Tracks.load(tracks_path(session_dir, tid)) for tid in track_ids]
    t, frames = tracks[0].time, tracks[0].frame
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or np.any(np.diff(frames) != 1):
        raise SystemExit("포즈는 시간순으로 중복과 누락 없는 프레임이어야 합니다.")
    for tr in tracks[1:]:
        if not np.array_equal(tr.frame, frames) or not np.array_equal(tr.time, t):
            raise SystemExit("개체별 포즈 프레임 또는 시각이 서로 다릅니다.")
    source_hashes = {}
    for key in ("path", "video"):
        source_path = session["source"].get(key)
        if source_path and Path(source_path).is_file():
            source_hashes[key] = _hash(source_path)
    video = session["source"].get("video")
    if video:
        if not Path(video).exists():
            raise SystemExit("포즈에 연결된 원본 영상을 찾지 못했습니다.")
        cap = open_video(video)
        try:
            info = video_info(cap)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        local_frames=frames-int(session["source"].get("frame_offset",0))
        if local_frames[0] < 0 or local_frames[-1] >= info["frames"] or not math.isfinite(fps) or fps <= 0:
            raise SystemExit("포즈 프레임이 연결된 원본 영상 범위에 맞지 않습니다.")
        stamps = camera_timestamps(video, info["frames"])
        expected_time = stamps[local_frames] if stamps is not None else local_frames / fps
        if not np.allclose(t, expected_time, rtol=0, atol=0.00011):
            raise SystemExit("포즈의 프레임 시각이 원본 영상의 시간 기준과 일치하지 않습니다.")
    from .evaluation import observed_durations
    dur,capture_gap_s = observed_durations(t)
    median_dt = float(np.median(np.diff(t)))
    # A large capture gap is not evidence of a continuing behaviour.
    time_valid = dur <= 2 * median_dt
    noses, valid_noses, valid_features = [], [], []
    for tr in tracks:
        nose = find_part(tr.keypoints, "nose", "snout")
        if not nose:
            raise SystemExit("각 포즈 트랙에 nose 또는 snout 좌표가 필요합니다.")
        coords = tr.keypoints[nose]
        good = np.isfinite(coords).all(axis=1) & (coords[:, 2] >= min_p) & (coords[:, 2] <= 1)
        good &= arena.shape.contains(coords[:, :2])
        noses.append(coords[:, :2])
        valid_noses.append(good)
        all_good = np.isfinite(tr.xy).all(axis=1) & time_valid
        for points in tr.keypoints.values():
            all_good &= np.isfinite(points).all(axis=1) & (points[:, 2] >= min_p) & (points[:, 2] <= 1)
        valid_features.append(_context_valid(all_good, t))
    noses = np.stack(noses)
    known = np.logical_and.reduce(valid_noses) & time_valid
    for i in range(animals):
        for j in range(i + 1, animals):
            dist = np.linalg.norm(noses[i] - noses[j], axis=1)
            known &= np.isfinite(dist) & (dist > min_separation_px)
    near = np.stack([zone.shape.contains(xy) for xy in noses])
    approach = np.where(known, near.sum(axis=0), -1)
    feeding = np.full(len(t), -1, dtype=int)
    probabilities = None
    model_sha = ""
    if model_path:
        import joblib

        # Only load model files the operator trusts: joblib uses pickle.
        bundle = joblib.load(model_path)
        if bundle.get("behavior") != "feeding" or "model" not in bundle or "features" not in bundle:
            raise SystemExit("behavior가 'feeding'인 behavtrack 분류 모델이 필요합니다.")
        model = bundle["model"]
        classes = np.asarray(model.classes_)
        positive_class = np.flatnonzero(classes == 1)
        if len(positive_class) != 1 or len(classes) != 2:
            raise SystemExit("섭식 모델에 양성·음성 두 클래스가 필요합니다.")
        probabilities = []
        for tr in tracks:
            names, x = features(tr, arena.calibration.scale, cfg["analysis"]["keypoint_min_p"])
            if names != bundle["features"]:
                raise SystemExit("섭식 모델의 학습 특징과 포즈 신체 부위가 다릅니다.")
            probabilities.append(model.predict_proba(x)[:, positive_class[0]])
        probabilities = np.stack(probabilities)
        pred_known = np.logical_and.reduce(valid_features) & known
        pred_known &= np.isfinite(probabilities).all(axis=0)
        pred_known &= ((probabilities <= negative) | (probabilities >= positive)).all(axis=0)
        feeding = np.where(pred_known, (probabilities >= positive).sum(axis=0), -1)
        actual=feeding_predictions(session,dict(zip(track_ids,tracks)),bundle,min_p,negative,positive,min_separation_px)
        feeding,probabilities=actual["count"],actual["probabilities"]
        model_sha = _hash(model_path)
    summary = {"cage_id": cage_id, "group": group, "animals": animals, "recorded_at": recorded_at,
               "source_session": str(session_dir.resolve()), "source_session_sha256": _hash(session_dir / "session.json"),
               "source_tracks_sha256": {tid: _hash(tracks_path(session_dir, tid)) for tid in track_ids},
               "source": session["source"], "source_file_sha256": source_hashes, "zone": zone_name, "zone_shape": zone.shape.to_dict(),
               "method": "multi_animal_pose_and_classifier" if model_path else "multi_animal_pose_only",
               "model": str(Path(model_path).resolve()) if model_path else None, "model_sha256": model_sha,
               "min_pose_confidence": min_p, "negative_threshold": negative, "positive_threshold": positive,
               "min_separation_px": min_separation_px, "frames": len(t), "total_s": float(dur.sum()),
               "approach": _stats(approach, dur), "feeding_prediction": _stats(feeding, dur),
               "warning": "Experimental predictions; validate against blinded independent reference videos. Feeding prediction is not food mass or appetite."}
    summary["postprocessing"]={"mode":"feeding_auto","negative":negative,"positive":positive,"min_bout_s":0,
                               "min_pose_p":min_p,"context_s":1.1,"min_separation_px":min_separation_px}
    summary["probability_meaning"]="uncalibrated_model_score"
    summary["capture_gap_s"]=capture_gap_s
    output.mkdir(parents=True, exist_ok=False)
    (output / "auto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "auto_frames.csv").open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "duration_s", "approach_count", "feeding_predicted_count"] +
                   ([f"feeding_probability_{tid}" for tid in track_ids] if probabilities is not None else []))
        for i in range(len(t)):
            w.writerow([int(frames[i]), float(t[i]), float(dur[i]),
                        "" if approach[i] < 0 else int(approach[i]),
                        "" if feeding[i] < 0 else int(feeding[i])] +
                       ([float(v) if known[i] and valid_features[k][i] and math.isfinite(v) else ""
                         for k, v in enumerate(probabilities[:, i])] if probabilities is not None else []))
    print(f"자동 분석: {output} (접근 판정 가능 {summary['approach']['known_s']:.1f}초, 섭식 추정 가능 {summary['feeding_prediction']['known_s']:.1f}초)")
    if model_path is None:
        print("섭식 분류 모델이 없어 섭식은 모두 판정 불가입니다.")


def summarize_auto(outputs, destination):
    if len({str(Path(p).resolve()) for p in outputs}) != len(outputs):
        raise SystemExit("동일한 자동 분석 결과 폴더가 중복되었습니다.")
    rows = []
    for path in map(Path, outputs):
        info = json.loads((path / "auto_summary.json").read_text(encoding="utf-8"))
        row = {k: info[k] for k in ("cage_id", "group", "recorded_at", "animals", "source_session", "method", "zone", "frames", "total_s", "model_sha256", "min_pose_confidence", "negative_threshold", "positive_threshold", "min_separation_px")}
        row["output_folder"] = str(path.resolve())
        for kind in ("approach", "feeding_prediction"):
            for key, value in info[kind].items():
                row[f"{kind}_{key}"] = value
        rows.append(row)
    with open(destination, "x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
