"""Approach ② extension: supervised behavior classifiers (grooming, rearing, digging, ...).

1) label example frames with `behavtrack annotate` → CSV: behavior,start_frame,end_frame[,track]
2) `behavtrack classifier train` fits a random forest on per-frame features
3) `behavtrack classifier predict` labels new sessions; the result joins the analysis and report

Features come from whatever the track contains: blob shape and pixel motion (approach ①) and/or
body-part distances and speeds (approach ②). Pose features make subtle behaviors far more reliable.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
import threading
import importlib.metadata
from pathlib import Path

import numpy as np

from .analysis.common import drop_short_runs, frame_durations
from .config import arenas_from, complete
from .session import load_session, tracks_path
from .tracks import Tracks
from .activity import heavy

WINDOWS_S = (0.2, 0.5, 1.0)
REVIEWED = "reviewed_v2"  # only frames displayed by the current annotator


def _filled(v) -> np.ndarray:
    v = np.asarray(v, float).copy()
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros_like(v)
    if not ok.all():
        v[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), v[ok])
    return v


def _rolling(v, w: int):
    """Centered moving mean and std."""
    if w <= 1:
        return v, np.zeros_like(v)
    k = w // 2
    p = np.pad(v, (k, w - 1 - k), mode="edge")
    c1 = np.cumsum(np.insert(p, 0, 0.0))
    c2 = np.cumsum(np.insert(p * p, 0, 0.0))
    mean = (c1[w:] - c1[:-w]) / w
    return mean, np.sqrt(np.clip((c2[w:] - c2[:-w]) / w - mean ** 2, 0, None))


def _features_block(tracks: Tracks, scale: float, min_p: float = 0.6) -> tuple[list[str], np.ndarray]:
    t = tracks.time.astype(float)
    n = len(t)
    if n < 3:
        raise SystemExit("프레임이 너무 적습니다.")
    dt = float(np.median(np.diff(t))) or 1 / 30
    base: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.column_stack([_filled(tracks.x), _filled(tracks.y)]) / scale
        base["speed"] = np.hypot(*np.gradient(xy, t, axis=0).T)
        base["accel"] = np.gradient(base["speed"], t)
        ex = tracks.extra
        if "area" in ex:
            base["area"] = _filled(ex["area"]) / scale ** 2
        if "major" in ex and "minor" in ex:
            major, minor = _filled(ex["major"]) / scale, _filled(ex["minor"]) / scale
            base["length"], base["width"] = major, minor
            base["elongation"] = major / np.maximum(minor, 1e-6)
        if "motion" in ex:
            base["motion"] = _filled(ex["motion"])
        if "angle" in ex:
            a = np.radians(_filled(ex["angle"]))
            base["turn"] = np.abs(np.angle(np.exp(2j * np.diff(a, prepend=a[:1])))) / 2 / dt
        kps = {}
        for name, arr in sorted(tracks.keypoints.items()):
            p = arr[:, :2].copy()
            p[~(arr[:, 2] >= min_p)] = np.nan
            kps[name] = np.column_stack([_filled(p[:, 0]), _filled(p[:, 1])]) / scale
            base[f"p_{name}"] = np.nan_to_num(arr[:, 2])
            base[f"v_{name}"] = np.hypot(*np.gradient(kps[name], t, axis=0).T)
        for a, b in itertools.combinations(sorted(kps), 2):
            base[f"d_{a}_{b}"] = np.hypot(*(kps[a] - kps[b]).T)
    names, cols = [], []
    for k, v in base.items():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        names.append(k)
        cols.append(v)
        for w_s in WINDOWS_S:
            # Actual elapsed-time windows, including irregular camera intervals.
            left = np.searchsorted(t, t-w_s/2, side="left")
            right = np.searchsorted(t, t+w_s/2, side="right")
            c1, c2 = np.r_[0,np.cumsum(v)], np.r_[0,np.cumsum(v*v)]
            mean = (c1[right]-c1[left])/(right-left)
            std = np.sqrt(np.maximum(0,(c2[right]-c2[left])/(right-left)-mean**2))
            names += [f"{k}_mean{w_s:g}s", f"{k}_std{w_s:g}s"]
            cols += [mean, std]
    return names, np.column_stack(cols)


def valid_blocks(tracks, min_p=.6):
    t = tracks.time
    if len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or np.any(np.diff(tracks.frame)<=0):
        raise SystemExit("프레임과 실제 촬영 시각은 중복 없이 증가해야 합니다.")
    valid = np.isfinite(tracks.xy).all(axis=1)
    for arr in tracks.keypoints.values():
        valid &= np.isfinite(arr).all(axis=1) & (arr[:,2]>=min_p) & (arr[:,2]<=1)
    for arr in tracks.extra.values():
        valid &= np.isfinite(arr)
    gap = np.r_[True, (np.diff(t)>max(.5,2*np.median(np.diff(t)))) | (np.diff(tracks.frame)!=1)]
    blocks, start = [], None
    for i in range(len(t)):
        if start is not None and (gap[i] or not valid[i]):
            blocks.append(np.arange(start,i)); start=None
        if valid[i] and start is None:
            start=i
    if start is not None:
        blocks.append(np.arange(start,len(t)))
    return [b for b in blocks if len(b)>=3]


def feature_valid(tracks, min_p=.6):
    valid = np.zeros(len(tracks),bool)
    for b in valid_blocks(tracks,min_p):
        # 1s centered window plus two derivative neighbors; never crosses a gap.
        edge = max(WINDOWS_S)/2 + 2*float(np.median(np.diff(tracks.time[b])))
        valid[b] = (tracks.time[b]-tracks.time[b[0]]>=edge) & (tracks.time[b[-1]]-tracks.time[b]>=edge)
    return valid


def features(tracks, scale, min_p=.6):
    if not math.isfinite(scale) or scale<=0:
        raise SystemExit("좌표 환산 단위를 확인하세요.")
    # Compute no interpolation/window across missing data, identity breaks or gaps.
    names, base = _features_block(tracks,scale,min_p)
    x = np.zeros_like(base)
    for b in valid_blocks(tracks,min_p):
        _, block = _features_block(tracks.subset(b),scale,min_p)
        x[b] = block
    return names,x


def feature_schema(session, names):
    cfg = complete(session["config"])
    calibration = [a.get("calibration",{}) for a in cfg["arenas"]]
    units = ["cm" if c.get("px_per_cm") or c.get("homography") else "px" for c in calibration]
    return {"version":2,"names":names,"coordinate_input":"px","position_units":sorted(set(units)),
            "speed_units":[u+"/s" for u in sorted(set(units))],"windows_s":list(WINDOWS_S),
            "windows":"centered_actual_time", "missing":"exclude_context_no_cross_gap_interpolation",
            "min_pose_p":cfg["analysis"]["keypoint_min_p"], "view":session.get("source",{}).get("view","unspecified"),
            "pose_format":session.get("source",{}).get("tracker"),
            "body_center_rule":"named_center_else_mean_non_tail_parts",
            "source_type":session.get("source",{}).get("type"),"context_rule":"half_max_window_plus_two_median_steps"}


def check_schema(bundle, session, names):
    if names != bundle["features"]:
        raise SystemExit("모델과 포즈 특징/신체 부위가 다릅니다.")
    if bundle.get("feature_schema") and feature_schema(session,names) != bundle["feature_schema"]:
        raise SystemExit("모델·특징·좌표 단위·시간창·촬영 구도 조건이 호환되지 않습니다.")


def predict_frames(bundle, tracks, X, min_p=.6, threshold=None, mode="predict", negative=.2, positive=.8):
    classes = np.asarray(bundle["model"].classes_)
    pos = np.flatnonzero(classes==1)
    if len(pos)!=1 or len(classes)!=2:
        raise SystemExit("행동 모델에는 양성·음성 두 클래스가 필요합니다.")
    prob = bundle["model"].predict_proba(X)[:,pos[0]]
    valid = feature_valid(tracks,min_p) & np.isfinite(prob)
    labels = np.full(len(tracks),-1,int)
    if mode == "feeding_auto":
        valid &= (prob<=negative) | (prob>=positive)
        labels[valid] = (prob[valid]>=positive).astype(int)
    elif mode == "predict":
        thr = bundle.get("threshold",.5) if threshold is None else threshold
        if not math.isfinite(thr) or not 0<=thr<=1:
            raise ValueError("행동 임계값은 0~1이어야 합니다.")
        from .evaluation import observed_durations
        d,_ = observed_durations(tracks.time)
        # Unknown frames split bouts and are never converted to negative.
        positive_mask = drop_short_runs(valid & (prob>=thr), d, bundle.get("min_bout_s",.2))
        labels[valid] = positive_mask[valid].astype(int)
    else:
        raise ValueError("지원하지 않는 운영 후처리입니다.")
    return labels,prob


def read_labels(path, behavior: str, track: str | None = None) -> list[tuple[int, int]]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["behavior"].strip() != behavior:
                continue
            if track and (row.get("track") or "").strip() not in ("", track):
                continue
            a,b=float(row["start_frame"]),float(row["end_frame"])
            if not math.isfinite(a) or not math.isfinite(b) or a!=int(a) or b!=int(b) or a<0 or a>b:
                raise SystemExit(f"{path}: 시작·종료 프레임은 유효한 정수 구간이어야 합니다.")
            out.append((int(a),int(b)))
    return out


def frame_mask(frames, intervals) -> np.ndarray:
    m = np.zeros(len(frames), bool)
    for a, b in intervals:
        m |= (frames >= a) & (frames <= b)
    return m


def _session_features(session_dir, track_id=None):
    sess = load_session(session_dir)
    cfg = complete(sess["config"])
    arenas = {a.name: a for a in arenas_from(cfg)}
    for tid, info in sess["tracks"].items():
        if track_id and tid != track_id:
            continue
        tracks = Tracks.load(tracks_path(session_dir, tid))
        names, X = features(tracks, arenas[info["arena"]].calibration.scale, cfg["analysis"]["keypoint_min_p"])
        yield tid, tracks, names, X


@heavy
def train(pairs, behavior: str, out_path, track_id: str | None = None, min_bout_s: float = 0.2,
          n_estimators: int = 300, seed: int = 0, cancel=None, progress=None) -> dict:
    try:
        import joblib
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.model_selection import GroupKFold
    except ImportError:
        raise SystemExit("분류기에는 scikit-learn이 필요합니다:  pip install scikit-learn") from None
    cancel = cancel or threading.Event()
    progress = progress or (lambda *args:None)
    from .integrity import digest, utcnow
    Xs, ys, gs, ref = [], [], [], None
    schema, provenance = None, []
    seen_sessions,seen_videos=set(),set()
    all_ids_known = True
    for si, (session_dir, labels) in enumerate(pairs):
        sess = load_session(session_dir)
        if (Path(session_dir)/"session.json").exists():
            session_key=str(Path(session_dir).resolve())
            video=sess.get("source",{}).get("video")
            video_hash=digest(video) if video and Path(video).is_file() else None
            if session_key in seen_sessions or video_hash and video_hash in seen_videos:
                raise SystemExit(f"{session_dir}: 같은 영상/세션이 중복 포함됐습니다.")
            seen_sessions.add(session_key)
            if video_hash:seen_videos.add(video_hash)
            provenance.append({"session":str(Path(session_dir).resolve()),"labels":str(Path(labels).resolve()),
                               "session_sha256":digest(Path(session_dir)/"session.json"),"labels_sha256":digest(labels),
                               "tracks_sha256":{t:digest(tracks_path(session_dir,t)) for t in sess["tracks"]},"source":sess.get("source")})
        for ti, (tid, tracks, names, X) in enumerate(_session_features(session_dir, track_id)):
            if ref is None:
                ref = names
            elif names != ref:
                raise SystemExit("세션마다 특징이 다릅니다 (추적 방식이나 신체 부위가 달라요). 같은 방식으로 추적한 세션만 함께 학습하세요.")
            y = frame_mask(tracks.frame, read_labels(labels, behavior, tid))
            if "config" in sess:
                current = feature_schema(sess,names)
                if schema is not None and schema != current:
                    raise SystemExit("학습 세션의 특징 정의·단위·시간창·촬영 구도가 다릅니다.")
                schema = current
            reviewed = read_labels(labels, REVIEWED, tid)
            if not reviewed:
                raise SystemExit(f"{labels}: 확인된 reviewed_v2 구간이 없습니다. 새 라벨 파일로 다시 채점하세요.")
            use = frame_mask(tracks.frame, reviewed)
            if np.any(y & ~use):
                raise SystemExit(f"{labels}: 행동 구간에 확인하지 않은 프레임이 있습니다. 빠진 구간을 재생하며 확인하세요.")
            info = sess["tracks"][tid]
            if "config" in sess:
                use &= feature_valid(tracks,complete(sess["config"])["analysis"]["keypoint_min_p"])
            animal = (sess.get("cage_id") or sess.get("cage") or ((sess.get("animal_id") or "") if len(sess["tracks"]) == 1 else "")).strip()
            # Multi-animal individual names are only session-local and do not establish identity.
            if not animal:
                all_ids_known = False
                animal = f"unknown:{si}:{ti}"
            Xs.append(X[use])
            ys.append(y[use])
            gs.extend([animal] * int(use.sum()))
    if not Xs:
        raise SystemExit("학습할 트랙이 없습니다.")
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.asarray(gs)
    if y.sum() < 10 or (~y).sum() < 10:
        raise SystemExit(f"'{behavior}' 라벨이 너무 적습니다 (양성 {int(y.sum())}, 음성 {int((~y).sum())} 프레임).")
    if Path(out_path).exists():
        raise FileExistsError("기존 모델을 덮어쓰지 않습니다. 새 후보 이름을 지정하세요.")
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", min_samples_leaf=2,
                                 n_jobs=1, random_state=seed)
    # Keep all sessions from the same explicitly identified animal together.
    cv = {}
    folds = min(5, len(np.unique(g))) if all_ids_known else 0
    if folds >= 2:
        pred, scored = np.zeros(len(y), bool), np.zeros(len(y), bool)
        for tr, te in GroupKFold(folds).split(X, y, g):
            if cancel.is_set():
                raise RuntimeError("학습 취소: 기존 운영 모델은 유지됩니다.")
            if y[tr].any() and (~y[tr]).any():
                pred[te] = clone(clf).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] >= 0.5
                scored[te] = True
        if scored.all() and y[scored].any() and (~y[scored]).any():
            p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            cv = {"precision": float(p), "recall": float(r), "f1": float(f), "folds": folds,
                  "scope":"development_group_cv_raw_probability_not_operational_validation",
                  "animals": int(len(np.unique(g))), "positive_frames": int(y.sum()), "frames": int(len(y))}
        else:
            print("  ⚠ 일부 평가 분할의 학습 데이터에 양성 또는 음성이 없어 교차검증을 보고하지 않습니다.")
    else:
        print("  ⚠ 고유 animal_id가 있는 여러 동물이 필요해 동물 단위 교차검증을 보고하지 않습니다.")
    clf.set_params(warm_start=True)
    for trees in range(min(25,n_estimators),n_estimators+25,25):
        if cancel.is_set():
            raise RuntimeError("학습 취소: 기존 운영 모델은 유지됩니다.")
        clf.set_params(n_estimators=min(trees,n_estimators)).fit(X,y)
        progress(min(trees,n_estimators),n_estimators,"랜덤 포레스트 학습")
        if trees>=n_estimators:
            break
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(out_path)+".partial")
    joblib.dump({"model": clf, "features": ref, "behavior": behavior, "threshold": 0.5, "min_bout_s": min_bout_s,
                 "cv": cv, "frames": int(len(y)), "positive_frames": int(y.sum()), "state":"candidate",
                 "feature_schema":schema,"training_data":provenance,"seed":seed,"created_utc":utcnow(),
                 "code_sha256":{name:digest(Path(__file__).parent/name) for name in ("classifier.py","feeding_auto.py","evaluation.py","analysis/common.py","config.py")},
                 "packages":{p:importlib.metadata.version(p) for p in ("numpy","scikit-learn","joblib")},
                 "probability_meaning":"uncalibrated_random_forest_score","groups":sorted(set(gs)),
                 "postprocessing":{"predict":{"threshold":.5,"min_bout_s":min_bout_s},
                                   "feeding_auto":{"negative":.2,"positive":.8,"min_bout_s":0}}}, temporary)
    if cancel.is_set():
        raise RuntimeError("학습 취소: 미완성 후보는 적용할 수 없습니다.")
    # Use exclusive destination, even if another job completed during training.
    with Path(out_path).open("xb") as target, temporary.open("rb") as source:
        import shutil
        shutil.copyfileobj(source,target)
    temporary.unlink()
    return cv


@heavy
def predict(session_dir, model_path, threshold: float | None = None, analyze: bool = True) -> list[Path]:
    try:
        import joblib
    except ImportError:
        raise SystemExit("분류기에는 scikit-learn이 필요합니다:  pip install scikit-learn") from None
    bundle = joblib.load(model_path)
    thr = bundle["threshold"] if threshold is None else threshold
    written = []
    for tid, tracks, names, X in _session_features(session_dir):
        sess = load_session(session_dir)
        check_schema(bundle,sess,names)
        from .research_data import mark_behavior_exposure
        mark_behavior_exposure(session_dir,"predict")
        label,prob = predict_frames(bundle,tracks,X,complete(sess["config"])["analysis"]["keypoint_min_p"],thr)
        path = Path(session_dir) / f"behavior_{bundle['behavior']}_{tid}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "probability", "label"])
            for fr, t, p, lab in zip(tracks.frame, tracks.time, prob, label):
                w.writerow([int(fr), f"{t:.9f}", f"{p:.6f}" if lab>=0 else "", int(lab) if lab>=0 else ""])
        written.append(path)
    if analyze:
        from .pipeline import analyze_session

        analyze_session(session_dir)
    return written
