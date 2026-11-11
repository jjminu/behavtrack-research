"""Approach ②: import pose-estimation output as a session.

Supported: DeepLabCut CSV (single- and multi-animal) and SLEAP analysis HDF5 (File > Export Analysis HDF5).
After import every analysis runs on body parts: zones, freezing from body-part motion, NOR with the
nose, EPM head dips, animal-to-animal contact, and the behavior classifier.
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import cv2
import numpy as np

from .analysis.keypoints import find_part
from .config import apply_overrides, arenas_from
from .session import create_session_dir, save_session, tracks_path
from .sources import camera_timestamps, open_video, video_info, frame_offset
from .tracks import Tracks

NAN = float("nan")


def _num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return NAN


def read_dlc_csv(path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, k = {}, 0
    while k < len(rows) and rows[k] and rows[k][0].strip().lower() in ("scorer", "individuals", "bodyparts", "coords"):
        header[rows[k][0].strip().lower()] = rows[k]
        k += 1
    if "bodyparts" not in header or "coords" not in header:
        raise SystemExit(f"DeepLabCut CSV 형식이 아닙니다 (bodyparts/coords 헤더 없음): {path}")
    ncol = len(header["coords"])
    data = [r for r in rows[k:] if r and r[0].strip()]
    numbers = np.array([_num(r[0]) for r in data])
    if not np.isfinite(numbers).all() or np.any(numbers != np.floor(numbers)):
        raise SystemExit("포즈 프레임 번호가 올바른 정수가 아닙니다.")
    frames = numbers.astype(int)
    vals = np.array([[_num(v) for v in (r + [""] * ncol)[1:ncol]] for r in data], float).reshape(len(data), ncol - 1)
    inds = header.get("individuals", [""] + ["single"] * (ncol - 1))
    cols: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for j in range(1, ncol):
        ind = inds[j] or "single"
        part=cols.setdefault(ind, {}).setdefault(header["bodyparts"][j], {})
        coordinate=header["coords"][j].strip().lower()
        if coordinate in part:
            raise SystemExit("포즈의 동물/신체 부위/좌표 ID가 중복입니다.")
        part[coordinate] = vals[:, j - 1]
    out = {}
    for ind, parts in cols.items():
        out[ind] = {bp: np.column_stack([c["x"], c["y"], c.get("likelihood", np.ones(len(data)))])
                    for bp, c in parts.items() if "x" in c and "y" in c}
    return frames, out


def read_sleap_h5(path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    try:
        import h5py
    except ImportError:
        raise SystemExit("SLEAP 파일을 읽으려면 h5py가 필요합니다:  pip install h5py") from None
    with h5py.File(path, "r") as f:
        if "tracks" not in f:
            raise SystemExit("SLEAP analysis HDF5가 아닙니다. DeepLabCut 결과라면 CSV로 저장(save_as_csv=True)해서 주세요.")
        locs = f["tracks"][:].T  # frames, nodes, 2, tracks
        nodes = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["track_names"][:]] if "track_names" in f else []
        scores = f["point_scores"][:].T if "point_scores" in f else None  # frames, nodes, tracks
    n_tracks = locs.shape[3]
    if len(names) != n_tracks:
        names = [f"track_{i}" for i in range(n_tracks)]
    if len(set(names))!=len(names):
        raise SystemExit("SLEAP 동물 트랙 ID가 중복입니다.")
    out = {}
    for k, name in enumerate(names):
        parts = {}
        for j, node in enumerate(nodes):
            x, y = locs[:, j, 0, k], locs[:, j, 1, k]
            p = scores[:, j, k] if scores is not None else np.isfinite(x).astype(float)
            parts[node] = np.column_stack([x, y, np.where(np.isfinite(x), p, 0.0)])
        out[name or f"track_{k}"] = parts
    return np.arange(locs.shape[0]), out


def read_pose(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return read_dlc_csv(path), "deeplabcut"
    if suffix in (".h5", ".hdf5", ".analysis.h5"):
        return read_sleap_h5(path), "sleap"
    raise SystemExit("지원 형식: DeepLabCut .csv, SLEAP analysis .h5")


def confidence_provided(path, kind):
    if kind == "deeplabcut":
        with open(path, encoding="utf-8-sig", newline="") as f:
            coord = next((row for row in csv.reader(f) if row and row[0].lower() == "coords"), [])
        return bool(coord.count("x")) and coord.count("likelihood") == coord.count("x")
    import h5py
    with h5py.File(path, "r") as f:
        return "point_scores" in f


def read_segment_poses(path, video):
    """Join complete per-segment predictions, with explicitly checked animal identities."""
    from .integrity import read_json, digest
    from .recording_io import load_times
    if not video or not Path(video).is_dir():
        raise ValueError("조각별 포즈 목록은 원본 촬영 세션 폴더에 연결하세요.")
    info, _ = load_times(video)
    spec = read_json(path)
    entries = spec.get("segments", [])
    segments = info.get("segments") or [{"file":"raw.mp4", "start_frame":0, "frames":info["frames"]}]
    if [e.get("segment") for e in entries] != [s["file"] for s in segments]:
        raise ValueError("조각별 포즈 목록은 원본의 모든 영상 조각을 순서대로 한 번씩 포함해야 합니다.")
    parts_by_animal, provenance, frames, kind, schema = {}, [], [], None, None
    for entry, segment in zip(entries, segments):
        source = Path(entry["pose"])
        source = source if source.is_absolute() else Path(path).parent/source
        (local, animals), current_kind = read_pose(source)
        if not np.array_equal(local, np.arange(segment["frames"])):
            raise ValueError(f"{source}: 원본 조각의 0~{segment['frames']-1} 프레임 전체가 필요합니다.")
        mapping = entry.get("identity_mapping", {})
        if len(animals)>1 and (entry.get("identity_checked") is not True or set(mapping)!=set(animals)):
            raise ValueError(f"{source}: 조각 경계의 동물 식별을 직접 확인하고 identity_mapping 및 identity_checked를 기록하세요.")
        mapped = {mapping.get(name,name):parts for name,parts in animals.items()}
        if len(mapped)!=len(animals) or any(not isinstance(name,str) or not name.strip() for name in mapped):
            raise ValueError("조각 간 동물 ID가 중복되거나 비어 있습니다.")
        current_schema = {name:sorted(parts) for name,parts in mapped.items()}
        if schema is not None and (schema!=current_schema or kind!=current_kind):
            raise ValueError("조각별 포즈 형식·동물 ID·신체 부위가 다릅니다. 정체성 불명 구간은 임의로 연결하지 마세요.")
        schema, kind = current_schema, current_kind
        for name,parts in mapped.items():
            for part,values in parts.items():
                parts_by_animal.setdefault(name,{}).setdefault(part,[]).append(values)
        frames.append(local+segment["start_frame"])
        provenance.append({"segment":segment["file"],"pose":str(source.resolve()),"sha256":digest(source),
                           "identity_mapping":mapping,"identity_checked":entry.get("identity_checked",False),
                           "confidence_provided":confidence_provided(source,kind)})
    joined = {name:{part:np.concatenate(values) for part,values in parts.items()} for name,parts in parts_by_animal.items()}
    return (np.concatenate(frames),joined),kind,provenance


def body_center(parts: dict[str, np.ndarray], min_p: float) -> np.ndarray:
    """Reference point: a body-centre part if the model has one, else the mean of non-tail parts."""
    def good(arr):
        xy = arr[:, :2].copy()
        xy[~(arr[:, 2] >= min_p)] = np.nan
        return xy

    name = find_part(parts, "center", "centroid", "body", "spine", "back", "mid", exclude=("tail",))
    if name:
        return good(parts[name])
    use = [good(a) for n, a in parts.items() if "tail" not in n.lower()] or [good(a) for a in parts.values()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(use), axis=0)


def import_pose(pose_path, cfg: dict, out_root, fps: float | None = None, video=None, name: str | None = None,
                meta: dict | None = None, analyze: bool = True) -> Path:
    meta = meta or {}
    cfg = apply_overrides(cfg, meta.get("overrides"))
    segment_sources = None
    if Path(pose_path).suffix.lower()==".json":
        (frames, animals), kind, segment_sources = read_segment_poses(pose_path,video)
    else:
        (frames, animals), kind = read_pose(pose_path)
    if not len(frames) or np.any(frames < 0) or np.any(np.diff(frames) <= 0):
        raise SystemExit("포즈 프레임 번호에 누락된 번호, 중복 또는 역순이 있습니다.")
    ref, timestamps = None, None
    if video:
        cap = open_video(video)
        info = video_info(cap)
        fps = fps or info["fps"]
        timestamps = camera_timestamps(video, info["frames"])
        ok, ref = cap.read()
        cap.release()
        if np.any(frames>=info["frames"]):
            raise SystemExit("포즈 프레임 번호가 원본 영상 범위를 벗어납니다.")
    if not fps or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise SystemExit("--fps 또는 --video 가 필요합니다 (시간 계산용).")
    if timestamps is not None and (np.any(frames < 0) or np.any(frames >= len(timestamps)) or np.any(np.diff(frames) <= 0)):
        raise SystemExit("포즈 프레임 번호가 원본 카메라 영상의 타임스탬프와 일치하지 않습니다.")
    arenas = arenas_from(cfg)
    min_p = cfg["analysis"]["keypoint_min_p"]

    placed = []
    for ind, parts in animals.items():
        center = body_center(parts, min_p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(center, axis=0)
        if len(arenas) == 1:
            arena = arenas[0]
        else:
            inside = [a for a in arenas if np.isfinite(med).all() and a.shape.contains(med)[0]]
            if not inside:
                print(f"  ⚠ '{ind}' 이(가) 어느 아레나에도 속하지 않아 건너뜁니다.")
                continue
            arena = inside[0]
        placed.append((ind, parts, center, arena))
    if not placed:
        raise SystemExit("가져올 동물이 없습니다.")

    offset = frame_offset(video) if video else 0
    session_dir = create_session_dir(out_root, name or Path(pose_path).stem)
    per_arena = {}
    for _, _, _, arena in placed:
        per_arena[arena.name] = per_arena.get(arena.name, 0) + 1
    index = {}
    for ind, parts, center, arena in placed:
        tid = arena.name if per_arena[arena.name] == 1 else f"{arena.name}-{ind}"
        times = timestamps[frames] if timestamps is not None else frames / float(fps)
        from .session import safe_name
        tid = safe_name(tid)
        if tid in index:
            raise SystemExit("포즈 트랙 ID가 중복됩니다. 원본 이름을 확인하세요.")
        Tracks(frames + offset, times, center[:, 0], center[:, 1], {}, parts).save(tracks_path(session_dir, tid))
        index[tid] = {"arena": arena.name, "individual": ind}
    if ref is None:
        pts = np.vstack([a.shape.outline() for a in arenas])
        ref = np.full((int(pts[:, 1].max()) + 40, int(pts[:, 0].max()) + 40, 3), 230, np.uint8)
    cv2.imwrite(str(session_dir / "reference.png"), ref)
    source = {"type": "pose", "path": str(Path(pose_path).resolve()), "video": str(Path(video).resolve()) if video else None,
              "fps": float(fps), "frames": int(len(frames)), "tracker": kind,
              "timebase": "camera_timestamps" if timestamps is not None else "video_fps"}
    from .integrity import digest, atomic_json
    source.update(frame_offset=offset, pose_sha256=digest(pose_path), coordinate_units="px",
                  pose_parts=sorted(next(iter(animals.values()))), view=meta.get("view", "unspecified"))
    if segment_sources is not None:
        source["pose_segments"] = segment_sources
        source["pose_confidence_provided"] = all(s["confidence_provided"] for s in segment_sources)
    else:
        source["pose_confidence_provided"] = confidence_provided(pose_path,kind)
    recording = Path(video) if video and Path(video).is_dir() else (Path(video).parent if video else None)
    if recording and (recording/"recording.json").exists():
        from .recording_io import load_times, frame_reference
        from .integrity import read_json
        info, _ = load_times(recording)
        spec = read_json(recording/"protocol.json")
        source.update(recording=str(recording.resolve()), source_sha256=info["sha256"], view=spec["criteria"]["view"],
                      recorded_at=info["started_utc"], recording_id=recording.name)
        from datetime import datetime,timedelta,timezone
        zone=timezone(timedelta(minutes=spec.get("utc_offset_minutes",540)))
        source["study_date"]=datetime.fromisoformat(info["started_utc"]).astimezone(zone).date().isoformat()
        source["utc_offset_minutes"]=spec.get("utc_offset_minutes",540)
        meta = dict(meta, cage_id=spec["cage"], group=spec["group"], treatment=spec.get("treatment"),
                    administration=spec.get("administration"), animals=spec["criteria"]["animals"])
        with (session_dir/"frame_map.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["session_frame", "source_frame", "segment", "segment_frame", "time_s", "video_sha256"])
            for local, t in zip(frames, times):
                ref = frame_reference(recording, int(local+offset), info)
                writer.writerow([local+offset, local, ref["segment"], ref["segment_frame"], f"{t:.9f}", ref["video_sha256"]])
    elif video:
        source["video_sha256"] = digest(video)
    save_session(session_dir, cfg, source, index, meta)
    print(f"[{Path(pose_path).name}] → {session_dir}  ({', '.join(index)})")
    if analyze:
        from .pipeline import analyze_session

        analyze_session(session_dir)
    return session_dir
