"""Approach ② metrics that need body parts from pose estimation (DeepLabCut, SLEAP, DLC-Live).

Body parts are matched by name, so any model works as long as names contain e.g. "nose"/"snout",
"ear", "head", "tail_base"/"tailbase".
"""

from __future__ import annotations

import warnings

import numpy as np

from .common import Prepared, bout_summary, drop_short_runs


def find_part(names, *keys: str, exclude: tuple[str, ...] = ()) -> str | None:
    for k in keys:
        for n in names:
            low = n.lower()
            if k in low and not any(e in low for e in exclude):
                return n
    return None


def nose_name(prep: Prepared) -> str | None:
    return find_part(prep.kp, "nose", "snout")


def head_center(prep: Prepared) -> np.ndarray | None:
    head = find_part(prep.kp, "head", exclude=("tail",))
    if head:
        return prep.kp[head]
    ears = [n for n in prep.kp if "ear" in n.lower()]
    if len(ears) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack([prep.kp[e] for e in ears]), axis=0)
    return None


def tail_base_name(prep: Prepared) -> str | None:
    return find_part(prep.kp, "tailbase", "tail_base", "tail-base", "tail base", "tail1", "tail",
                     exclude=("end", "tip", "mid"))


def object_exploration(prep: Prepared, shape, nose_cm: float, centroid_cm: float,
                       exclude_on_object: bool = True, min_s: float = 0.0) -> tuple[np.ndarray, str]:
    """Exploring an object = nose within nose_cm of its edge and pointing at it (pose), or, without
    pose, the body centre within centroid_cm of the edge. Sitting on the object is not exploring."""
    nose = nose_name(prep)
    if nose:
        d = shape.boundary_distance(prep.kp[nose]) / prep.scale
        near = np.nan_to_num(d, nan=np.inf) <= nose_cm
        head = head_center(prep)
        if head is not None:
            dh = shape.boundary_distance(head) / prep.scale
            near &= ~np.isfinite(dh) | (d < dh)
        method = "nose"
    else:
        d = shape.boundary_distance(prep.xy) / prep.scale
        near = np.nan_to_num(d, nan=np.inf) <= centroid_cm
        method = "centroid_proximity"
    if exclude_on_object:
        near &= ~shape.contains(prep.xy)
    return drop_short_runs(near, prep.dur, min_s), method


def head_dips(prep: Prepared, body_state: np.ndarray, maze_shapes) -> dict:
    """EPM head dip: nose over the maze edge while the body is on an open arm or the centre."""
    nose = nose_name(prep)
    if not nose:
        return {}
    xy = prep.kp[nose]
    on_maze = np.zeros(len(xy), bool)
    for s in maze_shapes:
        on_maze |= s.contains(xy)
    dip = np.isfinite(xy).all(axis=1) & ~on_maze & body_state
    b = bout_summary(drop_short_runs(dip, prep.dur, 0.1), prep.t, prep.dur)
    return {"head_dips": b["bouts"], "head_dip_time_s": b["time_s"]}


def pair_metrics(a: Prepared, b: Prepared, social_cfg: dict) -> dict:
    """Two animals in the same arena: distance, proximity, nose-nose and nose-anogenital contact."""
    common, ia, ib = np.intersect1d(a.tracks.frame, b.tracks.frame, return_indices=True)
    if len(common) == 0:
        return {}
    t, dur = a.t[ia], a.dur[ia]

    def contact(pa, pb, limit):
        d = np.hypot(*(a.cal.to_cm(pa) - b.cal.to_cm(pb)).T)
        return bout_summary(np.nan_to_num(d, nan=np.inf) < limit, t, dur), d

    close, d = contact(a.xy[ia], b.xy[ib], social_cfg["proximity_cm"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            f"mean_distance_{a.units}": float(np.nanmean(d)) if np.isfinite(d).any() else None,
            "proximity_time_s": close["time_s"],
            "proximity_bouts": close["bouts"],
        }
    na, nb, ta, tb = nose_name(a), nose_name(b), tail_base_name(a), tail_base_name(b)
    lim = social_cfg["contact_cm"]
    if na and nb:
        s, _ = contact(a.kp[na][ia], b.kp[nb][ib], lim)
        out["nose_nose_time_s"], out["nose_nose_bouts"] = s["time_s"], s["bouts"]
    if na and tb:
        s, _ = contact(a.kp[na][ia], b.kp[tb][ib], lim)
        out["a_sniffs_b_anogenital_s"] = s["time_s"]
    if nb and ta:
        s, _ = contact(a.kp[ta][ia], b.kp[nb][ib], lim)
        out["b_sniffs_a_anogenital_s"] = s["time_s"]
    return out
