"""Metrics shared by every paradigm: cleaning, distance/speed, zone occupancy, freezing, time bins."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..arena import Arena
from ..geometry import Calibration
from ..tracks import Tracks

NAN = float("nan")


# ---------------------------------------------------------------- boolean series helpers

def runs(mask) -> list[tuple[int, int]]:
    """(start, stop) index pairs of consecutive True values."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def drop_short_runs(mask, dur, min_s: float) -> np.ndarray:
    """Remove True runs shorter than min_s seconds."""
    out = np.asarray(mask, bool).copy()
    if min_s and min_s > 0:
        for a, b in runs(out):
            if dur[a:b].sum() < min_s:
                out[a:b] = False
    return out


def debounce(state, dur, min_s: float) -> np.ndarray:
    """Ignore flicker at zone borders: a run (True or False) shorter than min_s keeps the previous state."""
    out = np.asarray(state, bool).copy()
    if not min_s or min_s <= 0 or len(out) < 2:
        return out
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(out.astype(np.int8))) + 1, [len(out)]])
    for a, b in zip(bounds[1:-1], bounds[2:]):
        if dur[a:b].sum() < min_s:
            out[a:b] = out[a - 1]
    return out


def bout_summary(mask, t, dur) -> dict:
    r = runs(mask)
    lengths = [float(dur[a:b].sum()) for a, b in r]
    return {
        "time_s": float(sum(lengths)),
        "bouts": len(r),
        "mean_bout_s": float(np.mean(lengths)) if lengths else 0.0,
        "latency_s": float(t[r[0][0]] - t[0]) if r else None,
    }


def rising_edges(state) -> np.ndarray:
    state = np.asarray(state, bool)
    return np.flatnonzero(~state[:-1] & state[1:]) + 1


# ---------------------------------------------------------------- position cleaning

def frame_durations(t) -> np.ndarray:
    """Seconds each frame represents (works with the uneven timing of webcams)."""
    if len(t) == 0:
        return np.zeros(0)
    if len(t) == 1:
        return np.array([1 / 30])
    d = np.diff(t)
    pos = d[d > 0]
    return np.append(np.clip(d, 0, None), float(np.median(pos)) if len(pos) else 1 / 30)


def fill_gaps(v, t, max_gap_s: float) -> np.ndarray:
    """Linear interpolation over short tracking dropouts (no extrapolation at the ends)."""
    v = np.array(v, float, copy=True)
    flat = v.ndim == 1
    if flat:
        v = v[:, None]
    valid = np.isfinite(v).all(axis=1)
    if valid.sum() >= 2 and max_gap_s > 0:
        for a, b in runs(~valid):
            if a == 0 or b >= len(v) or t[b] - t[a - 1] > max_gap_s:
                continue
            for c in range(v.shape[1]):
                v[a:b, c] = np.interp(t[a:b], [t[a - 1], t[b]], [v[a - 1, c], v[b, c]])
    return v[:, 0] if flat else v


def rolling_median(v, win: int) -> np.ndarray:
    """Centered median filter that removes single-frame spikes; missing frames stay missing."""
    v = np.asarray(v, float)
    win = int(win) | 1
    if win < 3 or len(v) < win:
        return v.copy()
    flat = v.ndim == 1
    x = v[:, None] if flat else v
    pad = win // 2
    p = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(sliding_window_view(p, win, axis=0), axis=-1)
    med[~np.isfinite(x).all(axis=1)] = np.nan
    return med[:, 0] if flat else med


def path_steps(xy, t, min_step: float, max_speed: float | None) -> np.ndarray:
    """Distance travelled at each frame. Movement is counted only once the animal has moved
    min_step away from the last counted point, so centroid jitter of a resting animal adds nothing.
    Jumps faster than max_speed (tracking errors) and jumps across long dropouts are not counted."""
    step = np.zeros(len(t))
    anchor, t_anchor, gap = None, 0.0, False
    for i in range(len(t)):
        p = xy[i]
        if not (np.isfinite(p[0]) and np.isfinite(p[1])):
            gap = anchor is not None
            continue
        if anchor is None or gap:
            anchor, t_anchor, gap = p, t[i], False
            continue
        d = float(np.hypot(p[0] - anchor[0], p[1] - anchor[1]))
        if d >= min_step:
            dt = t[i] - t_anchor
            if not (max_speed and dt > 0 and d / dt > max_speed):
                step[i] = d
            anchor, t_anchor = p, t[i]
    return step


def windowed_speed(xy, t, window_s: float, max_speed: float | None = None) -> np.ndarray:
    n = len(t)
    if n < 2:
        return np.full(n, NAN)
    dt = float(np.median(np.diff(t))) or 1 / 30
    k = max(1, int(round(window_s / 2 / dt)))
    i = np.arange(n)
    lo, hi = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
    d = np.hypot(*(xy[hi] - xy[lo]).T)
    span = t[hi] - t[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        sp = np.where(span > 0, d / span, NAN)
    if max_speed:
        sp[sp > max_speed] = NAN
    return sp


# ---------------------------------------------------------------- prepared session

@dataclass
class Prepared:
    t: np.ndarray
    dur: np.ndarray
    xy: np.ndarray  # cleaned reference point, image px
    xy_cm: np.ndarray  # same in calibrated units (cm, or px when uncalibrated)
    valid: np.ndarray
    step: np.ndarray
    speed: np.ndarray
    cal: Calibration
    tracks: Tracks  # trimmed to the analysis window
    kp: dict[str, np.ndarray] = field(default_factory=dict)  # cleaned keypoints, image px (N, 2)
    kp_raw: dict[str, np.ndarray] = field(default_factory=dict)  # same before smoothing (keeps fast movement)

    @property
    def units(self) -> str:
        return self.cal.units

    @property
    def scale(self) -> float:
        return self.cal.scale

    @property
    def total_s(self) -> float:
        return float(self.dur.sum())

    @property
    def tracked_s(self) -> float:
        return float(self.dur[self.valid].sum())


def prepare(tracks: Tracks, arena: Arena, cfg: dict) -> Prepared:
    a = cfg["analysis"]
    sel = np.ones(len(tracks), bool)
    if a.get("start_s") is not None:
        sel &= tracks.time >= float(a["start_s"])
    if a.get("end_s") is not None:
        sel &= tracks.time < float(a["end_s"])
    tr = tracks.subset(sel)
    t = tr.time.astype(float)
    dur = frame_durations(t)
    smooth = int(round(a["smooth_s"] / (float(np.median(dur)) if len(dur) else 1 / 30)))

    kp, kp_raw = {}, {}
    for name, arr in tr.keypoints.items():
        xy = arr[:, :2].copy()
        xy[~(arr[:, 2] >= a["keypoint_min_p"])] = np.nan
        kp_raw[name] = fill_gaps(xy, t, a["max_gap_s"])
        kp[name] = rolling_median(kp_raw[name], smooth)

    pos = a.get("position") or "centroid"
    if pos in ("centroid", "auto"):
        xy = rolling_median(fill_gaps(tr.xy, t, a["max_gap_s"]), smooth)
    elif pos in kp:
        xy = kp[pos]
    else:
        raise SystemExit(f"analysis.position='{pos}' 를 찾을 수 없습니다 (가능: centroid, {', '.join(kp)})")

    cal = arena.calibration
    xy_cm = cal.to_cm(xy) if len(xy) else np.zeros((0, 2))
    max_speed = a["max_speed_cm_s"] if cal.calibrated else None
    min_step = a["min_step_cm"] if cal.calibrated else 3.0
    return Prepared(
        t=t, dur=dur, xy=xy, xy_cm=xy_cm, valid=np.isfinite(xy).all(axis=1),
        step=path_steps(xy_cm, t, min_step, max_speed),
        speed=windowed_speed(xy_cm, t, a["speed_window_s"], max_speed),
        cal=cal, tracks=tr, kp=kp, kp_raw=kp_raw,
    )


# ---------------------------------------------------------------- zones

def forward_filled(raw, valid) -> np.ndarray:
    """During tracking dropouts keep the last known state (so a dropout is not a zone exit)."""
    n = len(raw)
    if n == 0:
        return np.zeros(0, bool)
    last = np.maximum.accumulate(np.where(valid, np.arange(n), 0))
    seen = np.maximum.accumulate(np.asarray(valid, bool))
    return np.asarray(raw, bool)[last] & seen


def membership(prep: Prepared, shape, min_s: float) -> np.ndarray:
    return debounce(forward_filled(shape.contains(prep.xy), prep.valid), prep.dur, min_s)


def zone_stats(state, prep: Prepared) -> dict:
    state = np.asarray(state, bool)
    if len(state) == 0:
        return {"time_s": 0.0, "time_pct": 0.0, "entries": 0, "latency_s": None,
                f"distance_{prep.units}": 0.0, "mean_visit_s": 0.0}
    time = float(prep.dur[state & prep.valid].sum())
    rising = rising_edges(state)
    first = 0 if state[0] else (int(rising[0]) if len(rising) else None)
    visits = len(rising) + int(state[0])
    tracked = prep.tracked_s
    return {
        "time_s": time,
        "time_pct": 100 * time / tracked if tracked else 0.0,
        "entries": int(len(rising)),
        "latency_s": None if first is None else float(prep.t[first] - prep.t[0]),
        f"distance_{prep.units}": float(prep.step[state].sum()),
        "mean_visit_s": time / visits if visits else 0.0,
    }


# ---------------------------------------------------------------- freezing

def keypoint_motion(prep: Prepared, window_s: float) -> np.ndarray:
    """Largest body-part position spread (SD, cm) within a sliding window. Unlike speed it also
    catches back-and-forth movement such as grooming, which nets almost no displacement."""
    if not prep.kp_raw or len(prep.t) < 2:
        return np.full(len(prep.t), NAN)
    win = int(round(window_s / (float(np.median(prep.dur)) or 1 / 30))) | 1
    spreads = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for xy in prep.kp_raw.values():
            cm = prep.cal.to_cm(xy)
            p = np.pad(cm, ((win // 2, win // 2), (0, 0)), mode="edge")
            sd = np.nanstd(sliding_window_view(p, win, axis=0), axis=-1)
            spreads.append(np.hypot(sd[:, 0], sd[:, 1]))
        return np.nanmax(np.vstack(spreads), axis=0)


def freezing(prep: Prepared, cfg: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Freezing = no movement except breathing for at least min_duration_s.

    motion:    fraction of the animal's pixels that changed between frames (blob tracker; most
               like VideoFreeze / ezTrack, and grooming is correctly NOT counted as freezing)
    keypoints: largest body-part movement within a short window (pose estimation)
    speed:     centroid speed (fallback; cannot tell grooming from freezing)
    """
    f = cfg["freezing"]
    method = f["method"]
    motion = prep.tracks.extra.get("motion")
    if method == "auto":
        if motion is not None and np.isfinite(motion).any():
            method = "motion"
        else:
            method = "keypoints" if prep.kp else "speed"
    if method == "motion":
        if motion is None:
            raise SystemExit("트랙에 motion 값이 없습니다. freezing.method를 keypoints 또는 speed로 바꿔주세요.")
        value, thr = rolling_median(motion, 5), f["motion_threshold"]
        observed = np.isfinite(motion) & np.isfinite(prep.tracks.xy).all(axis=1)
    elif method == "keypoints":
        value, thr = keypoint_motion(prep, cfg["analysis"]["speed_window_s"]), f["keypoint_motion_cm"]
        observed = prep.valid.copy()
        # Short pose gaps have already been interpolated; long gaps remain unknown.
        for points in prep.kp_raw.values():
            observed &= np.isfinite(points).all(axis=1)
    else:
        value, thr = prep.speed, f["speed_threshold_cm_s"]
        observed = np.isfinite(prep.tracks.xy).all(axis=1)
    if method in ("keypoints", "speed") and len(observed):
        radius = max(1, int(round(cfg["analysis"]["speed_window_s"] / 2 / (float(np.median(prep.dur)) or 1 / 30))))
        missing = np.pad((~observed).astype(int), (radius, radius), mode="edge")
        cs = np.r_[0, np.cumsum(missing)]
        observed &= (cs[2 * radius + 1:] - cs[:-(2 * radius + 1)]) == 0
    observed &= np.isfinite(value)
    still = observed & (value < thr)
    return drop_short_runs(still, prep.dur, f["min_duration_s"]), observed, method


# ---------------------------------------------------------------- summary

def common_metrics(prep: Prepared, freeze: np.ndarray, freeze_observed: np.ndarray, freeze_method: str, cfg: dict) -> dict:
    u = prep.units
    total, tracked = prep.total_s, prep.tracked_s
    moving = np.nan_to_num(prep.speed) > cfg["analysis"]["moving_threshold_cm_s"]
    moving_s = float(prep.dur[moving].sum())
    dist = float(prep.step.sum())
    fz = bout_summary(freeze, prep.t, prep.dur)
    observed_s = float(prep.dur[freeze_observed].sum())
    return {
        "duration_s": total,
        "tracked_pct": 100 * tracked / total if total else 0.0,
        f"distance_{u}": dist,
        f"mean_speed_{u}_s": dist / tracked if tracked else 0.0,
        f"moving_speed_{u}_s": float(prep.step[moving].sum()) / moving_s if moving_s else 0.0,
        "moving_time_s": moving_s,
        "moving_pct": 100 * moving_s / tracked if tracked else 0.0,
        "freezing_time_s": fz["time_s"],
        "freezing_observed_s": observed_s,
        "freezing_unknown_s": max(0.0, total - observed_s),
        "freezing_pct": 100 * fz["time_s"] / observed_s if observed_s else None,
        "freezing_bouts": fz["bouts"],
        "freezing_mean_bout_s": fz["mean_bout_s"],
        "freezing_latency_s": fz["latency_s"],
        "freezing_method": freeze_method,
    }


def time_bins(prep: Prepared, freeze, freeze_observed, states: dict, cfg: dict) -> list[dict]:
    bin_s = cfg["analysis"].get("bin_s")
    if not bin_s or len(prep.t) == 0:
        return []
    u = prep.units
    rel = prep.t - prep.t[0]
    end = float(rel[-1] + prep.dur[-1])
    moving = np.nan_to_num(prep.speed) > cfg["analysis"]["moving_threshold_cm_s"]
    rows = []
    for k in range(int(np.ceil(end / bin_s - 1e-9))):
        sel = (rel >= k * bin_s) & (rel < (k + 1) * bin_s)
        d = prep.dur[sel]
        total = float(d.sum())
        observed_s = float(d[freeze_observed[sel]].sum())
        row = {
            "bin": k + 1,
            "start_s": k * bin_s,
            "end_s": min((k + 1) * bin_s, end),
            f"distance_{u}": float(prep.step[sel].sum()),
            "moving_s": float(d[moving[sel]].sum()),
            "freezing_s": float(d[freeze[sel]].sum()),
            "freezing_observed_s": observed_s,
            "freezing_unknown_s": max(0.0, total - observed_s),
            "freezing_pct": 100 * float(d[freeze[sel]].sum()) / observed_s if observed_s else None,
        }
        for name, st in states.items():
            row[f"{name}_time_s"] = float(d[(st & prep.valid)[sel]].sum())
        rows.append(row)
    return rows
