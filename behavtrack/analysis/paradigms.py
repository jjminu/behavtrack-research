"""Paradigm-specific metrics, computed on top of the common ones."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..arena import Arena
from ..geometry import Circle
from .common import (Prepared, bout_summary, debounce, forward_filled, rising_edges, zone_stats)
from .keypoints import head_dips, nose_name, object_exploration


@dataclass
class Context:
    prep: Prepared
    arena: Arena
    params: dict
    states: dict[str, np.ndarray]  # debounced zone membership per frame
    zones: dict[str, dict]  # zone statistics
    freeze: np.ndarray
    freeze_observed: np.ndarray
    cfg: dict

    def union(self, names) -> np.ndarray:
        raw = np.zeros(len(self.prep.t), bool)
        for n in names:
            raw |= self.arena.zone(n).shape.contains(self.prep.xy)
        return debounce(forward_filled(raw, self.prep.valid), self.prep.dur, self.cfg["analysis"]["min_zone_s"])

    def stats(self, state) -> dict:
        return zone_stats(state, self.prep)


def _ratio(a, b):
    return a / b if b else None


def _index(target, other):
    """(target - other) / (target + other): +1 = only target, 0 = no preference, -1 = only other."""
    return _ratio(target - other, target + other)


def open_field(ctx: Context) -> dict:
    c = ctx.zones.get("center")
    if c is None:
        return {}
    u = ctx.prep.units
    total = float(ctx.prep.step.sum())
    tracked = ctx.prep.tracked_s
    return {
        "center_time_s": c["time_s"],
        "center_time_pct": c["time_pct"],
        "center_entries": c["entries"],
        "center_latency_s": c["latency_s"],
        f"center_distance_{u}": c[f"distance_{u}"],
        "center_distance_pct": 100 * c[f"distance_{u}"] / total if total else 0.0,
        "periphery_time_s": tracked - c["time_s"],
        "periphery_time_pct": 100 - c["time_pct"] if tracked else 0.0,
    }


def plus_maze(ctx: Context) -> dict:
    """EPM and zero maze. Arm entries count each arm separately (centroid-based)."""
    opens = [z.name for z in ctx.arena.zones_with_prefix("open")]
    closed = [z.name for z in ctx.arena.zones_with_prefix("closed")]
    if not opens or not closed:
        return {}
    u = ctx.prep.units
    o_state = ctx.union(opens)
    o, c = ctx.stats(o_state), ctx.stats(ctx.union(closed))
    oe = sum(ctx.zones[n]["entries"] for n in opens)
    ce = sum(ctx.zones[n]["entries"] for n in closed)
    out = {
        "open_time_s": o["time_s"],
        "closed_time_s": c["time_s"],
        "open_time_pct": 100 * _ratio(o["time_s"], o["time_s"] + c["time_s"]) if o["time_s"] + c["time_s"] else None,
        "open_entries": oe,
        "closed_entries": ce,
        "total_arm_entries": oe + ce,
        "open_entries_pct": 100 * oe / (oe + ce) if oe + ce else None,
        "open_latency_s": o["latency_s"],
        f"open_distance_{u}": o[f"distance_{u}"],
    }
    body = o_state.copy()
    if "center" in ctx.states:
        out["center_time_s"] = ctx.zones["center"]["time_s"]
        body |= ctx.states["center"]
    out.update(head_dips(ctx.prep, body, [z.shape for z in ctx.arena.zones]))
    return out


def y_maze(ctx: Context) -> dict:
    """Spontaneous alternation: consecutive entries into three different arms / (entries - 2)."""
    arms = sorted(z.name for z in ctx.arena.zones_with_prefix("arm"))
    if len(arms) < 3:
        return {}
    label = np.full(len(ctx.prep.t), -1)
    for k, a in enumerate(arms):
        label[ctx.states[a] & (label < 0)] = k
    if len(label) == 0:
        return {}
    starts = np.flatnonzero((label[1:] != label[:-1]) & (label[1:] >= 0)) + 1
    seq = label[starts].tolist()
    if label[0] >= 0 and ctx.params.get("count_start_arm"):
        seq.insert(0, int(label[0]))
    n = len(seq)
    alt = sum(len({seq[i], seq[i + 1], seq[i + 2]}) == 3 for i in range(n - 2))
    letters = [arms[k].split("_", 1)[-1] for k in seq]
    return {
        "arm_entries": n,
        "alternations": alt,
        "alternation_pct": 100 * alt / (n - 2) if n > 2 else None,
        "same_arm_returns": sum(seq[i] == seq[i + 1] for i in range(n - 1)),
        "alternate_arm_returns": sum(seq[i] == seq[i + 2] and seq[i] != seq[i + 1] for i in range(n - 2)),
        "arm_sequence": "".join(letters) if all(len(x) == 1 for x in letters) else "-".join(letters),
    }


def nor(ctx: Context) -> dict:
    objs = ctx.arena.zones_with_prefix("object")
    if not objs:
        return {}
    p, prep = ctx.params, ctx.prep
    out, times, masks, method = {}, {}, {}, ""
    for z in objs:
        mask, method = object_exploration(prep, z.shape, p["explore_cm"], p["centroid_explore_cm"],
                                          p["exclude_on_object"], p["min_explore_s"])
        b = bout_summary(mask, prep.t, prep.dur)
        out[f"{z.name}_explore_s"] = b["time_s"]
        out[f"{z.name}_bouts"] = b["bouts"]
        out[f"{z.name}_latency_s"] = b["latency_s"]
        times[z.name], masks[z.name] = b["time_s"], mask
    total = sum(times.values())
    out["exploration_method"] = method
    out["total_explore_s"] = total
    novel = p.get("novel_object")
    if novel:
        if novel not in times:
            raise SystemExit(f"params.novel_object='{novel}' 가 물체 구역에 없습니다 (가능: {', '.join(times)})")
        n_t = times[novel]
        out["discrimination_index"] = _index(n_t, total - n_t)
        out["novel_preference_pct"] = 100 * n_t / total if total else None
        crit = p.get("criterion_s")
        if crit:
            any_obj = np.zeros(len(prep.t), bool)
            for m in masks.values():
                any_obj |= m
            cum = np.cumsum(prep.dur * any_obj)
            hit = np.flatnonzero(cum >= float(crit))
            if len(hit):
                k = hit[0] + 1
                nn = float(prep.dur[:k][masks[novel][:k]].sum())
                out["time_to_criterion_s"] = float(prep.t[hit[0]] - prep.t[0])
                out["discrimination_index_at_criterion"] = _index(nn, float(cum[hit[0]]) - nn)
            else:
                out["time_to_criterion_s"] = None
    elif len(objs) == 2:
        a, b = objs[0].name, objs[1].name
        out[f"index_{b}_vs_{a}"] = _index(times[b], times[a])
        out[f"{b}_preference_pct"] = 100 * times[b] / total if total else None
    return out


def light_dark(ctx: Context) -> dict:
    dark = ctx.arena.zone("dark")
    if dark is None:
        return {}
    prep, u = ctx.prep, ctx.prep.units
    in_dark = dark.shape.contains(prep.xy)
    visible = prep.valid & np.isfinite(prep.tracks.xy).all(axis=1)
    if ctx.params.get("undetected_as") == "dark":
        # the covered dark box hides the animal from the camera: not visible = in the dark box
        state = debounce(in_dark | ~visible, prep.dur, ctx.cfg["analysis"]["min_zone_s"])
        counted = np.ones(len(state), bool)
    else:
        state = debounce(forward_filled(in_dark, visible), prep.dur, ctx.cfg["analysis"]["min_zone_s"])
        counted = visible
    dark_s = float(prep.dur[state & counted].sum())
    light_s = float(prep.dur[~state & counted].sum())
    valid_edge = counted[1:] & counted[:-1]
    dark_in = rising_edges(state)
    dark_in = dark_in[valid_edge[dark_in - 1]]
    light_in = rising_edges(~state)
    light_in = light_in[valid_edge[light_in - 1]]
    first = 0 if len(state) and state[0] else (int(dark_in[0]) if len(dark_in) else None)
    return {
        "light_time_s": light_s,
        "light_time_pct": 100 * light_s / (light_s + dark_s) if light_s + dark_s else None,
        "dark_time_s": dark_s,
        "undetected_time_s": float(prep.dur[~visible].sum()),
        "undetected_handling": ctx.params.get("undetected_as", "unknown"),
        "transitions": int(len(dark_in) + len(light_in)),
        "dark_entries": int(len(dark_in)),
        "light_entries": int(len(light_in)),
        "dark_latency_s": None if first is None else float(prep.t[first] - prep.t[0]),
        f"light_distance_{u}": float(prep.step[~state].sum()),
    }


def three_chamber(ctx: Context) -> dict:
    p, prep = ctx.params, ctx.prep
    out, chamber, sniff, method = {}, {}, {}, ""
    for side in ("A", "B"):
        z = ctx.zones.get(f"chamber_{side}")
        if z:
            chamber[side] = z["time_s"]
            out[f"chamber_{side}_time_s"] = z["time_s"]
            out[f"chamber_{side}_entries"] = z["entries"]
        cup = ctx.arena.zone(f"cup_{side}")
        if cup:
            mask, method = object_exploration(prep, cup.shape, p["explore_cm"], p["interaction_cm"])
            b = bout_summary(mask, prep.t, prep.dur)
            sniff[side] = b["time_s"]
            out[f"cup_{side}_sniff_s"] = b["time_s"]
            out[f"cup_{side}_sniff_bouts"] = b["bouts"]
    if "chamber_center" in ctx.zones:
        out["chamber_center_time_s"] = ctx.zones["chamber_center"]["time_s"]
    if method:
        out["interaction_method"] = method
    side = p.get("social_side")
    if side in ("A", "B"):
        other = "B" if side == "A" else "A"
        out["social_side"] = side
        if side in chamber and other in chamber:
            out["social_chamber_time_s"], out["other_chamber_time_s"] = chamber[side], chamber[other]
            out["chamber_preference_index"] = _index(chamber[side], chamber[other])
        if side in sniff and other in sniff:
            out["social_sniff_s"], out["other_sniff_s"] = sniff[side], sniff[other]
            out["sniff_preference_index"] = _index(sniff[side], sniff[other])
    else:
        if len(chamber) == 2:
            out["chamber_index_A_vs_B"] = _index(chamber["A"], chamber["B"])
        if len(sniff) == 2:
            out["sniff_index_A_vs_B"] = _index(sniff["A"], sniff["B"])
    return out


def cpp(ctx: Context) -> dict:
    out = {}
    for name in ("chamber_A", "chamber_B", "neutral"):
        if name in ctx.zones:
            out[f"{name}_time_s"] = ctx.zones[name]["time_s"]
            out[f"{name}_time_pct"] = ctx.zones[name]["time_pct"]
            out[f"{name}_entries"] = ctx.zones[name]["entries"]
    paired = ctx.params.get("paired_chamber")
    if paired:
        if paired not in ("chamber_A", "chamber_B"):
            raise SystemExit("params.paired_chamber 는 chamber_A 또는 chamber_B 입니다.")
        other = "chamber_B" if paired == "chamber_A" else "chamber_A"
        pt, ot = ctx.zones[paired]["time_s"], ctx.zones[other]["time_s"]
        out["paired_chamber"] = paired
        out["preference_score_s"] = pt - ot
        out["preference_ratio"] = _ratio(pt, pt + ot)
    return out


OPPOSITE = {"Q_NE": "Q_SW", "Q_SW": "Q_NE", "Q_NW": "Q_SE", "Q_SE": "Q_NW"}


def target_maze(ctx: Context, target_name: str) -> dict:
    """Morris water maze (platform) and Barnes maze (escape hole)."""
    target = ctx.arena.zone(target_name)
    if target is None:
        return {}
    p, prep, u = ctx.params, ctx.prep, ctx.prep.units
    n = len(prep.t)
    nose = nose_name(prep)
    if nose and target_name == "target":
        reach = target.shape.contains(prep.kp[nose])
    else:
        reach = np.nan_to_num(target.shape.boundary_distance(prep.xy) / prep.scale, nan=np.inf) <= p.get("approach_cm", 0.0)
    hit = np.flatnonzero(reach)
    found = len(hit) > 0
    upto = np.arange(n) <= hit[0] if (found and p.get("stop_at_target")) else np.ones(n, bool)
    path = float(prep.step[: hit[0] + 1].sum()) if found else float(prep.step.sum())
    latency = float(prep.t[hit[0]] - prep.t[0]) if found else None
    tc = target.shape.centroid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(prep.xy[upto, 0] - tc[0], prep.xy[upto, 1] - tc[1]) / prep.scale
        mean_dist = float(np.nanmean(dist)) if np.isfinite(dist).any() else None
    crossings = debounce(reach, prep.dur, ctx.cfg["analysis"]["min_zone_s"])
    out = {
        "found_target": bool(found),
        "latency_s": latency,
        f"path_to_target_{u}": path,
        f"speed_to_target_{u}_s": _ratio(path, latency) if found else None,
        f"mean_target_distance_{u}": mean_dist,
        "target_crossings": int(len(rising_edges(crossings)) + int(bool(n and crossings[0]))),
    }
    quads = [z for z in ctx.arena.zones if z.name.startswith("Q_")]
    if quads:
        tq = p.get("target_quadrant", "auto")
        if tq in (None, "auto"):
            tq = next((q.name for q in quads if q.shape.contains(tc)[0]), quads[0].name)
        base = float(prep.dur[prep.valid & upto].sum())
        for q in quads:
            out[f"{q.name}_pct"] = 100 * float(prep.dur[ctx.states[q.name] & prep.valid & upto].sum()) / base if base else None
        out["target_quadrant"] = tq
        out["target_quadrant_pct"] = out.get(f"{tq}_pct")
        if tq in OPPOSITE:
            out["opposite_quadrant_pct"] = out.get(f"{OPPOSITE[tq]}_pct")
    if target_name == "platform" and isinstance(ctx.arena.shape, Circle):
        c, r = ctx.arena.shape.center, ctx.arena.shape.radius / prep.scale
        from_center = np.hypot(prep.xy[:, 0] - c[0], prep.xy[:, 1] - c[1]) / prep.scale
        wall = np.nan_to_num(from_center, nan=-1) > r - p.get("wall_zone_cm", 10.0)
        base = float(prep.dur[prep.valid & upto].sum())
        out["thigmotaxis_pct"] = 100 * float(prep.dur[wall & prep.valid & upto].sum()) / base if base else None
    return out


def fear_conditioning(ctx: Context) -> dict:
    """Freezing per epoch. Epochs are seconds from the start of the recording."""
    p, prep = ctx.params, ctx.prep
    epochs = list(p.get("epochs") or [])
    onsets = p.get("tone_onsets") or []
    if onsets:
        d = float(p.get("tone_duration_s", 20.0))
        tones = [{"name": f"tone_{i + 1}", "start": s, "end": s + d} for i, s in enumerate(onsets)]
        epochs = [{"name": "baseline", "start": 0, "end": onsets[0]}, *tones, *epochs]
    out = {}
    for e in epochs:
        sel = (prep.t >= float(e["start"])) & (prep.t < float(e["end"]))
        total = float(prep.dur[sel & ctx.freeze_observed].sum())
        fz = float(prep.dur[sel & ctx.freeze].sum())
        out[f"freezing_observed_s_{e['name']}"] = total
        out[f"freezing_unknown_s_{e['name']}"] = float(prep.dur[sel & ~ctx.freeze_observed].sum())
        out[f"freezing_pct_{e['name']}"] = 100 * fz / total if total else None
        out[f"freezing_s_{e['name']}"] = fz
    return out


PARADIGM_METRICS = {
    "open_field": open_field,
    "epm": plus_maze,
    "ezm": plus_maze,
    "y_maze": y_maze,
    "nor": nor,
    "light_dark": light_dark,
    "three_chamber": three_chamber,
    "cpp": cpp,
    "mwm": lambda ctx: target_maze(ctx, "platform"),
    "barnes": lambda ctx: target_maze(ctx, "target"),
    "fear_conditioning": fear_conditioning,
    "custom": lambda ctx: {},
}
