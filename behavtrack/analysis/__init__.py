"""Analysis of one animal's track: common metrics + paradigm metrics + classifier behaviors."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..arena import Arena
from ..tracks import Tracks
from .common import Prepared, bout_summary, common_metrics, freezing, membership, prepare, time_bins, zone_stats
from .keypoints import pair_metrics
from .paradigms import PARADIGM_METRICS, Context


@dataclass
class Result:
    track_id: str
    arena: Arena
    paradigm: str
    prep: Prepared
    metrics: dict
    zones: dict[str, dict]
    bins: list[dict]
    states: dict[str, np.ndarray]
    freeze: np.ndarray
    freeze_observed: np.ndarray
    behaviors: dict[str, np.ndarray] = field(default_factory=dict)

    def flat(self) -> dict:
        """One row for the summary table."""
        row = dict(self.metrics)
        for zname, stats in self.zones.items():
            for k, v in stats.items():
                row[f"zone_{zname}_{k}"] = v
        return row

    def save(self, session_dir) -> None:
        d = Path(session_dir)
        with open(d / f"metrics_{self.track_id}.json", "w", encoding="utf-8") as f:
            json.dump({"track": self.track_id, "arena": self.arena.name, "paradigm": self.paradigm,
                       "units": self.prep.units, "metrics": self.metrics, "zones": self.zones},
                      f, ensure_ascii=False, indent=2, default=_json_default)
        if self.bins:
            write_rows(d / f"bins_{self.track_id}.csv", self.bins)
        write_rows(d / f"frames_{self.track_id}.csv", self.frame_rows())

    def frame_rows(self) -> list[dict]:
        p, u = self.prep, self.prep.units
        cols = {
            "frame": p.tracks.frame, "time_s": p.t, "x_px": p.xy[:, 0], "y_px": p.xy[:, 1],
            f"x_{u}": p.xy_cm[:, 0], f"y_{u}": p.xy_cm[:, 1], f"speed_{u}_s": p.speed,
            "freezing": [int(v) if known else None for v, known in zip(self.freeze, self.freeze_observed)],
            "freezing_observed": self.freeze_observed.astype(int),
            **{f"in_{k}": v.astype(int) for k, v in self.states.items()},
            **{k: v.astype(int) for k, v in self.behaviors.items()},
        }
        names = list(cols)
        return [dict(zip(names, vals)) for vals in zip(*cols.values())]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return v


def write_rows(path, rows: list[dict]) -> None:
    if not rows:
        return
    names = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in names})


def analyze(tracks: Tracks, arena: Arena, cfg: dict, track_id: str = "A",
            behaviors: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> Result:
    prep = prepare(tracks, arena, cfg)
    min_s = cfg["analysis"]["min_zone_s"]
    states = {z.name: membership(prep, z.shape, min_s) for z in arena.zones}
    zones = {name: zone_stats(st, prep) for name, st in states.items()}
    freeze, freeze_observed, method = freezing(prep, cfg)
    metrics = common_metrics(prep, freeze, freeze_observed, method, cfg)
    paradigm = cfg.get("paradigm", "custom")
    ctx = Context(prep, arena, cfg.get("params", {}), states, zones, freeze, freeze_observed, cfg)
    metrics.update(PARADIGM_METRICS.get(paradigm, PARADIGM_METRICS["custom"])(ctx))

    aligned = {}
    for name, (frames, labels) in (behaviors or {}).items():
        lookup = dict(zip(frames.tolist(), labels.tolist()))
        values=np.array([lookup.get(int(f),-1) for f in prep.tracks.frame],int)
        mask = values==1
        known_s=float(prep.dur[values>=0].sum())
        b = bout_summary(mask, prep.t, prep.dur)
        metrics[f"{name}_time_s"] = b["time_s"]
        metrics[f"{name}_pct"] = 100 * b["time_s"] / known_s if known_s else None
        metrics[f"{name}_known_s"]=known_s
        metrics[f"{name}_unknown_s"]=float(prep.dur[values<0].sum())
        metrics[f"{name}_bouts"] = b["bouts"]
        metrics[f"{name}_latency_s"] = b["latency_s"]
        aligned[name] = values
    return Result(track_id, arena, paradigm, prep, metrics, zones, time_bins(prep, freeze, freeze_observed, states, cfg),
                  states, freeze, freeze_observed, aligned)


def social(results: dict[str, Result], cfg: dict) -> dict[str, dict]:
    """Pairwise metrics for animals that share an arena (multi-animal pose tracking)."""
    out = {}
    ids = list(results)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if results[a].arena.name == results[b].arena.name:
                m = pair_metrics(results[a].prep, results[b].prep, cfg["social"])
                if m:
                    out[f"{a}~{b}"] = m
    return out


__all__ = ["Result", "analyze", "social", "write_rows"]
