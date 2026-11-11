"""Experiment configuration (JSON): defaults, species presets, loading, command-line overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .arena import Arena
from .templates import PARADIGMS

DEFAULTS: dict = {
    "version": 1,
    "paradigm": "custom",
    "animal": {"species": "mouse", "color": "dark"},  # color: dark | light | any (vs. the floor)
    "tracking": {
        "method": "background",  # background (empty-arena image) | threshold (fixed intensity)
        "threshold": 30,  # gray-level difference from the background that counts as animal
        "intensity_threshold": 80,  # threshold method: darker (or brighter) than this is animal
        "blur": 5,
        "process_width": 960,  # frames are downscaled to this width for speed
        "tail_removal_cm": 0.8,  # morphological opening; removes the thin tail from the body blob
        "min_area_cm2": 2.0,
        "max_area_cm2": 80.0,
        "min_area_px": 60,  # used only when the arena is not calibrated
        "bg_samples": 40,
        "bg_update_rate": 0.0,  # >0 slowly adapts the background to lighting drift
        "motion_pixel_threshold": 12,  # gray-level change counted as movement (freezing)
        "arena_margin_cm": 3.0,  # mazes: tracking area extends this far beyond the drawn zones
    },
    "freezing": {
        "method": "auto",  # auto | motion (pixel change) | keypoints | speed
        "motion_threshold": 0.04,  # fraction of the animal's pixels that changed
        "keypoint_motion_cm": 0.3,  # body-part position spread within speed_window_s (pose data)
        "speed_threshold_cm_s": 1.0,
        "min_duration_s": 1.0,
    },
    "analysis": {
        "start_s": None,
        "end_s": None,
        "position": "centroid",  # centroid, or a body part name when pose keypoints exist
        "keypoint_min_p": 0.6,
        "smooth_s": 0.2,
        "max_gap_s": 0.5,
        "min_step_cm": 0.5,  # movement below this is treated as jitter (distance)
        "speed_window_s": 0.5,
        "max_speed_cm_s": 100.0,
        "moving_threshold_cm_s": 2.0,
        "min_zone_s": 0.2,
        "bin_s": 300,
    },
    "social": {"proximity_cm": 6.0, "contact_cm": 2.0},
    "params": {},
    "arenas": [],
}

SPECIES: dict[str, dict] = {
    "mouse": {},
    "rat": {
        "tracking": {"tail_removal_cm": 1.5, "min_area_cm2": 8.0, "max_area_cm2": 400.0},
        "analysis": {"min_step_cm": 1.0, "max_speed_cm_s": 150.0},
        "social": {"proximity_cm": 12.0, "contact_cm": 3.0},
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def complete(raw: dict) -> dict:
    """Fill a (possibly partial) config with defaults, the species preset and paradigm parameters."""
    species = (raw.get("animal") or {}).get("species", "mouse")
    cfg = deep_merge(DEFAULTS, SPECIES.get(species, {}))
    par = PARADIGMS.get(raw.get("paradigm", cfg["paradigm"]))
    if par:
        cfg["params"] = deep_merge(cfg["params"], par.params)
    return deep_merge(cfg, raw)


def new_config(paradigm: str, species: str = "mouse", color: str = "dark") -> dict:
    return complete({"paradigm": paradigm, "animal": {"species": species, "color": color}})


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return complete(json.load(f))


def save_config(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: dict, items: list[str] | None) -> dict:
    """Apply "section.key=value" overrides, e.g. params.novel_object=object_2 or analysis.end_s=300."""
    cfg = copy.deepcopy(cfg)
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set 형식은 key=value 입니다: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(value.strip())
    return cfg


def arenas_from(cfg: dict) -> list[Arena]:
    """Arenas of a config. Template zones ("auto": true) are rebuilt from the current params, so e.g.
    params.center_fraction takes effect on re-analysis. Remove "auto" from a zone to edit it by hand."""
    arenas = [Arena.from_dict(a) for a in cfg.get("arenas", [])]
    if not arenas:
        raise SystemExit("설정 파일에 아레나가 없습니다. 먼저 `behavtrack setup`으로 아레나를 그려주세요.")
    par = PARADIGMS.get(cfg.get("paradigm"))
    if par and par.auto_zones:
        for a in arenas:
            if any(z.auto for z in a.zones):
                a.zones = [z for z in a.zones if not z.auto] + par.auto_zones(a, cfg.get("params", {}))
    return arenas
