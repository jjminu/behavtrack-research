"""Synthetic data with known ground truth: a rendered 'mouse' in an open field.

Used by `behavtrack demo` (try the whole pipeline without a camera) and by the tests.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

PX_PER_CM = 10.0
SIZE_CM = 40.0
OFFSET = np.array([40.0, 40.0])  # arena top-left corner in px
FRAME_HW = (480, 480)


def corners_px(size_cm: float = SIZE_CM) -> np.ndarray:
    s = size_cm * PX_PER_CM
    return OFFSET + np.array([[0, 0], [s, 0], [s, s], [0, s]])


def simulate_path(seconds: float = 60, fps: float = 30, seed: int = 1, freezes=((12, 16), (34, 40)),
                  grooms=((48, 52),), speed_cm_s: float = 12.0, size_cm: float = SIZE_CM, margin: float = 6.0) -> dict:
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    still = np.zeros(n, bool)
    groom = np.zeros(n, bool)
    for a, b in freezes:
        still[(t >= a) & (t < b)] = True
    for a, b in grooms:
        groom[(t >= a) & (t < b)] = True
    pos, heading = np.zeros((n, 2)), np.zeros(n)
    p, h = np.array([size_cm * 0.25, size_cm * 0.3]), 0.0
    target = rng.uniform(margin, size_cm - margin, 2)
    for i in range(n):
        if not (still[i] or groom[i]):
            d = target - p
            if np.hypot(*d) < 1.5:
                target = rng.uniform(margin, size_cm - margin, 2)
                d = target - p
            diff = (math.atan2(d[1], d[0]) - h + math.pi) % (2 * math.pi) - math.pi
            h += float(np.clip(diff, -0.12, 0.12))
            p = np.clip(p + speed_cm_s / fps * np.array([math.cos(h), math.sin(h)]), margin / 2, size_cm - margin / 2)
        pos[i], heading[i] = p, h
    return {"t": t, "pos": pos, "heading": heading, "freeze": still, "groom": groom, "fps": fps}


def posture(path_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame drawn body angle (rad), length and width (cm); grooming wiggles, freezing breathes."""
    n = len(path_data["t"])
    i = np.arange(n)
    angle = path_data["heading"].copy()
    length, width = np.full(n, 3.5), np.full(n, 1.6)
    g, f = path_data["groom"], path_data["freeze"]
    angle[g] += np.radians(18 * np.sin(i[g] * 0.9))
    length[g] *= 0.85 + 0.08 * np.sin(i[g] * 0.9 * 1.3)
    width[g] *= 1.1
    width[f & ~g] *= 1 + 0.01 * np.sin(i[f & ~g] * 0.8)
    return angle, length, width


def body_points(path_data: dict) -> dict[str, np.ndarray]:
    """Ground-truth body parts (cm) for synthetic pose data, consistent with the rendered video."""
    angle, length, _ = posture(path_data)
    pos = path_data["pos"]
    u = np.column_stack([np.cos(angle), np.sin(angle)])
    v = np.column_stack([-u[:, 1], u[:, 0]])
    k = (length / 3.5)[:, None]
    return {
        "nose": pos + 4.4 * k * u,
        "left_ear": pos + 2.4 * k * u + 0.8 * v,
        "right_ear": pos + 2.4 * k * u - 0.8 * v,
        "center": pos,
        "tail_base": pos - 3.4 * k * u,
    }


def render(path_data: dict, out_path, noise: float = 3.0, seed: int = 2, floor: int = 205, wall: int = 110,
           fur: int = 45) -> Path:
    rng = np.random.default_rng(seed)
    fps = path_data["fps"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_HW[1], FRAME_HW[0]))
    base = np.full(FRAME_HW, wall, np.uint8)
    c = corners_px().astype(np.int32)
    cv2.fillPoly(base, [c], floor)
    k = PX_PER_CM
    angles, lengths, widths = posture(path_data)
    for i in range(len(path_data["t"])):
        img = base.copy()
        p = OFFSET + path_data["pos"][i] * k
        ang, length, width = math.degrees(angles[i]), lengths[i], widths[i]
        u = np.array([math.cos(angles[i]), math.sin(angles[i])])
        # tail: thin wavy line behind the body (should be removed by the tracker)
        tail = [p - u * (length - 0.2) * k]
        for s in np.linspace(0.5, 7.0, 14):
            wave = 0.6 * math.sin(s * 1.2 + i * 0.2)
            tail.append(p - u * (length + s) * k + np.array([-u[1], u[0]]) * wave * k)
        cv2.polylines(img, [np.array(tail, np.int32)], False, fur + 10, max(1, int(0.3 * k)), cv2.LINE_AA)
        for center, axes in ((p, (length, width)), (p + u * 3.0 * k, (1.5, 1.05))):
            cv2.ellipse(img, tuple(int(v * 16) for v in center), tuple(int(a * k * 16) for a in axes), ang, 0, 360,
                        fur, -1, cv2.LINE_AA, 4)
        img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.int16)
        img += rng.normal(0, noise, img.shape).astype(np.int16)
        writer.write(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    writer.release()
    return out_path


def write_dlc_csv(path_data: dict, out_path, drop_fraction: float = 0.02, seed: int = 3, jitter_px: float = 1.0) -> Path:
    """DeepLabCut-style CSV (3 header rows) with the ground-truth body parts in px."""
    rng = np.random.default_rng(seed)
    parts = body_points(path_data)
    n = len(path_data["t"])
    names = list(parts)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer"] + ["DLC_synthetic"] * 3 * len(names))
        w.writerow(["bodyparts"] + [b for b in names for _ in range(3)])
        w.writerow(["coords"] + ["x", "y", "likelihood"] * len(names))
        for i in range(n):
            row = [i]
            for b in names:
                xy = OFFSET + parts[b][i] * PX_PER_CM + rng.normal(0, jitter_px, 2)
                lik = 0.05 if rng.random() < drop_fraction else 0.99
                row += [f"{xy[0]:.2f}", f"{xy[1]:.2f}", f"{lik:.3f}"]
            w.writerow(row)
    return Path(out_path)


def open_field_config(paradigm_key: str = "open_field") -> dict:
    """Config matching the synthetic arena (what `behavtrack setup` would produce)."""
    from .config import new_config
    from .geometry import Calibration, Polygon
    from .setup_gui import build_arena
    from .templates import paradigm

    cfg = new_config(paradigm_key)
    cal = Calibration.from_rect(corners_px(), SIZE_CM, SIZE_CM)
    arena = build_arena(paradigm(paradigm_key), "A", Polygon(corners_px()), cal, [], cfg["params"])
    cfg["arenas"] = [arena.to_dict()]
    return cfg
