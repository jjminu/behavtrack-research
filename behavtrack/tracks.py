"""Per-frame tracking data, one CSV per animal. Every tracker writes it and every analysis reads it.

Columns: frame, time_s, x, y (reference point in image px), tracker extras (area, motion, ...),
and for pose data <bodypart>_x, <bodypart>_y, <bodypart>_p (likelihood).
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE_COLUMNS = ("frame", "time_s", "x", "y")


def _fmt(v) -> str:
    return "" if v is None or not math.isfinite(v) else f"{v:.6g}"


@dataclass
class Tracks:
    frame: np.ndarray
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    keypoints: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N, 3): x, y, likelihood

    def __len__(self) -> int:
        return len(self.time)

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def subset(self, sel) -> "Tracks":
        return Tracks(
            self.frame[sel], self.time[sel], self.x[sel], self.y[sel],
            {k: v[sel] for k, v in self.extra.items()},
            {k: v[sel] for k, v in self.keypoints.items()},
        )

    def save(self, path) -> None:
        with TrackWriter(path, list(self.extra), list(self.keypoints)) as w:
            for i in range(len(self)):
                w.write_values(
                    int(self.frame[i]), float(self.time[i]), self.x[i], self.y[i],
                    [self.extra[k][i] for k in self.extra],
                    [self.keypoints[k][i] for k in self.keypoints],
                )

    @classmethod
    def load(cls, path) -> "Tracks":
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # empty file
            data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float, ndmin=2)
        if data.size == 0:
            data = np.empty((0, len(header)))
        col = {h: data[:, i] for i, h in enumerate(header)}
        kp_names = [h[:-2] for h in header
                    if h.endswith("_x") and f"{h[:-2]}_y" in col and f"{h[:-2]}_p" in col]
        kp_cols = {f"{k}{s}" for k in kp_names for s in ("_x", "_y", "_p")}
        extra = {h: col[h] for h in header if h not in BASE_COLUMNS and h not in kp_cols}
        keypoints = {k: np.column_stack([col[f"{k}_x"], col[f"{k}_y"], col[f"{k}_p"]]) for k in kp_names}
        return cls(col["frame"].astype(int), col["time_s"], col["x"], col["y"], extra, keypoints)


class TrackWriter:
    """Streams rows to disk while tracking, so a crash or power cut keeps what was recorded."""

    def __init__(self, path, extra_fields: list[str], keypoint_names: list[str]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.extra = list(extra_fields)
        self.kps = list(keypoint_names)
        self.w.writerow([*BASE_COLUMNS, *self.extra, *[f"{k}{s}" for k in self.kps for s in ("_x", "_y", "_p")]])
        self.rows = 0

    def write(self, frame: int, time_s: float, det) -> None:
        nan = float("nan")
        self.write_values(
            frame, time_s, det.x, det.y,
            [det.extra.get(k, nan) for k in self.extra],
            [det.keypoints.get(k, (nan, nan, nan)) for k in self.kps],
        )

    def write_values(self, frame, time_s, x, y, extras, keypoints) -> None:
        row = [int(frame), f"{time_s:.9f}", _fmt(x), _fmt(y), *[_fmt(v) for v in extras]]
        for kp in keypoints:
            row += [_fmt(float(v)) for v in kp]
        self.w.writerow(row)
        self.rows += 1
        if self.rows % 300 == 0:
            self.f.flush()

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
