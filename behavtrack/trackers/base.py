"""Common tracker interface: a tracker turns one video frame into one Detection per arena."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NAN = float("nan")


@dataclass
class Detection:
    x: float = NAN  # reference point in full-resolution image px
    y: float = NAN
    extra: dict[str, float] = field(default_factory=dict)
    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # name -> x, y, p
    contour: np.ndarray | None = None  # outline for display only

    @property
    def found(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y)


class Tracker:
    extra_fields: tuple[str, ...] = ()
    keypoint_names: tuple[str, ...] = ()
    needs_background = False

    @property
    def ready(self) -> bool:
        return True

    def process(self, frame) -> list[Detection]:
        raise NotImplementedError
