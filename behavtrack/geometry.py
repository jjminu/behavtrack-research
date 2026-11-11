"""Regions (arena, zones) in image pixel coordinates and pixel-to-cm calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

_CHUNK = 20000  # points per vectorized batch, keeps memory bounded for long sessions


def as_points(xy) -> np.ndarray:
    return np.atleast_2d(np.asarray(xy, dtype=float)).reshape(-1, 2)


def apply_homography(h: np.ndarray, xy) -> np.ndarray:
    xy = as_points(xy)
    p = np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(h, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def order_corners(points) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left (image coordinates)."""
    pts = as_points(points)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


class Shape:
    """A closed region in image pixel coordinates."""

    def contains(self, xy) -> np.ndarray:
        raise NotImplementedError

    def boundary_distance(self, xy) -> np.ndarray:
        """Distance in px from each point to the outline (negative inside, NaN for missing points)."""
        raise NotImplementedError

    def outline(self) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def mask(self, hw) -> np.ndarray:
        m = np.zeros(tuple(hw[:2]), np.uint8)
        cv2.fillPoly(m, [np.round(self.outline()).astype(np.int32)], 255)
        return m

    def draw(self, img, color, thickness: int = 2) -> None:
        cv2.polylines(img, [np.round(self.outline()).astype(np.int32)], True, color, thickness, cv2.LINE_AA)

    def centroid(self) -> np.ndarray:
        pts = self.outline().astype(np.float32)
        m = cv2.moments(pts)
        if abs(m["m00"]) < 1e-9:
            return pts.mean(axis=0).astype(float)
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

    def area_px(self) -> float:
        return float(abs(cv2.contourArea(self.outline().astype(np.float32))))

    @staticmethod
    def from_dict(d: dict) -> "Shape":
        kind = d["type"]
        if kind == "polygon":
            return Polygon(d["points"])
        if kind == "circle":
            return Circle(tuple(d["center"]), float(d["radius"]))
        if kind == "rect":  # x, y, width, height
            x, y, w, h = d["rect"]
            return Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        raise ValueError(f"unknown shape type: {kind}")


class Polygon(Shape):
    def __init__(self, points):
        self.points = as_points(points)
        if len(self.points) < 3:
            raise ValueError("a polygon needs at least 3 points")

    def outline(self) -> np.ndarray:
        return self.points

    def contains(self, xy) -> np.ndarray:
        xy = as_points(xy)
        out = np.zeros(len(xy), bool)
        x0, y0 = self.points[:, 0][None], self.points[:, 1][None]
        x1, y1 = np.roll(x0, -1, axis=1), np.roll(y0, -1, axis=1)
        for s in range(0, len(xy), _CHUNK):
            px, py = xy[s:s + _CHUNK, 0:1], xy[s:s + _CHUNK, 1:2]
            straddle = (y0 > py) != (y1 > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            out[s:s + _CHUNK] = (straddle & (px < x_cross)).sum(axis=1) % 2 == 1
        return out & np.isfinite(xy).all(axis=1)

    def boundary_distance(self, xy) -> np.ndarray:
        xy = as_points(xy)
        a = self.points
        ab = np.roll(a, -1, axis=0) - a
        ab2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
        out = np.empty(len(xy))
        for s in range(0, len(xy), _CHUNK):
            p = xy[s:s + _CHUNK, None, :]
            t = np.clip(((p - a) * ab).sum(axis=2) / ab2, 0, 1)
            d = np.hypot(*(p - (a + t[..., None] * ab)).transpose(2, 0, 1))
            out[s:s + _CHUNK] = d.min(axis=1)
        out[self.contains(xy)] *= -1
        out[~np.isfinite(xy).all(axis=1)] = np.nan
        return out

    def to_dict(self) -> dict:
        return {"type": "polygon", "points": np.round(self.points, 2).tolist()}


class Circle(Shape):
    def __init__(self, center, radius: float):
        self.center = np.asarray(center, float).reshape(2)
        self.radius = float(radius)

    def outline(self, n: int = 96) -> np.ndarray:
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return self.center + self.radius * np.column_stack([np.cos(a), np.sin(a)])

    def contains(self, xy) -> np.ndarray:
        d = self.boundary_distance(xy)
        return np.nan_to_num(d, nan=1.0) <= 0

    def boundary_distance(self, xy) -> np.ndarray:
        xy = as_points(xy)
        return np.hypot(xy[:, 0] - self.center[0], xy[:, 1] - self.center[1]) - self.radius

    def mask(self, hw) -> np.ndarray:
        m = np.zeros(tuple(hw[:2]), np.uint8)
        cv2.circle(m, tuple(int(round(v)) for v in self.center), int(round(self.radius)), 255, -1)
        return m

    def centroid(self) -> np.ndarray:
        return self.center.copy()

    def area_px(self) -> float:
        return math.pi * self.radius ** 2

    def to_dict(self) -> dict:
        return {"type": "circle", "center": np.round(self.center, 2).tolist(), "radius": round(self.radius, 2)}


def rect_subregion(corners, u0: float, v0: float, u1: float, v1: float) -> Polygon:
    """Part of a (possibly perspective-distorted) rectangle, in fractions of its width (u) and height (v)."""
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    h = cv2.getPerspectiveTransform(src, order_corners(corners).astype(np.float32))
    return Polygon(apply_homography(h, [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))


def grown_hull(shapes: list[Shape], margin_px: float) -> Polygon:
    """Convex hull around shapes, grown outward by margin_px (used as the arena of mazes)."""
    pts = np.vstack([s.outline() for s in shapes]).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if margin_px <= 0:
        return Polygon(hull)
    c = hull.mean(axis=0)
    d = hull - c
    norm = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    return Polygon(hull + d / norm * margin_px)


@dataclass
class Calibration:
    """Pixel-to-cm mapping. A homography (from 4 arena corners) also corrects camera tilt."""

    px_per_cm: float | None = None
    homography: np.ndarray | None = None  # 3x3, image px -> cm

    @property
    def calibrated(self) -> bool:
        return bool(self.px_per_cm)

    @property
    def units(self) -> str:
        return "cm" if self.calibrated else "px"

    @property
    def scale(self) -> float:
        """Approximate pixels per cm (1.0 when uncalibrated, so lengths stay in px)."""
        return float(self.px_per_cm) if self.px_per_cm else 1.0

    def to_cm(self, xy) -> np.ndarray:
        if self.homography is not None:
            return apply_homography(self.homography, xy)
        return as_points(xy) / self.scale

    @classmethod
    def from_rect(cls, corners, width_cm: float, height_cm: float) -> "Calibration":
        src = order_corners(corners).astype(np.float32)
        dst = np.float32([[0, 0], [width_cm, 0], [width_cm, height_cm], [0, height_cm]])
        perimeter = float(np.hypot(*(np.roll(src, -1, axis=0) - src).T).sum())
        return cls(perimeter / (2 * (width_cm + height_cm)), cv2.getPerspectiveTransform(src, dst))

    @classmethod
    def from_line(cls, p0, p1, length_cm: float) -> "Calibration":
        return cls(math.dist(p0, p1) / length_cm)

    @classmethod
    def from_circle(cls, circle: Circle, diameter_cm: float) -> "Calibration":
        return cls(2 * circle.radius / diameter_cm)

    def to_dict(self) -> dict:
        return {
            "px_per_cm": None if self.px_per_cm is None else round(float(self.px_per_cm), 5),
            "homography": None if self.homography is None else np.asarray(self.homography).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        d = d or {}
        h = d.get("homography")
        return cls(d.get("px_per_cm"), None if h is None else np.asarray(h, float))
