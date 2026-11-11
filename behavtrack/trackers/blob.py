"""Approach ①: classic computer vision.

The animal is the largest blob that differs from the empty-arena background (or that is darker /
brighter than a fixed intensity). Besides the centroid it reports body area, orientation,
length/width and a pixel-change "motion" score that drives freezing detection.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .base import NAN, Detection, Tracker

EXTRA = ("area", "angle", "major", "minor", "motion")


def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def _disk(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class _ArenaROI:
    """Bounding box, mask and size limits of one arena at processing resolution."""

    def __init__(self, arena, full_hw, proc_hw, f: float, cfg: dict):
        mask = cv2.resize(arena.shape.mask(full_hw), (proc_hw[1], proc_hw[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise SystemExit(f"아레나 '{arena.name}'가 화면 밖에 있습니다. 설정을 다시 확인해주세요.")
        self.x0, self.y0, self.x1, self.y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        self.mask = mask[self.y0:self.y1, self.x0:self.x1]
        if arena.calibration.calibrated:
            s = arena.calibration.scale * f  # px per cm at processing resolution
            self.min_area = cfg["min_area_cm2"] * s * s
            self.max_area = cfg["max_area_cm2"] * s * s
            self.open_k = _disk(_odd(cfg["tail_removal_cm"] * s))
            self.grow_k = _disk(_odd(0.5 * s))
        else:
            self.min_area, self.max_area = cfg["min_area_px"] * f * f, math.inf
            self.open_k, self.grow_k = _disk(5), _disk(7)
        self.close_k = _disk(5)
        self.prev_mask = None

    def crop(self, img):
        return img[self.y0:self.y1, self.x0:self.x1]


class BlobTracker(Tracker):
    extra_fields = EXTRA

    def __init__(self, arenas, cfg: dict, frame_hw):
        t = cfg["tracking"]
        self.method = t["method"]
        self.color = cfg["animal"]["color"]
        self.threshold = int(t["threshold"])
        self.intensity_threshold = int(t["intensity_threshold"])
        self.motion_threshold = int(t["motion_pixel_threshold"])
        self.bg_rate = float(t["bg_update_rate"])
        self.blur = _odd(t["blur"])
        h, w = frame_hw[:2]
        self.f = min(1.0, t["process_width"] / w) if t.get("process_width") else 1.0
        self.proc_hw = (int(round(h * self.f)), int(round(w * self.f)))
        self.rois = [_ArenaROI(a, (h, w), self.proc_hw, self.f, t) for a in arenas]
        self.needs_background = self.method == "background"
        self.bg = None
        self.bg_f = None
        self.prev = None
        self.keep_debug = False
        self.last_binary = None

    @property
    def ready(self) -> bool:
        return not self.needs_background or self.bg is not None

    def preprocess(self, frame) -> np.ndarray:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != self.proc_hw:
            gray = cv2.resize(gray, (self.proc_hw[1], self.proc_hw[0]), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (self.blur, self.blur), 0)

    def set_background(self, img) -> None:
        """Accepts a preprocessed background or any empty-arena image (color or full size)."""
        if img.ndim == 2 and img.shape[:2] == self.proc_hw:
            self.bg = img.astype(np.uint8).copy()
        else:
            self.bg = self.preprocess(img)
        self.bg_f = None

    def binarize(self, gray) -> np.ndarray:
        if self.method == "background":
            if self.color == "dark":
                d = cv2.subtract(self.bg, gray)
            elif self.color == "light":
                d = cv2.subtract(gray, self.bg)
            else:
                d = cv2.absdiff(gray, self.bg)
            return cv2.threshold(d, self.threshold, 255, cv2.THRESH_BINARY)[1]
        mode = cv2.THRESH_BINARY_INV if self.color == "dark" else cv2.THRESH_BINARY
        return cv2.threshold(gray, self.intensity_threshold, 255, mode)[1]

    def _blobs(self, roi: _ArenaROI, binary) -> np.ndarray:
        b = cv2.bitwise_and(roi.crop(binary), roi.mask)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, roi.open_k)
        return cv2.morphologyEx(b, cv2.MORPH_CLOSE, roi.close_k)

    def foreground_mask(self, gray) -> np.ndarray:
        """All animal-like pixels (used to refine the background estimate)."""
        binary = self.binarize(gray)
        out = np.zeros(self.proc_hw, np.uint8)
        for roi in self.rois:
            out[roi.y0:roi.y1, roi.x0:roi.x1] |= cv2.dilate(self._blobs(roi, binary), roi.grow_k)
        return out

    def process(self, frame) -> list[Detection]:
        return self.process_gray(self.preprocess(frame))

    def process_gray(self, gray) -> list[Detection]:
        if not self.ready:
            return [Detection(extra=dict.fromkeys(EXTRA, NAN)) for _ in self.rois]
        binary = self.binarize(gray)
        debug = np.zeros(self.proc_hw, np.uint8) if self.keep_debug else None
        dets = []
        for roi in self.rois:
            b = self._blobs(roi, binary)
            if debug is not None:
                debug[roi.y0:roi.y1, roi.x0:roi.x1] |= b
            contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best, best_area = None, 0.0
            for c in contours:
                a = cv2.contourArea(c)
                if roi.min_area <= a <= roi.max_area and a > best_area:
                    best, best_area = c, a
            dets.append(self._describe(roi, best, gray))
        if self.bg_rate > 0 and self.bg is not None:
            self._update_background(gray)
        self.prev = gray
        self.last_binary = debug
        return dets

    def _describe(self, roi: _ArenaROI, contour, gray) -> Detection:
        if contour is None:
            roi.prev_mask = None
            return Detection(extra=dict.fromkeys(EXTRA, NAN))
        mask = np.zeros(roi.mask.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        m = cv2.moments(mask, binaryImage=True)
        area = m["m00"]
        cx, cy = m["m10"] / area, m["m01"] / area
        mu20, mu02, mu11 = m["mu20"] / area, m["mu02"] / area, m["mu11"] / area
        common = (mu20 + mu02) / 2
        spread = math.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
        major, minor = 4 * math.sqrt(common + spread), 4 * math.sqrt(max(common - spread, 0.0))
        angle = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))
        motion = NAN
        if self.prev is not None:
            region = mask if roi.prev_mask is None else cv2.bitwise_or(mask, roi.prev_mask)
            region = cv2.dilate(region, roi.grow_k)
            diff = cv2.absdiff(roi.crop(gray), roi.crop(self.prev))
            changed = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
            motion = cv2.countNonZero(cv2.bitwise_and(changed, region)) / area
        roi.prev_mask = mask
        inv = 1.0 / self.f
        outline = (contour.reshape(-1, 2) + [roi.x0, roi.y0]) * inv
        return Detection(
            x=(cx + roi.x0) * inv, y=(cy + roi.y0) * inv,
            extra={"area": area * inv * inv, "angle": angle, "major": major * inv, "minor": minor * inv, "motion": motion},
            contour=outline.astype(np.int32),
        )

    def _update_background(self, gray) -> None:
        """Blend the current frame into the background everywhere except around the animal."""
        if self.bg_f is None:
            self.bg_f = self.bg.astype(np.float32)
        keep = np.full(self.proc_hw, 255, np.uint8)
        for roi in self.rois:
            if roi.prev_mask is not None:
                grown = cv2.dilate(roi.prev_mask, roi.grow_k, iterations=3)
                keep[roi.y0:roi.y1, roi.x0:roi.x1][grown > 0] = 0
        cv2.accumulateWeighted(gray, self.bg_f, self.bg_rate, mask=keep)
        self.bg = cv2.convertScaleAbs(self.bg_f)
