"""Overlays for the live window, annotated videos and setup previews (OpenCV text is ASCII only)."""

from __future__ import annotations

import cv2
import numpy as np

ZONE_COLORS = [(0, 200, 255), (255, 160, 0), (60, 220, 60), (255, 0, 200), (0, 90, 255), (200, 200, 0), (180, 120, 255)]
ANIMAL_COLORS = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]


def text(img, s: str, org, scale: float = 0.5, color=(255, 255, 255), thickness: int = 1) -> None:
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def lines(img, rows: list[str], origin=(10, 22), scale: float = 0.55) -> None:
    for i, row in enumerate(rows):
        text(img, row, (origin[0], origin[1] + int(i * 26 * scale / 0.55)), scale)


def layout(img, arenas, labels: bool = True) -> None:
    for a in arenas:
        a.shape.draw(img, (255, 255, 255), 2)
        for i, z in enumerate(a.zones):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            z.shape.draw(img, color, 1)
            if labels:
                c = z.shape.centroid()
                text(img, z.name, (c[0] - 20, c[1]), 0.4, color)
        if labels:
            x, y = a.shape.outline().min(axis=0)
            text(img, a.name, (x + 4, y + 16), 0.5)


def detections(img, dets, trails=None) -> None:
    for i, d in enumerate(dets):
        color = ANIMAL_COLORS[i % len(ANIMAL_COLORS)]
        if trails is not None and len(trails[i]) > 1:
            pts = np.array([p for p in trails[i] if p is not None], np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        if d.contour is not None:
            cv2.drawContours(img, [d.contour], -1, color, 1, cv2.LINE_AA)
        for x, y, p in d.keypoints.values():
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255) if p >= 0.6 else (0, 0, 128), -1)
        if d.found:
            cv2.circle(img, (int(d.x), int(d.y)), 4, color, -1)
