"""Empty-arena background estimation for the blob tracker."""

from __future__ import annotations

import warnings

import cv2
import numpy as np


def median_of(grays: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(grays), axis=0).astype(np.uint8)


def refine(grays: list[np.ndarray], tracker, first: np.ndarray) -> np.ndarray:
    """Second pass: ignore pixels the animal covers in each sample, so an animal that stays in one
    spot for much of the video (e.g. freezing in a corner) does not leak into the background."""
    tracker.set_background(first)
    stack = np.stack(grays)
    covered = np.stack([tracker.foreground_mask(g) > 0 for g in grays])
    out = first.copy()
    for r in range(0, stack.shape[1], 64):
        sub = stack[:, r:r + 64].astype(np.float32)
        sub[covered[:, r:r + 64]] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(sub, axis=0)
        ok = np.isfinite(med)
        rows = out[r:r + 64]
        rows[ok] = np.round(med[ok]).astype(np.uint8)
    return out


def from_video(path, tracker, n: int = 40, start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    """Median of n frames spread over the video (animal removed by the refine pass)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i0 = int(start_s * fps)
    i1 = min(total, int(end_s * fps)) if end_s else total
    grays = []
    for i in np.unique(np.linspace(i0, max(i0, i1 - 1), n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            grays.append(tracker.preprocess(frame))
    cap.release()
    if len(grays) < 3:
        raise SystemExit(f"배경을 만들 프레임을 읽지 못했습니다: {path}")
    return refine(grays, tracker, median_of(grays))


def load_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"배경 이미지를 읽을 수 없습니다: {path}")
    return img
