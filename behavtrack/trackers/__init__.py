"""Trackers: ① BlobTracker (classic CV, default) and ② DLCLiveTracker (real-time pose)."""

from __future__ import annotations

from .base import Detection, Tracker
from .blob import BlobTracker


def make_tracker(kind: str, arenas, cfg: dict, frame_hw, model=None, bodyparts=None, model_type=None) -> Tracker:
    if kind == "blob":
        return BlobTracker(arenas, cfg, frame_hw)
    if kind == "dlclive":
        if not model:
            raise SystemExit("--tracker dlclive 에는 --model (내보낸 DeepLabCut 모델 폴더)가 필요합니다.")
        from .dlclive import DLCLiveTracker

        return DLCLiveTracker(arenas, cfg, model, bodyparts, model_type)
    raise SystemExit(f"알 수 없는 추적기: {kind} (blob 또는 dlclive)")


__all__ = ["BlobTracker", "Detection", "Tracker", "make_tracker"]
