"""CPU multi-mouse pose integration and conservative frame-level approach counting.

SuperAnimal's detections are per-frame candidates, not stable mouse identities.
The user must validate model performance on the actual camera and animals.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def approach_from_noses(noses, arena, food, expected=5, min_confidence=0.8, min_separation_px=2.0):
    """Count only when all expected noses are visible and valid, otherwise return unknown."""
    a = np.asarray(noses, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("코 좌표는 마리별 x, y, 신뢰도 형식이어야 합니다.")
    if len(a) != expected:
        return None, "detected_count_mismatch"
    if not np.isfinite(a).all() or np.any(a[:, 2] < min_confidence) or np.any(a[:, 2] > 1):
        return None, "low_nose_confidence"
    if not np.all(arena.contains(a[:, :2])):
        return None, "outside_cage"
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if np.linalg.norm(a[i, :2] - a[j, :2]) <= min_separation_px:
                return None, "duplicate_or_overlapping_noses"
    return int(np.count_nonzero(food.contains(a[:, :2]))), "estimated"


class SuperAnimalCPU:
    """Load pinned DLC model weights once; infer per-frame on CPU only."""

    def __init__(self, model_dir, expected=5, min_confidence=0.8, min_box_confidence=0.6):
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
        from deeplabcut.pose_estimation_pytorch.config import PoseConfig

        model_dir = Path(model_dir)
        pose, detector = model_dir / "pose.pt", model_dir / "detector.pt"
        if not pose.is_file() or not detector.is_file():
            raise FileNotFoundError("공식 다중 마우스 모델 파일 두 개가 필요합니다.")
        if expected < 2 or not 0 < min_confidence <= 1 or not 0 < min_box_confidence <= 1:
            raise ValueError("마릿수 또는 신뢰도 기준을 확인하세요.")
        self.expected = expected
        self.min_confidence = min_confidence
        self.min_box_confidence = min_box_confidence
        self.model_cfg = PoseConfig.build_for_superanimal_inference(
            super_animal="superanimal_topviewmouse", model_name="hrnet_w32",
            detector_name="fasterrcnn_mobilenet_v3_large_fpn", max_individuals=expected + 1, device="cpu")
        names = list(self.model_cfg["metadata"]["bodyparts"])
        try:
            self.nose_idx = next(i for i, name in enumerate(names) if str(name).lower() in ("nose", "snout"))
        except StopIteration:
            raise ValueError("이 모델의 신체 부위에 nose 또는 snout이 없습니다.") from None
        self.pose, self.detector = get_inference_runners(
            model_config=self.model_cfg, snapshot_path=pose, detector_path=detector,
            max_individuals=expected + 1, num_bodyparts=len(names), num_unique_bodyparts=0,
            batch_size=1, detector_batch_size=1, device="cpu", min_bbox_score=min_box_confidence)
        if self.detector is None:
            raise RuntimeError("다중 개체 검출기를 초기화하지 못했습니다.")

    def predict(self, frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detection = self.detector.inference([rgb])[0]
        boxes = np.asarray(detection.get("bboxes", []))
        scores = np.asarray(detection.get("bbox_scores", []), dtype=float).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(scores):
            return {"detected": 0, "noses": [], "reason": "invalid_detection_output"}
        # Do not call the five returned instances identities; their order may change.
        if len(scores) != self.expected or not np.isfinite(scores).all() or np.any(scores < self.min_box_confidence):
            return {"detected": len(scores), "noses": [], "reason": "detected_count_mismatch"}
        result = self.pose.inference([(rgb, detection)])[0]
        parts = np.asarray(result.get("bodyparts", []), dtype=float)
        if parts.ndim != 3 or parts.shape[0] < self.expected or parts.shape[1] <= self.nose_idx or parts.shape[2] < 3:
            return {"detected": len(scores), "noses": [], "reason": "invalid_pose_output"}
        noses = parts[:self.expected, self.nose_idx, :3]
        # Replace non-finite floats so JSON does not encode NaN or Infinity.
        safe = [[float(v) if math.isfinite(v) else None for v in nose] for nose in noses]
        return {"detected": len(scores), "noses": safe, "reason": "candidate"}


class GroupDetectorCPU:
    """Whole-frame animal detection only: no pose model or individual identities."""

    def __init__(self, model_dir, expected=5, detector_path=None):
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_detector_inference_runner
        from deeplabcut.pose_estimation_pytorch.config import PoseConfig

        config = PoseConfig.build_for_superanimal_inference(
            super_animal="superanimal_topviewmouse", model_name="hrnet_w32",
            detector_name="fasterrcnn_mobilenet_v3_large_fpn", max_individuals=expected + 1, device="cpu")
        self.detector = get_detector_inference_runner(
            config, Path(detector_path) if detector_path else Path(model_dir) / "detector.pt", batch_size=1, device="cpu",
            max_individuals=expected + 1, min_bbox_score=0.6)

    def predict(self, frame_bgr):
        import cv2
        result = self.detector.inference([cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)])[0]
        # DLC's detector postprocessor returns COCO (x, y, width, height).
        # BehavTrack's counting and drawing consume (x1, y1, x2, y2).
        boxes = np.array(result.get("bboxes", []), dtype=float, copy=True).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        def safe(values):
            return json_safe(np.asarray(values, dtype=float).tolist())
        return {"boxes": safe(boxes), "scores": safe(result.get("bbox_scores", [])), "box_format": "xyxy"}


def json_safe(value):
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value if math.isfinite(value) else None
