"""Approach ②, real time: run a trained DeepLabCut model on every frame with DeepLabCut-Live.

Install separately (it brings TensorFlow or PyTorch):  pip install deeplabcut-live
Export the trained model with deeplabcut.export_model(...) and pass that folder as --model.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .base import Detection, Tracker


def bodyparts_from_model(model_dir) -> list[str]:
    """Read body part names from an exported model's pose_cfg.yaml (without needing PyYAML)."""
    cfg = Path(model_dir) / "pose_cfg.yaml"
    if not cfg.exists():
        return []
    names, inside = [], False
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if re.match(r"^all_joints_names\s*:", line):
            inside = True
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("["):
                return [n.strip(" '\"") for n in inline.strip("[]").split(",") if n.strip()]
            continue
        if inside:
            m = re.match(r"^\s*-\s*(.+)$", line)
            if not m:
                break
            names.append(m.group(1).strip(" '\""))
    return names


class DLCLiveTracker(Tracker):
    """One animal per arena: each frame's pose is assigned to the arena that contains it."""

    def __init__(self, arenas, cfg: dict, model_dir, bodyparts: list[str] | None = None, model_type: str | None = None):
        try:
            from dlclive import DLCLive
        except ImportError:
            raise SystemExit("DeepLabCut-Live가 설치되어 있지 않습니다:  pip install deeplabcut-live") from None
        kwargs = {"model_type": model_type} if model_type else {}
        self.dlc = DLCLive(str(model_dir), **kwargs)
        self.arenas = arenas
        self.keypoint_names = tuple(bodyparts or bodyparts_from_model(model_dir))
        if not self.keypoint_names:
            raise SystemExit("모델의 신체 부위 이름을 찾지 못했습니다. --bodyparts nose,left_ear,... 로 지정해주세요.")
        self.min_p = float(cfg["analysis"]["keypoint_min_p"])
        self.started = False

    def process(self, frame) -> list[Detection]:
        if not self.started:
            self.dlc.init_inference(frame)
            self.started = True
        pose = np.asarray(self.dlc.get_pose(frame), float).reshape(-1, 3)
        if len(pose) != len(self.keypoint_names):
            raise SystemExit(f"모델 출력 부위 수({len(pose)})와 --bodyparts 수({len(self.keypoint_names)})가 다릅니다.")
        good = pose[:, 2] >= self.min_p
        dets = [Detection() for _ in self.arenas]
        if not good.any():
            return dets
        center = pose[good, :2].mean(axis=0)
        for i, arena in enumerate(self.arenas):
            if arena.shape.contains(center)[0] or len(self.arenas) == 1:
                kps = {n: (float(x), float(y), float(p)) for n, (x, y, p) in zip(self.keypoint_names, pose)}
                dets[i] = Detection(x=float(center[0]), y=float(center[1]), keypoints=kps)
                break
        return dets
