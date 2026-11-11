"""CPU model worker. One JSON request per sampled frame over local standard I/O."""
from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path


def emit(value):
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def main():
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--detect-only", action="store_true")
        parser.add_argument("--animals", type=int, default=5)
        parser.add_argument("--detector-path")
        args = parser.parse_args()
        if not 1 <= args.animals <= 20:
            raise ValueError("마릿수는 1~20 사이여야 합니다.")
        os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch"))
        with contextlib.redirect_stdout(sys.stderr):
            import cv2
            import numpy as np
            from .ai_pose import SuperAnimalCPU, GroupDetectorCPU

            model_class = GroupDetectorCPU if args.detect_only else SuperAnimalCPU
            kwargs={"detector_path":args.detector_path} if args.detect_only and args.detector_path else {}
            model = model_class(Path(__file__).resolve().parents[1] / "models" / "topviewmouse", expected=args.animals,**kwargs)
        import importlib.metadata
        emit({"ready": True, "runtime": {"python": sys.version, "device": "cpu",
              "packages": {name: importlib.metadata.version(name) for name in ("deeplabcut", "torch", "torchvision", "numpy", "opencv-python-headless")}}})
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if request.get("stop"):
                    break
                raw = base64.b64decode(request["image"], validate=True)
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("영상 프레임을 읽을 수 없습니다.")
                with contextlib.redirect_stdout(sys.stderr):
                    result = model.predict(frame)
                emit({"id": request["id"], "result": result})
            except Exception as exc:
                print(traceback.format_exc(), file=sys.stderr, flush=True)
                emit({"id": request.get("id") if isinstance(request, dict) else None, "error": str(exc)})
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        emit({"ready": False, "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
