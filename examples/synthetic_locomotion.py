"""Generate a calibrated locomotion report from an artificial video."""

import argparse
from math import isclose
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from behavtrack import cage_locomotion, research


def create_report(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    size = (160, 120)
    fps = 10
    protocol = research.protocol(
        {"group": "합성 영상 데모", "cage": "DEMO-01", "animals": 2, "food_rect": [0, 0, 1, 1]},
        size,
    )
    with TemporaryDirectory(dir=output) as work:
        recording = research.ResearchRecording(Path(work), protocol, 0, fps)
        try:
            for index in range(8 * fps + 1):
                # Each target moves 40 px, then 80 px. The scale is 4 px/cm.
                displacement = min(index, 40) + 2 * max(index - 40, 0)
                frame = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
                cv2.circle(frame, (18 + displacement, 32), 7, (230, 230, 230), -1)
                cv2.circle(frame, (142 - displacement, 88), 7, (230, 230, 230), -1)
                recording.record(frame, recording.start_monotonic + index / fps)
        finally:
            recording.finish()

        settings = {
            "calibration": {
                "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "width_cm": 40,
                "height_cm": 30,
                "planar_confirmed": True,
            },
            "roi": [0, 0, 1, 1],
            "setup_id": "synthetic-overhead",
            "comparison_block": "demo",
            "detector": "bright_foreground",
            "white_threshold": 140,
            "bin_seconds": 2,
        }
        result = cage_locomotion.analyze(recording.folder, settings, Path(work), Path(work))
        if result["state"] != "complete":
            raise RuntimeError(result["error"])
        run = Path(result["path"])
        report = cage_locomotion.read_report(run)
        summary = report["summary"]
        expected = {"distance_total": 60.0, "mean_speed": 3.75, "valid_s": 8.0}
        for metric, reference in expected.items():
            if not isclose(summary[metric], reference, abs_tol=0.05):
                raise RuntimeError(f"{metric}: expected {reference}, measured {summary[metric]}")

        copyfile(run / "chart.png", output / "chart.png")
        copyfile(run / "report.html", output / "report.html")
        print(f"{summary['distance_total']:.3f} cm · {summary['mean_speed']:.3f} cm/s · "
              f"{summary['coverage_pct']:.1f}% coverage")
        print(output / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo-output"))
    create_report(parser.parse_args().output.resolve())
