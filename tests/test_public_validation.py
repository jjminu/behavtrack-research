"""Checks on the public-dataset label conversion and scoring."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")


def test_reference_intervals_and_unreviewed_markers():
    spec = importlib.util.spec_from_file_location("public_validation", Path(__file__).parents[1] / "공개영상_행동검증.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def row(kind, start, end):
        return {"ID": "OFT_1", "Experimenter": "Jin", "type": kind, "from": str(start), "to": str(end)}
    rows = [row("StartEnd", 0, 1), row("Supported", 2, 3), row("_DEFAULT", 4, 5), row("StartEnd", 9, 10)]
    t = np.arange(0, 11)
    truth, valid = module.labels_for(rows, "Jin", t, module.TARGETS["rearing"])
    assert np.flatnonzero(truth).tolist() == [2, 3]
    assert np.flatnonzero(valid).tolist() == [2, 3, 6, 7, 8]
    assert module.counts(truth, truth, valid) == {"tp": 2, "fn": 0, "fp": 0, "tn": 3}
