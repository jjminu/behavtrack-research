"""Paradigm metrics on hand-built tracks with known answers."""

import numpy as np
import pytest

from behavtrack.analysis import analyze
from behavtrack.analysis.common import debounce, zone_stats
from behavtrack.arena import Zone
from behavtrack.geometry import Circle

from conftest import FPS, make_arena, make_tracks, rect


def run(cfg, arena, tracks):
    return analyze(tracks, arena, cfg).metrics


def test_debounce_ignores_border_flicker():
    state = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], bool)
    out = debounce(state, np.full(len(state), 1 / FPS), min_s=0.1)  # 3 frames
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_open_field_center():
    cfg, arena = make_arena("open_field", rect(0, 0, 400, 400), [])
    tr = make_tracks([(5, (20, 20)), (3, (200, 200)), (4, (380, 20)), (2, (210, 190))])
    m = run(cfg, arena, tr)
    assert m["center_time_s"] == pytest.approx(5, abs=0.05)
    assert m["center_entries"] == 2
    assert m["center_latency_s"] == pytest.approx(5, abs=0.05)
    assert m["periphery_time_s"] == pytest.approx(9, abs=0.05)


def test_epm():
    zones = [Zone("open_1", rect(0, 100, 100, 150)), Zone("open_2", rect(150, 100, 250, 150)),
             Zone("closed_1", rect(100, 0, 150, 100)), Zone("closed_2", rect(100, 150, 150, 250)),
             Zone("center", rect(100, 100, 150, 150))]
    cfg, arena = make_arena("epm", None, zones)
    c, o1, o2, c1, c2 = (125, 125), (50, 125), (200, 125), (125, 50), (125, 200)
    tr = make_tracks([(2, c), (3, o1), (1, c), (4, c1), (1, c), (2, o2), (5, c2)])
    m = run(cfg, arena, tr)
    assert m["open_time_s"] == pytest.approx(5, abs=0.05)
    assert m["closed_time_s"] == pytest.approx(9, abs=0.05)
    assert m["open_time_pct"] == pytest.approx(100 * 5 / 14, abs=0.5)
    assert (m["open_entries"], m["closed_entries"]) == (2, 2)
    assert m["open_entries_pct"] == pytest.approx(50)


def test_y_maze_alternation():
    zones = [Zone("arm_A", rect(0, 0, 100, 100)), Zone("arm_B", rect(200, 0, 300, 100)),
             Zone("arm_C", rect(400, 0, 500, 100)), Zone("center", rect(0, 200, 500, 300))]
    cfg, arena = make_arena("y_maze", None, zones)
    pos = {"A": (50, 50), "B": (250, 50), "C": (450, 50)}
    segs = [(1, (250, 250))]
    for arm in "ABCAACBA":
        segs += [(1, pos[arm]), (1, (250, 250))]
    m = run(cfg, arena, make_tracks(segs))
    assert m["arm_sequence"] == "ABCAACBA"
    assert m["alternations"] == 4
    assert m["alternation_pct"] == pytest.approx(100 * 4 / 6)
    assert m["same_arm_returns"] == 1


def test_nor_centroid_proximity():
    zones = [Zone("object_1", Circle((100, 100), 20)), Zone("object_2", Circle((300, 100), 20))]
    cfg, arena = make_arena("nor", rect(0, 0, 400, 400), zones, novel_object="object_2")
    tr = make_tracks([(4, (100, 150)), (5, (200, 300)), (12, (300, 150)), (3, (200, 300))])
    m = run(cfg, arena, tr)
    assert m["exploration_method"] == "centroid_proximity"
    assert m["object_1_explore_s"] == pytest.approx(4, abs=0.05)
    assert m["object_2_explore_s"] == pytest.approx(12, abs=0.05)
    assert m["discrimination_index"] == pytest.approx(0.5, abs=0.01)


def test_nor_nose_must_point_at_object():
    zones = [Zone("object_1", Circle((100, 100), 20))]
    cfg, arena = make_arena("nor", rect(0, 0, 400, 400), zones, exclude_on_object=False)
    n = 3 * FPS
    def kp(nose, ear):
        return np.tile([*nose, 0.99], (n, 1)), np.tile([*ear, 0.99], (n, 1))
    facing = kp((100, 125), (100, 150))  # nose 0.5 cm from the edge, ears behind it
    away = kp((100, 128), (100, 122))  # nose near, but ears even closer: looking away
    nose = np.vstack([facing[0], away[0]])
    ears = np.vstack([facing[1], away[1]])
    tr = make_tracks([(3, (100, 170)), (3, (100, 110))],
                     keypoints={"nose": nose, "left_ear": ears, "right_ear": ears})
    m = run(cfg, arena, tr)
    assert m["exploration_method"] == "nose"
    assert m["object_1_explore_s"] == pytest.approx(3, abs=0.1)


def test_light_dark_hidden_animal_counts_as_dark():
    cfg, arena = make_arena("light_dark", rect(0, 0, 400, 200), [Zone("dark", rect(200, 0, 400, 200))])
    tr = make_tracks([(5, (100, 100)), (2, (300, 100)), (10, None), (3, (100, 100))])
    m = run(cfg, arena, tr)
    assert m["dark_time_s"] == pytest.approx(2, abs=0.05)
    assert m["undetected_time_s"] == pytest.approx(10, abs=0.05)
    assert m["transitions"] == 1
    cfg["params"]["undetected_as"] = "dark"
    m = run(cfg, arena, tr)
    assert m["dark_time_s"] == pytest.approx(12, abs=0.05)
    assert m["light_time_s"] == pytest.approx(8, abs=0.05)
    assert m["transitions"] == 2
    assert m["dark_latency_s"] == pytest.approx(5, abs=0.05)


def test_three_chamber():
    zones = [Zone("cup_A", Circle((100, 100), 40)), Zone("cup_B", Circle((500, 100), 40))]
    cfg, arena = make_arena("three_chamber", rect(0, 0, 600, 200), zones, social_side="A")
    assert {z.name for z in arena.zones} >= {"chamber_A", "chamber_center", "chamber_B"}
    tr = make_tracks([(6, (100, 160)), (4, (300, 100)), (5, (450, 20)), (2, (500, 160))])
    m = run(cfg, arena, tr)
    assert m["social_chamber_time_s"] == pytest.approx(6, abs=0.05)
    assert m["other_chamber_time_s"] == pytest.approx(7, abs=0.05)
    assert m["social_sniff_s"] == pytest.approx(6, abs=0.05)
    assert m["other_sniff_s"] == pytest.approx(2, abs=0.05)
    assert m["sniff_preference_index"] == pytest.approx(0.5, abs=0.01)


def test_cpp():
    zones = [Zone("chamber_A", rect(0, 0, 200, 200)), Zone("chamber_B", rect(200, 0, 400, 200))]
    cfg, arena = make_arena("cpp", rect(0, 0, 400, 200), zones, paired_chamber="chamber_A")
    m = run(cfg, arena, make_tracks([(10, (100, 100)), (5, (300, 100))]))
    assert m["preference_score_s"] == pytest.approx(5, abs=0.05)
    assert m["preference_ratio"] == pytest.approx(2 / 3, abs=0.01)


def test_water_maze():
    cfg, arena = make_arena("mwm", Circle((300, 300), 300), [Zone("platform", Circle((450, 150), 25))], px_per_cm=5.0)
    tr = make_tracks([(3, (280, 570)), (2, (150, 310)), (5, (450, 150))])
    m = run(cfg, arena, tr)
    assert m["found_target"] is True
    assert m["latency_s"] == pytest.approx(5, abs=0.05)
    assert m["target_quadrant"] == "Q_NE"
    assert m["target_quadrant_pct"] == pytest.approx(50, abs=0.5)
    assert m["opposite_quadrant_pct"] == pytest.approx(50, abs=0.5)
    assert m["thigmotaxis_pct"] == pytest.approx(30, abs=0.5)


def test_fear_conditioning_epochs():
    cfg, arena = make_arena("fear_conditioning", rect(0, 0, 400, 300), [], tone_onsets=[10, 40], tone_duration_s=10)
    n = 60 * FPS
    t = np.arange(n) / FPS
    motion = np.where(((t >= 10) & (t < 20)) | ((t >= 40) & (t < 50)), 0.0, 0.5)
    tr = make_tracks([(60, (200, 150))], extra={"motion": motion})
    m = run(cfg, arena, tr)
    assert m["freezing_pct_baseline"] == pytest.approx(0)
    assert m["freezing_pct_tone_1"] == pytest.approx(100, abs=2)
    assert m["freezing_pct_tone_2"] == pytest.approx(100, abs=2)
    assert m["freezing_time_s"] == pytest.approx(20, abs=0.3)


def test_zone_stats_handles_empty():
    cfg, arena = make_arena("open_field", rect(0, 0, 400, 400), [])
    res = analyze(make_tracks([(1, None)]), arena, cfg)
    assert res.metrics["tracked_pct"] == 0
    assert zone_stats(np.zeros(0, bool), res.prep)["entries"] == 0


def test_freezing_missing_motion_is_unknown_in_total_bins_and_epochs():
    cfg, arena = make_arena("fear_conditioning", rect(0, 0, 400, 300), [], epochs=[{"name": "test", "start": 0, "end": 10}])
    cfg["analysis"]["bin_s"] = 10
    motion = np.full(10 * FPS, np.nan)
    motion[:2 * FPS] = 0
    tr = make_tracks([(2, (200, 150)), (8, None)], extra={"motion": motion})
    res = analyze(tr, arena, cfg)
    assert res.metrics["freezing_observed_s"] == pytest.approx(2)
    assert res.metrics["freezing_unknown_s"] == pytest.approx(8)
    assert res.metrics["freezing_pct"] == pytest.approx(100)
    assert res.metrics["freezing_pct_test"] == pytest.approx(100)
    assert res.bins[0]["freezing_pct"] == pytest.approx(100)
    assert res.frame_rows()[-1]["freezing"] is None
    empty = analyze(make_tracks([(2, None)]), arena, cfg)
    assert empty.metrics["freezing_pct"] is None
