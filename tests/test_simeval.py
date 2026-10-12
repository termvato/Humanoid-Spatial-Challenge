import numpy as np

from roomdiff.simeval import evaluate, map_to_world, world_to_map

# sim world: z up, floor at -0.77; level frame: y up, floor at 0 (rotation x by -90 deg, shift)
T = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0.77], [0, -1, 0, 0], [0, 0, 0, 1]])
FLOOR = -0.77


def _box(xy):
    return {"centre": list(xy), "bottom": 0.0, "top": 0.5, "size": [0.5, 0.5]}


def test_world_map_round_trip():
    m = world_to_map(T, (1.5, -2.0), FLOOR)
    assert np.allclose(m, (1.5, -2.0))  # this T keeps x, y on the map plane
    assert np.allclose(map_to_world(T, m)[:2], (1.5, -2.0)) and np.isclose(map_to_world(T, m)[2], FLOOR)


def test_evaluate_kinds_and_false_alarms():
    gt = {
        "cab": {"change": "moved", "visit1": {"pos": [-3, 3]}, "visit2": {"pos": [3, -2]}},
        "dolly": {"change": "removed", "visit1": {"pos": [3, 3]}, "visit2": None},
        "bins": {"change": "added", "visit1": None, "visit2": {"pos": [-2.6, -2]}},
        "bins_top": {"change": "added", "visit1": None, "visit2": {"pos": [-2.6, -2]}},
        "corner": {"change": "unobserved", "visit1": {"pos": [-3.8, -2.7]}, "visit2": {"pos": [-3.8, -2.7]}},
        "mug": {"change": "unchanged", "visit1": {"pos": [0, 0]}, "visit2": {"pos": [0, 0]}},
    }
    changes = [
        {"id": 0, "kind": "moved", "before": _box((-3.1, 3.1)), "after": _box((3.0, -2.2))},
        {"id": 1, "kind": "added", "before": None, "after": _box((-2.5, -2.0))},
        {"id": 2, "kind": "added", "before": None, "after": _box((0.1, 0.0))},  # on the mug: false alarm
        {"id": 3, "kind": "added", "before": None, "after": _box((1.0, 4.0))},  # nothing there
    ]
    r = evaluate(changes, gt, T, FLOOR, look_targets=[(-3.6, -2.5)])
    assert r["changed"] == 3 and r["found"] == 2 and r["kind_ok"] == 2  # dolly missed
    assert r["props"]["cab"]["kind_ok"] and not r["props"]["dolly"]["found"]
    assert "bins_top" not in r["props"]  # stacked: counted once
    assert r["props"]["corner"]["look_target"]
    assert r["props"]["mug"]["detections"] == [2]
    assert r["false_alarms"] == [2, 3]
