"""Tests for the visibility map on an analytic scene: a wall at z=3 and a box at z=2."""

import numpy as np

from roomdiff.visibility import FREE, LABELS, OCC, UNKNOWN, Grid, classify, compare, components, observe

K = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])  # 100 x 100 depth image
SIZE = 0.1


def _depth(box: bool) -> np.ndarray:
    d = np.full((100, 100), 3.0)
    if box:
        d[40:60, 40:60] = 2.0  # box front face covers x, y in [-0.2, 0.2] at z = 2
    return d


def _grid() -> Grid:
    return Grid(origin=np.array([-1.0, -1.0, 0.5]), size=SIZE, shape=(21, 21, 26))  # z 0.5 .. 3.0


def _state(box: bool) -> np.ndarray:
    g = _grid()
    c = g.centres()
    hits, frees = np.zeros(len(c), int), np.zeros(len(c), int)
    for dx in (-0.02, 0.0, 0.02):  # three nearly identical views
        T = np.eye(4)
        T[0, 3] = dx
        h, f = observe(c, _depth(box), K, T, tau=0.05)
        hits += h
        frees += f
    return classify(hits, frees)


def _at(g: Grid, x: float, y: float, z: float) -> int:
    return int(g.index(np.array([[x, y, z]]))[0])


def test_observe_hit_free_and_hidden():
    g = _grid()
    s = _state(box=True)
    assert s[_at(g, 0, 0, 2.0)] == OCC  # box face
    assert s[_at(g, 0, 0, 1.0)] == FREE  # air in front of the box
    assert s[_at(g, 0, 0, 2.5)] == UNKNOWN  # behind the box
    assert s[_at(g, 0.6, 0, 3.0)] == OCC  # wall beside the box


def test_observe_outside_view_is_unknown():
    g = _grid()
    s = _state(box=False)
    # at z = 0.6 the 100 px image spans only x in [-0.3, 0.3]
    assert s[_at(g, 0.9, 0, 0.6)] == UNKNOWN


def test_compare_removed_box_and_unobserved_wall_behind_it():
    g = _grid()
    lab = compare(_state(box=True), _state(box=False), g.shape)
    assert lab[_at(g, 0, 0, 2.0)] == LABELS["removed"]
    assert lab[_at(g, 0, 0, 3.0)] == LABELS["unobserved"]  # wall was hidden before: no evidence
    assert lab[_at(g, 0.6, 0, 3.0)] == LABELS["unchanged"]
    assert not np.any(lab == LABELS["added"])


def test_compare_is_symmetric_for_added():
    g = _grid()
    lab = compare(_state(box=False), _state(box=True), g.shape)
    assert lab[_at(g, 0, 0, 2.0)] == LABELS["added"]


def test_components_size_filter_and_order():
    shape = (10, 10, 10)
    m = np.zeros(shape, bool)
    m[0:3, 0:3, 0:3] = True  # 27
    m[6:8, 6:8, 6:8] = True  # 8
    m[9, 0, 9] = True  # 1
    comps = components(m.ravel(), shape, min_voxels=5)
    assert [len(c) for c in comps] == [27, 8]


def test_grid_index_outside_is_minus_one():
    g = _grid()
    assert g.index(np.array([[10.0, 0, 0]]))[0] == -1
    assert _at(g, -1.0, -1.0, 0.5) == 0
