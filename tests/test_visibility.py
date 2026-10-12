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


def test_components_split_and_size_filter():
    shape = (12, 4, 4)
    m = np.zeros(shape, bool)
    m[0:3, 1, 1] = True
    m[5:9, 1, 1] = True  # separate, 4 voxels
    comps = components(m.ravel(), shape, min_voxels=1)
    assert [len(c) for c in comps] == [4, 3]  # largest first
    assert len(components(m.ravel(), shape, min_voxels=4)) == 1


def test_free_valid_blocks_free_evidence_only():
    K1 = np.array([[10.0, 0, 5], [0, 10, 5], [0, 0, 1]])
    d = np.full((10, 10), 4.0)
    c = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])  # in front of the surface, on it
    h, f = observe(c, d, K1, np.eye(4), tau=0.05, free_valid=np.zeros((10, 10), bool))
    assert not f[0] and h[1]


def test_footprint_keeps_voxel_beside_a_thin_surface():
    d = np.full((100, 100), 3.0)
    d[:, 53:55] = 2.0  # thin pole at z = 2, just right of the centre ray (x ~ 0.03 .. 0.05)
    c = np.array([[0.0, 0.0, 2.0], [-0.5, 0.0, 2.0]])  # voxel touching the pole; voxel far from it
    _, f = observe(c, d, K, np.eye(4), tau=0.05)
    assert f.all()  # centre rays both see the wall behind
    _, f = observe(c, d, K, np.eye(4), tau=0.05, voxel=SIZE)
    assert not f[0] and f[1]  # the whole voxel is not seen past when the pole cuts its footprint
