import numpy as np

from roomdiff.grid2d import FREE_CELL, LOOK, OCC_CELL, UNKNOWN_CELL, Grid2D
from roomdiff.viewpoints import drop_unviewable, suggest, visible


def _maps():
    """4 x 4 m room at 0.1 m: free floor, a wall across the middle with a gap at the right end,
    and an unknown 'go look' corner at the bottom left, behind the wall from the top half."""
    a = np.full((40, 40), FREE_CELL, np.int8)
    a[[0, -1], :] = OCC_CELL
    a[:, [0, -1]] = OCC_CELL
    a[20, :32] = OCC_CELL  # wall, gap at cols 32-38
    a[2:6, 2:6] = UNKNOWN_CELL  # corner never seen again
    c = np.zeros_like(a)
    c[2:6, 2:6] = LOOK
    return Grid2D(a, 0.1, (0.0, 0.0)), Grid2D(c, 0.1, (0.0, 0.0)), np.where(a == OCC_CELL, 2.0, 0.0)


def test_visible_blocked_by_tall_obstacles_only():
    h = np.zeros((10, 10), np.float32)
    h[5, :] = 2.0  # wall
    assert not visible(h, np.array([0, 0]), np.array([[9, 0]]), camera_height=1.0)[0]
    assert visible(h, np.array([0, 0]), np.array([[4, 9]]), camera_height=1.0)[0]
    h[5, :] = 0.3  # low bed edge halfway: the ray is at 0.5 m there
    assert visible(h, np.array([0, 0]), np.array([[9, 0]]), camera_height=1.0)[0]
    assert not visible(h, np.array([0, 0]), np.array([[9, 0]]), camera_height=0.5)[0]


def test_suggest_stands_on_the_seeing_side_and_faces_the_region():
    after, change, height = _maps()
    vps = suggest(after, change, height, robot_radius=0.2, range_m=(0.3, 3.0))
    assert len(vps) == 1
    v = vps[0]
    assert v.cells == 16 and v.seen == 1.0
    x, y, yaw = v.pose
    assert y < 2.0  # same side of the wall as the corner (the wall is at y = 2.0 m)
    r, c = int(y / 0.1), int(x / 0.1)
    assert after.data[r, c] == FREE_CELL
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    to_target = np.array(v.target) - [x, y]
    assert heading @ to_target / np.linalg.norm(to_target) > 0.99


def test_suggest_reports_unreachable_region():
    after, change, height = _maps()
    after.data[1:20, 1:39] = np.where(after.data[1:20, 1:39] == FREE_CELL, UNKNOWN_CELL, after.data[1:20, 1:39])
    v = suggest(after, change, height, robot_radius=0.2)[0]
    assert v.pose is None and v.seen == 0.0

    change2, kept = drop_unviewable(change, [v])
    assert kept == [] and not (change2.data == LOOK).any()


def test_nearby_regions_share_one_stop():
    after, change, height = _maps()
    change.data[2:6, 8:10] = LOOK  # second region next to the first, visible from the same place
    after.data[2:6, 8:10] = UNKNOWN_CELL
    vps = suggest(after, change, height, robot_radius=0.2, range_m=(0.3, 3.0))
    assert len(vps) == 2
    assert vps[0].pose[:2] == vps[1].pose[:2]
