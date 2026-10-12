import cv2
import numpy as np

from roomdiff.changemap import Blob, ChangeMap
from roomdiff.grid2d import BLOCKED, FREE_CELL, LOOK, OCC_CELL, UNKNOWN_CELL, robot_maps, save_ros
from roomdiff.visibility import FREE, LABELS, OCC, UNKNOWN, Grid


def _map():
    """10 x 10 x 10 voxels of 0.1 m, world = level frame (+y up, floor at y = 0)."""
    g = Grid(origin=np.zeros(3), size=0.1, shape=(10, 10, 10))
    n = 1000
    before, after, lab = (np.full(n, UNKNOWN, np.uint8) for _ in range(3))
    lab[:] = 0
    ijk = np.stack(np.unravel_index(np.arange(n), g.shape), 1)
    i, j, k = ijk.T
    floor = (j == 0) & (i < 8)
    before[floor] = after[floor] = OCC  # floor seen in both, x < 0.8 m
    air = (j >= 2) & (j <= 5) & (i < 8)
    before[air] = after[air] = FREE
    box = (i == 3) & (k == 3) & (j >= 2) & (j <= 4)  # added box column at (0.3, 0.3)
    after[box] = OCC
    lab[box] = LABELS["added"]
    boot = (i == 8) & (k == 5) & (j == 2)  # seen occupied before, never after
    before[boot] = OCC
    lab[boot] = LABELS["unobserved"]
    blobs = [Blob(0, "added", np.flatnonzero(box), np.zeros(3), np.zeros(3), int(box.sum()))]
    return ChangeMap(g, lab, before, after, blobs, 0.05), np.eye(4)


def _cell(m, x, z):
    c = np.floor((np.array([x, -z]) - m.origin) / m.resolution).astype(int)
    return m.data[c[1], c[0]]


def test_robot_maps_occupancy_and_change():
    cm, T = _map()
    maps = robot_maps(cm, T, res=0.1, band=(0.1, 1.0), min_look_voxels=1)
    assert _cell(maps.before, 0.3, 0.3) == FREE_CELL
    assert _cell(maps.after, 0.3, 0.3) == OCC_CELL
    assert _cell(maps.change, 0.3, 0.3) == BLOCKED
    assert _cell(maps.after, 0.5, 0.5) == FREE_CELL
    assert _cell(maps.before, 0.8, 0.5) == OCC_CELL  # the boot
    assert _cell(maps.after, 0.8, 0.5) == UNKNOWN_CELL
    assert _cell(maps.change, 0.8, 0.5) == LOOK
    c = np.floor((np.array([0.3, -0.3]) - maps.after.origin) / 0.1).astype(int)
    assert maps.height[c[1], c[0]] == np.float32(0.45)  # box voxels up to y = 0.4, top face at 0.45


def test_save_ros_writes_map_server_files(tmp_path):
    cm, T = _map()
    maps = robot_maps(cm, T, res=0.1, band=(0.1, 1.0))
    y = save_ros(maps.after, tmp_path / "after")
    text = y.read_text()
    assert "mode: trinary" in text and "resolution: 0.1" in text
    img = cv2.imread(str(tmp_path / "after.pgm"), cv2.IMREAD_UNCHANGED)
    assert img.shape == maps.after.data.shape
    assert set(np.unique(img)) <= {0, 205, 254}
    assert (np.flipud(img) == 0).sum() == (maps.after.data == OCC_CELL).sum()
