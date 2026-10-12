import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roomdiff.level import camera_height, level_frame, scale_from_camera_height, vertical_axis


def _room(rng):
    """4 x 3 m floor at y=0, a 2 m high wall at z=0, a 0.5 m bed top; world +y up."""
    floor = np.c_[rng.uniform(0, 4, 6000), np.zeros(6000), rng.uniform(0, 3, 6000)]
    wall = np.c_[rng.uniform(0, 4, 8000), rng.uniform(0, 2, 8000), np.zeros(8000)]
    bed = np.c_[rng.uniform(1, 3, 9000), np.full(9000, 0.5), rng.uniform(1, 2, 9000)]
    return np.concatenate([floor, wall, bed])


def _cameras(n, roll_deg=0.0, pitch_deg=30.0, height=1.5):
    """Cameras walking a circle, looking outwards and down; OpenCV axes (y down)."""
    T = np.tile(np.eye(4), (n, 1, 1))
    for i, yaw in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], float)  # camera y down = world -y
        R = Rotation.from_euler("y", yaw).as_matrix() @ base @ Rotation.from_euler("x", -pitch_deg, degrees=True).as_matrix() \
            @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
        T[i, :3, :3] = R
        T[i, :3, 3] = [2 + np.cos(yaw), height, 1.5 + np.sin(yaw)]
    return T


@pytest.mark.parametrize("roll", [0.0, 90.0])
def test_level_recovers_floor_in_unknown_frame(roll):
    rng = np.random.default_rng(0)
    pts, T_wc = _room(rng), _cameras(40, roll_deg=roll)
    # hide gravity: apply an arbitrary rigid motion and scale to everything
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("xyz", [40, -70, 25], degrees=True).as_matrix()
    M[:3, 3] = [3, -1, 2]
    s = 0.6
    pts_m = s * pts @ M[:3, :3].T + M[:3, 3]
    T_m = M @ T_wc
    T_m[:, :3, 3] = s * (T_wc[:, :3, 3] @ M[:3, :3].T) + M[:3, 3]
    T = level_frame(pts_m, T_m)
    p = pts_m @ T[:3, :3].T + T[:3, 3]
    floor = p[:6000]
    assert np.abs(floor[:, 1]).max() < 0.01  # floor flat at y = 0
    assert np.median(p[6000:14000, 1]) > 0  # wall above it
    assert camera_height(T_m, T) == pytest.approx(1.5 * s, rel=0.02)
    assert scale_from_camera_height(T_m, T) == pytest.approx(1 / s, rel=0.02)


def test_vertical_axis_picks_constant_image_axis():
    T = _cameras(40, roll_deg=90.0, pitch_deg=0.0)
    v = vertical_axis(T)
    assert abs(v @ [0, 1, 0]) > 0.99
