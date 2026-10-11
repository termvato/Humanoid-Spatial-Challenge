"""Registration recovers a known yaw + shift on a synthetic room, despite a removed object."""

import numpy as np
from scipy.spatial.transform import Rotation

from roomdiff.align import register, transform


def _room(rng: np.random.Generator, n: int = 40_000) -> np.ndarray:
    """Floor, two walls and a few boxes: enough asymmetry for a unique fit."""
    parts = [
        np.c_[rng.uniform(0, 4, n), np.zeros(n), rng.uniform(0, 3, n)],  # floor y = 0
        np.c_[rng.uniform(0, 4, n), rng.uniform(0, 2.5, n), np.zeros(n)],  # wall z = 0
        np.c_[np.zeros(n), rng.uniform(0, 2.5, n), rng.uniform(0, 3, n)],  # wall x = 0
    ]
    for lo, hi in [((0.5, 0, 0.3), (1.3, 0.9, 0.8)), ((2.5, 0, 1.5), (3.0, 1.8, 2.2)), ((0.2, 0, 2.0), (0.8, 0.5, 2.6))]:
        pts = rng.uniform(lo, hi, (n // 4, 3))
        axis = rng.integers(0, 3, len(pts))  # snap each point to a face of the box
        side = rng.integers(0, 2, len(pts))
        pts[np.arange(len(pts)), axis] = np.where(side, np.array(hi)[axis], np.array(lo)[axis])
        parts.append(pts)
    return np.concatenate(parts)


def test_register_recovers_yaw_and_shift_with_an_object_removed():
    rng = np.random.default_rng(0)
    before = _room(rng)
    T_true = np.eye(4)
    T_true[:3, :3] = Rotation.from_euler("y", 37, degrees=True).as_matrix()
    T_true[:3, 3] = [0.8, 0.3, -1.2]  # ARKit origins differ in height too
    keep = ~((before[:, 0] > 2.4) & (before[:, 2] > 1.4) & (before[:, 1] > 0.01))  # one box removed
    after = transform(before[keep], T_true) + rng.normal(0, 0.005, (keep.sum(), 3))

    reg = register(before, after)
    err_rot = np.degrees(Rotation.from_matrix(reg.T[:3, :3] @ T_true[:3, :3].T).magnitude())
    assert err_rot < 1.0
    assert np.linalg.norm(reg.T[:3, 3] - T_true[:3, 3]) < 0.03
    assert reg.tilt_deg < 1.0
    assert reg.fitness > 0.8


def test_register_handles_large_yaw():
    rng = np.random.default_rng(1)
    before = _room(rng)
    T_true = np.eye(4)
    T_true[:3, :3] = Rotation.from_euler("y", -150, degrees=True).as_matrix()
    T_true[:3, 3] = [-2.0, -0.4, 3.0]
    reg = register(before, transform(before, T_true))
    err_rot = np.degrees(Rotation.from_matrix(reg.T[:3, :3] @ T_true[:3, :3].T).magnitude())
    assert err_rot < 1.0
    assert np.linalg.norm(reg.T[:3, 3] - T_true[:3, 3]) < 0.03
