"""Find gravity and the floor in a reconstruction that does not know them (pi3, VGGT, ...).

Feed-forward models put the world in the first camera's frame, so "up" is unknown.
Two cheap cues fix it:
1. A hand-held phone keeps one image axis roughly vertical (portrait or landscape,
   rotated or not), so that axis is nearly constant across frames, while the
   horizontal axis turns with the walker and averages out. The axis with the larger
   mean is vertical. It is tilted when the phone looks down, so it is only a guide.
2. The floor is the lowest large plane whose normal is near that guide.

Metric scale is not observable from images alone. `scale_from_camera_height`
assumes the phone was held at a typical height; LiDAR, robot RGB-D and Isaac
inputs are already metric and skip it.
"""

from __future__ import annotations

import numpy as np


def vertical_axis(T_wc: np.ndarray) -> np.ndarray:
    """Unit vector along the image axis that stays vertical (sign not yet fixed)."""
    x = T_wc[:, :3, 0].mean(0)
    y = T_wc[:, :3, 1].mean(0)
    v = x if np.linalg.norm(x) > np.linalg.norm(y) else y
    return v / np.linalg.norm(v)


def _rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Rotation taking `up` to +y (Rodrigues)."""
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(up, y), float(up @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def level_frame(points: np.ndarray, T_wc: np.ndarray, n_planes: int = 8, max_tilt_deg: float = 45.0,
                min_share: float = 0.02, dist_frac: float = 0.005) -> np.ndarray:
    """4x4 world -> level transform: floor at y = 0, +y up.

    Fits the `n_planes` largest planes (RANSAC, open3d). The floor is the lowest plane
    holding >= `min_share` of the points whose normal is within `max_tilt_deg` of the
    image-axis cue (the cue is tilted when the phone looks down, so it is only a guide).
    """
    import open3d as o3d

    down = vertical_axis(T_wc)  # OpenCV image y points down
    span = float(np.ptp(points @ down))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    best = None
    for _ in range(n_planes):
        if len(pc.points) < 100:
            break
        (a, b, c, _), inl = pc.segment_plane(dist_frac * span, 3, 500)
        n = np.array([a, b, c])
        n /= np.linalg.norm(n)
        if n @ down > 0:
            n = -n
        p = np.asarray(pc.points)[inl]
        if len(inl) >= min_share * len(points) and n @ -down > np.cos(np.radians(max_tilt_deg)):
            h = float(np.median(p @ down))  # larger = lower
            if best is None or h > best[0]:
                best = (h, n, p)
        pc = pc.select_by_index(inl, invert=True)
    if best is None:
        raise ValueError("no floor plane found")
    _, up, p = best
    T = np.eye(4)
    T[:3, :3] = _rotation_to_y(up)
    T[1, 3] = -float(np.median(p @ up))
    return T


def camera_height(T_wc: np.ndarray, T_level: np.ndarray) -> float:
    """Median camera height above the floor, in reconstruction units."""
    c = T_wc[:, :3, 3] @ T_level[:3, :3].T + T_level[:3, 3]
    return float(np.median(c[:, 1]))


def scale_from_camera_height(T_wc: np.ndarray, T_level: np.ndarray, metres: float = 1.5) -> float:
    """Metres per reconstruction unit, assuming the phone was held `metres` above the floor.

    1.5 m default: the median phone height in the pair01 ARKit capture was 1.57 m, and
    1.5 m gives pi3 a scale 2% off the ARKit Sim3 scale there (one person, one room).
    """
    return metres / camera_height(T_wc, T_level)
