"""Coloured point clouds of both visits in the level frame (metres, floor y = 0, +y up), for the 3D viewer.

The level frame is the robot map frame turned on its side: map x = x, map y = -z, height = y
(report.py, grid2d.py), so change boxes from changes.json drop straight in.
"""

from __future__ import annotations

import numpy as np

from roomdiff.changemap import frame_points
from roomdiff.frames import Frames


def visit_cloud(frames: Frames, T_level: np.ndarray, scale: float, voxel: float = 0.03, step: int = 4,
                max_height: float = 2.6) -> tuple[np.ndarray, np.ndarray]:
    """(points (N,3) float32 in metres, colours (N,3) uint8) of all frames, one point per `voxel` cell
    (mean position and colour). Points above `max_height` (ceiling) are dropped so the room can be seen
    from above."""
    pts, cols = [], []
    for i in range(len(frames)):
        p = frame_points(frames, i, step)
        d = frames.depth[i][::step, ::step]
        ok = frames.valid[i][::step, ::step] & (d > 0)
        pts.append(p)
        cols.append(frames.rgb[i][::step, ::step][ok] if frames.rgb is not None else np.full((len(p), 3), 160))
    p = (np.concatenate(pts) @ T_level[:3, :3].T + T_level[:3, 3]) * scale
    c = np.concatenate(cols).astype(np.float32)
    keep = p[:, 1] < max_height
    p, c = p[keep], c[keep]
    key = np.floor(p / voxel).astype(np.int64)
    _, inv, n = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    mean_p = np.zeros((len(n), 3))
    mean_c = np.zeros((len(n), 3))
    np.add.at(mean_p, inv, p)
    np.add.at(mean_c, inv, c)
    return (mean_p / n[:, None]).astype(np.float32), np.clip(mean_c / n[:, None], 0, 255).astype(np.uint8)


def clouds(frames: Frames, T_level: np.ndarray, scale: float, voxel: float = 0.03) -> dict[str, np.ndarray]:
    """npz-ready arrays: before_xyz, before_rgb, after_xyz, after_rgb."""
    out = {}
    for v, name in ((1, "before"), (2, "after")):
        p, c = visit_cloud(frames.of_video(v), T_level, scale, voxel)
        out[f"{name}_xyz"], out[f"{name}_rgb"] = p, c
    return out
