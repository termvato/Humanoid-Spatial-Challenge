"""Register two captures of the same room into one frame.

Each ARKit session has its own world origin (where recording started), but both
have +y = up (gravity). So the unknown is only 4-DoF: heading (yaw) and a shift.

1. Global: for each yaw in a sweep, rasterise both clouds to top-down occupancy
   images of wall-height points and find the best shift by FFT cross-correlation.
   The yaw with the highest overlap wins. Exhaustive, so no local minimum.
2. Refine: point-to-plane ICP restricted to yaw + shift (`icp_4dof`), coarse to fine.
   Unconstrained 6-DoF ICP tilted pair01 by 6 deg: the floor is only ~9% of the
   points, so roll and pitch are weakly constrained, and ARKit gravity is more
   accurate than that.

Generic 6-DoF FPFH + RANSAC was tried first and returned a 168 deg flipped room
on pair01 (logged in docs/spikes.md S7); gravity removes that failure mode.
Changed objects are a small part of the room, so they act as outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class Registration:
    T: np.ndarray  # 4x4, maps points of the source capture into the target frame
    fitness: float  # fraction of source points with a target neighbour within one voxel
    rmse: float  # metres, over those inlier pairs
    tilt_deg: float  # angle between the two up axes after alignment; ~0 if gravity agrees
    yaw_deg: float  # heading found by the global sweep


def _pcd(points: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    p = p.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    return p


def floor_height(points: np.ndarray) -> float:
    return float(np.percentile(points[:, 1], 1))


def _bev(points: np.ndarray, lo: np.ndarray, shape: tuple[int, int], res: float) -> np.ndarray:
    """Binary top-down occupancy (x, z) of points between 0.2 and 2.0 m above the floor."""
    y0 = floor_height(points)
    p = points[(points[:, 1] > y0 + 0.2) & (points[:, 1] < y0 + 2.0)]
    ij = np.floor((p[:, [0, 2]] - lo) / res).astype(int)
    ok = np.all((ij >= 0) & (ij < np.array(shape)), axis=1)
    img = np.zeros(shape, np.float32)
    img[ij[ok, 0], ij[ok, 1]] = 1.0
    return img


def yaw_sweep(source: np.ndarray, target: np.ndarray, step_deg: float = 2.0, res: float = 0.05) -> np.ndarray:
    """Best 4-DoF pose (4x4) of source in target frame by exhaustive yaw + FFT shift search."""
    c_src, c_tgt = source.mean(0), target.mean(0)
    src0, tgt0 = source - c_src, target - c_tgt  # centre both so rotation stays in the canvas
    half = np.abs(np.concatenate([src0, tgt0])[:, [0, 2]]).max() + 0.5
    lo = np.array([-half, -half])
    n = int(np.ceil(2 * half / res))
    tgt_img = _bev(tgt0, lo, (n, n), res)
    best = (-1.0, 0.0, np.zeros(2))
    for yaw in np.arange(0, 360, step_deg):
        R = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        src_img = _bev(src0 @ R.T, lo, (n, n), res)
        corr = fftconvolve(tgt_img, src_img[::-1, ::-1], mode="full")
        k = np.unravel_index(np.argmax(corr), corr.shape)
        if corr[k] > best[0]:
            shift = (np.array(k) - (n - 1)) * res  # target cell = source cell + shift
            best = (float(corr[k]), yaw, shift)
    _, yaw, shift = best
    R = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    dy = floor_height(tgt0) - floor_height(src0 @ R.T)
    T[:3, 3] = c_tgt + np.array([shift[0], dy, shift[1]]) - R @ c_src
    return T


def icp_4dof(src: np.ndarray, tgt: np.ndarray, tgt_normals: np.ndarray, T: np.ndarray, radius: float,
             iters: int = 30) -> np.ndarray:
    """Point-to-plane ICP over yaw (about +y) and translation only.

    Linearised per step: R(dθ) p ≈ p + dθ (ŷ × p), with ŷ × p = (p_z, 0, -p_x).
    Residual n · (p + dθ (ŷ × p) + dt - q), solved by least squares over pairs
    closer than `radius`.
    """
    tree = cKDTree(tgt)
    T = T.copy()
    for _ in range(iters):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(p, distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        p, q, n = p[ok], tgt[j[ok]], tgt_normals[j[ok]]
        cross = np.stack([p[:, 2], np.zeros(len(p)), -p[:, 0]], axis=1)
        A = np.column_stack([np.sum(n * cross, axis=1), n])
        b = -np.sum(n * (p - q), axis=1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        step = np.eye(4)
        step[:3, :3] = Rotation.from_euler("y", x[0]).as_matrix()
        step[:3, 3] = x[1:]
        T = step @ T
        if abs(x[0]) < 1e-5 and np.linalg.norm(x[1:]) < 1e-5:
            break
    return T


def register(source: np.ndarray, target: np.ndarray, voxel: float = 0.05) -> Registration:
    """Pose of `source` points in the `target` frame. Inputs are (N,3) world points, +y up."""
    reg = o3d.pipelines.registration
    T = yaw_sweep(source, target)
    yaw = float(np.degrees(np.arctan2(T[0, 2], T[2, 2])))
    src, tgt = _pcd(source, voxel), _pcd(target, voxel)
    src_p, tgt_p, tgt_n = np.asarray(src.points), np.asarray(tgt.points), np.asarray(tgt.normals)
    for radius in (4 * voxel, 2 * voxel, voxel):  # coarse-to-fine
        T = icp_4dof(src_p, tgt_p, tgt_n, T, radius)
    ev = reg.evaluate_registration(src, tgt, voxel, T)  # downsampled points are up to ~0.7 voxel apart
    up = np.array([0.0, 1.0, 0.0])
    tilt = np.degrees(np.arccos(np.clip(up @ (T[:3, :3] @ up), -1, 1)))
    return Registration(T=np.asarray(T).copy(), fitness=ev.fitness, rmse=ev.inlier_rmse,
                        tilt_deg=float(tilt), yaw_deg=yaw)


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]
