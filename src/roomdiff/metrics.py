"""Geometry metrics for comparing reconstruction models against ARKit + LiDAR."""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t with dst ~ s R src + t (Umeyama 1991). src, dst: (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def ate(est: np.ndarray, ref: np.ndarray) -> tuple[float, float, np.ndarray]:
    """RMSE of camera centres after Sim(3) alignment. Returns (rmse, scale, 4x4 est->ref)."""
    s, R, t = umeyama(est, ref)
    aligned = s * est @ R.T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))
    return rmse, s, sim3_matrix(s, R, t)


def depth_errors(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, scale: float | None = None) -> dict:
    """AbsRel and delta<1.25 of pred vs ref on mask.

    scale=None: per-image median scaling (shape quality only). Otherwise the given
    global scale (tests multi-view consistent, metric-after-Sim3 depth).
    """
    p, r = pred[mask].astype(np.float64), ref[mask].astype(np.float64)
    keep = (p > 0) & (r > 0)
    p, r = p[keep], r[keep]
    s = np.median(r) / np.median(p) if scale is None else scale
    p = p * s
    ratio = np.maximum(p / r, r / p)
    return {"abs_rel": float(np.mean(np.abs(p - r) / r)), "delta1": float(np.mean(ratio < 1.25)), "n": len(r)}
