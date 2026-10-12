"""3-state visibility map from depth frames: what was seen, seen through, or never seen.

For every voxel centre and every depth frame, project the centre into the frame:
- hit:  measured depth agrees with the voxel depth (within `tau`): a surface is there
- free: the camera saw past the voxel (measured depth > voxel depth + tau): it is empty
- nothing otherwise (outside the image, invalid depth, or hidden behind a surface)

Per video, counts become a state: OCC, FREE or UNKNOWN. Comparing two videos:
- removed:    OCC before, FREE after   (and no OCC voxel next to it after)
- added:      FREE before, OCC after   (and no OCC voxel next to it before)
- unchanged:  OCC in both
- unobserved: OCC in one, UNKNOWN in the other. This is the case a 2D detector
  gets wrong: the object may have changed, but the evidence does not exist.

The neighbour check absorbs ~1 voxel of registration error at surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

UNKNOWN, FREE, OCC = 0, 1, 2
LABELS = {"unchanged": 1, "removed": 2, "added": 3, "unobserved": 4}


@dataclass
class Grid:
    origin: np.ndarray  # (3,) world position of voxel (0,0,0) centre
    size: float  # voxel edge, metres
    shape: tuple[int, int, int]

    @classmethod
    def around(cls, points: np.ndarray, size: float, pad: float = 0.1) -> Grid:
        lo, hi = points.min(0) - pad, points.max(0) + pad
        shape = tuple(int(n) for n in np.ceil((hi - lo) / size).astype(int))
        return cls(origin=lo + size / 2, size=size, shape=shape)

    def centres(self) -> np.ndarray:
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), -1)
        return self.origin + idx.reshape(-1, 3) * self.size

    def centres_of(self, flat: np.ndarray) -> np.ndarray:
        """Centres of the given flat voxel indices."""
        return self.origin + np.stack(np.unravel_index(flat, self.shape), axis=1) * self.size

    def index(self, points: np.ndarray) -> np.ndarray:
        """Flat voxel index of each point, -1 if outside the grid."""
        ijk = np.round((points - self.origin) / self.size).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        flat = np.full(len(points), -1)
        flat[ok] = np.ravel_multi_index(ijk[ok].T, self.shape)
        return flat


def observe(
    centres: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    tau: float = 0.05,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (hit, free) per voxel centre for one depth frame (OpenCV camera axes).

    `valid` masks depth pixels to trust (confidence, range, edges); default: depth > 0.
    """
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pc = (centres - t) @ R  # world -> camera: R^T (x - t), written for row vectors
    z = pc[:, 2]
    hit = np.zeros(len(centres), bool)
    free = np.zeros(len(centres), bool)
    front = z > 0.05
    u = np.round(K[0, 0] * pc[front, 0] / z[front] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[front, 1] / z[front] + K[1, 2]).astype(int)
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sel = np.flatnonzero(front)[inside]
    d = depth[v[inside], u[inside]]
    ok = d > 0
    if valid is not None:
        ok &= valid[v[inside], u[inside]]
    zs = z[sel]
    t = tau
    hit[sel] = ok & (np.abs(zs - d) <= t)
    free[sel] = ok & (zs < d - t)
    return hit, free


def classify(hits: np.ndarray, frees: np.ndarray, min_views: int = 2, occ_ratio: float = 0.5) -> np.ndarray:
    """Per-voxel state from view counts: OCC, FREE or UNKNOWN."""
    n = hits + frees
    ratio = np.divide(hits, n, out=np.zeros(len(n)), where=n > 0)
    state = np.full(len(n), UNKNOWN, np.uint8)
    state[(frees >= min_views) & (ratio < 0.2)] = FREE
    state[(hits >= min_views) & (ratio >= occ_ratio)] = OCC
    return state


def compare(before: np.ndarray, after: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Label volume (flat) with LABELS values; 0 = empty or irrelevant."""
    occ_b = ndimage.binary_dilation((before == OCC).reshape(shape)).ravel()
    occ_a = ndimage.binary_dilation((after == OCC).reshape(shape)).ravel()
    out = np.zeros(len(before), np.uint8)
    out[(before == OCC) & (after == OCC)] = LABELS["unchanged"]
    out[(before == OCC) & (after == FREE) & ~occ_a] = LABELS["removed"]
    out[(before == FREE) & (after == OCC) & ~occ_b] = LABELS["added"]
    out[(before == OCC) & (after == UNKNOWN)] = LABELS["unobserved"]
    out[(before == UNKNOWN) & (after == OCC)] = LABELS["unobserved"]
    return out


def components(mask: np.ndarray, shape: tuple[int, int, int], min_voxels: int = 10) -> list[np.ndarray]:
    """Connected components (26-neighbour) of a flat mask, as arrays of flat indices, largest first."""
    lab, n = ndimage.label(mask.reshape(shape), structure=np.ones((3, 3, 3)))
    flat = lab.ravel()
    order = np.argsort(flat)
    bounds = np.searchsorted(flat[order], np.arange(1, n + 2))
    comps = [order[bounds[i] : bounds[i + 1]] for i in range(n)]
    return sorted((c for c in comps if len(c) >= min_voxels), key=len, reverse=True)
