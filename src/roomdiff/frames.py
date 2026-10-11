"""One container for posed depth frames, whatever produced them.

LiDAR (Stray), pi3, DA3, VGGT, MapAnything all reduce to: per frame a depth map,
a validity mask, intrinsics at the depth resolution, a camera-to-world pose in
OpenCV axes, and which video and source frame it came from. The change map and
the evaluation only ever see this type, so model comparisons share all other code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from roomdiff.io.stray import StrayCapture


@dataclass
class Frames:
    depth: np.ndarray  # (N,H,W) float32, units of the producer (metres for LiDAR, arbitrary for pi3)
    valid: np.ndarray  # (N,H,W) bool
    K: np.ndarray  # (N,3,3) at depth resolution
    T_wc: np.ndarray  # (N,4,4) camera-to-world, OpenCV axes
    video: np.ndarray  # (N,) 1 = before, 2 = after
    src: np.ndarray  # (N,) frame index in the source video
    rgb: np.ndarray | None = None  # (N,H,W,3) uint8, optional
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.depth)

    def select(self, idx: np.ndarray) -> Frames:
        return Frames(self.depth[idx], self.valid[idx], self.K[idx], self.T_wc[idx], self.video[idx],
                      self.src[idx], None if self.rgb is None else self.rgb[idx], dict(self.meta))

    def of_video(self, v: int) -> Frames:
        return self.select(np.flatnonzero(self.video == v))

    def transformed(self, T: np.ndarray) -> Frames:
        """Same frames with every pose left-multiplied by T (moves the world frame)."""
        f = self.select(np.arange(len(self)))
        f.T_wc = T @ self.T_wc
        return f

    def scaled(self, s: float) -> Frames:
        """Scale depth and camera positions by s (e.g. arbitrary model units -> metres)."""
        f = self.select(np.arange(len(self)))
        f.depth = self.depth * s
        f.T_wc = self.T_wc.copy()
        f.T_wc[:, :3, 3] *= s
        return f

    def median_depth(self) -> float:
        return float(np.median(self.depth[self.valid]))


def depth_edges(depth: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """True at depth discontinuities: (3x3 max - 3x3 min) / depth > rtol. Same rule as pi3's depth_edge."""
    hi = ndimage.maximum_filter(depth, size=3)
    lo = ndimage.minimum_filter(depth, size=3)
    return (hi - lo) > rtol * np.maximum(depth, 1e-6)


def concat(a: Frames, b: Frames) -> Frames:
    rgb = None if a.rgb is None or b.rgb is None else np.concatenate([a.rgb, b.rgb])
    return Frames(*(np.concatenate([getattr(a, k), getattr(b, k)])
                    for k in ("depth", "valid", "K", "T_wc", "video", "src")), rgb, dict(a.meta))


def from_recon_npz(path: str | Path, min_conf: float = 0.1, edge_rtol: float = 0.03) -> Frames:
    """Load a model reconstruction written by infra (reconstruct): one shared frame for both videos."""
    z = np.load(path)
    depth = z["depth"].astype(np.float32)
    conf = z["conf"].astype(np.float32)
    valid = (conf > min_conf) & (depth > 0)
    for i in range(len(depth)):
        valid[i] &= ~depth_edges(depth[i], edge_rtol)
    return Frames(depth, valid, z["K"].astype(np.float64), z["T_wc"].astype(np.float64), z["video"], z["src"],
                  z["rgb"] if "rgb" in z else None, {"source": str(path)})


def from_stray(cap: StrayCapture, video: int, stride: int = 5, min_conf: int = 2, max_depth: float = 4.0) -> Frames:
    """LiDAR frames of one Stray capture (in its own ARKit world frame)."""
    idx = np.arange(0, len(cap), stride)
    depth = np.stack([cap.depth(i) for i in idx])
    conf = np.stack([cap.confidence(i) for i in idx])
    valid = (conf >= min_conf) & (depth > 0) & (depth <= max_depth)
    return Frames(depth, valid, np.stack([cap.K_depth(i) for i in idx]), cap.T_wc[idx].copy(),
                  np.full(len(idx), video), cap.frames[idx].copy(), None, {"source": str(cap.root)})
