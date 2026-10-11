"""Loader for Stray Scanner captures (iPhone RGB video + LiDAR depth + ARKit poses).

Format spec: github.com/strayrobots/scanner/blob/main/docs/format.md

Conventions returned by this module:
- Poses are camera-to-world 4x4 matrices in the OpenCV camera convention
  (x right, y down, z forward). Stray's odometry.csv is already in that
  convention: applying `arkit_to_opencv` on top smeared pair01 (frames 30 apart
  disagreed by 18 cm median vs 2 cm without it; docs/spikes.md S7). World axes
  are ARKit's: +y up (gravity), origin where recording started.
  `arkit_to_opencv` is kept for raw ARKit transforms from other sources.
- Intrinsics are per frame, in pixels of the RGB video.
- Depth is metres, float32, at the LiDAR resolution (192 x 256).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Flips y and z of the camera frame: OpenGL camera axes -> OpenCV camera axes.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def arkit_to_opencv(T_wc_gl: np.ndarray) -> np.ndarray:
    """Convert camera-to-world pose(s) from ARKit/OpenGL to OpenCV camera axes."""
    return T_wc_gl @ _GL_TO_CV


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics for an image resized by (sx, sy). Works on (3,3) or (N,3,3)."""
    S = np.diag([sx, sy, 1.0])
    return S @ K


@dataclass
class StrayCapture:
    root: Path
    timestamps: np.ndarray  # (N,) seconds
    frames: np.ndarray  # (N,) int frame index, matches depth/{frame:06d}.png
    T_wc: np.ndarray  # (N,4,4) camera-to-world, OpenCV axes, metres
    K: np.ndarray  # (N,3,3) intrinsics at RGB resolution
    rgb_size: tuple[int, int]  # (width, height) of rgb.mp4

    def __len__(self) -> int:
        return len(self.frames)

    def depth(self, i: int) -> np.ndarray:
        """Depth of the i-th pose row in metres, (192, 256) float32."""
        path = self.root / "depth" / f"{self.frames[i]:06d}.png"
        mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mm is None:
            raise FileNotFoundError(path)
        return mm.astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        """ARKit depth confidence of the i-th row: 0, 1 or 2 (2 = high)."""
        path = self.root / "confidence" / f"{self.frames[i]:06d}.png"
        conf = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            raise FileNotFoundError(path)
        return conf

    def K_depth(self, i: int, depth_shape: tuple[int, int] = (192, 256)) -> np.ndarray:
        """Intrinsics of row i rescaled to the depth map resolution."""
        h, w = depth_shape
        return scale_intrinsics(self.K[i], w / self.rgb_size[0], h / self.rgb_size[1])


def _read_odometry(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [r for r in reader if r]
    cols: dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        vals = [r[j].strip() if j < len(r) else "" for r in rows]
        if all(v == "" for v in vals):
            continue  # optional columns, e.g. distortion centre on non-Pro devices
        cols[name] = np.array([float(v) if v else np.nan for v in vals])
    return cols


def _video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w == 0 or h == 0:
        raise ValueError(f"cannot read video size of {path}")
    return w, h


def load_stray(root: str | Path, rgb_size: tuple[int, int] | None = None) -> StrayCapture:
    """Load a Stray Scanner capture folder.

    `rgb_size` overrides reading the size from rgb.mp4 (used by tests and when
    the video is missing). Older app versions have no per-frame intrinsics in
    odometry.csv; then camera_matrix.csv is used for every frame.
    """
    root = Path(root)
    odo = _read_odometry(root / "odometry.csv")
    n = len(odo["timestamp"])

    t = np.stack([odo["x"], odo["y"], odo["z"]], axis=1)
    q = np.stack([odo["qx"], odo["qy"], odo["qz"], odo["qw"]], axis=1)
    T_gl = np.tile(np.eye(4), (n, 1, 1))
    T_gl[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    T_gl[:, :3, 3] = t

    if all(k in odo for k in ("fx", "fy", "cx", "cy")):
        K = np.zeros((n, 3, 3))
        K[:, 0, 0], K[:, 1, 1] = odo["fx"], odo["fy"]
        K[:, 0, 2], K[:, 1, 2] = odo["cx"], odo["cy"]
        K[:, 2, 2] = 1.0
    else:
        K0 = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
        K = np.tile(K0, (n, 1, 1))

    size = rgb_size if rgb_size is not None else _video_size(root / "rgb.mp4")
    return StrayCapture(
        root=root,
        timestamps=odo["timestamp"],
        frames=odo["frame"].astype(int),
        T_wc=T_gl,  # already OpenCV camera axes, see module docstring
        K=K,
        rgb_size=size,
    )


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    conf: np.ndarray | None = None,
    min_conf: int = 2,
    max_depth: float = 5.0,
) -> np.ndarray:
    """World points (M,3) from a depth map, OpenCV camera axes.

    Pixels with depth <= 0, depth > max_depth, or confidence < min_conf are dropped.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))  # OpenCV: pixel centres at integers
    keep = (depth > 0) & (depth <= max_depth)
    if conf is not None:
        keep &= conf >= min_conf
    z = depth[keep]
    x = (u[keep] - K[0, 2]) * z / K[0, 0]
    y = (v[keep] - K[1, 2]) * z / K[1, 1]
    pts_c = np.stack([x, y, z], axis=1)
    return pts_c @ T_wc[:3, :3].T + T_wc[:3, 3]
