"""The 3D-first pipeline on one pair: load, align video 2 onto video 1, build the change map.

Two inputs, one change map: a feed-forward reconstruction of both videos (`run`, plain
video; scale-free, already in one frame) or two Stray Scanner LiDAR captures
(`run_lidar`; metric, gravity-aligned, each in its own world, so they are registered).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roomdiff.align import Registration, refine_between_videos, register
from roomdiff.changemap import ChangeMap, build, frame_points
from roomdiff.frames import Frames, concat, from_recon_npz, from_stray
from roomdiff.io.stray import load_stray
from roomdiff.level import level_frame, scale_from_camera_height


@dataclass
class Pair:
    frames: Frames  # both videos, video 2 aligned onto video 1
    cm: ChangeMap
    voxel: float
    icp: tuple[float, float]  # fitness before / after the between-video refinement


def points(frames: Frames, step: int = 4) -> np.ndarray:
    return np.concatenate([frame_points(frames, i, step) for i in range(len(frames))])


def run(recon: str | Path, div: float = 80, min_voxels: int = 5, icp: bool = True, min_conf: float = 0.1,
        max_incidence: float | None = None, slant: bool = False,
        footprint: bool = False) -> Pair:
    """Voxel = median depth / `div` (scale-free); defaults are the settings tuned in S9."""
    F = from_recon_npz(recon, min_conf=min_conf)
    voxel = F.median_depth() / div
    before, after = F.of_video(1), F.of_video(2)
    f0 = f1 = float("nan")
    if icp:
        T, f0, f1 = refine_between_videos(points(after), points(before), voxel)
        if f1 > f0:
            after = after.transformed(T)
    cm = build(before, after, voxel=voxel, min_voxels=min_voxels, max_incidence=max_incidence,
               slant=slant, footprint=footprint)
    return Pair(concat(before, after), cm, voxel, (f0, f1))


def capture_dir(pair_dir: str | Path, which: str) -> Path:
    """The Stray capture folder inside <pair_dir>/<which> (before or after)."""
    return next(p for p in (Path(pair_dir) / which).iterdir() if (p / "odometry.csv").exists())


def lidar_frames(pair_dir: str | Path, stride: int = 5) -> tuple[Frames, Frames]:
    """Before (video 1) and after (video 2) LiDAR frames, each in its own ARKit world."""
    return (from_stray(load_stray(capture_dir(pair_dir, "before")), 1, stride=stride),
            from_stray(load_stray(capture_dir(pair_dir, "after")), 2, stride=stride))


def register_lidar(source: Frames, target: Frames, voxel: float = 0.05) -> Registration:
    """Gravity-aware 4-DoF registration of two LiDAR visits (S7), from every other frame's points."""
    return register(points(source.select(slice(None, None, 2)), step=2),
                    points(target.select(slice(None, None, 2)), step=2), voxel=voxel)


def run_lidar(pair_dir: str | Path, voxel: float = 0.05, stride: int = 5, min_voxels: int = 15) -> Pair:
    """LiDAR change map in metres: after registered onto before, voxels and surface band of `voxel`."""
    before, after = lidar_frames(pair_dir, stride)
    reg = register_lidar(after, before, voxel)
    after = after.transformed(reg.T)
    cm = build(before, after, voxel=voxel, min_voxels=min_voxels)
    return Pair(concat(before, after), cm, voxel, (reg.fitness, reg.fitness))


def level(pair: Pair, metric: bool, scale: float = 0.0) -> tuple[np.ndarray, float]:
    """World -> level transform (floor y = 0, +y up) and metres per unit: 1 for metric input,
    `scale` if given, else from the phone-height assumption (level.scale_from_camera_height)."""
    T = level_frame(points(pair.frames, step=8), pair.frames.T_wc)
    if metric:
        return T, 1.0
    return T, scale or scale_from_camera_height(pair.frames.T_wc, T)
