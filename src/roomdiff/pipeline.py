"""The 3D-first pipeline on one reconstructed pair: load, align video 2 onto video 1, build the change map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roomdiff.align import refine_between_videos
from roomdiff.changemap import ChangeMap, build, frame_points
from roomdiff.frames import Frames, concat, from_recon_npz


@dataclass
class Pair:
    frames: Frames  # both videos, video 2 aligned onto video 1
    cm: ChangeMap
    voxel: float
    icp: tuple[float, float]  # fitness before / after the between-video refinement


def points(frames: Frames, step: int = 4) -> np.ndarray:
    return np.concatenate([frame_points(frames, i, step) for i in range(len(frames))])


def run(recon: str | Path, div: float = 80, min_voxels: int = 5, icp: bool = True) -> Pair:
    """Voxel = median depth / `div` (scale-free); defaults are the settings tuned in S9."""
    F = from_recon_npz(recon)
    voxel = F.median_depth() / div
    before, after = F.of_video(1), F.of_video(2)
    f0 = f1 = float("nan")
    if icp:
        T, f0, f1 = refine_between_videos(points(after), points(before), voxel)
        if f1 > f0:
            after = after.transformed(T)
    cm = build(before, after, voxel=voxel, min_voxels=min_voxels)
    return Pair(concat(before, after), cm, voxel, (f0, f1))
