"""Build a SceneDiff pair folder from chosen keyframes instead of uniform sampling.

For each video: score every frame (sharpness rank x finger penalty), compute the
10 cm voxels each frame's LiDAR depth sees, and pick n frames by weighted set
cover (`roomdiff.keyframes.select_coverage`). The chosen frames are copied as
00000.jpg.. so SceneDiff runs them with resample 1.

Usage: uv run python tools/make_keyframe_pair.py data/raw/pair01 data/scenediff_own/pair01 \
           data/scenediff_own/pair01_kf --n 14
Writes into out_dir: video{1,2}_frames/, keyframes.json, finger_masks.npz, keyframes.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from roomdiff.io.stray import backproject, load_stray
from roomdiff.keyframes import finger_mask, quality, select_coverage, sharpness

VIDEOS = {1: "before", 2: "after"}


def capture_dir(pair: Path, which: str) -> Path:
    return next(p for p in (pair / which).iterdir() if (p / "odometry.csv").exists())


def frame_voxels(cap, i: int, voxel: float = 0.1, px_step: int = 4) -> np.ndarray:
    """Ids of voxels hit by frame i's depth (subsampled pixels), in that capture's world frame."""
    d = cap.depth(i)[::px_step, ::px_step]
    c = cap.confidence(i)[::px_step, ::px_step]
    K = cap.K_depth(i).copy()
    K[:2] /= px_step  # subsampled pixel k is full pixel k * px_step
    pts = backproject(d, K, cap.T_wc[i], c, max_depth=4.0)
    ijk = np.floor(pts / voxel).astype(np.int64)
    return np.unique((ijk[:, 0] + 10_000) * 100_000_000 + (ijk[:, 1] + 10_000) * 10_000 + ijk[:, 2] + 10_000)


def n_covered(seen: list[np.ndarray], idx) -> int:
    return len(np.unique(np.concatenate([seen[i] for i in idx])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_pair", type=Path, help="data/raw/pairNN with before/ and after/ Stray captures")
    ap.add_argument("frames_pair", type=Path, help="folder with upright video{1,2}_frames/")
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--min-gap", type=int, default=15)
    args = ap.parse_args()

    report, masks, sheet = {}, {}, []
    for v, which in VIDEOS.items():
        cap = load_stray(capture_dir(args.raw_pair, which))
        frames = sorted((args.frames_pair / f"video{v}_frames").glob("*.jpg"))
        n = min(len(frames), len(cap))
        sharp, occ, fmasks = np.zeros(n), np.zeros(n), []
        for i in range(n):
            img = cv2.imread(str(frames[i]))
            sharp[i] = sharpness(img)
            m = finger_mask(img)
            occ[i] = m.mean()
            fmasks.append(m)
        q = quality(sharp, occ)
        seen = [frame_voxels(cap, i) for i in range(n)]
        picks = select_coverage(seen, q, args.n, args.min_gap)

        dst = args.out / f"video{v}_frames"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for k, i in enumerate(picks):
            shutil.copy(frames[i], dst / f"{k:05d}.jpg")
            masks[f"video{v}_{k}"] = fmasks[i]
        uniform = np.arange(0, n, 90)
        report[f"video{v}"] = {
            "frames": picks.tolist(),
            "finger_fraction": np.round(occ[picks], 3).tolist(),
            "quality": np.round(q[picks], 3).tolist(),
            "voxels_covered": n_covered(seen, picks),
            "voxels_covered_uniform_1_in_90": n_covered(seen, uniform),
            "uniform_mean_quality": round(float(q[uniform].mean()), 3),
            "chosen_mean_quality": round(float(q[picks].mean()), 3),
        }
        tiles = []
        for k, i in enumerate(picks):
            t = cv2.resize(cv2.imread(str(frames[i])), (180, 240))
            if occ[i] > 0.005:
                over = cv2.resize(fmasks[i].astype(np.uint8), (180, 240), interpolation=cv2.INTER_NEAREST) > 0
                t[over] = (0.5 * t[over] + [0, 0, 127]).astype(np.uint8)
            cv2.putText(t, f"{which} {k}: f{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            tiles.append(t)
        tiles += [np.zeros_like(tiles[0])] * (args.n - len(tiles))
        sheet.append(np.vstack([np.hstack(tiles[: args.n // 2]), np.hstack(tiles[args.n // 2 :])]))

    np.savez_compressed(args.out / "finger_masks.npz", **masks)
    (args.out / "keyframes.json").write_text(json.dumps(report, indent=2))
    cv2.imwrite(str(args.out / "keyframes.jpg"), np.vstack(sheet))
    print(json.dumps({v: {k: r[k] for k in r if k not in ("quality",)} for v, r in report.items()}, indent=1))


if __name__ == "__main__":
    main()
