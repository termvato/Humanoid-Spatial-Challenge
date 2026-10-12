"""Build the 3-state change map of one own pair from iPhone LiDAR + ARKit poses.

Steps: load both Stray captures, register before -> after with the LiDAR clouds,
carve a voxel grid with every `stride`-th depth frame of each video, compare,
and cluster removed / added voxels into change blobs. Each blob is drawn on the
real video frame that saw it best, so a person can check it.

Usage: uv run python tools/visibility_map.py data/raw/pair01 --out outputs/s7
Writes: result.json, labels.npz (label + per-video state volumes), map.ply, bev.png, blobs.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from roomdiff.align import register, transform
from roomdiff.io.stray import StrayCapture, backproject, load_stray
from roomdiff.visibility import LABELS, Grid, classify, compare, components, observe

COLOURS = {  # RGB 0-1
    "unchanged": (0.65, 0.65, 0.65),
    "removed": (0.9, 0.15, 0.15),
    "added": (0.15, 0.75, 0.2),
    "unobserved": (0.6, 0.3, 0.9),
}


def capture_dir(pair: Path, which: str) -> Path:
    return next(p for p in (pair / which).iterdir() if (p / "odometry.csv").exists())


def cloud(cap: StrayCapture, T: np.ndarray, stride: int, max_depth: float) -> np.ndarray:
    pts = [backproject(cap.depth(i), cap.K_depth(i), T @ cap.T_wc[i], cap.confidence(i), max_depth=max_depth)
           for i in range(0, len(cap), stride)]
    return np.concatenate(pts)


def carve(cap: StrayCapture, T: np.ndarray, centres: np.ndarray, stride: int, tau: float, max_depth: float):
    hits = np.zeros(len(centres), np.uint16)
    frees = np.zeros(len(centres), np.uint16)
    per_frame_hits: dict[int, np.ndarray] = {}
    for i in range(0, len(cap), stride):
        d = cap.depth(i)
        valid = (cap.confidence(i) >= 2) & (d > 0) & (d <= max_depth)
        h, f = observe(centres, d, cap.K_depth(i), T @ cap.T_wc[i], tau=tau, valid=valid)
        hits += h
        frees += f
        per_frame_hits[i] = np.flatnonzero(h)
    return hits, frees, per_frame_hits


def best_view(cap: StrayCapture, T: np.ndarray, pts: np.ndarray, frames: list[int]) -> tuple[int, float]:
    """Frame where the blob projects most fully inside the image and closest to its centre."""
    best, score = frames[0], -1.0
    w, h = cap.rgb_size
    for i in frames:
        Twc = T @ cap.T_wc[i]
        pc = (pts - Twc[:3, 3]) @ Twc[:3, :3]
        if np.any(pc[:, 2] < 0.2):
            continue
        uv = (cap.K[i] @ (pc / pc[:, 2:3]).T).T[:, :2]
        inside = np.mean((uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h))
        centred = 1 - np.linalg.norm(uv.mean(0) / [w, h] - 0.5)
        s = inside + 0.3 * centred - 0.05 * np.median(pc[:, 2])
        if s > score:
            best, score = i, s
    return best, score


def read_frame(video: Path, idx: int) -> np.ndarray:
    """Frame idx, or the nearest earlier decodable one (container frame counts can overstate by one)."""
    v = cv2.VideoCapture(str(video))
    try:
        for k in range(idx, max(idx - 5, -1), -1):
            v.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = v.read()
            if ok:
                return img
    finally:
        v.release()
    raise IndexError(f"{video} frame {idx}")


def draw_blob(cap: StrayCapture, T: np.ndarray, frame: int, pts: np.ndarray, colour, text: str) -> np.ndarray:
    img = read_frame(cap.root / "rgb.mp4", int(cap.frames[frame]))
    Twc = T @ cap.T_wc[frame]
    pc = (pts - Twc[:3, 3]) @ Twc[:3, :3]
    uv = (cap.K[frame] @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.float32)
    hull = cv2.convexHull(uv).astype(np.int32)
    bgr = tuple(int(255 * c) for c in colour[::-1])
    over = img.copy()
    cv2.fillPoly(over, [hull], bgr)
    img = cv2.addWeighted(img, 0.6, over, 0.4, 0)
    cv2.polylines(img, [hull], True, bgr, 8)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Stray stores portrait video sideways
    img = cv2.resize(img, (360, 480))
    cv2.rectangle(img, (0, 0), (360, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/s7"))
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=5, help="use every N-th depth frame for carving")
    ap.add_argument("--tau", type=float, default=0.05, help="surface thickness, metres")
    ap.add_argument("--max-depth", type=float, default=4.0)
    ap.add_argument("--min-voxels", type=int, default=15)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    before = load_stray(capture_dir(args.pair, "before"))
    after = load_stray(capture_dir(args.pair, "after"))
    I = np.eye(4)
    cb, ca = cloud(before, I, 10, args.max_depth), cloud(after, I, 10, args.max_depth)
    reg = register(cb, ca, voxel=args.voxel)
    T_b = reg.T  # before world -> after world; after is the reference frame
    print(f"registration: fitness {reg.fitness:.3f}, rmse {reg.rmse:.4f} m, tilt {reg.tilt_deg:.2f} deg")
    if reg.tilt_deg > 5:
        raise SystemExit("registration failed: up axes disagree, map would be meaningless")

    grid = Grid.around(np.concatenate([transform(cb, T_b), ca]), args.voxel)
    centres = grid.centres()
    hb, fb, pfb = carve(before, T_b, centres, args.stride, args.tau, args.max_depth)
    ha, fa, pfa = carve(after, I, centres, args.stride, args.tau, args.max_depth)
    sb, sa = classify(hb, fb), classify(ha, fa)
    lab = compare(sb, sa, grid.shape)

    counts = {k: int(np.sum(lab == v)) for k, v in LABELS.items()}
    blobs, tiles = [], []
    for kind, cap, T, pf in (("removed", before, T_b, pfb), ("added", after, I, pfa)):
        for c in components(lab == LABELS[kind], grid.shape, args.min_voxels):
            pts = centres[c]
            seen_in = [i for i, h in pf.items() if np.isin(c, h).any()] or list(pf)
            frame, _ = best_view(cap, T, pts, seen_in)
            lo, hi = pts.min(0), pts.max(0)
            blob = {"id": len(blobs), "type": kind, "voxels": len(c), "volume_l": round(len(c) * args.voxel**3 * 1000, 1),
                    "centre": np.round(pts.mean(0), 3).tolist(), "size": np.round(hi - lo + args.voxel, 3).tolist(),
                    "best_frame": int(cap.frames[frame])}
            blobs.append(blob)
            text = f"B{blob['id']} {kind} {blob['volume_l']} L  ({'before' if cap is before else 'after'} f{blob['best_frame']})"
            tiles.append(draw_blob(cap, T, frame, pts, COLOURS[kind], text))

    # top-down view: for each (x, z) column, the most informative label (change > unobserved > unchanged)
    vol = lab.reshape(grid.shape)
    prio = np.array([0, 1, 4, 4, 2])[vol]  # by label value: empty, unchanged, removed, added, unobserved
    top_idx = prio.argmax(axis=1)
    top = np.take_along_axis(vol, top_idx[:, None, :], axis=1)[:, 0, :]
    rgb = np.ones((*top.shape, 3))
    for k, v in LABELS.items():
        rgb[top == v] = COLOURS[k]
    fig, ax = plt.subplots(figsize=(7, 7))
    x0, z0 = grid.origin[0], grid.origin[2]
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower",
              extent=[x0, x0 + grid.shape[0] * args.voxel, z0, z0 + grid.shape[2] * args.voxel])
    for b in blobs:
        ax.annotate(f"B{b['id']}", (b["centre"][0], b["centre"][2]), color="black", fontsize=11, weight="bold")
    for k, c in COLOURS.items():
        ax.plot([], [], "s", color=c, label=f"{k} ({counts[k]} voxels)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Top-down change map, iPhone LiDAR, 5 cm voxels")
    fig.tight_layout()
    fig.savefig(args.out / "bev.png", dpi=110)

    np.savez_compressed(args.out / "labels.npz", labels=lab.reshape(grid.shape), state_before=sb.reshape(grid.shape),
                        state_after=sa.reshape(grid.shape), origin=grid.origin, voxel=args.voxel, T_before_to_after=T_b)
    keep = lab > 0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres[keep]))
    inv = {v: k for k, v in LABELS.items()}
    pcd.colors = o3d.utility.Vector3dVector(np.array([COLOURS[inv[v]] for v in lab[keep]]))
    o3d.io.write_point_cloud(str(args.out / "map.ply"), pcd)
    if tiles:
        rows = [np.hstack(tiles[i : i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i : i + 4])))
                for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(args.out / "blobs.jpg"), np.vstack(rows))

    result = {
        "pair": args.pair.name, "params": {k: v for k, v in vars(args).items() if k not in ("pair", "out")},
        "registration": {"fitness": round(reg.fitness, 3), "rmse_m": round(reg.rmse, 4),
                         "tilt_deg": round(reg.tilt_deg, 2), "yaw_deg": round(reg.yaw_deg, 1), "T_before_to_after": np.round(T_b, 4).tolist()},
        "grid_shape": list(grid.shape), "voxel_counts": counts, "blobs": blobs,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("registration", "voxel_counts", "seconds")}))
    print(f"{len(blobs)} blobs:", [(b["id"], b["type"], b["volume_l"]) for b in blobs])


if __name__ == "__main__":
    main()
