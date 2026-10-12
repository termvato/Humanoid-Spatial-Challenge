"""Score one model reconstruction of an own pair against ARKit poses and LiDAR depth (S5, S9a).

Reports, for the model's joint reconstruction of both videos:
  - camera ATE after Sim(3) vs ARKit (both videos in the after-capture world, so the
    cross-video alignment is tested too), per video and joint
  - depth AbsRel / delta1 vs LiDAR, per-image median scaled (shape) and with the
    single Sim(3) scale (multi-view consistent metric depth)
  - the 3D change map from the model's depth, its blobs compared with the LiDAR
    map's suitcase blobs (S7), metric (Sim(3) scale) and scale-free (voxel = median depth / 40)

Usage: uv run python tools/score_recon.py outputs/recon/pi3_pair01.npz --name pi3 --out outputs/s5
Needs data/raw/<pair> (both Stray captures; they are registered here, as in S7).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from roomdiff.changemap import blob_mask, build
from roomdiff.frames import from_recon_npz
from roomdiff.io.stray import load_stray
from roomdiff.metrics import ate, depth_errors
from roomdiff.pipeline import capture_dir, lidar_frames, register_lidar

# S7 LiDAR suitcase blobs (after-capture world, metres): ground truth locations of the move
SUITCASE = {"removed": np.array([-1.958, -1.107, 2.097]), "added": np.array([-2.847, -1.129, 2.146])}
MATCH_RADIUS = 0.4


def score_blobs(blobs_world: list[dict]) -> dict:
    found = {}
    fps = []
    for b in blobs_world:
        hit = next((k for k, c in SUITCASE.items() if b["kind"] == k and np.linalg.norm(b["centre"] - c) < MATCH_RADIUS),
                   None)
        if hit:
            found[hit] = True
        else:
            fps.append(b)
    return {"suitcase_removed": bool(found.get("removed")), "suitcase_added": bool(found.get("added")),
            "false_alarm_blobs": len(fps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recon", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pair", type=Path, default=Path("data/raw/pair01"))
    ap.add_argument("--out", type=Path, default=Path("outputs/s5"))
    ap.add_argument("--min-voxels", type=int, default=15)
    ap.add_argument("--min-conf", type=float, default=0.1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    F = from_recon_npz(args.recon, min_conf=args.min_conf)
    before, after = load_stray(capture_dir(args.pair, "before")), load_stray(capture_dir(args.pair, "after"))
    T_b = register_lidar(*lidar_frames(args.pair)).T  # before -> after world
    caps = {1: (before, T_b), 2: (after, np.eye(4))}
    ref = np.stack([(caps[v][1] @ caps[v][0].T_wc[min(s, len(caps[v][0]) - 1)])[:3, 3]
                    for v, s in zip(F.video, F.src)])
    est = F.T_wc[:, :3, 3]
    rmse, s, S = ate(est, ref)
    per_video = {f"video{v}": round(ate(est[F.video == v], ref[F.video == v])[0], 4) for v in (1, 2)}

    # metric frames in the after-capture world
    R = S[:3, :3] / s
    Tr = np.eye(4)
    Tr[:3, :3], Tr[:3, 3] = R, S[:3, 3]
    M = F.scaled(s).transformed(Tr)

    # depth vs LiDAR (LiDAR stored sideways: rotate clockwise like the RGB)
    errs_med, errs_glob = [], []
    for i in range(0, len(F), max(1, len(F) // 40)):
        cap = caps[int(F.video[i])][0]
        j = min(int(F.src[i]), len(cap) - 1)
        lid = cv2.rotate(cap.depth(j), cv2.ROTATE_90_CLOCKWISE)
        conf = cv2.rotate(cap.confidence(j), cv2.ROTATE_90_CLOCKWISE)
        h, w = lid.shape
        pred = cv2.resize(F.depth[i], (w, h), interpolation=cv2.INTER_NEAREST)
        pv = cv2.resize(F.valid[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        m = (conf >= 2) & (lid > 0) & (lid < 4) & pv
        if m.sum() < 100:
            continue
        errs_med.append(depth_errors(pred, lid, m))
        errs_glob.append(depth_errors(pred, lid, m, scale=s))

    def mean(es, k):
        return round(float(np.mean([e[k] for e in es])), 4)

    result = {
        "model": args.name, "frames": len(F),
        "ate_m": round(rmse, 4), "ate_per_video_m": per_video, "sim3_scale": round(s, 4),
        "depth_median_scaled": {"abs_rel": mean(errs_med, "abs_rel"), "delta1": mean(errs_med, "delta1")},
        "depth_sim3_scaled": {"abs_rel": mean(errs_glob, "abs_rel"), "delta1": mean(errs_glob, "delta1")},
    }

    # change maps
    tiles = []
    for variant in ("metric", "scale_free"):
        G = M if variant == "metric" else F
        voxel = 0.05 if variant == "metric" else F.median_depth() / 40
        cm = build(G.of_video(1), G.of_video(2), voxel=voxel, min_voxels=args.min_voxels)
        to_world = np.eye(4) if variant == "metric" else S
        blobs = [{"id": b.id, "kind": b.kind, "voxels": len(b.idx),
                  "centre": (to_world[:3, :3] @ b.centre + to_world[:3, 3])} for b in cm.blobs]
        result[f"changemap_{variant}"] = {"voxel": round(voxel, 4), "counts": cm.counts(), **score_blobs(blobs),
                                          "blobs": [{**b, "centre": np.round(b["centre"], 3).tolist()} for b in blobs]}
        if variant == "metric" and F.rgb is not None:
            for b in cm.blobs[:12]:
                Gv = G.of_video(1 if b.kind == "removed" else 2)
                Fv = F.of_video(1 if b.kind == "removed" else 2)
                areas = [blob_mask(Gv, i, cm.grid.centres_of(b.idx), voxel, cm.tau).sum() for i in range(len(Gv))]
                i = int(np.argmax(areas))
                m = blob_mask(Gv, i, cm.grid.centres_of(b.idx), voxel, cm.tau)
                img = cv2.cvtColor(Fv.rgb[i], cv2.COLOR_RGB2BGR)
                col = (60, 60, 230) if b.kind == "removed" else (60, 200, 60)
                over = img.copy()
                over[m] = col
                img = cv2.addWeighted(img, 0.6, over, 0.4, 0)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 3)
                cv2.putText(img, f"B{b.id} {b.kind} {len(b.idx)}v", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                tiles.append(img)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(args.out / f"{args.name}_blobs.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]))
    result["seconds"] = round(time.perf_counter() - t0, 1)
    (args.out / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if not k.startswith("changemap")}
    for k in ("changemap_metric", "changemap_scale_free"):
        summary[k] = {kk: vv for kk, vv in result[k].items() if kk != "blobs"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
