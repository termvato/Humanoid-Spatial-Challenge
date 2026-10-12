"""Full 3D-first pipeline on one reconstructed pair, with a picture per detected change.

recon npz -> robust ICP of video 2 onto video 1 -> change map -> SAM object masks -> tiles.

Usage: uv run --group gpu python tools/run_pair.py outputs/recon/pi3_pair01.npz --div 80 --out outputs/s9/pair01
Writes: objects.jpg (best view per change, before/after side by side where available), result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roomdiff.changemap import collect_views
from roomdiff.objects import SamRefiner
from roomdiff.pipeline import run


def tile(rgb: np.ndarray, mask: np.ndarray, colour, text: str) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    over = img.copy()
    over[mask] = colour
    img = cv2.addWeighted(img, 0.65, over, 0.35, 0)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, colour, 3)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recon", type=Path)
    ap.add_argument("--div", type=float, default=80)
    ap.add_argument("--min-voxels", type=int, default=5)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pair = run(args.recon, div=args.div, min_voxels=args.min_voxels)
    F, cm, voxel, (f0, f1) = pair.frames, pair.cm, pair.voxel, pair.icp
    # every frame of the pair is "scored" here: resample 1 over the stride-10 frames we have
    views = collect_views(cm, F, resample=int(F.src[1] - F.src[0]), refine=SamRefiner())
    blobs = {b.id: b for b in cm.blobs}
    ranked = sorted(views, key=lambda k: -blobs[k].score)[: args.top]
    tiles, report = [], []
    for k in ranked:
        b = blobs[k]
        i, m = max(views[k].items(), key=lambda kv: kv[1].sum())
        col = (60, 60, 230) if b.kind == "removed" else (60, 200, 60)
        tiles.append(tile(F.rgb[i], m, col, f"#{len(report)} {b.kind} {len(b.idx)} vox  "
                                              f"{'before' if F.video[i] == 1 else 'after'} f{F.src[i]}"))
        report.append({"rank": len(report), "kind": b.kind, "voxels": len(b.idx), "frame": int(F.src[i]),
                       "video": int(F.video[i]), "views": len(views[k])})
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(args.out / "objects.jpg"), np.vstack([np.hstack(tiles[j:j + 6])
                                                             for j in range(0, len(tiles), 6)]))
    res = {"icp_fitness": [round(f0, 3), round(f1, 3)], "voxel": voxel, "counts": cm.counts(),
           "blobs": len(cm.blobs), "with_views": len(views), "top": report}
    (args.out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "top"}))


if __name__ == "__main__":
    main()
