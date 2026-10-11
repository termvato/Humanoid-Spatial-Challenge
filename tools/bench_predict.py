"""3D-first predictions for SceneDiff benchmark pairs, in its object_masks.pkl format (S9b).

For each reconstruction outputs/recon/<model>_<scene>.npz: build the change map
scale-free (voxel = median depth / --div), project blobs to 2D masks on the
evaluator's frames, write <out>/pred/<scene>/object_masks.pkl, plus a summary.
Then upload <out>/pred to the volume and run the official evaluator (infra: evalpred).

Usage: uv run python tools/bench_predict.py --model pi3 --scenes a,b,c --div 25 --out outputs/s9/pi3_d25
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from roomdiff.changemap import build, to_scenediff
from roomdiff.frames import from_recon_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi3")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--div", type=float, default=25)
    ap.add_argument("--min-voxels", type=int, default=15)
    ap.add_argument("--min-conf", type=float, default=0.1)
    ap.add_argument("--merge", type=int, default=0, help="join blobs within N voxels")
    ap.add_argument("--refine", choices=["none", "sam"], default="none")
    ap.add_argument("--recon-dir", type=Path, default=Path("outputs/recon"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    refine = None
    if args.refine == "sam":
        from roomdiff.objects import SamRefiner

        refine = SamRefiner()
    summary = {}
    for scene in args.scenes.split(","):
        F = from_recon_npz(args.recon_dir / f"{args.model}_{scene}.npz", min_conf=args.min_conf)
        voxel = F.median_depth() / args.div
        cm = build(F.of_video(1), F.of_video(2), voxel=voxel, min_voxels=args.min_voxels, merge=args.merge)
        pred = to_scenediff(cm, F, refine=refine)
        d = args.out / "pred" / scene
        d.mkdir(parents=True, exist_ok=True)
        (d / "object_masks.pkl").write_bytes(pickle.dumps(pred))
        summary[scene] = {"blobs": len(cm.blobs), "exported": len(pred) - 2, "counts": cm.counts()}
        print(scene, summary[scene])
    (args.out / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       "scenes": summary}, indent=2))


if __name__ == "__main__":
    main()
