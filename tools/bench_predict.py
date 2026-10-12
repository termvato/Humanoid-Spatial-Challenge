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

import numpy as np

from roomdiff.align import refine_between_videos
from roomdiff.changemap import build, collect_views, export, frame_points
from roomdiff.frames import concat, from_recon_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi3")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--div", type=float, default=25)
    ap.add_argument("--min-voxels", type=int, default=15)
    ap.add_argument("--min-conf", type=float, default=0.1)
    ap.add_argument("--conf-drop", type=float, default=0.0, help="drop this fraction of least confident pixels "
                                                                 "(replaces --min-conf when > 0)")
    ap.add_argument("--merge", type=int, default=0, help="join blobs within N voxels")
    ap.add_argument("--refine", choices=["none", "sam"], default="none")
    ap.add_argument("--match", type=float, default=0.0, help="moved matching: min colour similarity (0 = off)")
    ap.add_argument("--match-desc", choices=["hist", "dino"], default="hist")
    ap.add_argument("--match-size", type=float, default=0.4, help="moved matching: min voxel count ratio")
    ap.add_argument("--tau-rel", type=float, default=0.0, help="surface tolerance growth with depth")
    ap.add_argument("--icp", action="store_true", help="refine video 2 onto video 1 with robust ICP")
    ap.add_argument("--recon-dir", type=Path, default=Path("outputs/recon"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    refine = None
    if args.refine == "sam":
        from roomdiff.objects import SamRefiner

        refine = SamRefiner()
    descriptor = None
    if args.match_desc == "dino":
        from roomdiff.objects import DinoDescriptor

        descriptor = DinoDescriptor()
    summary = {}
    for scene in args.scenes.split(","):
        F = from_recon_npz(args.recon_dir / f"{args.model}_{scene}.npz", min_conf=args.min_conf, conf_drop=args.conf_drop)
        voxel = F.median_depth() / args.div
        before, after = F.of_video(1), F.of_video(2)
        icp = None
        if args.icp:
            pb = np.concatenate([frame_points(before, i) for i in range(len(before))])
            pa = np.concatenate([frame_points(after, i) for i in range(len(after))])
            T, f0, f1 = refine_between_videos(pa, pb, voxel)
            icp = {"fitness_before": round(f0, 3), "fitness_after": round(f1, 3), "applied": f1 > f0}
            if f1 > f0:
                after = after.transformed(T)
                F = concat(before, after)
        cm = build(before, after, voxel=voxel, min_voxels=args.min_voxels, merge=args.merge,
                   tau_rel=args.tau_rel)
        match = ({"min_sim": args.match, "min_size_ratio": args.match_size,
                  "kind": "cos" if args.match_desc == "dino" else "hist"} if args.match > 0 else None)
        cache = Path("outputs/s9/cache") / (f"{args.model}_{scene}_d{args.div:g}_v{args.min_voxels}_g{args.merge}"
                                            f"_c{args.min_conf:g}_{args.refine}{'_icp' if args.icp else ''}"
                                            f"{f'_t{args.tau_rel:g}' if args.tau_rel else ''}"
                                            f"{f'_q{args.conf_drop:g}' if args.conf_drop else ''}.pkl")
        if cache.exists():
            views = pickle.loads(cache.read_bytes())
        else:
            views = collect_views(cm, F, refine=refine)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(views))
        pred = export(cm, F, views, match=match, descriptor=descriptor)
        d = args.out / "pred" / scene
        d.mkdir(parents=True, exist_ok=True)
        (d / "object_masks.pkl").write_bytes(pickle.dumps(pred))
        summary[scene] = {"icp": icp, "blobs": len(cm.blobs), "exported": len(pred) - 2, "counts": cm.counts()}
        print(scene, summary[scene])
    (args.out / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       "scenes": summary}, indent=2))


if __name__ == "__main__":
    main()
