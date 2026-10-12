"""3D-first predictions for SceneDiff benchmark pairs, in its object_masks.pkl format (S9b).

For each reconstruction outputs/recon/<model>_<scene>.npz: the 3D-first pipeline
(defaults = the final tuned settings: voxel median depth / 80, ICP, SAM, DINOv2
moved matching at 0.5), blobs projected to 2D masks on the evaluator's frames,
written to <out>/pred/<scene>/object_masks.pkl plus a summary. tools/s9_sweep.sh
uploads them and runs the official evaluator.

Usage: uv run --group gpu python tools/bench_predict.py --scenes a,b,c --out outputs/s9/run
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from roomdiff.changemap import collect_views, export
from roomdiff.objects import DinoDescriptor, SamRefiner
from roomdiff.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi3")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--div", type=float, default=80)
    ap.add_argument("--min-voxels", type=int, default=5)
    ap.add_argument("--min-conf", type=float, default=0.1)
    ap.add_argument("--no-icp", action="store_true", help="skip the robust ICP of video 2 onto video 1")
    ap.add_argument("--no-sam", action="store_true", help="coarse projected masks instead of SAM outlines")
    ap.add_argument("--match", type=float, default=0.5, help="moved matching: min DINOv2 cosine (0 = off)")
    ap.add_argument("--match-size", type=float, default=0.4, help="moved matching: min voxel count ratio")
    ap.add_argument("--max-incidence", type=float, default=0.0, help="no free evidence beyond this angle (0 = off)")
    ap.add_argument("--recon-dir", type=Path, default=Path("outputs/recon"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    refine = None if args.no_sam else SamRefiner()
    descriptor = DinoDescriptor() if args.match > 0 else None
    match = {"min_sim": args.match, "min_size_ratio": args.match_size} if args.match > 0 else None
    summary = {}
    for scene in args.scenes.split(","):
        pair = run(args.recon_dir / f"{args.model}_{scene}.npz", div=args.div, min_voxels=args.min_voxels,
                   icp=not args.no_icp, min_conf=args.min_conf,
                   max_incidence=args.max_incidence or None)
        # SAM masks are the slow part: cached per scene and setting (name kept from the S9 sweeps)
        cache = Path("outputs/s9/cache") / (f"{args.model}_{scene}_d{args.div:g}_v{args.min_voxels}_g0"
                                            f"_c{args.min_conf:g}_{'none' if args.no_sam else 'sam'}"
                                            f"{'' if args.no_icp else '_icp'}{f'_i{args.max_incidence:g}' if args.max_incidence else ''}.pkl")
        if cache.exists():
            views = pickle.loads(cache.read_bytes())
        else:
            views = collect_views(pair.cm, pair.frames, refine=refine)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(views))
        pred = export(pair.cm, pair.frames, views, match=match, descriptor=descriptor)
        d = args.out / "pred" / scene
        d.mkdir(parents=True, exist_ok=True)
        (d / "object_masks.pkl").write_bytes(pickle.dumps(pred))
        summary[scene] = {"icp": [round(f, 3) for f in pair.icp], "blobs": len(pair.cm.blobs),
                          "exported": len(pred) - 2, "counts": pair.cm.counts()}
        print(scene, summary[scene])
    (args.out / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       "scenes": summary}, indent=2))


if __name__ == "__main__":
    main()
