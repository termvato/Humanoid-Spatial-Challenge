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

from roomdiff.changemap import collect_views, drop_small, export
from roomdiff.label import COMPARE, changed, crop, other_view
from roomdiff.objects import DinoDescriptor, SamRefiner
from roomdiff.pipeline import run


def vlm_verdicts(pair, views: dict, cache: Path) -> dict[int, bool | None]:
    """Blob id -> VLM 'changed' (True), 'same' (False) or unclear (None), from its best view next to the
    closest view of the same spot in the other video (first visit first). Cached per scene and setting."""
    import cv2
    import modal

    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    F, cm = pair.frames, pair.cm
    kinds = {b.id: b.kind for b in cm.blobs}
    ids, items = [], []
    for bid in views:
        o = other_view(F, cm, views, bid)
        if o is None:
            continue
        f, g, m = o
        own, oth = crop(F.rgb[f], views[bid][f]), crop(F.rgb[g], m)
        pair_imgs = [own, oth] if kinds[bid] == "removed" else [oth, own]
        ids.append(bid)
        items.append([cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1].tobytes() for im in pair_imgs])
    answers = modal.Cls.from_name("roomdiff-vlm", "VLM")().ask.remote(items, COMPARE) if items else []
    out = {b: changed(a) for b, a in zip(ids, answers, strict=True)}
    cache.write_text(json.dumps(out))
    return out


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
    ap.add_argument("--slant", action="store_true", help="free margin measured along the surface normal")
    ap.add_argument("--footprint", action="store_true", help="free only if the whole voxel is seen past")
    ap.add_argument("--min-extent", type=float, default=0.0, help="drop blobs whose largest side is under this many voxels")
    ap.add_argument("--vlm-check", action="store_true", help="drop blobs a VLM judges unchanged (S16)")
    ap.add_argument("--vlm-rescore", type=float, default=0.0,
                    help="multiply the score of blobs a VLM judges unchanged by this, instead of dropping them (S17)")
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
                   max_incidence=args.max_incidence or None, slant=args.slant,
                   footprint=args.footprint)
        # SAM masks are the slow part: cached per scene and setting (name kept from the S9 sweeps)
        cache = Path("outputs/s9/cache") / (f"{args.model}_{scene}_d{args.div:g}_v{args.min_voxels}_g0"
                                            f"_c{args.min_conf:g}_{'none' if args.no_sam else 'sam'}"
                                            f"{'' if args.no_icp else '_icp'}{f'_i{args.max_incidence:g}' if args.max_incidence else ''}"
                                            f"{'_slant' if args.slant else ''}{'_fp' if args.footprint else ''}.pkl")
        if cache.exists():
            views = pickle.loads(cache.read_bytes())
        else:
            views = collect_views(pair.cm, pair.frames, refine=refine)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(views))
        if args.min_extent:
            views = drop_small(views, pair.cm, args.min_extent)
        if args.vlm_check or args.vlm_rescore:
            verdict = vlm_verdicts(pair, views, cache.with_suffix(".vlm.json"))
            if args.vlm_rescore:
                for b in pair.cm.blobs:
                    if verdict.get(b.id) is False:
                        b.score *= args.vlm_rescore
            else:
                views = {k: v for k, v in views.items() if verdict.get(k) is not False}
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
