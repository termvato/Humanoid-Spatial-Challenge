"""Fuse SceneDiff (appearance) detections with the 3D-first map (geometry): head A as a geometric verifier.

For each SceneDiff object, `support` = mean over its masks of the fraction of mask
pixels covered by 3D change masks in the same frame (dilated a little to absorb
misregistration). Modes:
  soft:  score = cost * (floor + support)
  gate:  drop objects with support < min_support, keep the rest unchanged
  add:   soft, plus 3D objects that overlap no SceneDiff object, scored below SceneDiff's range

Usage: uv run python tools/fuse.py --scenes a,b --sd outputs/sd/tune --geo outputs/s9/tune_pi3_d80_icp_sam/pred \
          --cache-tag d80_v5_g0_c0.1_sam_icp --mode soft --out outputs/s9/tune_fuse_soft
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as rle


def dec(m: dict) -> np.ndarray:
    m = dict(m)
    if isinstance(m["counts"], str):
        m["counts"] = m["counts"].encode()
    return rle.decode(m).astype(bool)


def geo_union(views: dict, video: np.ndarray, src: np.ndarray, shape, resample: int = 30, grow: int = 7) -> dict:
    """(video, resample index) -> union of 3D change masks, resized to `shape` and dilated by `grow` px."""
    out: dict = {}
    for per_frame in views.values():
        for i, m in per_frame.items():
            key = (int(video[i]), int(src[i]) // resample)
            r = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            out[key] = np.maximum(out.get(key, 0), r)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return {key: cv2.dilate(m, k) > 0 for key, m in out.items()}


def support(obj: dict, union: dict) -> float:
    fr = []
    for vk, frames in obj.items():
        v = int(vk[-1])
        for k, f in frames.items():
            m = dec(f["mask"])
            u = union.get((v, int(k)))
            fr.append(0.0 if u is None or not m.any() else float((m & u).sum() / m.sum()))
    return float(np.mean(fr)) if fr else 0.0


def overlaps_any(obj: dict, others: list[dict], thr: float = 0.2) -> bool:
    for o in others:
        for vk, frames in obj.items():
            for k, f in frames.items():
                g = o.get(vk, {}).get(k)
                if g is None:
                    continue
                a, b = dec(f["mask"]), dec(g["mask"])
                if a.shape != b.shape:
                    b = cv2.resize(b.astype(np.uint8), a.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                if (a & b).sum() / max((a | b).sum(), 1) > thr:
                    return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--sd", type=Path, required=True)
    ap.add_argument("--geo", type=Path, required=True, help="3D-first pred dir (for --mode add)")
    ap.add_argument("--cache-tag", required=True, help="suffix of outputs/s9/cache/pi3_<scene>_<tag>.pkl")
    ap.add_argument("--model", default="pi3")
    ap.add_argument("--mode", choices=["soft", "gate", "add"], default="soft")
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--min-support", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {}
    for scene in args.scenes.split(","):
        sd = pickle.loads((args.sd / scene / "object_masks.pkl").read_bytes())
        views = pickle.loads(Path(f"outputs/s9/cache/{args.model}_{scene}_{args.cache_tag}.pkl").read_bytes())
        z = np.load(f"outputs/recon/{args.model}_{scene}.npz")
        union = geo_union(views, z["video"], z["src"], (sd["H"], sd["W"]))
        out = {"H": sd["H"], "W": sd["W"]}
        sups = {}
        for oid, obj in sd.items():
            if oid in ("H", "W"):
                continue
            s = support(obj, union)
            sups[str(oid)] = round(s, 3)
            if args.mode == "gate":
                if s >= args.min_support:
                    out[oid] = obj
                continue
            out[oid] = {vk: {k: {**f, "cost": f["cost"] * (args.floor + s)} for k, f in fr.items()}
                        for vk, fr in obj.items()}
        added = 0
        if args.mode == "add":
            geo = pickle.loads((args.geo / scene / "object_masks.pkl").read_bytes())
            sd_objs = [o for k, o in sd.items() if k not in ("H", "W")]
            lo = min([f["cost"] for o in out.values() if isinstance(o, dict)
                      for fr in o.values() for f in fr.values()] or [1.0])
            gmax = max([f["cost"] for k, o in geo.items() if k not in ("H", "W")
                        for fr in o.values() for f in fr.values()] or [1.0])
            for gid, gobj in geo.items():
                if gid in ("H", "W") or overlaps_any(gobj, sd_objs):
                    continue
                out[f"g{gid}"] = {vk: {k: {**f, "cost": 0.9 * lo * f["cost"] / gmax} for k, f in fr.items()}
                                  for vk, fr in gobj.items()}
                added += 1
        d = args.out / "pred" / scene
        d.mkdir(parents=True, exist_ok=True)
        (d / "object_masks.pkl").write_bytes(pickle.dumps(out))
        report[scene] = {"support": sups, "kept": len(out) - 2, "added_from_3d": added}
        print(scene, report[scene]["kept"], "objects, added", added)
    (args.out / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       "scenes": report}, indent=2))


if __name__ == "__main__":
    main()
