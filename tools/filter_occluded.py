"""Drop SceneDiff predictions that sit mostly on a lens occluder (finger).

An object is dropped when, pooled over all its frames, >= `--max-inside` of its
mask pixels fall inside that frame's finger mask. Those pixels were never
observed, so a "change" there is an artifact, not evidence.

Usage: uv run python tools/filter_occluded.py <pred_dir> <kf_dir> [--max-inside 0.5]
  pred_dir: SceneDiff output with object_masks.pkl
  kf_dir:   keyframe pair with finger_masks.npz (tools/make_keyframe_pair.py)
Writes pred_dir/filtered/object_masks.pkl and prints dropped ids.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from show_pair import decode

from roomdiff.keyframes import inside_fraction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("kf_dir", type=Path)
    ap.add_argument("--max-inside", type=float, default=0.5)
    args = ap.parse_args()

    pred = pickle.loads((args.pred_dir / "object_masks.pkl").read_bytes())
    fingers = np.load(args.kf_dir / "finger_masks.npz")
    kept, dropped = {}, []
    for oid, per_video in pred.items():
        if oid in ("H", "W"):
            kept[oid] = per_video
            continue
        inside = total = 0.0
        for v in (1, 2):
            for k, hit in per_video.get(f"video_{v}", {}).items():
                m = decode(hit["mask"])
                key = f"video{v}_{k}"
                if key in fingers:
                    inside += inside_fraction(m, fingers[key]) * m.sum()
                total += m.sum()
        if total and inside / total >= args.max_inside:
            dropped.append((oid, round(inside / total, 2)))
        else:
            kept[oid] = per_video
    out = args.pred_dir / "filtered"
    out.mkdir(exist_ok=True)
    (out / "object_masks.pkl").write_bytes(pickle.dumps(kept))
    print(f"kept {len([k for k in kept if k not in ('H', 'W')])}, dropped {dropped}")


if __name__ == "__main__":
    main()
