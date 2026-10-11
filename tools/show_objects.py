"""One tile per predicted object and video: the frame where its mask is largest, outlined.

Usage: uv run python tools/show_objects.py <pair_dir> <pred_dir> --out file.jpg
pair_dir frames must match prediction indices (resample 1, e.g. a keyframe pair).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from show_pair import decode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_dir", type=Path)
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pred = pickle.loads((args.pred_dir / "object_masks.pkl").read_bytes())
    tiles = []
    for oid, pv in pred.items():
        if oid in ("H", "W"):
            continue
        in_v = [v for v in (1, 2) if pv.get(f"video_{v}")]
        kind = "moved/matched" if len(in_v) == 2 else ("removed" if in_v == [1] else "added")
        for v in in_v:
            k, hit = max(pv[f"video_{v}"].items(), key=lambda kv: decode(kv[1]["mask"]).sum())
            img = cv2.imread(str(sorted((args.pair_dir / f"video{v}_frames").glob("*.jpg"))[k]))
            m = decode(hit["mask"])
            m = cv2.resize(m.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 255, 255), 12)
            img = cv2.resize(img, (300, 400))
            cv2.rectangle(img, (0, 0), (300, 26), (0, 0, 0), -1)
            label = f"P{oid} {kind}  {'before' if v == 1 else 'after'} k{k}"
            cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(img)
    cols = 6
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(args.out)


if __name__ == "__main__":
    main()
