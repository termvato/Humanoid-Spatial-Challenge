"""One tile per SceneDiff prediction and video: the frame where its mask is largest (red before, green after).

Works on our own pairs, where prediction frame indices are every `--resample`-th video frame.
Usage: uv run python tools/show_sd.py outputs/sd/own/pair02/pair02/object_masks.pkl \
          data/scenediff_own/pair02/original_video1.mp4 data/scenediff_own/pair02/original_video2.mp4 \
          --resample 90 --out outputs/sd/own/pair02/objects.jpg
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as rle


def decode(m: dict) -> np.ndarray:
    m = dict(m)
    if isinstance(m["counts"], str):
        m["counts"] = m["counts"].encode()
    return rle.decode(m).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", type=Path)
    ap.add_argument("video1", type=Path)
    ap.add_argument("video2", type=Path)
    ap.add_argument("--resample", type=int, default=90)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = pickle.loads(args.pred.read_bytes())
    H, W = d["H"], d["W"]
    caps = {1: cv2.VideoCapture(str(args.video1)), 2: cv2.VideoCapture(str(args.video2))}

    def frame(v: int, i: int) -> np.ndarray:
        caps[v].set(cv2.CAP_PROP_POS_FRAMES, i * args.resample)
        return cv2.resize(caps[v].read()[1], (W, H))

    objs = sorted(((k, o) for k, o in d.items() if k not in ("H", "W")),
                  key=lambda ko: -max(f["cost"] for fr in ko[1].values() for f in fr.values()))
    tiles = []
    for k, o in objs:
        kind = "moved" if len(o) == 2 else ("removed" if "video_1" in o else "added")
        for vk in sorted(o):
            v = int(vk[-1])
            fi, f = max(o[vk].items(), key=lambda kv: decode(kv[1]["mask"]).sum())
            im = frame(v, int(fi))
            over = im.copy()
            over[decode(f["mask"])] = (60, 60, 230) if v == 1 else (60, 200, 60)
            im = cv2.addWeighted(im, 0.6, over, 0.4, 0)
            im = cv2.resize(im, (320, round(320 * H / W)))
            cv2.rectangle(im, (0, 0), (im.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(im, f"P{k} {kind} {'before' if v == 1 else 'after'} c{f['cost']:.2f}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(im)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
    cv2.imwrite(str(args.out), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    print(len(objs), "objects")


if __name__ == "__main__":
    main()
