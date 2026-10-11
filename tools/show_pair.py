"""Contact sheet for one SceneDiff pair: frames, ground truth, and predictions.

Rows: before (video 1) and after (video 2). Columns: sampled frames.
Ground-truth changed objects are filled (colour = change type); predictions are
outlined in white with their id. Writes a PNG.

Usage: uv run python tools/show_pair.py <pair_dir> [--pred <pred_dir>] [--out file.png]
  pair_dir: benchmark pair (segments.pkl, video{1,2}_frames/)
  pred_dir: SceneDiff output for the pair (object_masks.pkl)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as rle

COLOURS = {"removed": (60, 60, 230), "added": (60, 200, 60), "moved": (230, 160, 30)}  # BGR


def change_type(obj: dict) -> str:
    if obj["in_video1"] and obj["in_video2"]:
        return "moved"
    return "removed" if obj["in_video1"] else "added"


def decode(m: dict) -> np.ndarray:
    m = dict(m)
    if isinstance(m["counts"], str):
        m["counts"] = m["counts"].encode()
    return rle.decode(m).astype(bool)


def overlay(img: np.ndarray, masks: list[tuple[np.ndarray, tuple, str]], fill: bool) -> np.ndarray:
    out = img.copy()
    for mask, colour, text in masks:
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        if fill:
            out[mask] = (0.45 * out[mask] + 0.55 * np.array(colour)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour if fill else (0, 255, 255), 2 if fill else 5)
        if text and contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(out, text, (x, max(y - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_dir", type=Path)
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/pair.png"))
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--resample", type=int, default=30, help="SceneDiff dataset.resample_rate of the run")
    args = ap.parse_args()

    gt_path = args.pair_dir / "segments.pkl"  # absent for our own captures
    gt = pickle.loads(gt_path.read_bytes()) if gt_path.exists() else {"objects": []}
    objs = gt["objects"]
    pred = pickle.loads((args.pred / "object_masks.pkl").read_bytes()) if args.pred else None

    rows = []
    for v in (1, 2):
        frames = sorted((args.pair_dir / f"video{v}_frames").glob("*.jpg"))
        pred_frames = sorted({k for o, pv in (pred or {}).items() if o not in ("H", "W")
                              for k in pv.get(f"video_{v}", {})})
        if pred_frames:  # show where SceneDiff fired, spread over the video
            idx = np.linspace(0, len(pred_frames) - 1, min(args.cols, len(pred_frames))).astype(int)
            picks = [pred_frames[i] * args.resample for i in idx]
        else:
            picks = (np.linspace(0, len(frames) - 1, args.cols).astype(int) // args.resample) * args.resample
        tiles = []
        for f in picks:
            img = cv2.imread(str(frames[f]))
            gt_masks = []
            for i, o in enumerate(objs):
                seg = gt.get(f"video{v}_objects", {}).get(o["original_obj_idx"], {}).get(str(f))
                if seg is not None:
                    gt_masks.append((decode(seg), COLOURS[change_type(o)], ""))
            img = overlay(img, gt_masks, fill=True)
            if pred is not None:
                pm = []
                for oid, per_video in pred.items():
                    if oid in ("H", "W"):
                        continue
                    hit = per_video.get(f"video_{v}", {}).get(f // args.resample)
                    if hit is not None:
                        pm.append((decode(hit["mask"]), (0, 255, 255), f"P{oid}"))
                img = overlay(img, pm, fill=False)
            img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
            cv2.putText(img, f"video {v}  frame {f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(img)
        while len(tiles) < args.cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(tiles))

    legend = np.full((40, rows[0].shape[1], 3), 30, np.uint8)
    x = 10
    for name, c in COLOURS.items():
        cv2.rectangle(legend, (x, 12), (x + 18, 30), c, -1)
        cv2.putText(legend, f"GT {name}", (x + 24, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        x += 150
    cv2.putText(legend, "yellow outline = SceneDiff prediction (P#)", (x, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack([legend, *rows]))
    print(args.out)


if __name__ == "__main__":
    main()
