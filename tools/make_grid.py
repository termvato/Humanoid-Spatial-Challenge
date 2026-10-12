"""Robot maps from a reconstructed pair: before / after occupancy + change layer, as ROS map_server files.

recon npz -> pipeline.run (align + change map) -> level frame (floor, up) -> metric scale -> grids.

Usage: uv run python tools/make_grid.py outputs/recon/pi3_pair02.npz --out outputs/grid/pair02
       (--scale M: metres per unit if known, e.g. from ARKit or Pi3X; default: phone held at 1.5 m)
Writes: before.{pgm,yaml}, after.{pgm,yaml}, look_mask.{pgm,yaml}, change.npz, map.png, result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roomdiff.grid2d import BLOCKED, FREED, LOOK, OCC_CELL, render, robot_maps, save_change_mask, save_ros
from roomdiff.level import camera_height, level_frame, scale_from_camera_height
from roomdiff.pipeline import points, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recon", type=Path)
    ap.add_argument("--div", type=float, default=80)
    ap.add_argument("--scale", type=float, default=0.0, help="metres per reconstruction unit (0 = from camera height)")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--band", type=float, nargs=2, default=(0.10, 1.80), help="robot height band, metres")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pair = run(args.recon, div=args.div)
    T = level_frame(points(pair.frames, step=8), pair.frames.T_wc)
    scale = args.scale or scale_from_camera_height(pair.frames.T_wc, T)
    maps = robot_maps(pair.cm, T, scale=scale, res=args.res, band=tuple(args.band))
    save_ros(maps.before, args.out / "before")
    save_ros(maps.after, args.out / "after")
    save_change_mask(maps.change, args.out / "look_mask", LOOK)
    np.savez_compressed(args.out / "change.npz", change=maps.change.data, resolution=args.res,
                        origin=np.array(maps.change.origin))
    cv2.imwrite(str(args.out / "map.png"), render(maps))
    c = maps.change.data
    res = {"scale_m_per_unit": round(scale, 4), "scale_source": "given" if args.scale else "camera height 1.5 m",
           "camera_height_units": round(camera_height(pair.frames.T_wc, T), 4), "cells": list(c.shape),
           "occupied_after": int((maps.after.data == OCC_CELL).sum()), "blocked": int((c == BLOCKED).sum()),
           "freed": int((c == FREED).sum()), "look": int((c == LOOK).sum()), "icp": pair.icp}
    (args.out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
