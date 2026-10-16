"""Everything for one pair from a reconstruction npz or a Stray LiDAR folder (roomdiff.analyse).

Usage: uv run --group gpu python tools/analyse.py outputs/recon/pi3_pair02.npz --out outputs/pair02
       uv run --group gpu python tools/analyse.py outputs/sim/room01/visits.npz --metric --no-icp --out outputs/sim/room01/result
       uv run python tools/analyse.py data/raw/pair02 --out outputs/pair02_lidar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roomdiff.analyse import analyse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="recon npz or Stray pair folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metric", action="store_true", help="npz depth is already metric (simulation)")
    ap.add_argument("--scale", type=float, default=0.0, help="metres per unit if known (default: phone at 1.5 m)")
    ap.add_argument("--div", type=float, default=80, help="voxel = median depth / div")
    ap.add_argument("--no-icp", action="store_true", help="poses are exact (simulation): skip ICP between visits")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--max-incidence", type=float, default=0.0, help="no free evidence beyond this angle (0 = off)")
    ap.add_argument("--naive", action="store_true", help="S26 ablation: point-distance diff instead of carving")
    ap.add_argument("--plain-carving", action="store_true",
                    help="free if the voxel centre is seen past (benchmark setting); default is conservative "
                         "carving: whole voxel seen past, margin along the surface normal (D16)")
    ap.add_argument("--min-extent", type=float, default=0.0, help="drop blobs whose largest side is under this many voxels")
    ap.add_argument("--match", type=float, default=0.5, help="moved matching: min DINOv2 cosine (0 = off)")
    ap.add_argument("--res", type=float, default=0.05, help="map cell, metres")
    ap.add_argument("--band", type=float, nargs=2, default=(0.10, 1.80), help="robot height band, metres")
    ap.add_argument("--camera-height", type=float, default=1.0, help="robot camera height for viewpoints, metres")
    args = ap.parse_args()
    res = analyse(args.input, args.out, metric=args.metric, scale_arg=args.scale, div=args.div, no_icp=args.no_icp,
                  no_sam=args.no_sam, max_incidence=args.max_incidence, plain_carving=args.plain_carving, naive=args.naive,
                  min_extent=args.min_extent, match=args.match, cell=args.res, band=tuple(args.band),
                  camera_height=args.camera_height, progress=print)
    print(json.dumps({k: v for k, v in res.items() if k != "world_to_level"}))


if __name__ == "__main__":
    main()
