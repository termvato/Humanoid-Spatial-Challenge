"""Everything for one pair: 3D change list, pictures, robot maps and viewpoints.

Input: a reconstruction npz (plain video via pi3, or simulated RGB-D with --metric) or a
Stray LiDAR pair folder (before/, after/; metric, no colour so no SAM or moved matching).

Usage: uv run --group gpu python tools/analyse.py outputs/recon/pi3_pair02.npz --out outputs/pair02
       uv run --group gpu python tools/analyse.py outputs/sim/room01/visits.npz --metric --no-icp --out outputs/sim/room01/result
       uv run python tools/analyse.py data/raw/pair02 --out outputs/pair02_lidar
Writes:
  changes.json     added / removed / moved, map-frame boxes, moved distance
  changes.jpg      one picture per change (red where it was, green where it is)
  crops/           per change and blob, up to 3 outlined crops for the labeller (tools/label.py)
  before|after.{pgm,yaml}, look_mask.{pgm,yaml}   ROS map_server maps
  viewpoints.json  where to stand to look at each unobserved region (x, y, yaw)
  map.png          after map with changes and viewpoints
  result.json      settings, counts, and the world -> map transform
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roomdiff.changemap import blob_mask, collect_views, drop_small, moved_pairs
from roomdiff.grid2d import (
    BLOCKED,
    FREED,
    LOOK,
    OCC_CELL,
    RobotMaps,
    render,
    robot_maps,
    save_change_mask,
    save_ros,
)
from roomdiff.label import best_views, closest_poses, crop
from roomdiff.pipeline import level, run, run_lidar
from roomdiff.report import Change, changes, to_json
from roomdiff.viewpoints import Viewpoint, drop_unviewable, suggest

PX = 4  # map.png pixels per cell
RED, GREEN, BLUE = (60, 60, 230), (60, 200, 60), (200, 80, 0)


def tile(rgb: np.ndarray, mask: np.ndarray, colour, text: str) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    over = img.copy()
    over[mask] = colour
    img = cv2.addWeighted(img, 0.65, over, 0.35, 0)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, colour, 3)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def change_tiles(chs: list[Change], views: dict, rgb: np.ndarray, kinds: dict[int, str]) -> list[np.ndarray]:
    tiles = []
    for i, c in enumerate(chs):
        for bid in c.blobs:
            f, m = max(views[bid].items(), key=lambda kv: kv[1].sum())
            where = "was" if kinds[bid] == "removed" else "now"
            text = f"#{i} {c.kind}" + (f" {c.distance:.2f} m" if c.distance else "") + f" ({where})"
            tiles.append(tile(rgb[f], m, RED if kinds[bid] == "removed" else GREEN, text))
    return tiles


def write_crops(out: Path, chs: list[Change], views: dict, pair, cm, min_px: int = 50) -> None:
    """Per change and blob, up to 3 outlined crops <change>_<blob>_<k>.jpg from its own video, and for
    the best one the same 3D spot from the closest camera in the other video, <change>_<blob>_0x.jpg."""
    F = pair.frames
    out.mkdir(exist_ok=True)
    blobs = {b.id: b for b in cm.blobs}

    def save(name, f, m):
        cv2.imwrite(str(out / name), cv2.cvtColor(crop(F.rgb[f], m), cv2.COLOR_RGB2BGR))
    for i, c in enumerate(chs):
        for bid in c.blobs:
            picks = best_views(views[bid])
            for k, f in enumerate(picks):
                save(f"{i}_{bid}_{k}.jpg", f, views[bid][f])
            other = np.flatnonzero(F.video != F.video[picks[0]])
            centres = cm.grid.centres_of(blobs[bid].idx)
            for g in closest_poses(F.T_wc, picks[0], other, F.median_depth())[:10]:
                m = blob_mask(F, g, centres, cm.grid.size, cm.tau)
                if m.sum() >= min_px:
                    save(f"{i}_{bid}_0x.jpg", g, m)
                    break


def draw_map(maps: RobotMaps, chs: list[Change], vps: list[Viewpoint]) -> np.ndarray:
    img = np.ascontiguousarray(render(maps, PX))
    g = maps.after

    def px(x, y):
        return int((x - g.origin[0]) / g.resolution * PX), int((g.data.shape[0] - (y - g.origin[1]) / g.resolution) * PX)

    for i, c in enumerate(chs):
        if c.kind == "moved":
            cv2.arrowedLine(img, px(*c.before.centre), px(*c.after.centre), (120, 0, 160), 2, tipLength=0.08)
        b = c.after or c.before
        cv2.putText(img, f"#{i}", px(*b.centre), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    for v in vps:
        x, y, yaw = v.pose
        cv2.circle(img, px(x, y), 2 * PX, BLUE, -1)
        cv2.arrowedLine(img, px(x, y), px(x + 0.4 * np.cos(yaw), y + 0.4 * np.sin(yaw)), BLUE, 2, tipLength=0.4)
    return img


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
    ap.add_argument("--slant", action="store_true", help="free margin measured along the surface normal")
    ap.add_argument("--footprint", action="store_true", help="free only if the whole voxel is seen past")
    ap.add_argument("--min-extent", type=float, default=0.0, help="drop blobs whose largest side is under this many voxels")
    ap.add_argument("--match", type=float, default=0.5, help="moved matching: min DINOv2 cosine (0 = off)")
    ap.add_argument("--res", type=float, default=0.05, help="map cell, metres")
    ap.add_argument("--band", type=float, nargs=2, default=(0.10, 1.80), help="robot height band, metres")
    ap.add_argument("--camera-height", type=float, default=1.0, help="robot camera height for viewpoints, metres")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lidar = args.input.is_dir()
    pair = run_lidar(args.input) if lidar else run(args.input, div=args.div, icp=not args.no_icp,
                                                    max_incidence=args.max_incidence or None, slant=args.slant,
                                                    footprint=args.footprint)
    T, scale = level(pair, metric=lidar or args.metric, scale=args.scale)
    F, cm = pair.frames, pair.cm
    colour = F.rgb is not None
    refine = None
    if colour and not args.no_sam:
        from roomdiff.objects import SamRefiner

        refine = SamRefiner()
    views = collect_views(cm, F, resample=1, refine=refine)  # every frame we have
    if args.min_extent:
        views = drop_small(views, cm, args.min_extent)
    partner = {}
    if colour and args.match > 0:
        from roomdiff.objects import DinoDescriptor

        partner = moved_pairs(cm, F, views, DinoDescriptor(), min_sim=args.match, min_size_ratio=0.4)
    chs = changes(cm, views, partner, T, scale)

    maps = robot_maps(cm, T, scale=scale, res=args.res, band=tuple(args.band))
    vps_all = suggest(maps.after, maps.change, maps.height, camera_height=args.camera_height)
    change_layer, vps = drop_unviewable(maps.change, vps_all)  # back-of-wall artefacts are not goals
    maps = RobotMaps(maps.before, maps.after, change_layer, maps.height)

    (args.out / "changes.json").write_text(json.dumps(to_json(chs), indent=2))
    if colour and chs:
        tiles = change_tiles(chs, views, F.rgb, {b.id: b.kind for b in cm.blobs})
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(args.out / "changes.jpg"), np.vstack([np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]))
        write_crops(args.out / "crops", chs, views, pair, cm)
    save_ros(maps.before, args.out / "before")
    save_ros(maps.after, args.out / "after")
    save_change_mask(maps.change, args.out / "look_mask", LOOK)
    (args.out / "viewpoints.json").write_text(json.dumps(
        [{"region": v.region, "cells": v.cells, "target": [round(t, 3) for t in v.target],
          "pose": [round(t, 3) for t in v.pose], "seen": round(v.seen, 3)} for v in vps], indent=2))
    cv2.imwrite(str(args.out / "map.png"), draw_map(maps, chs, vps))
    c = maps.change.data
    res = {"input": str(args.input), "scale_m_per_unit": round(scale, 4),
           "scale_source": "metric" if (lidar or args.metric) else ("given" if args.scale else "camera height 1.5 m"),
           "world_to_level": np.round(T, 6).tolist(), "voxel": pair.voxel, "alignment_fitness": pair.icp,
           "changes": {k: sum(ch.kind == k for ch in chs) for k in ("added", "removed", "moved")},
           "map_cells": list(c.shape), "occupied_after": int((maps.after.data == OCC_CELL).sum()),
           "blocked": int((c == BLOCKED).sum()), "freed": int((c == FREED).sum()), "look": int((c == LOOK).sum()),
           "viewpoints": len(vps), "not_viewable_dropped": len(vps_all) - len(vps)}
    (args.out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "world_to_level"}))


if __name__ == "__main__":
    main()
