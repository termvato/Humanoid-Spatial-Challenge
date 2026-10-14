"""Everything for one pair: 3D change list, pictures, robot maps, viewpoints and point clouds.

Input: a reconstruction npz (plain video via pi3, or simulated RGB-D with metric=True) or a
Stray LiDAR pair folder (before/, after/; metric, no colour so no SAM or moved matching).
Writes into `out`:
  changes.json     added / removed / moved, map-frame boxes, moved distance
  changes.jpg      one picture per change (red where it was, green where it is)
  crops/           per change and blob, up to 3 outlined crops and their other-video views (label.py)
  before|after.{pgm,yaml}, look_mask.{pgm,yaml}   ROS map_server maps
  viewpoints.json  where to stand to look at each unobserved region (x, y, yaw)
  map.png          after map with changes and viewpoints
  points.npz       coloured point cloud of each visit in the level frame, for the 3D viewer
  result.json      settings, counts, the world -> map transform, 3D overlap of the visits (S23)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from roomdiff.align import refine_between_videos
from roomdiff.changemap import collect_views, drop_small, moved_pairs
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
from roomdiff.label import best_views, crop, other_view
from roomdiff.pipeline import Pair, level, points, run, run_lidar
from roomdiff.report import Change, changes, to_json
from roomdiff.roomcheck import WARN_FIT_BELOW
from roomdiff.scene3d import clouds
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
            f = best_views(views[bid], k=1, rgb=rgb)[0]
            m = views[bid][f]
            where = "was" if kinds[bid] == "removed" else "now"
            text = f"#{i} {c.kind}" + (f" {c.distance:.2f} m" if c.distance else "") + f" ({where})"
            tiles.append(tile(rgb[f], m, RED if kinds[bid] == "removed" else GREEN, text))
    return tiles


def write_crops(out: Path, chs: list[Change], views: dict, pair, cm) -> None:
    """Per change and blob, up to 3 outlined crops <change>_<blob>_<k>.jpg from its own video, and for each
    the same 3D spot from the closest camera in the other video, <change>_<blob>_<k>x.jpg (S20)."""
    F = pair.frames
    out.mkdir(exist_ok=True)

    def save(name, f, m):
        cv2.imwrite(str(out / name), cv2.cvtColor(crop(F.rgb[f], m), cv2.COLOR_RGB2BGR))
    for i, c in enumerate(chs):
        for bid in c.blobs:
            for k, f in enumerate(best_views(views[bid], rgb=F.rgb)):
                save(f"{i}_{bid}_{k}.jpg", f, views[bid][f])
                o = other_view(F, cm, views, bid, f=f)
                if o:
                    save(f"{i}_{bid}_{k}x.jpg", o[1], o[2])


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


def overlap_3d(pair: Pair) -> float:
    """S23: share of each visit's points within 2 voxels of the other after robust ICP, the lower of both
    directions. Same room: >= 0.6 on 21 of 22 pairs; two different rooms: <= 0.22."""
    f = pair.frames
    a, b = points(f.of_video(2)), points(f.of_video(1))
    return float(min(max(pair.icp), refine_between_videos(b, a, pair.voxel)[2]))


def analyse(input: Path, out: Path, metric: bool = False, scale_arg: float = 0.0, div: float = 80,
            no_icp: bool = False, no_sam: bool = False, max_incidence: float = 0.0, plain_carving: bool = False,
            min_extent: float = 0.0, match: float = 0.5, cell: float = 0.05, band: tuple[float, float] = (0.10, 1.80),
            camera_height: float = 1.0, progress: Callable[[str], None] = lambda s: None) -> dict:
    """Run the pipeline on `input` (recon npz or Stray pair folder) and write everything into `out`.
    Returns the result.json content. Defaults are the adopted settings (D16: conservative carving)."""
    out.mkdir(parents=True, exist_ok=True)

    lidar = input.is_dir()
    progress("3D change map")
    pair = run_lidar(input) if lidar else run(input, div=div, icp=not no_icp, max_incidence=max_incidence or None,
                                              slant=not plain_carving, footprint=not plain_carving)
    progress("level")
    T, scale = level(pair, metric=lidar or metric, scale=scale_arg)
    F, cm = pair.frames, pair.cm
    colour = F.rgb is not None
    refine = None
    if colour and not no_sam:
        from roomdiff.objects import SamRefiner

        refine = SamRefiner()
    progress("object outlines (SAM)")
    views = collect_views(cm, F, resample=1, refine=refine)  # every frame we have
    if min_extent:
        views = drop_small(views, cm, min_extent)
    partner = {}
    if colour and match > 0:
        from roomdiff.objects import DinoDescriptor

        partner = moved_pairs(cm, F, views, DinoDescriptor(), min_sim=match, min_size_ratio=0.4)
    chs = changes(cm, views, partner, T, scale)

    maps = robot_maps(cm, T, scale=scale, res=cell, band=tuple(band))
    vps_all = suggest(maps.after, maps.change, maps.height, camera_height=camera_height)
    change_layer, vps = drop_unviewable(maps.change, vps_all)  # back-of-wall artefacts are not goals
    maps = RobotMaps(maps.before, maps.after, change_layer, maps.height)

    (out / "changes.json").write_text(json.dumps(to_json(chs), indent=2))
    if colour and chs:
        tiles = change_tiles(chs, views, F.rgb, {b.id: b.kind for b in cm.blobs})
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(out / "changes.jpg"), np.vstack([np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]))
        write_crops(out / "crops", chs, views, pair, cm)
    np.savez_compressed(out / "points.npz", **clouds(F, T, scale))
    save_ros(maps.before, out / "before")
    save_ros(maps.after, out / "after")
    save_change_mask(maps.change, out / "look_mask", LOOK)
    (out / "viewpoints.json").write_text(json.dumps(
        [{"region": v.region, "cells": v.cells, "target": [round(t, 3) for t in v.target],
          "pose": [round(t, 3) for t in v.pose], "seen": round(v.seen, 3)} for v in vps], indent=2))
    cv2.imwrite(str(out / "map.png"), draw_map(maps, chs, vps))
    c = maps.change.data
    res = {"input": str(input), "scale_m_per_unit": round(scale, 4),
           "scale_source": "metric" if (lidar or metric) else ("given" if scale_arg else "camera height 1.5 m"),
           "world_to_level": np.round(T, 6).tolist(), "voxel": pair.voxel, "alignment_fitness": pair.icp,
           "changes": {k: sum(ch.kind == k for ch in chs) for k in ("added", "removed", "moved")},
           "map_cells": list(c.shape), "occupied_after": int((maps.after.data == OCC_CELL).sum()),
           "blocked": int((c == BLOCKED).sum()), "freed": int((c == FREED).sum()), "look": int((c == LOOK).sum()),
           "viewpoints": len(vps), "not_viewable_dropped": len(vps_all) - len(vps)}
    if not lidar:
        res["overlap_3d"] = round(overlap_3d(pair), 3)
        if res["overlap_3d"] < WARN_FIT_BELOW:
            res["warning"] = (f"The two videos barely overlap in 3D ({res['overlap_3d']:.2f} < {WARN_FIT_BELOW}): "
                              "they may show different rooms, or the reconstruction failed. Treat the changes as unreliable.")
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res

