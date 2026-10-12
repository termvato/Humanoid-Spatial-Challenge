"""2D occupancy grids for a robot from the 3D change map, in the ROS map_server format.

Each visit becomes a standard trinary occupancy grid (-1 unknown, 0 free, 100 occupied)
over the height band a robot body occupies. A separate change layer marks cells that
became blocked, became free, or held something in visit 1 that visit 2 never looked at
again (the robot should go and look). Files are map_server PGM + YAML, so Nav2 loads
them as a map, or as a costmap filter mask, without any glue code.

Axes: the level frame has +y up and the floor at y = 0 (see `level.py`). Map axes are
x_map = x, y_map = -z, which keeps a right-handed frame with z_map up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from roomdiff.changemap import ChangeMap
from roomdiff.visibility import LABELS, OCC, UNKNOWN, components

UNKNOWN_CELL, FREE_CELL, OCC_CELL = -1, 0, 100
NONE, BLOCKED, FREED, LOOK = 0, 1, 2, 3  # change layer values


@dataclass
class Grid2D:
    data: np.ndarray  # (rows, cols) int8, row 0 at y_map = origin[1] (bottom of the map)
    resolution: float  # metres per cell
    origin: tuple[float, float]  # map position of cell (0, 0)'s lower-left corner


@dataclass
class RobotMaps:
    before: Grid2D
    after: Grid2D
    change: Grid2D  # NONE / BLOCKED / FREED / LOOK


def _map_xy(cm: ChangeMap, flat: np.ndarray, T_level: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Map-plane metres (x, y) and height above the floor of voxels `flat`."""
    p = cm.grid.centres_of(flat) @ T_level[:3, :3].T + T_level[:3, 3]
    p = p * scale
    return np.stack([p[:, 0], -p[:, 2]], 1), p[:, 1]


def robot_maps(cm: ChangeMap, T_level: np.ndarray, scale: float = 1.0, res: float = 0.05,
               band: tuple[float, float] = (0.10, 1.80), min_look_voxels: int = 15) -> RobotMaps:
    """Before / after occupancy and a change layer over the height band `band` (metres above the floor).

    Occupied: any occupied voxel in the band. Free: seen (floor or air in the band)
    and not occupied. Unknown: never seen. Change layer from the 3D change blobs
    (they already passed the dilation and size checks), plus LOOK for voxels occupied
    before and unknown after (components of >= `min_look_voxels`), kept only where
    visit 2 saw nothing in that cell's column.
    """
    known = np.flatnonzero((cm.before != UNKNOWN) | (cm.after != UNKNOWN))
    xy, h = _map_xy(cm, known, T_level, scale)
    lo = xy.min(0) - res
    shape = tuple((np.ceil((xy.max(0) + res - lo) / res)).astype(int)[::-1])  # (rows, cols)

    def cells(xy_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.floor((xy_ - lo) / res).astype(int)
        return c[:, 1], c[:, 0]

    in_band = (h >= band[0]) & (h <= band[1])
    seen_band = (h >= -band[0]) & (h <= band[1])

    def visit(state: np.ndarray) -> Grid2D:
        s = state[known]
        g = np.full(shape, UNKNOWN_CELL, np.int8)
        g[cells(xy[seen_band & (s != UNKNOWN)])] = FREE_CELL
        g[cells(xy[in_band & (s == OCC)])] = OCC_CELL
        return Grid2D(g, res, (float(lo[0]), float(lo[1])))

    def mark(layer: np.ndarray, flat: np.ndarray, value: int) -> None:
        if len(flat) == 0:
            return
        p, hh = _map_xy(cm, flat, T_level, scale)
        r, c = cells(p[(hh >= band[0]) & (hh <= band[1])])
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        layer[r[ok], c[ok]] = value

    change = np.zeros(shape, np.int8)
    look = (cm.labels == LABELS["unobserved"]) & (cm.before == OCC)
    for comp in components(look, cm.grid.shape, min_voxels=min_look_voxels):
        mark(change, comp, LOOK)
    for b in cm.blobs:
        mark(change, b.idx, BLOCKED if b.kind == "added" else FREED)
    after = visit(cm.after)
    change[(change == LOOK) & (after.data != UNKNOWN_CELL)] = NONE  # the column was seen again: nothing to look at
    return RobotMaps(visit(cm.before), after, Grid2D(change, res, (float(lo[0]), float(lo[1]))))


def save_ros(grid: Grid2D, stem: str | Path) -> Path:
    """Write map_server files <stem>.pgm + <stem>.yaml (trinary: free 254, occupied 0, unknown 205)."""
    stem = Path(stem)
    img = np.full(grid.data.shape, 205, np.uint8)
    img[grid.data == FREE_CELL] = 254
    img[grid.data == OCC_CELL] = 0
    cv2.imwrite(str(stem.with_suffix(".pgm")), np.flipud(img))  # image row 0 is the top of the map
    stem.with_suffix(".yaml").write_text(
        f"image: {stem.with_suffix('.pgm').name}\nmode: trinary\nresolution: {grid.resolution}\n"
        f"origin: [{grid.origin[0]:.4f}, {grid.origin[1]:.4f}, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return stem.with_suffix(".yaml")


def save_change_mask(change: Grid2D, stem: str | Path, value: int) -> Path:
    """One change class as a map_server mask (occupied where the class is): e.g. LOOK cells for a goal layer."""
    return save_ros(Grid2D(np.where(change.data == value, OCC_CELL, FREE_CELL).astype(np.int8),
                           change.resolution, change.origin), stem)


def render(maps: RobotMaps, scale_px: int = 4) -> np.ndarray:
    """BGR picture: after map in grey levels, change cells coloured (red blocked, green freed, orange look)."""
    g = maps.after.data
    img = np.full((*g.shape, 3), 160, np.uint8)
    img[g == FREE_CELL] = 255
    img[g == OCC_CELL] = 40
    c = maps.change.data
    img[c == BLOCKED] = (40, 40, 220)
    img[c == FREED] = (60, 180, 60)
    img[c == LOOK] = (0, 150, 255)
    img = np.flipud(img)
    return cv2.resize(img, (img.shape[1] * scale_px, img.shape[0] * scale_px), interpolation=cv2.INTER_NEAREST)
