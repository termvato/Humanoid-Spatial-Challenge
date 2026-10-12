"""Suggested viewpoints: where the robot should stand to look at each "go look" region.

A small next-best-view on the 2.5D maps. Each connected group of LOOK cells is a region.
Candidate standpoints are cells known free in the latest map, with clearance for the
robot body, within camera range of the region. A candidate sees a region cell if the
ray from the camera (at `camera_height`) down to the floor at that cell passes above
every obstacle on the way (per-cell obstacle heights; unknown cells may be open, so
they do not block). The best candidate sees the most region cells; ties go to the
nearer one. The robot faces the region centroid. Regions are handled largest first,
and a pose already chosen is reused when it sees >= `reuse` of a later region, so
nearby regions share one stop.

A region no standpoint can see (pose None) is usually the back of a wall: model
depth makes walls a few cm thick, and the hidden back layer counts as "seen before,
not after". `drop_unviewable` removes such regions from the look layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from roomdiff.grid2d import FREE_CELL, LOOK, NONE, OCC_CELL, Grid2D


@dataclass
class Viewpoint:
    region: int
    cells: int  # region size in cells
    target: tuple[float, float]  # region centroid, map metres
    pose: tuple[float, float, float] | None  # x, y, yaw (rad); None if no standpoint sees the region
    seen: float  # fraction of region cells visible from the pose
    members: np.ndarray  # (N, 2) rows, cols of the region's cells


def _centres(grid: Grid2D, rc: np.ndarray) -> np.ndarray:
    """Map metres (x, y) of cell centres; rc is (N, 2) rows, cols."""
    return np.asarray(grid.origin) + (rc[:, ::-1] + 0.5) * grid.resolution


def visible(height: np.ndarray, start: np.ndarray, ends: np.ndarray, camera_height: float) -> np.ndarray:
    """For each end cell (rows, cols), True if the ray from `start` at `camera_height` to the floor at the
    end cell clears every obstacle top in between (`height`: metres per cell, 0 = no obstacle)."""
    out = np.empty(len(ends), bool)
    for k, e in enumerate(ends):
        n = int(np.abs(e - start).max()) + 1
        t = np.linspace(0, 1, n)[1:-1]
        r = np.rint(start[0] + t * (e[0] - start[0])).astype(int)
        c = np.rint(start[1] + t * (e[1] - start[1])).astype(int)
        out[k] = bool(np.all(height[r, c] < camera_height * (1 - t)))
    return out


def suggest(after: Grid2D, change: Grid2D, height: np.ndarray, camera_height: float = 1.0,
            robot_radius: float = 0.25, range_m: tuple[float, float] = (0.5, 3.0), min_cells: int = 4,
            max_candidates: int = 400, reuse: float = 0.9) -> list[Viewpoint]:
    """One viewpoint per LOOK region of >= `min_cells` cells, largest region first."""
    res = after.resolution
    occ = after.data == OCC_CELL
    clear = ndimage.distance_transform_edt(~occ) * res >= robot_radius
    stand = np.argwhere((after.data == FREE_CELL) & clear)
    lab, n = ndimage.label(change.data == LOOK, structure=np.ones((3, 3)))
    regions = sorted((np.argwhere(lab == k) for k in range(1, n + 1)), key=len, reverse=True)
    out, chosen = [], []  # chosen: standpoint cells already used
    for k, region in enumerate(r for r in regions if len(r) >= min_cells):
        centre = region.mean(0)
        target = tuple(float(v) for v in _centres(after, centre[None])[0])
        best, best_key = None, None
        for s in chosen:
            seen = float(visible(height, s, region, camera_height).mean())
            if seen >= reuse and (best_key is None or seen > best_key[0]):
                best, best_key = s, (seen, 0.0)
        if best is None:
            d = np.linalg.norm(stand - centre, axis=1) * res
            cand = stand[(d >= range_m[0]) & (d <= range_m[1])]
            if len(cand) > max_candidates:  # even subsample keeps the search cheap
                cand = cand[np.linspace(0, len(cand) - 1, max_candidates).astype(int)]
            for s in cand:
                seen = float(visible(height, s, region, camera_height).mean())
                key = (seen, -float(np.linalg.norm(s - centre)))
                if best_key is None or key > best_key:
                    best, best_key = s, key
            if best is not None and best_key[0] > 0:
                chosen.append(best)
        if best is None or best_key[0] == 0:
            out.append(Viewpoint(k, len(region), target, None, 0.0, region))
            continue
        x, y = _centres(after, best[None])[0]
        yaw = float(np.arctan2(target[1] - y, target[0] - x))
        out.append(Viewpoint(k, len(region), target, (float(x), float(y), yaw), best_key[0], region))
    return out


def drop_unviewable(change: Grid2D, vps: list[Viewpoint]) -> tuple[Grid2D, list[Viewpoint]]:
    """Change layer whose LOOK cells are only those of viewable regions (too small or unseeable ones
    are dropped), and the viewable viewpoints."""
    data = change.data.copy()
    data[data == LOOK] = NONE
    kept = [v for v in vps if v.pose is not None]
    for v in kept:
        data[v.members[:, 0], v.members[:, 1]] = LOOK
    return Grid2D(data, change.resolution, change.origin), kept
