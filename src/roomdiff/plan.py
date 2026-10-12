"""Paths on the robot map: 8-connected A* through known free cells, kept `radius` clear of obstacles.

Unknown cells are not entered (the robot only drives where a visit saw free floor).
Kept minimal on purpose; a real robot would hand the map to Nav2 instead.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

from roomdiff.grid2d import FREE_CELL, OCC_CELL, Grid2D


def drivable(grid: Grid2D, radius: float) -> np.ndarray:
    """Free cells at least `radius` metres from any occupied cell."""
    clear = ndimage.distance_transform_edt(grid.data != OCC_CELL) * grid.resolution
    return (grid.data == FREE_CELL) & (clear >= radius)


def to_cell(grid: Grid2D, xy: tuple[float, float]) -> tuple[int, int]:
    return int((xy[1] - grid.origin[1]) / grid.resolution), int((xy[0] - grid.origin[0]) / grid.resolution)


def to_xy(grid: Grid2D, rc: tuple[int, int]) -> tuple[float, float]:
    return grid.origin[0] + (rc[1] + 0.5) * grid.resolution, grid.origin[1] + (rc[0] + 0.5) * grid.resolution


def nearest_drivable(ok: np.ndarray, rc: tuple[int, int]) -> tuple[int, int]:
    cells = np.argwhere(ok)
    return tuple(int(v) for v in cells[np.argmin(np.abs(cells - rc).sum(1))])


def astar(ok: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Cell path from start to goal over `ok` cells (both snapped to the nearest ok cell), or None."""
    start, goal = nearest_drivable(ok, start), nearest_drivable(ok, goal)
    steps = [(dr, dc, float(np.hypot(dr, dc))) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    g = {start: 0.0}
    prev: dict = {}
    todo = [(0.0, start)]
    while todo:
        _, cur = heapq.heappop(todo)
        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]
        for dr, dc, w in steps:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < ok.shape[0] and 0 <= nxt[1] < ok.shape[1]) or not ok[nxt]:
                continue
            ng = g[cur] + w
            if ng < g.get(nxt, np.inf):
                g[nxt], prev[nxt] = ng, cur
                heapq.heappush(todo, (ng + float(np.hypot(goal[0] - nxt[0], goal[1] - nxt[1])), nxt))
    return None
