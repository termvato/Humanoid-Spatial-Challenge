"""The 3D change list: what changed and where, in the robot's map frame (metres).

Each change is added, removed, or moved (a removed + added pair that looks like one
object, see changemap.moved_pairs). Positions are box centres on the map plane
(x_map = x, y_map = -z of the level frame, as in grid2d), with height above the floor
and box size, so the list lines up with the robot maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from roomdiff.changemap import Blob, ChangeMap


@dataclass
class Box:
    centre: tuple[float, float]  # map x, y (m)
    bottom: float  # height above the floor (m)
    top: float
    size: tuple[float, float]  # extent along map x, y (m)


@dataclass
class Change:
    kind: str  # added, removed, moved
    before: Box | None  # where it was (removed, moved)
    after: Box | None  # where it is now (added, moved)
    distance: float | None  # moved only, metres between box centres
    blobs: list[int]  # change-map blob ids
    views: dict[int, list[int]]  # blob id -> frame indices where it is outlined
    label: str | None = None  # semantic name, filled by a labeller


def box(cm: ChangeMap, b: Blob, T_level: np.ndarray, scale: float) -> Box:
    p = (cm.grid.centres_of(b.idx) @ T_level[:3, :3].T + T_level[:3, 3]) * scale
    xy = np.stack([p[:, 0], -p[:, 2]], 1)
    half = cm.grid.size * scale / 2
    lo, hi = xy.min(0) - half, xy.max(0) + half
    c = (lo + hi) / 2
    return Box((round(float(c[0]), 3), round(float(c[1]), 3)), round(float(p[:, 1].min() - half), 3),
               round(float(p[:, 1].max() + half), 3), (round(float(hi[0] - lo[0]), 3), round(float(hi[1] - lo[1]), 3)))


def changes(cm: ChangeMap, views: dict[int, dict], partner: dict[int, int], T_level: np.ndarray,
            scale: float = 1.0) -> list[Change]:
    """Changes for blobs that have views, moved pairs merged; largest (by voxels) first."""
    blobs = {b.id: b for b in cm.blobs}
    absorbed = set(partner.values())
    out = []
    for bid in sorted(views, key=lambda k: -len(blobs[k].idx)):
        if bid in absorbed:
            continue
        b = blobs[bid]
        frames = {bid: sorted(views[bid])}
        if bid in partner:
            a = blobs[partner[bid]]
            frames[a.id] = sorted(views[a.id])
            bb, ba = box(cm, b, T_level, scale), box(cm, a, T_level, scale)
            dist = round(float(np.hypot(ba.centre[0] - bb.centre[0], ba.centre[1] - bb.centre[1])), 3)
            out.append(Change("moved", bb, ba, dist, [b.id, a.id], frames))
        elif b.kind == "removed":
            out.append(Change("removed", box(cm, b, T_level, scale), None, None, [b.id], frames))
        else:
            out.append(Change("added", None, box(cm, b, T_level, scale), None, [b.id], frames))
    return out


def to_json(chs: list[Change]) -> list[dict]:
    return [{"id": i, **asdict(c)} for i, c in enumerate(chs)]
