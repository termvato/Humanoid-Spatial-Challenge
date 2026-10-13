"""The 3D change list: what changed and where, in the robot's map frame (metres).

Each change is added, removed, or moved (a removed + added pair that looks like one
object, see changemap.moved_pairs). Positions are box centres on the map plane
(x_map = x, y_map = -z of the level frame, as in grid2d), with height above the floor
and box size, so the list lines up with the robot maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from roomdiff.changemap import ChangeMap


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
    blobs: list[int]  # change-map blob ids, "was" side (removed) first
    views: dict[int, list[int]]  # blob id -> frame indices where it is outlined
    was: list[int] = field(default_factory=list)  # the blobs on the "was" side (removed)
    label: str | None = None  # semantic name, filled by a labeller


def box(cm: ChangeMap, idx: np.ndarray, T_level: np.ndarray, scale: float) -> Box:
    """Map-frame box around voxels `idx`."""
    p = (cm.grid.centres_of(idx) @ T_level[:3, :3].T + T_level[:3, 3]) * scale
    xy = np.stack([p[:, 0], -p[:, 2]], 1)
    half = cm.grid.size * scale / 2
    lo, hi = xy.min(0) - half, xy.max(0) + half
    c = (lo + hi) / 2
    return Box((round(float(c[0]), 3), round(float(c[1]), 3)), round(float(p[:, 1].min() - half), 3),
               round(float(p[:, 1].max() + half), 3), (round(float(hi[0] - lo[0]), 3), round(float(hi[1] - lo[1]), 3)))


def gap(a: Box, b: Box) -> float:
    """Shortest distance between two boxes in 3D (0 if they touch or overlap), metres."""
    d = [max(0.0, abs(a.centre[k] - b.centre[k]) - (a.size[k] + b.size[k]) / 2) for k in (0, 1)]
    d.append(max(0.0, a.bottom - b.top, b.bottom - a.top))
    return float(np.linalg.norm(d))


def _groups(ids: list[int], boxes: dict[int, Box], merge: float) -> list[list[int]]:
    """Blobs whose boxes are within `merge` metres, chained (single linkage); input order kept."""
    root = {i: i for i in ids}

    def find(i):
        while root[i] != i:
            i = root[i]
        return i
    for j, a in enumerate(ids):
        for b in ids[j + 1:]:
            if gap(boxes[a], boxes[b]) <= merge:
                root[find(b)] = find(a)
    out: dict[int, list[int]] = {}
    for i in ids:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def changes(cm: ChangeMap, views: dict[int, dict], partner: dict[int, int], T_level: np.ndarray,
            scale: float = 1.0, merge: float = 0.10) -> list[Change]:
    """Changes for blobs that have views, largest (by voxels) first.

    Fragments of one object are merged: blobs of the same kind whose boxes are within `merge`
    metres form one group (S18; 0 = off). A removed group and an added group become one moved
    change when `partner` (removed blob -> added blob) links the largest blob of each, so a stray
    fragment pair cannot turn a removed object into a moved one.
    """
    blobs = {b.id: b for b in cm.blobs}
    order = sorted(views, key=lambda k: -len(blobs[k].idx))
    boxes = {k: box(cm, blobs[k].idx, T_level, scale) for k in order}
    groups = {kind: _groups([k for k in order if blobs[k].kind == kind], boxes, merge if merge > 0 else -1.0)
              for kind in ("removed", "added")}
    head = {g[0]: g for kind in groups for g in groups[kind]}  # largest blob -> its group
    pairs = {r: a for r, a in partner.items() if r in head and a in head}
    paired = set(pairs.values())

    def union(ids):
        return box(cm, np.concatenate([blobs[k].idx for k in ids]), T_level, scale)

    def frames(ids):
        return {k: sorted(views[k]) for k in ids}
    out = []
    for h, g in sorted(head.items(), key=lambda hg: -sum(len(blobs[k].idx) for k in hg[1])):
        if h in paired:
            continue
        if h in pairs:
            a = head[pairs[h]]
            bb, ba = union(g), union(a)
            dist = round(float(np.hypot(ba.centre[0] - bb.centre[0], ba.centre[1] - bb.centre[1])), 3)
            out.append(Change("moved", bb, ba, dist, g + a, frames(g + a), list(g)))
        elif blobs[h].kind == "removed":
            out.append(Change("removed", union(g), None, None, list(g), frames(g), list(g)))
        else:
            out.append(Change("added", None, union(g), None, list(g), frames(g)))
    return out


def to_json(chs: list[Change]) -> list[dict]:
    return [{"id": i, **asdict(c)} for i, c in enumerate(chs)]
