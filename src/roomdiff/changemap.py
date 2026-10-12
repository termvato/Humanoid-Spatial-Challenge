"""3D-first change detection: carve both visits into one voxel grid, compare, cluster.

Input is two `Frames` sets already in one world frame (a joint reconstruction, or
LiDAR captures after `align.register`). Output is a `ChangeMap`: per-voxel labels
(`visibility.LABELS`) and change blobs. `blob_masks` projects a blob back into a
frame as a 2D mask, which is how 3D results are scored with SceneDiff's 2D evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from roomdiff.frames import Frames, incidence_cos
from roomdiff.visibility import LABELS, Grid, classify, compare, components, observe

MAX_VOXELS = 200_000_000  # dense label arrays only; carving is sparse


@dataclass
class Blob:
    id: int
    kind: str  # "removed" or "added"
    idx: np.ndarray  # flat voxel indices
    centre: np.ndarray
    size: np.ndarray
    score: float  # higher = more confident


@dataclass
class ChangeMap:
    grid: Grid
    labels: np.ndarray  # flat, LABELS values
    before: np.ndarray  # flat per-voxel state
    after: np.ndarray
    blobs: list[Blob]
    tau: float

    def counts(self) -> dict[str, int]:
        return {k: int(np.sum(self.labels == v)) for k, v in LABELS.items()}


def frame_points(frames: Frames, i: int, step: int = 4) -> np.ndarray:
    """World points of frame i's valid depth, on a `step` pixel lattice."""
    d = frames.depth[i][::step, ::step]
    ok = frames.valid[i][::step, ::step] & (d > 0)
    v, u = np.nonzero(ok)
    u, v = u * step, v * step
    K = frames.K[i]
    z = d[ok]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    T = frames.T_wc[i]
    return pc @ T[:3, :3].T + T[:3, 3]


def carve(frames: Frames, centres: np.ndarray, tau: float, max_incidence: float | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """Hit and free view counts per voxel. `max_incidence` (deg): pixels seeing their surface more
    obliquely than this give no free evidence (a ray skimming a surface lands far behind it)."""
    hits = np.zeros(len(centres), np.uint16)
    frees = np.zeros(len(centres), np.uint16)
    cos_min = None if max_incidence is None else np.cos(np.radians(max_incidence))
    for i in range(len(frames)):
        fv = None if cos_min is None else incidence_cos(frames.depth[i], frames.K[i]) >= cos_min
        h, f = observe(centres, frames.depth[i], frames.K[i], frames.T_wc[i], tau=tau, valid=frames.valid[i],
                       free_valid=fv)
        hits += h
        frees += f
    return hits, frees


def build(before: Frames, after: Frames, voxel: float, tau: float | None = None, min_voxels: int = 15,
          min_views: int = 2, bounds_pct: float = 1.0, max_incidence: float | None = None) -> ChangeMap:
    """Change map of two visits in one world frame. `tau` defaults to one voxel; see `carve` for max_incidence."""
    tau = voxel if tau is None else tau
    pts = np.concatenate([frame_points(f, i, step=2) for f in (before, after) for i in range(len(f))])
    lo, hi = np.percentile(pts, bounds_pct, axis=0), np.percentile(pts, 100 - bounds_pct, axis=0)
    grid = Grid.around(np.stack([lo, hi]), voxel)
    if np.prod(grid.shape) > MAX_VOXELS:
        raise ValueError(f"grid {grid.shape} too large for voxel {voxel}")
    # Sparse: only voxels within one voxel of an observed surface can be OCC in either visit.
    near = np.zeros(grid.shape, bool)
    flat = grid.index(pts)
    near.ravel()[flat[flat >= 0]] = True
    cand = np.flatnonzero(ndimage.binary_dilation(near))
    centres = grid.centres_of(cand)
    n = int(np.prod(grid.shape))
    sb = np.zeros(n, np.uint8)
    sa = np.zeros(n, np.uint8)
    sb[cand] = classify(*carve(before, centres, tau, max_incidence), min_views=min_views)
    sa[cand] = classify(*carve(after, centres, tau, max_incidence), min_views=min_views)
    lab = compare(sb, sa, grid.shape)
    blobs = []
    for kind in ("removed", "added"):
        for c in components(lab == LABELS[kind], grid.shape, min_voxels):
            p = grid.centres_of(c)
            blobs.append(Blob(len(blobs), kind, c, p.mean(0), p.max(0) - p.min(0) + voxel, float(len(c))))
    return ChangeMap(grid, lab, sb, sa, blobs, tau)


def blob_mask(frames: Frames, i: int, centres: np.ndarray, voxel: float, tau: float) -> np.ndarray:
    """2D mask of voxel centres in frame i, keeping only voxels not hidden behind observed depth."""
    h, w = frames.depth[i].shape
    T = frames.T_wc[i]
    pc = (centres - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    keep = z > 1e-3
    K = frames.K[i]
    u = np.round(K[0, 0] * pc[keep, 0] / z[keep] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[keep, 1] / z[keep] + K[1, 2]).astype(int)
    zk = z[keep]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zk = u[inside], v[inside], zk[inside]
    d = frames.depth[i][v, u]
    ok = frames.valid[i][v, u]
    visible = ~ok | (zk <= d + tau)  # behind a trusted surface -> occluded in this view
    mask = np.zeros((h, w), np.uint8)
    if not visible.any():
        return mask.astype(bool)
    mask[v[visible], u[visible]] = 1
    r = int(np.ceil(0.5 * K[0, 0] * voxel / np.median(zk[visible]))) + 1  # voxel footprint in pixels
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask.astype(bool)


def to_scenediff(cm: ChangeMap, frames: Frames, resample: int = 30, min_px: int = 50, refine=None,
                 match: dict | None = None, views: dict | None = None, descriptor=None) -> dict:
    """Predictions in SceneDiff's object_masks.pkl format, for its official evaluator.

    Removed blobs are drawn in video 1 frames, added in video 2, only on frames the
    evaluator scores (source index divisible by `resample`). `refine(key, rgb, coarse)`
    optionally replaces the coarse projected mask (e.g. `objects.SamRefiner`); a None
    return drops that view. `match`: kwargs for `objects.match_moved`, with `descriptor`
    (needs rgb); a matched removed + added pair is exported as one moved object.
    `views`: precomputed `collect_views` output (e.g. cached SAM masks).
    """
    views = collect_views(cm, frames, resample, min_px, refine) if views is None else views
    return export(cm, frames, views, resample, match, descriptor)


def collect_views(cm: ChangeMap, frames: Frames, resample: int = 30, min_px: int = 50,
                  refine=None) -> dict[int, dict[int, np.ndarray]]:
    """Per blob, the mask in each scored frame of its video: blob id -> frame index -> bool mask."""
    voxel = cm.grid.size
    views: dict[int, dict[int, np.ndarray]] = {}
    for i in np.flatnonzero(frames.src % resample == 0):  # frame-major so a refiner embeds each image once
        v = int(frames.video[i])
        for b in cm.blobs:
            if (b.kind == "removed") != (v == 1):
                continue
            m = blob_mask(frames, i, cm.grid.centres_of(b.idx), voxel, cm.tau)
            if m.sum() < min_px:
                continue
            if refine is not None:
                m = refine(int(i), frames.rgb[i], m)
                if m is None or m.sum() < min_px:
                    continue
            views.setdefault(b.id, {})[int(i)] = m
    return views


def moved_pairs(cm: ChangeMap, frames: Frames, views: dict, descriptor, **match) -> dict[int, int]:
    """Removed blob id -> added blob id for pairs that look like one object in a new place.

    A blob's descriptor is the normalised mean of `descriptor(rgb, mask)` over its views;
    `match` goes to `objects.match_moved` (min_sim, min_size_ratio). Needs rgb.
    """
    from roomdiff.objects import match_moved

    if frames.rgb is None:
        return {}
    blobs = {b.id: b for b in cm.blobs}

    def summary(bid):
        d = np.mean([descriptor(frames.rgb[i], m) for i, m in views[bid].items()], axis=0)
        return {"desc": d / (np.linalg.norm(d) + 1e-9), "voxels": len(blobs[bid].idx)}
    removed = {k: summary(k) for k in views if blobs[k].kind == "removed"}
    added = {k: summary(k) for k in views if blobs[k].kind == "added"}
    return {r: a for r, a, _ in match_moved(removed, added, **match)}


def export(cm: ChangeMap, frames: Frames, views: dict, resample: int = 30, match: dict | None = None,
           descriptor=None) -> dict:
    """SceneDiff-format objects from per-blob views, optionally merging matched removed + added as moved.

    `descriptor(rgb, mask) -> vector`: appearance descriptor for matching (e.g. `objects.DinoDescriptor`),
    required when `match` is given. A blob's descriptor is the normalised mean over its views.
    """
    from pycocotools import mask as rle

    h, w = frames.depth.shape[1:]
    blobs = {b.id: b for b in cm.blobs}
    partner = {} if match is None else moved_pairs(cm, frames, views, descriptor, **match)

    def encode(m):
        enc = rle.encode(np.asfortranarray(m.astype(np.uint8)))
        enc["counts"] = enc["counts"].decode()
        return enc

    out: dict = {"H": h, "W": w}
    absorbed = set(partner.values())
    for bid in views:
        if bid in absorbed:
            continue
        members = [bid] + ([partner[bid]] if bid in partner else [])
        score = float(np.mean([blobs[k].score for k in members]))
        obj: dict = {}
        for k in members:
            v = 1 if blobs[k].kind == "removed" else 2
            for i, m in views[k].items():
                obj.setdefault(f"video_{v}", {})[int(frames.src[i] // resample)] = {"mask": encode(m), "cost": score}
        out[bid] = obj
    return out
