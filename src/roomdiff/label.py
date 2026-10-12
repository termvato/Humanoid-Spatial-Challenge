"""Semantic labels for changes: crops of each change for a vision-language model, and its answers.

The 3D pipeline says where something changed; a VLM looking at the outlined region names it.
It may also answer "none" when the outline holds no distinct object (a wall patch, floor,
shadow, edge or reflection). `other_view` finds the same 3D spot in the other video, so a VLM
can compare the two (COMPARE): a check on false alarms.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

PROMPT = (
    "Each image shows the same spot from a different angle; the red box marks a region found by a "
    "change detector. What object is inside the red box? Answer with a short name (1-4 words, e.g. "
    "'black suitcase'). If the box holds no distinct movable object (only wall, floor, ceiling, door "
    "frame, shadow, reflection, or an image edge), answer exactly: none"
)
NONE = {"none", "nothing", "no object", "n/a"}
COMPARE = (  # S16: a one-word "changed / same" version answered "same" for most real changes
    "Look at the region inside the red box in image 1 and in image 2 (same place, two different times). "
    "Is there an object inside the box in one image that is missing from the other image? Answer yes or no."
)


def best_views(views: dict[int, np.ndarray], k: int = 3) -> list[int]:
    """Up to k frame indices with the largest masks, spread over the video (no two adjacent picks)."""
    order = sorted(views, key=lambda f: -int(views[f].sum()))
    picked: list[int] = []
    gap = max(1, (max(views) - min(views)) // (2 * k)) if views else 1
    for f in order:
        if all(abs(f - p) >= gap for p in picked):
            picked.append(f)
        if len(picked) == k:
            break
    return picked


def closest_poses(T_wc: np.ndarray, f: int, candidates: np.ndarray, scale: float) -> np.ndarray:
    """Candidate frames ordered by how close their camera is to frame f's: distance / `scale`
    (e.g. the median depth) plus the angle between viewing directions (radians)."""
    t, z = T_wc[f][:3, 3], T_wc[f][:3, 2]
    d = np.linalg.norm(T_wc[candidates][:, :3, 3] - t, axis=1) / scale
    a = np.arccos(np.clip(T_wc[candidates][:, :3, 2] @ z, -1, 1))
    return candidates[np.argsort(d + a)]


def other_view(frames, cm, views: dict, bid: int, min_px: int = 50, tries: int = 60
               ) -> tuple[int, int, np.ndarray] | None:
    """(own frame, other-video frame, mask there) for blob `bid`: its best own view and the closest camera
    in the other video where the blob projects to >= `min_px` visible pixels; None if there is none."""
    from roomdiff.changemap import blob_mask

    f = best_views(views[bid], k=1)[0]
    centres = cm.grid.centres_of(next(b for b in cm.blobs if b.id == bid).idx)
    other = np.flatnonzero(frames.video != frames.video[f])
    for g in closest_poses(frames.T_wc, f, other, frames.median_depth())[:tries]:
        m = blob_mask(frames, g, centres, cm.grid.size, cm.tau)
        if m.sum() >= min_px:
            return f, int(g), m
    return None


def crop(rgb: np.ndarray, mask: np.ndarray, context: float = 1.0, min_side: int = 224, pad: float = 0.15
         ) -> np.ndarray:
    """RGB crop around the mask with `context` times its size around it, upscaled so its short side is at
    least `min_side` px, with a thin red box `pad` (fraction of its size) outside the mask's bounds.
    A box, not the mask's contour: on small regions a contour hides what is inside."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    half = max(bx1 - bx0, by1 - by0, 32) * (1 + context) / 2
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    out = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    s = max(1.0, min_side / min(out.shape[:2]))
    if s > 1:
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    p = pad * max(bx1 - bx0, by1 - by0) + 2
    corners = [(bx0 - p - x0) * s, (by0 - p - y0) * s, (bx1 + p - x0) * s, (by1 + p - y0) * s]
    a0, b0, a1, b1 = (int(np.rint(v)) for v in corners)
    cv2.rectangle(out, (a0, b0), (a1, b1), (255, 0, 0), 2)
    return out


def changed(answer: str) -> bool | None:
    """COMPARE answer -> True (changed), False (same), None (unclear)."""
    a = answer.strip().lower()
    return True if a.startswith("yes") else False if a.startswith("no") else None


def parse(answer: str) -> str | None:
    """VLM answer -> label, or None when it says the outline holds no object."""
    a = re.sub(r"[^a-z0-9 \-]", "", answer.strip().lower().splitlines()[0] if answer.strip() else "").strip()
    a = re.sub(r"^(a|an|the) ", "", a)
    return None if not a or a in NONE or a.startswith("none") else a
