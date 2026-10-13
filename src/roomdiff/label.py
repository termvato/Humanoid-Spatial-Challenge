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
    "The red box in each image marks the same object, seen from different angles (a moved object is "
    "shown at its old and its new place). What object is inside the red box? Answer with a short name "
    "(1-4 words, e.g. 'black suitcase'). If the box holds no distinct movable object (only wall, floor, "
    "ceiling, door frame, shadow, reflection, or an image edge), answer exactly: none"
)
NONE = {"none", "nothing", "no object", "n/a"}
# S19: the VLM often names the structure instead of answering "none"; such answers are not objects
STRUCTURE = ("wall", "floor", "ceiling", "door frame", "doorframe", "door", "baseboard", "skirting board",
             "shadow", "reflection", "window", "door handle", "door knob", "doorknob", "light switch",
             "outlet", "socket", "tile", "tiles", "staircase", "stairs", "corner", "edge")
STILL_THERE = (  # S21: naming the object; "no" = changed
    "The red box in image 1 contains a {name}. Image 2 shows the same place at another time. "
    "Is the {name} inside the red box in image 2? Answer yes or no."
)
COMPARE = (  # S16: a one-word "changed / same" version answered "same" for most real changes
    "Look at the region inside the red box in image 1 and in image 2 (same place, two different times). "
    "Is there an object inside the box in one image that is missing from the other image? Answer yes or no."
)


def sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian inside the mask's bounding box (higher = sharper)."""
    ys, xs = np.nonzero(mask)
    g = cv2.cvtColor(np.ascontiguousarray(rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def touches_border(mask: np.ndarray, px: int = 2) -> bool:
    return bool(mask[:px].any() or mask[-px:].any() or mask[:, :px].any() or mask[:, -px:].any())


def best_views(views: dict[int, np.ndarray], k: int = 3, rgb: np.ndarray | None = None) -> list[int]:
    """Up to k frame indices, best first, spread over the video (no two adjacent picks).

    Without `rgb`: largest masks. With `rgb` (S19): masks cut by the image border are skipped (when
    any other view exists) and the area is weighted by sharpness relative to the median view, capped
    at 1, so a blurry close-up cannot win on size alone."""
    score = {f: float(m.sum()) for f, m in views.items()}
    if rgb is not None and views:
        inside = [f for f in views if not touches_border(views[f])]
        cand = inside or list(views)
        sharp = {f: sharpness(rgb[f], views[f]) for f in cand}
        med = float(np.median(list(sharp.values()))) or 1.0
        score = {f: score[f] * min(1.0, sharp[f] / med) for f in cand}
    order = sorted(score, key=lambda f: -score[f])
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


def other_view(frames, cm, views: dict, bid: int, min_px: int = 50, tries: int = 60, f: int | None = None
               ) -> tuple[int, int, np.ndarray] | None:
    """(own frame, other-video frame, mask there) for blob `bid`: own frame `f` (default its best view) and
    the closest camera in the other video where the blob projects to >= `min_px` visible pixels; None if
    there is none."""
    from roomdiff.changemap import blob_mask

    f = best_views(views[bid], k=1)[0] if f is None else f  # default: largest mask (S17 benchmark setting)
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


def gone(answer: str) -> bool | None:
    """STILL_THERE answer -> True (gone: changed), False (still there), None (unclear)."""
    v = changed(answer)
    return None if v is None else not v


def majority(votes: list[bool | None]) -> bool | None:
    """S20: changed if at least half of the clear answers say so; None if there is no clear answer."""
    v = [x for x in votes if x is not None]
    return None if not v else 2 * sum(v) >= len(v)


def parse(answer: str) -> str | None:
    """VLM answer -> label, or None when it says the outline holds no object."""
    a = re.sub(r"[^a-z0-9 \-]", "", answer.strip().lower().splitlines()[0] if answer.strip() else "").strip()
    a = re.sub(r"^(a|an|the) ", "", a)
    if not a or a in NONE or a.startswith("none"):
        return None
    head = re.split(r" (?:on|in|of|with|at|near|under) ", a)[0]  # "hat on shelf" -> "hat"
    return None if any(head == w or head.endswith(" " + w) for w in STRUCTURE) else a
