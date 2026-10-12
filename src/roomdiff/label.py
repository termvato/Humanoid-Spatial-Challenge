"""Semantic labels for changes: crops of each change for a vision-language model, and its answers.

The 3D pipeline says where something changed; a VLM looking at the outlined region names it.
It may also answer "none" when the outline holds no distinct object (a wall patch, floor,
shadow, edge or reflection), which is a check on false alarms.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

PROMPT = (
    "Each image shows the same spot from a different angle; the red outline marks a region found by a "
    "change detector. What object is inside the red outline? Answer with a short name (1-4 words, e.g. "
    "'black suitcase'). If the outline holds no distinct movable object (only wall, floor, ceiling, door "
    "frame, shadow, reflection, or an image edge), answer exactly: none"
)
NONE = {"none", "nothing", "no object", "n/a"}


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


def crop(rgb: np.ndarray, mask: np.ndarray, context: float = 1.0, min_side: int = 224) -> np.ndarray:
    """RGB crop around the mask with `context` times its size around it, red outline drawn, upscaled
    so its short side is at least `min_side` px."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min(), 32) * (1 + context) / 2
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    img = np.ascontiguousarray(rgb.copy())
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (255, 0, 0), max(2, round(min(h, w) / 240)))
    out = img[y0:y1, x0:x1]
    s = min_side / min(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else out


def parse(answer: str) -> str | None:
    """VLM answer -> label, or None when it says the outline holds no object."""
    a = re.sub(r"[^a-z0-9 \-]", "", answer.strip().lower().splitlines()[0] if answer.strip() else "").strip()
    a = re.sub(r"^(a|an|the) ", "", a)
    return None if not a or a in NONE or a.startswith("none") else a
