"""Keyframe selection: sharp, unoccluded frames that together cover the room.

SceneDiff's memory limits us to ~14 frames per video, so which 14 matters.
Uniform sampling (1 in 90) picked blurred frames and frames with a finger over
the lens, and skipped the only views of the hats. Two selectors:

- `select_windows`: image only. Split the video into n equal time windows and
  take the best-quality frame of each. Works for any video.
- `select_coverage`: greedy set cover. Each frame sees a set of voxels (from
  LiDAR depth and ARKit poses); repeatedly take the frame that adds the most
  unseen voxels, weighted by its quality. Needs the sensor path (D1).
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np

WORK_WIDTH = 240  # analysis resolution; frames are 1440 x 1920 portrait


def _small(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (WORK_WIDTH, round(WORK_WIDTH * h / w)), interpolation=cv2.INTER_AREA)


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian at a fixed working resolution. Higher = sharper."""
    g = cv2.cvtColor(_small(bgr), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def finger_mask(bgr: np.ndarray, blur_thresh: float = 2.0) -> np.ndarray:
    """Pixels likely covered by a finger: skin-coloured and out of focus.

    A finger on the lens is centimetres away, so it is both skin-toned (YCrCb box
    from the skin-detection literature) and featureless. Returned at the working
    resolution; resize to use elsewhere. Heuristic: tuned on pair01, fails on
    skin-toned blurry objects (a beige wall out of focus can trigger it).
    """
    small = _small(bgr)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    skin = (cr > 140) & (cr < 180) & (cb > 80) & (cb < 125)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    flat = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (15, 15)) < blur_thresh
    m = (skin & flat).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop speckle
    return m.astype(bool)


def quality(sharp: np.ndarray, occluded: np.ndarray, occ_penalty: float = 10.0) -> np.ndarray:
    """Per-frame quality in [0, 1]: sharpness rank within the video, times an occlusion penalty.

    Rank, not raw variance, because texture (white wall vs bookshelf) changes the
    raw value far more than blur does.
    """
    ranks = np.argsort(np.argsort(sharp)) / max(len(sharp) - 1, 1)
    return ranks * np.clip(1.0 - occ_penalty * occluded, 0.0, 1.0)


def select_windows(q: np.ndarray, n: int) -> np.ndarray:
    """Best frame in each of n equal windows. Returns sorted indices."""
    edges = np.linspace(0, len(q), n + 1).astype(int)
    return np.array([a + int(np.argmax(q[a:b])) for a, b in pairwise(edges) if b > a])


def select_coverage(seen: list[np.ndarray], q: np.ndarray, n: int, min_gap: int = 15) -> np.ndarray:
    """Greedy weighted set cover.

    seen[i]: ids of voxels visible in frame i. Gain of a frame = q[i] * (# voxels not
    yet covered). Frames closer than `min_gap` to a chosen frame are skipped so the
    picks are not near-duplicates. Returns sorted indices.
    """
    covered: set[int] = set()
    chosen: list[int] = []
    sets = [set(s.tolist()) for s in seen]
    for _ in range(n):
        best, best_gain = -1, 0.0
        for i, s in enumerate(sets):
            if any(abs(i - c) < min_gap for c in chosen):
                continue
            gain = q[i] * len(s - covered)
            if gain > best_gain:
                best, best_gain = i, gain
        if best < 0:
            break
        chosen.append(best)
        covered |= sets[best]
    return np.array(sorted(chosen), dtype=int)
