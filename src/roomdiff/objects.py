"""Object layer: turn coarse projected 3D blobs into crisp per-frame object masks with SAM.

Geometry says *where* something changed; its projected voxels are blobby and
fragmented. SAM, prompted with the blob's box and interior points, returns object
masks. We keep the candidate that best agrees with the geometry, and drop the
detection if none agrees (IoU < `min_agree`): SAM is never allowed to invent a
change, only to sharpen one.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAM_CKPT = Path.home() / ".cache" / "roomdiff" / "sam_vit_h.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def prompts_from_mask(mask: np.ndarray, n_points: int = 3, pad: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Box (x0,y0,x1,y1), padded by `pad` of its size, and up to n interior points (x,y) far from the edge."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = pad * (x1 - x0 + 1), pad * (y1 - y0 + 1)
    h, w = mask.shape
    box = np.array([max(0, x0 - px), max(0, y0 - py), min(w - 1, x1 + px), min(h - 1, y1 + py)])
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    d = dist.copy()
    for _ in range(n_points):
        y, x = np.unravel_index(np.argmax(d), d.shape)
        if d[y, x] <= 0:
            break
        pts.append((x, y))
        r = max(int(dist[y, x]), 3)
        cv2.circle(d, (int(x), int(y)), 2 * r, 0, -1)  # next point elsewhere in the blob
    return box, np.array(pts, dtype=np.float32)


def choose_mask(candidates: np.ndarray, coarse: np.ndarray, min_agree: float = 0.3) -> np.ndarray | None:
    """Candidate with the highest IoU against the coarse geometric mask, or None if all disagree."""
    scores = [iou(c, coarse) for c in candidates]
    k = int(np.argmax(scores))
    return candidates[k] if scores[k] >= min_agree else None


class SamRefiner:
    """Lazily loaded SAM ViT-H predictor; caches the image embedding per frame key."""

    def __init__(self, checkpoint: Path = SAM_CKPT, device: str = "cuda", min_agree: float = 0.3):
        self.checkpoint, self.device, self.min_agree = checkpoint, device, min_agree
        self.predictor = None
        self.key = None

    def _load(self) -> None:
        from segment_anything import SamPredictor, sam_model_registry

        if not self.checkpoint.exists():  # 2.6 GB, once
            import urllib.request

            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.checkpoint.with_suffix(".part")
            urllib.request.urlretrieve(SAM_URL, tmp)
            tmp.rename(self.checkpoint)
        sam = sam_model_registry["vit_h"](checkpoint=str(self.checkpoint)).to(self.device)
        self.predictor = SamPredictor(sam)

    def __call__(self, key, rgb: np.ndarray, coarse: np.ndarray) -> np.ndarray | None:
        if self.predictor is None:
            self._load()
        if key != self.key:
            self.predictor.set_image(rgb)
            self.key = key
        box, pts = prompts_from_mask(coarse)
        masks, _, _ = self.predictor.predict(
            point_coords=pts if len(pts) else None,
            point_labels=np.ones(len(pts)) if len(pts) else None,
            box=box, multimask_output=True,
        )
        return choose_mask(masks, coarse, self.min_agree)


def match_moved(removed: dict[int, dict], added: dict[int, dict], min_sim: float = 0.5,
                min_size_ratio: float = 0.4) -> list[tuple[int, int, float]]:
    """Pair removed and added objects that are the same thing in a new place (greedy, best first).

    Each value: {"desc": unit feature vector, "voxels": int}. Similarity = cosine; pairs
    whose voxel counts differ by more than 1/min_size_ratio are not considered.
    Returns (removed_id, added_id, similarity).
    """
    cands = []
    for r, a_r in removed.items():
        for a, a_a in added.items():
            ratio = min(a_r["voxels"], a_a["voxels"]) / max(a_r["voxels"], a_a["voxels"])
            if ratio < min_size_ratio:
                continue
            sim = float(a_r["desc"] @ a_a["desc"])
            if sim >= min_sim:
                cands.append((sim, r, a))
    used_r, used_a, pairs = set(), set(), []
    for sim, r, a in sorted(cands, reverse=True):
        if r in used_r or a in used_a:
            continue
        used_r.add(r)
        used_a.add(a)
        pairs.append((r, a, sim))
    return pairs


class DinoDescriptor:
    """Mean DINOv2 ViT-S/14 patch feature under a mask (L2-normalised). Lazily loaded via torch.hub."""

    def __init__(self, device: str = "cuda", size: int = 448):
        self.device, self.size = device, size
        self.model = None

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()
        h, w = rgb.shape[:2]
        s = self.size / max(h, w)
        nh, nw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
        x = cv2.resize(rgb, (nw, nh)).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(self.device)
        with torch.no_grad():
            f = self.model.forward_features(t)["x_norm_patchtokens"][0]  # (nh/14 * nw/14, 384)
        f = f.reshape(nh // 14, nw // 14, -1).cpu().numpy()
        m = cv2.resize(mask.astype(np.uint8), (nw // 14, nh // 14), interpolation=cv2.INTER_AREA) > 0
        if not m.any():
            m = cv2.resize(mask.astype(np.uint8), (nw // 14, nh // 14), interpolation=cv2.INTER_NEAREST) > 0
        v = f[m].mean(0) if m.any() else f.reshape(-1, f.shape[-1]).mean(0)
        return v / (np.linalg.norm(v) + 1e-9)
