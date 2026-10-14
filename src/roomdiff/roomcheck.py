"""Are the two videos of the same room? A cheap check before the 3D reconstruction (S22).

Every frame gets a DINOv2 ViT-S/14 image feature (CLS token). Each frame of one video is matched to
its most similar frame in the other; the score is the median best-match cosine, both directions,
the lower kept. Two videos of one room share many near-identical views; two rooms share few.
"""

from __future__ import annotations

import cv2
import numpy as np

REFUSE_BELOW = 0.45  # S22: image score; every same-room pair scored >= 0.63, most other rooms 0.1-0.4
WARN_FIT_BELOW = 0.40  # S23: 3D overlap after pi3; same room >= 0.605 (one outlier 0.074), other rooms <= 0.215


def frame_features(rgbs: np.ndarray | list[np.ndarray], device: str = "cuda", size: int = 224, model=None
                   ) -> np.ndarray:
    """(N, 384) L2-normalised CLS features of RGB uint8 frames."""
    import torch

    model = model or torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    out = []
    for rgb in rgbs:
        h, w = rgb.shape[:2]
        s = size / max(h, w)
        nh, nw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
        x = cv2.resize(rgb, (nw, nh)).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
        with torch.no_grad():
            v = model(t)[0].cpu().numpy()
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(out)


def overlap_score(f1: np.ndarray, f2: np.ndarray) -> float:
    """Median best-match cosine between two videos' frame features, the lower of both directions."""
    s = f2 @ f1.T
    return float(min(np.median(s.max(1)), np.median(s.max(0))))
