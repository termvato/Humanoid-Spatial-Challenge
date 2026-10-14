"""pi3 on both videos jointly: one shared 3D frame for both visits, written as a recon npz.

Same steps as the Modal reconstruction used for every result in docs/spikes.md
(infra/modal_scenediff.py::reconstruct): ffmpeg frames at 30 fps, every `stride`-th frame,
longest side 518 px (multiple of 14), pi3 in bf16, focal from the local point map. Frame
handling and the focal solver follow SceneDiff's utils.py (MIT, Yuqun Wu), which takes the
solver from MoGe (MIT, Microsoft).
"""

from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path

import numpy as np

PI3 = "yyfz233/Pi3"


def video_frames(video: str | Path, out_dir: Path, fps: int = 30) -> list[Path]:
    """Every frame at `fps` as JPEG (ffmpeg applies the phone's rotation tag)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-q:v", "2", "-r", str(fps),
                    "-start_number", "0", str(out_dir / "%05d.jpg")], check=True)
    files = sorted(out_dir.glob("*.jpg"))
    if not files:
        raise ValueError(f"no frames could be read from {video}")
    return files


def target_size(w: int, h: int, max_size: int = 518) -> tuple[int, int]:
    """(W, H): longest side `max_size`, the other a multiple of 14, aspect kept (SceneDiff)."""
    if h > w:
        return round(round(w * max_size / h) / 14) * 14, max_size
    return max_size, round(round(h * max_size / w) / 14) * 14


def load_images(files: list[Path], max_size: int = 518) -> np.ndarray:
    """(N, H, W, 3) uint8, every image resized to the first one's target size (Lanczos)."""
    from PIL import Image

    imgs = [Image.open(f).convert("RGB") for f in files]
    size = target_size(*imgs[0].size, max_size)
    return np.stack([np.asarray(im.resize(size, Image.Resampling.LANCZOS)) for im in imgs])


def _uv(width: int, height: int) -> np.ndarray:
    """Normalised view-plane coordinates (H, W, 2), spanning +-width/diagonal, +-height/diagonal."""
    a = width / height
    sx, sy = a / (1 + a * a) ** 0.5, 1 / (1 + a * a) ** 0.5
    u = np.linspace(-sx * (width - 1) / width, sx * (width - 1) / width, width)
    v = np.linspace(-sy * (height - 1) / height, sy * (height - 1) / height, height)
    return np.stack(np.meshgrid(u, v, indexing="xy"), -1)


def solve_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> tuple[float, float]:
    """Focal and depth shift minimising |focal * xy / (z + shift) - uv| (MoGe)."""
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv, xy, z, shift):
        p = xy / (z + shift)[:, None]
        f = (p * uv).sum() / np.square(p).sum()
        return (f * p - uv).ravel()
    shift = float(least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")["x"].squeeze())
    p = xy / (z + shift)[:, None]
    return float((p * uv).sum() / np.square(p).sum()), shift


def intrinsics(local_points: np.ndarray, conf: np.ndarray, lr: int = 64) -> np.ndarray:
    """(N, 3, 3) pinhole K per frame from pi3's local point maps (N, H, W, 3) and confidence (sigmoid),
    as SceneDiff's GeometryModel.compute_intrinsics: nearest-neighbour 64x64, points with conf > 0.1."""
    n, h, w, _ = local_points.shape
    a = w / h
    ys = np.floor(np.arange(lr) * h / lr).astype(int)
    xs = np.floor(np.arange(lr) * w / lr).astype(int)
    uv = _uv(w, h)[ys][:, xs]
    out = []
    for i in range(n):
        pts, m = local_points[i][ys][:, xs], conf[i][ys][:, xs] > 0.1
        focal = solve_focal_shift(uv[m], pts[m])[0] if m.sum() >= 2 else 1.0
        fx = focal * w / 2 * (1 + a * a) ** 0.5 / a
        fy = focal * h / 2 * (1 + a * a) ** 0.5
        out.append([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
    return np.array(out, np.float32)


def sample(n: int, stride: int, max_frames: int) -> list[int]:
    """Every `stride`-th of n frame indices; if that is more than `max_frames`, evenly spaced instead."""
    idx = list(range(0, n, stride))
    if max_frames and len(idx) > max_frames:
        idx = np.linspace(0, n - 1, max_frames).round().astype(int).tolist()
    return idx


def reconstruct(frames1: list[Path], frames2: list[Path], out: Path, stride: int = 10, max_frames: int = 0,
                device: str = "cuda") -> Path:
    """pi3 jointly on both videos' frames (from `video_frames`) -> `out` (npz: depth, conf, K, T_wc, rgb,
    video, src, stride).

    `max_frames` caps frames per video (0 = no cap): pi3 memory grows with the frame count
    (243 frames: 16 GB, 376: 22 GB on an A100)."""
    import torch
    from pi3.models.pi3 import Pi3

    files, vid, src = [], [], []
    for v, fl in ((1, frames1), (2, frames2)):
        idx = sample(len(fl), stride, max_frames)
        files += [fl[i] for i in idx]
        vid += [v] * len(idx)
        src += idx
    rgb = load_images(files)
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255)[None].to(device)
    model = Pi3.from_pretrained(PI3).to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        res = model(images)
    local = res["local_points"][0].float().cpu().numpy()
    conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
    poses = res["camera_poses"][0].float().cpu().numpy()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, res, images
    torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, depth=local[..., 2].astype(np.float16), conf=conf.astype(np.float16),
                        K=intrinsics(local, conf), T_wc=poses, rgb=rgb, video=np.array(vid), src=np.array(src),
                        stride=stride, peak_mem_gb=peak)
    return out
