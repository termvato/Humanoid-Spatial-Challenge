"""S5: run alternative reconstruction models on exactly the frames pi3 saw, same output format.

Each model gets its own image (dependencies conflict). Every function reads the
frames already extracted in the `scenediff` volume (every `stride`-th frame of
video{1,2}_frames, both videos jointly in one call) and writes
/vol/outputs/recon/<model>/<name>.npz with depth, conf (in [0,1]), K (at depth
resolution), T_wc (camera-to-world, OpenCV), rgb, video, src.

Confidence: pi3/Pi3X give logits (sigmoid). DA3, VGGT and MapAnything give
unbounded scores, so they are stored as the per-scene percentile rank; the shared
loader then drops conf <= 0.1, i.e. each model's least confident 10% of pixels.

Usage: uv run modal run infra/modal_models.py --model vggt --scenes a,b --own pair01
"""

from pathlib import Path

import modal

VOL = Path("/vol")
BENCH = VOL / "scenediff_benchmark" / "data"
OWN = VOL / "own"
OUT = VOL / "outputs" / "recon"

vol = modal.Volume.from_name("scenediff")
hf_cache = modal.Volume.from_name("hf-cache")

base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2", "pillow", "huggingface_hub", "safetensors", "einops", "scipy", "opencv-python-headless")
)
img_pi3x = base.run_commands(
    "git clone https://github.com/yyfz/Pi3 /opt/pi3", "pip install -r /opt/pi3/requirements.txt || true")
img_vggt = base.run_commands(
    "git clone https://github.com/facebookresearch/vggt /opt/vggt", "pip install -r /opt/vggt/requirements.txt")
img_da3 = base.pip_install("xformers==0.0.28.post3", index_url="https://download.pytorch.org/whl/cu121").run_commands(
    "git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /opt/da3", "cd /opt/da3 && pip install -e .", "pip install addict omegaconf trimesh imageio")
img_omega = base.run_commands(
    "git clone https://github.com/facebookresearch/vggt-omega /opt/vggt_omega",
    "cd /opt/vggt_omega && pip install -r requirements.txt && pip install -e .")
img_mapa = base.run_commands(
    "git clone https://github.com/facebookresearch/map-anything /opt/mapa", "cd /opt/mapa && pip install -e .")

app = modal.App("roomdiff-models")
GPU = {"gpu": "A100-80GB", "cpu": 8.0, "memory": 65536, "timeout": 3600,
       "volumes": {"/vol": vol, "/root/.cache/huggingface": hf_cache}}


def _frames(scene_dir: str, stride: int):
    d = Path(scene_dir)
    files, vid, src = [], [], []
    for v in (1, 2):
        fl = sorted((d / f"video{v}_frames").glob("*.jpg"))
        if not fl:
            raise FileNotFoundError(f"{d}/video{v}_frames empty: run the pi3 recon first (it extracts frames)")
        idx = list(range(0, len(fl), stride))
        files += [str(fl[i]) for i in idx]
        vid += [v] * len(idx)
        src += idx
    return files, vid, src


def _rank01(x):
    import numpy as np

    flat = x.reshape(-1)
    r = np.empty(len(flat), np.float32)
    r[np.argsort(flat, kind="stable")] = np.linspace(0, 1, len(flat), dtype=np.float32)
    return r.reshape(x.shape)


def _save(model: str, name: str, depth, conf, K, T_wc, rgb, vid, src, stride, extra: dict) -> str:
    import numpy as np
    import torch

    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{name}.npz", depth=depth.astype(np.float16), conf=conf.astype(np.float16), K=K.astype(np.float32),
        T_wc=T_wc.astype(np.float32), rgb=rgb.astype(np.uint8), video=np.array(vid), src=np.array(src),
        stride=stride, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, **extra)
    vol.commit()
    hf_cache.commit()
    return f"{model} {name}: {len(vid)} frames {depth.shape[1]}x{depth.shape[2]}, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"


def _w2c_to_c2w(E):
    import numpy as np

    n = len(E)
    M = np.tile(np.eye(4), (n, 1, 1))
    M[:, :3, :4] = E
    return np.linalg.inv(M)


def _load_518(files):
    """pi3-style loading: longest side 518, both sides multiples of 14, values in [0,1]."""
    import numpy as np
    import torch
    from PIL import Image

    ims = [Image.open(f).convert("RGB") for f in files]
    w0, h0 = ims[0].size
    s = 518 / max(w0, h0)
    w, h = round(w0 * s / 14) * 14, round(h0 * s / 14) * 14
    arr = np.stack([np.asarray(im.resize((w, h), Image.BICUBIC)) for im in ims])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0, arr


def _focal_from_points(local_points, mask):
    """Least-squares focal from a camera-frame point map (u - cx = f x / z), principal point at centre."""
    import numpy as np

    h, w = local_points.shape[:2]
    v, u = np.mgrid[0:h, 0:w]
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    m = mask & (z > 1e-6)
    a = np.concatenate([(x / z)[m], (y / z)[m]])
    b = np.concatenate([(u - (w - 1) / 2)[m], (v - (h - 1) / 2)[m]])
    return float((a @ b) / (a @ a))


@app.function(image=img_pi3x, **GPU)
def pi3x(scene_dir: str, name: str, stride: int = 10) -> str:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/opt/pi3")
    from pi3.models.pi3x import Pi3X

    files, vid, src = _frames(scene_dir, stride)
    imgs, rgb = _load_518(files)
    model = Pi3X.from_pretrained("yyfz233/Pi3X").cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        res = model(imgs[None].cuda())
    lp = res["local_points"][0].float().cpu().numpy()
    conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
    h, w = lp.shape[1:3]
    K = np.zeros((len(lp), 3, 3))
    for i in range(len(lp)):
        f = _focal_from_points(lp[i], conf[i] > 0.1)
        K[i] = [[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]
    T = res["camera_poses"][0].float().cpu().numpy()
    return _save("pi3x", name, lp[..., 2], conf, K, T, rgb, vid, src, stride, {})


@app.function(image=img_vggt, **GPU)
def vggt(scene_dir: str, name: str, stride: int = 10) -> str:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/opt/vggt")
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    files, vid, src = _frames(scene_dir, stride)
    # "pad" keeps the whole image (default "crop" cuts the long side of portrait frames);
    # the padding is cropped off the outputs below and K's principal point shifted to match.
    images = load_and_preprocess_images(files, mode="pad").cuda()
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    E, Kt = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    from PIL import Image

    w0, h0 = Image.open(files[0]).size
    if w0 >= h0:
        w, h = 518, round(h0 * 518 / w0 / 14) * 14
    else:
        w, h = round(w0 * 518 / h0 / 14) * 14, 518
    top, left = (518 - h) // 2, (518 - w) // 2
    crop = (slice(None), slice(top, top + h), slice(left, left + w))
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()[crop]
    conf = _rank01(pred["depth_conf"][0].float().cpu().numpy()[crop])
    rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)[crop]
    K = Kt[0].float().cpu().numpy()
    K[:, 0, 2] -= left
    K[:, 1, 2] -= top
    return _save("vggt", name, depth, conf, K, _w2c_to_c2w(E[0].float().cpu().numpy()),
                 rgb, vid, src, stride, {})


@app.function(image=img_da3, **GPU)
def da3(scene_dir: str, name: str, stride: int = 10, model_id: str = "depth-anything/DA3-GIANT-1.1") -> str:
    from depth_anything_3.api import DepthAnything3

    files, vid, src = _frames(scene_dir, stride)
    model = DepthAnything3.from_pretrained(model_id).to("cuda").eval()
    p = model.inference(files, process_res=518)
    return _save("da3", name, p.depth, _rank01(p.conf), p.intrinsics, _w2c_to_c2w(p.extrinsics),
                 p.processed_images, vid, src, stride, {"model_id": model_id})


@app.function(image=img_mapa, **GPU)
def mapanything(scene_dir: str, name: str, stride: int = 10) -> str:
    import numpy as np
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    files, vid, src = _frames(scene_dir, stride)
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda")
    views = load_images(files)
    preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    g = lambda k: np.concatenate([p[k].float().cpu().numpy() for p in preds])
    depth = g("depth_z")[..., 0]
    rgb = (np.clip(g("img_no_norm"), 0, 1) * 255).astype(np.uint8) if g("img_no_norm").max() <= 1.0 \
        else g("img_no_norm").astype(np.uint8)
    return _save("mapanything", name, depth, _rank01(g("conf")), g("intrinsics"), g("camera_poses"), rgb,
                 vid, src, stride, {})


@app.function(image=img_omega, secrets=[modal.Secret.from_name("huggingface")], **GPU)
def vggt_omega(scene_dir: str, name: str, stride: int = 10, res: int = 512) -> str:
    """VGGT-Omega-1B-512 (gated, CC BY-NC per the access request). Outputs at its own preprocessed size."""
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    files, vid, src = _frames(scene_dir, stride)
    ckpt = hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    images = load_and_preprocess_images(files, image_resolution=res).to("cuda")
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    E, Kt = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()
    conf = _rank01(pred["depth_conf"][0].float().cpu().numpy())
    im = pred["images"][0].float().cpu().numpy()
    rgb = (np.clip(im.transpose(0, 2, 3, 1), 0, 1) * 255).astype(np.uint8)
    E = E[0].float().cpu().numpy()[:, :3, :4]
    hf_cache.commit()
    return _save("vggt_omega", name, depth, conf, Kt[0].float().cpu().numpy(), _w2c_to_c2w(E),
                 rgb, vid, src, stride, {"res": res, "images_shape": list(pred["images"].shape)})


FUNCS = {"vggt_omega": vggt_omega, "pi3x": pi3x, "vggt": vggt, "da3": da3, "mapanything": mapanything}


@app.local_entrypoint()
def main(model: str, scenes: str = "", own: str = "", stride: int = 10) -> None:
    jobs = [(str(BENCH / s), s, stride) for s in scenes.split(",") if s]
    jobs += [(str(OWN / p), p, stride) for p in own.split(",") if p]
    for r in FUNCS[model].starmap(jobs, return_exceptions=True):
        print(r)
