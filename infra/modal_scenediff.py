"""SceneDiff and pi3 on Modal: the baseline, our reconstructions, and the official evaluator.

Functions (results persisted in the `scenediff` volume):
  download       CPU  fetch + unzip the benchmark (~12 GB) into the volume, once
  fetch_weights  CPU  DINOv3 weights for SceneDiff, once
  predict        GPU  SceneDiff (default config) on benchmark pairs
  predict_own    GPU  SceneDiff on one of our pairs in /vol/own/<pair>
  reconstruct    GPU  pi3 on both videos jointly -> /vol/outputs/recon/<name>.npz (our pipeline's input)
  mix            CPU  two-room fake pair (video 1 of one pair, video 2 of another), S23
  evaluate       CPU  official evaluator on predictions in /vol/outputs/<tag>/pred

Usage:
  uv run modal run infra/modal_scenediff.py::data_app.download
  uv run modal run infra/modal_scenediff.py::data_app.fetch_weights   # once; links expire
  uv run modal run infra/modal_scenediff.py::baseline --scenes a,b --tag sd_run
  uv run modal run infra/modal_scenediff.py::recon --scenes a,b --own pair02
  uv run modal run infra/modal_scenediff.py::own --pair pair02
  uv run modal run infra/modal_scenediff.py::mixed --pairs pair01:pair02,pair02:pair01
  uv run modal run infra/modal_scenediff.py::evalpred --tag run --scenes a,b

DINOv3 links are signed and personal: they live in ~/.config/spatial-project/dinov3.env
(outside the repo) and are only used by fetch_weights.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

SCENEDIFF_COMMIT = "62dcbd285fcfe984bb3db55dc01bdcd1a0055714"
REPO = "/root/scene_diff"
VOL = Path("/vol")
BENCH = VOL / "scenediff_benchmark"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1", "libgomp1", "libx11-6")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        f"git clone https://github.com/yuqunw/scene_diff {REPO}",
        f"cd {REPO} && git checkout {SCENEDIFF_COMMIT} && git submodule update --init --recursive",
        f"pip install -r {REPO}/requirements.txt",
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html",
        f"cd {REPO}/submodules/segment-anything-langsplat-modified && pip install -e .",
        f"cd {REPO} && bash checkpoints/download_sam_checkpoint.sh",
    )
)

app = modal.App("scenediff-s1", image=image)
# CPU-only data app: usable without a GPU-enabled Modal account.
data_app = modal.App("scenediff-data", image=image)
vol = modal.Volume.from_name("scenediff", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@data_app.function(volumes={"/vol": vol}, timeout=3 * 3600, cpu=4)
def download() -> None:
    if (BENCH / "splits" / "val_split.json").exists():
        print("benchmark already in volume")
        return
    zip_path = VOL / "scenediff_benchmark.zip"
    url = "https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip"
    subprocess.run(["wget", "-q", "-O", str(zip_path), url], check=True)
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(VOL)], check=True)
    zip_path.unlink()
    vol.commit()
    print(sorted(p.name for p in VOL.iterdir()))


def _workspace(scenes: list[str], data_root: Path = BENCH / "data") -> Path:
    """Repo-relative `data/` dir whose split files list only `scenes`.

    Both SceneDiff scripts read data/scenediff_benchmark/{data,splits}, and the
    evaluator hard-codes that path, so a subset is expressed by replacing the
    split files rather than by flags.
    """
    ws = Path(REPO) / "data" / "scenediff_benchmark"
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "data").exists():
        (ws / "data").symlink_to(data_root)
    (ws / "splits").mkdir(exist_ok=True)
    subset = {"varied": scenes, "kitchen": []}
    for name in ("val_split.json", "test_split.json"):
        (ws / "splits" / name).write_text(json.dumps(subset))
    return ws


WEIGHTS = VOL / "weights"
DINOV3_FILES = {
    "DINOV3_VITH16PLUS_URL": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "DINOV3_VITL16_URL": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "DINOV3_VITB16_URL": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}


@data_app.function(
    volumes={"/vol": vol}, timeout=3600, cpu=2,
    # Named secret, created once with:
    # modal secret create dinov3 --from-dotenv ~/.config/spatial-project/dinov3.env
    secrets=[modal.Secret.from_name("dinov3")],
)
def fetch_weights() -> None:
    """Copy DINOv3 weights from signed links (they expire) into the volume, once."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for env, name in DINOV3_FILES.items():
        dst = WEIGHTS / name
        if dst.exists():
            print("have", name)
            continue
        subprocess.run(["wget", "-q", "-O", str(dst) + ".part", os.environ[env]], check=True)
        Path(str(dst) + ".part").rename(dst)
        print("fetched", name, dst.stat().st_size)
    vol.commit()


def _set_dinov3_weights(model_name: str = "dinov3_vith16plus") -> None:
    """Point the SceneDiff config at a local weight file in the volume."""
    name = next(f for f in DINOV3_FILES.values() if f.startswith(model_name + "_"))
    cfg = Path(REPO) / "configs" / "scenediff_config.yml"
    text = cfg.read_text().replace('"TO_BE_FILLED"', json.dumps(str(WEIGHTS / name)))
    cfg.write_text(text)


@app.function(
    gpu="A100-40GB",
    cpu=8.0,  # SAM automatic mask post-processing is CPU-heavy; default is a fraction of a core
    memory=32768,
    volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
    timeout=4 * 3600,
)
def predict(scenes: list[str], tag: str) -> str:
    _set_dinov3_weights()
    _workspace(scenes)
    out = VOL / "outputs" / tag / "pred"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "scripts/predict_multiview.py", "--config", "configs/scenediff_config.yml",
         "--splits", "val", "--sets", "varied", "--output_dir", str(out)],
        cwd=REPO, check=True,
    )
    vol.commit()
    hf_cache.commit()
    return str(out)


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=32768)
def evaluate(scenes: list[str], tag: str) -> str:
    _workspace(scenes)
    out = VOL / "outputs" / tag
    subprocess.run(
        ["python", "scripts/evaluate_multiview.py", "--pred_dir", str(out / "pred"),
         "--gt_dir", "data/scenediff_benchmark/data", "--splits", "val", "--sets", "varied",
         "--output_path", str(out / "metrics.txt"), "--visualize", "False"],
        cwd=REPO, check=True,
    )
    vol.commit()
    # The evaluator appends threshold settings to the file name.
    return "\n".join(f.read_text() for f in sorted(out.glob("metrics*.txt")))


@app.function(volumes={"/vol": vol})
def list_val_varied() -> list[str]:
    return json.loads((BENCH / "splits" / "val_split.json").read_text())["varied"]


OWN = VOL / "own"  # our own captures, one folder per pair with original_video{1,2}.mp4


@app.function(
    gpu="A100-40GB",
    cpu=8.0,
    memory=32768,
    volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
    timeout=2 * 3600,
)
def predict_own(pair: str, resample: int = 90) -> str:
    """Run SceneDiff on one of our own pairs uploaded to /vol/own/<pair>.

    `resample` = keep 1 of every N frames at 30 fps. Benchmark videos are ~14 s
    and give ~14 frames at N=30; our ~40 s videos need N=90 for the same count.
    At N=30 (40 frames per video) stage 7 asked for 38.7 GB and ran out of memory.
    Changing N invalidates SceneDiff's per-pair SAM cache (sam_masks.pkl), so the
    cache is deleted when N differs from the last run.
    """
    _set_dinov3_weights()
    cfg = Path(REPO) / "configs" / "scenediff_config.yml"
    cfg.write_text(cfg.read_text().replace("resample_rate: 30", f"resample_rate: {resample}"))
    pair_dir = OWN / pair
    stamp = pair_dir / "resample.txt"
    if not stamp.exists() or stamp.read_text().strip() != str(resample):  # unknown or stale cache
        for f in pair_dir.glob("*.pkl"):
            f.unlink()
    stamp.write_text(str(resample))
    _workspace([pair], data_root=OWN)
    out = VOL / "outputs" / "own" / pair
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["python", "scripts/predict_multiview.py", "--config", "configs/scenediff_config.yml",
             "--splits", "val", "--sets", "varied", "--output_dir", str(out)],
            cwd=REPO, check=True,
        )
    finally:
        vol.commit()  # keep extracted frames and SAM cache even if a later stage fails
        hf_cache.commit()
    return str(out)


RECON = VOL / "outputs" / "recon"


@app.function(
    gpu="A100-40GB", cpu=8.0, memory=32768,
    volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
    timeout=3600,
)
def reconstruct(scene_dir: str, name: str, stride: int = 10) -> str:
    """pi3 on every `stride`-th frame of both videos jointly: one shared frame, no registration.

    Writes RECON/<name>.npz with, per frame: depth (z of pi3 local points), conf
    (sigmoid), K (SceneDiff's focal recovery), T_wc (pi3 camera-to-world, OpenCV),
    rgb at model resolution, video id (1/2) and source frame index. Scale is pi3's
    (arbitrary but shared by both videos).
    """
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, REPO)
    from modules.geometry_model import GeometryModel
    from utils import load_images_as_tensor_from_list, process_video_to_frames

    d = Path(scene_dir)
    videos = sorted(x for x in d.iterdir() if x.suffix.lower() in (".mp4", ".mov"))
    files, vid, src = [], [], []
    for v in (1, 2):
        fd = d / f"video{v}_frames"
        if not fd.exists():
            process_video_to_frames(str(videos[v - 1]), str(fd))
        fl = sorted(fd.glob("*.jpg"))
        idx = list(range(0, len(fl), stride))
        files += [fl[i] for i in idx]
        vid += [v] * len(idx)
        src += idx
    vol.commit()
    images = load_images_as_tensor_from_list(files)[None].cuda()
    H, W = images.shape[-2:]
    gm = GeometryModel({"models": {"pi3": {"name": "yyfz233/Pi3"}}})
    gm._load_pi3_model()
    res = gm.estimate_depth_and_poses(images)
    K = gm.compute_intrinsics(res, files, H, W)[0]
    out = {
        "depth": res["local_points"][0, ..., 2].float().cpu().numpy().astype(np.float16),
        "conf": res["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        "K": K.float().cpu().numpy(),
        "T_wc": res["poses"][0].float().cpu().numpy(),
        "rgb": (images[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8),
        "video": np.array(vid), "src": np.array(src), "stride": stride,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    RECON.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECON / f"{name}.npz", **out)
    vol.commit()
    hf_cache.commit()
    return f"{name}: {len(files)} frames {H}x{W}, peak {out['peak_mem_gb']:.1f} GB"


@app.local_entrypoint()
def recon(scenes: str = "", own: str = "", stride: int = 10) -> None:
    """Comma-separated benchmark scenes and/or own pairs, reconstructed in parallel."""
    jobs = [(str(BENCH / "data" / s), s) for s in scenes.split(",") if s]
    jobs += [(str(OWN / p), p) for p in own.split(",") if p]
    for r in reconstruct.starmap([(d, n, stride) for d, n in jobs]):
        print(r)


@app.function(volumes={"/vol": vol}, timeout=1800)
def mix(a: str, b: str, name: str) -> str:
    """S23: a fake pair of two rooms, /vol/own/<name> = video 1 (and its frames) of pair `a` + video 2 of `b`."""
    import shutil

    def src(p):
        return OWN / p if p.startswith("pair") else BENCH / "data" / p
    out = OWN / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for v, p in ((1, a), (2, b)):
        d = src(p)
        videos = sorted(x for x in d.iterdir() if x.suffix.lower() in (".mp4", ".mov"))
        shutil.copy(videos[v - 1], out / f"original_video{v}{videos[v - 1].suffix.lower()}")
        if (d / f"video{v}_frames").exists():
            shutil.copytree(d / f"video{v}_frames", out / f"video{v}_frames")
    vol.commit()
    return name


@app.local_entrypoint()
def mixed(pairs: str, stride: int = 10) -> None:
    """S23: comma-separated a:b items; reconstruct video 1 of a with video 2 of b as own/mix_<a>__<b>."""
    names = [f"mix_{a}__{b}" for a, b in (x.split(":") for x in pairs.split(","))]
    list(mix.starmap([(*x.split(":"), n) for x, n in zip(pairs.split(","), names, strict=True)]))
    for r in reconstruct.starmap([(str(OWN / n), n, stride) for n in names]):
        print(r)


@app.local_entrypoint()
def evalpred(tag: str, scenes: str) -> None:
    """Official evaluator on predictions already uploaded to /vol/outputs/<tag>/pred."""
    print(evaluate.remote(scenes.split(","), tag))


@app.local_entrypoint()
def own(pair: str = "pair01", resample: int = 90) -> None:
    print(predict_own.remote(pair, resample))


@app.local_entrypoint()
def val_list() -> None:
    print(",".join(list_val_varied.remote()))


@app.local_entrypoint()
def baseline(scenes: str, tag: str) -> None:
    """SceneDiff default config on a comma-separated scene list, then the official evaluator."""
    s = scenes.split(",")
    predict.remote(s, tag)
    print(evaluate.remote(s, tag))
