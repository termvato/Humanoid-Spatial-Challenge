"""Spike S1: run the SceneDiff baseline on Modal.

Stages (each a Modal function, results persisted in the `scenediff` volume):
  download   CPU   fetch + unzip the benchmark (~12 GB) into the volume, once
  predict    GPU   run scripts/predict_multiview.py on a subset of pairs
  evaluate   CPU   run scripts/evaluate_multiview.py, write metrics to the volume

Usage:
  uv run modal run infra/modal_scenediff.py::data_app.download
  uv run modal run infra/modal_scenediff.py::data_app.fetch_weights   # once; links expire
  uv run modal run infra/modal_scenediff.py::spike --n 5
  uv run modal volume get scenediff outputs/s1 outputs/

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


def _workspace(scenes: list[str]) -> Path:
    """Repo-relative `data/` dir whose split files list only `scenes`.

    Both SceneDiff scripts read data/scenediff_benchmark/{data,splits}, and the
    evaluator hard-codes that path, so a subset is expressed by replacing the
    split files rather than by flags.
    """
    ws = Path(REPO) / "data" / "scenediff_benchmark"
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "data").exists():
        (ws / "data").symlink_to(BENCH / "data")
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


@data_app.function(volumes={"/vol": vol}, cpu=2)
def smoke() -> None:
    """Import everything SceneDiff needs, on CPU, before paying for a GPU."""
    subprocess.run(
        ["python", "-c", ("import open3d, torch_scatter, segment_anything_ls; "
         "import sys; sys.path.insert(0, '.'); from modules import SceneDiff; print('imports ok')")],
        cwd=REPO, check=True,
    )


@app.function(volumes={"/vol": vol})
def list_val_varied() -> list[str]:
    return json.loads((BENCH / "splits" / "val_split.json").read_text())["varied"]


@app.local_entrypoint()
def spike(n: int = 5, tag: str = "s1", start: int = 0) -> None:
    scenes = list_val_varied.remote()[start:start + n]
    print("scenes:", scenes)
    predict.remote(scenes, tag)
    print(evaluate.remote(scenes, tag))
