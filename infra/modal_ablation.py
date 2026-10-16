"""S26 ablation on Modal: carving vs a naive point-distance diff, same reconstructions, same later steps.

Reconstructions go to the `roomdiff-ablation` volume first:
  uv run modal volume put roomdiff-ablation outputs/recon/pi3_pair01.npz recon/pi3_pair01.npz   (and the others)
Then:
  uv run modal run infra/modal_ablation.py
Results: own/<arm>/<pair>/changes.json and bench_<arm>/pred on the volume; download and score locally
(tools/score_own.py; tools/s9_sweep.sh's evaluation step for the benchmark).
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OWN = ("pair01", "pair02")
BENCH = (ROOT / "outputs/s9_test_scenes.txt").read_text().strip() if (ROOT / "outputs/s9_test_scenes.txt").exists() else ""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy<2.3", "scipy", "open3d==0.19.0", "opencv-python-headless", "pycocotools", "matplotlib", "pillow",
        "segment-anything", "huggingface_hub", "safetensors", "einops", "modal",
    )
    .env({"TORCH_HOME": "/root/.cache/roomdiff/torch"})
    .add_local_python_source("roomdiff", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "tools", "/root/tools", ignore=["**/__pycache__"])
)
app = modal.App("roomdiff-ablation", image=image)
data = modal.Volume.from_name("roomdiff-ablation", create_if_missing=True)
cache = modal.Volume.from_name("roomdiff-cache", create_if_missing=True)


@app.function(gpu="A10G", cpu=8.0, memory=32768, timeout=3 * 3600,
              volumes={"/data": data, "/root/.cache/roomdiff": cache})
def arm(kind: str, naive: bool, scenes: str = "") -> str:
    tag = "naive" if naive else "carving"
    if kind == "own":
        from roomdiff.analyse import analyse

        for p in OWN:
            analyse(Path(f"/data/recon/pi3_{p}.npz"), Path(f"/data/own/{tag}/{p}"), naive=naive)
            data.commit()
    else:
        work = Path(f"/data/work_{tag}")
        work.mkdir(parents=True, exist_ok=True)
        cmd = ["python", "/root/tools/bench_predict.py", "--scenes", scenes, "--vlm-rescore", "0",
               "--recon-dir", "/data/recon", "--out", f"/data/bench_{tag}"] + (["--naive"] if naive else [])
        subprocess.run(cmd, cwd=work, check=True)
    data.commit()
    cache.commit()
    return f"{kind} {tag} done"


@app.local_entrypoint()
def main():
    calls = [arm.spawn(k, n, BENCH) for k in ("own", "bench") for n in (False, True)]
    for c in calls:
        print(c.get())
