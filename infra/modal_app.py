"""roomdiff on Modal: the whole pipeline on a cloud GPU, and the public upload page.

  pipeline  GPU (A100 40 GB)  roomdiff.run on videos in the `roomdiff-results` volume (<job>/input/),
                              results written to <job>/, input videos deleted afterwards
  web       CPU               the upload page (roomdiff.web) with a daily run cap

Deploy once:  uv run modal deploy infra/modal_app.py
Then:         the printed URL (public page), `roomdiff run a.mp4 b.mp4 --out r --modal`, or `roomdiff serve --modal`.
Weights (pi3, Qwen2.5-VL, SAM, DINOv2) download on first use into the `hf-cache` and `roomdiff-cache` volumes.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DAILY_CAP = 20  # public page only
MAX_FRAMES = 150  # per video for pi3; keeps a run inside 40 GB

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy<2.3", "scipy", "open3d==0.19.0", "opencv-python-headless", "pycocotools", "matplotlib", "pillow",
        "segment-anything", "huggingface_hub", "safetensors", "einops",
        "transformers==4.51.3", "accelerate", "qwen-vl-utils==0.0.10",
        "git+https://github.com/yyfz/Pi3@9fa3ddb3f8d53041f8b2738df404f62223bbaa7b",
    )
    .env({"TORCH_HOME": "/root/.cache/roomdiff/torch"})
    .add_local_python_source("roomdiff", ignore=["**/__pycache__"])
)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    .add_local_python_source("roomdiff", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "examples", "/root/examples")
)
app = modal.App("roomdiff-app")
results = modal.Volume.from_name("roomdiff-results", create_if_missing=True)
cache = modal.Volume.from_name("roomdiff-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
jobs = modal.Dict.from_name("roomdiff-jobs", create_if_missing=True)


@app.function(image=gpu_image, gpu="A100-40GB", cpu=8.0, memory=32768, timeout=3600,
              volumes={"/results": results, "/root/.cache/roomdiff": cache, "/root/.cache/huggingface": hf_cache})
def pipeline(job: str, name1: str, name2: str, max_frames: int = MAX_FRAMES) -> None:
    import shutil

    from roomdiff.run import RoomMismatch, run_pair

    results.reload()
    out = Path("/results") / job
    state = {"state": "running", "stage": "starting"}

    def progress(stage: str) -> None:
        state["stage"] = stage
        jobs[job] = state

    progress("starting")
    try:
        summary = run_pair(out / "input" / name1, out / "input" / name2, out, max_frames=max_frames,
                           progress=progress, title="What changed")
        state.update(state="done", summary=summary)
    except RoomMismatch as e:
        state.update(state="refused", error=str(e))
    except Exception as e:  # noqa: BLE001 (reported to the page)
        state.update(state="failed", error=f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(out / "input", ignore_errors=True)  # do not keep people's videos
        for big in ("recon.npz", "points.npz"):
            (out / big).unlink(missing_ok=True)
        results.commit()
        cache.commit()
        hf_cache.commit()
        jobs[job] = state


@app.function(image=web_image, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from roomdiff.remote import ModalJobs
    from roomdiff.web import make_app

    return make_app(ModalJobs(daily_cap=DAILY_CAP, max_frames=MAX_FRAMES), examples=Path("/root/examples"),
                    note=f"Public demo: at most {DAILY_CAP} runs a day, videos up to 200 MB. Videos are deleted after the run.")
