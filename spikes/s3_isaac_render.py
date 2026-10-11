"""Spike S3: can this laptop render synthetic room pairs fast enough?

Loads a furnished Isaac environment headless, orbits a camera through it and
captures RGB + depth + instance segmentation with Replicator annotators.
Writes timing, peak VRAM and sample frames to outputs/s3/.

Run: ~/isaacsim/python.sh spikes/s3_isaac_render.py [--scene office] [--frames 120]
Decision rule (docs/spikes.md): >= 5 fps with clean instance masks at 640x480.
"""

import argparse
import json
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="office", choices=["office", "simple_room", "hospital"])
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "s3"))
args = parser.parse_args()

from isaacsim import SimulationApp

# The app must exist before any other isaacsim/omni import.
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

import numpy as np
import omni.replicator.core as rep
from isaacsim.core.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path
from PIL import Image

SCENES = {
    "office": "/Isaac/Environments/Office/office.usd",
    "simple_room": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "hospital": "/Isaac/Environments/Hospital/hospital.usd",
}


def gpu_mem_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def orbit_poses(n: int, radius: float = 3.0, height: float = 1.4):
    """Camera positions on a circle, all looking at the scene centre."""
    for i in range(n):
        a = 2 * np.pi * i / n
        yield (radius * np.cos(a), radius * np.sin(a), height)


def main() -> None:
    os.makedirs(args.out, exist_ok=True)
    root = get_assets_root_path()
    t0 = time.perf_counter()
    open_stage(root + SCENES[args.scene])
    for _ in range(30):  # let materials and textures stream in
        simulation_app.update()
    load_s = time.perf_counter() - t0

    cam = rep.create.camera(position=(3.0, 0.0, 1.4), look_at=(0.0, 0.0, 0.8))
    rp = rep.create.render_product(cam, (args.width, args.height))
    annots = {
        name: rep.AnnotatorRegistry.get_annotator(name)
        for name in ("rgb", "distance_to_image_plane", "instance_id_segmentation")
    }
    for a in annots.values():
        a.attach(rp)

    vram_peak = gpu_mem_mib()
    frame_times = []
    for i, pos in enumerate(orbit_poses(args.frames)):
        with cam:
            rep.modify.pose(position=pos, look_at=(0.0, 0.0, 0.8))
        t = time.perf_counter()
        rep.orchestrator.step(rt_subframes=1)
        data = {k: a.get_data() for k, a in annots.items()}
        frame_times.append(time.perf_counter() - t)
        vram_peak = max(vram_peak, gpu_mem_mib()) if i % 10 == 0 else vram_peak
        if i % 30 == 0:
            Image.fromarray(data["rgb"][..., :3]).save(f"{args.out}/rgb_{i:03d}.png")
            seg = data["instance_id_segmentation"]["data"].astype(np.uint32)
            rng = np.random.default_rng(0)
            lut = rng.integers(0, 255, size=(int(seg.max()) + 1, 3), dtype=np.uint8)
            Image.fromarray(lut[seg]).save(f"{args.out}/inst_{i:03d}.png")
            n_inst = len(np.unique(seg))

    ft = np.array(frame_times[5:])  # skip warm-up frames
    result = {
        "scene": args.scene,
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "load_s": round(load_s, 1),
        "fps_mean": round(1.0 / ft.mean(), 2),
        "frame_ms_p50": round(1000 * float(np.median(ft)), 1),
        "frame_ms_p95": round(1000 * float(np.percentile(ft, 95)), 1),
        "vram_peak_mib": vram_peak,
        "instances_last_sample": int(n_inst),
    }
    with open(f"{args.out}/result_{args.scene}.json", "w") as f:
        json.dump(result, f, indent=2)
    print("S3_RESULT", json.dumps(result))


try:
    main()
finally:
    simulation_app.close()
