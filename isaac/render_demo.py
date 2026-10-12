"""Render the demo plan (tools/demo.py plan): robot and props per frame, from an overview camera.

Run: ~/isaacsim/python.sh isaac/render_demo.py --plan outputs/sim/room01/result/demo_plan.json \
        --out outputs/sim/room01/result/frames
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--plan", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

import numpy as np
import omni.replicator.core as rep
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import scene

CAMERA = ((0.0, -1.6, 3.3), (0.0, 0.6, scene.FLOOR_Z))  # under the ceiling, wide: the whole room in view


def main() -> None:
    os.makedirs(args.out, exist_ok=True)
    with open(args.plan) as f:
        frames = json.load(f)["frames"]
    scene.load(app)
    robot = scene.add_robot(app)
    cam = rep.create.camera(position=CAMERA[0], look_at=CAMERA[1], focal_length=7.0)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rep.create.render_product(cam, (args.width, args.height)))
    last = None
    with open(f"{args.out}/progress.log", "w") as log:
        for i, fr in enumerate(frames):
            if fr["props"] != last:
                scene.place(app, {k: tuple(v) if v else None for k, v in fr["props"].items()})
                last = fr["props"]
            scene.set_robot(robot, *fr["robot"])
            rep.orchestrator.step(rt_subframes=2)
            Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(f"{args.out}/{i:05d}.png")
            if i % 50 == 0:
                log.write(f"frame {i} of {len(frames)}\n")
                log.flush()


try:
    main()
finally:
    app.close()
