"""Record two simulated visits of the test room as posed RGB-D, in our reconstruction format.

The camera stands in for a robot's head camera: 1 m above the floor, 15 deg down, facing
across the room and sweeping left and right while it loops the room. Visit 2 loops the other way on a different radius
and never looks at UNOBSERVED_CORNER. Depth (distance to image plane) and poses come
from the renderer, so they are exact and metric.

Run: ~/isaacsim/python.sh isaac/record_visits.py --out outputs/sim/room01
Writes: visits.npz (depth, conf, K, T_wc OpenCV camera-to-world, rgb, video, src: the
format roomdiff.frames.from_recon_npz reads), gt.json (prop positions and boxes per
visit, expected change per prop), check.json (floor height recovered from depth).
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="outputs/sim/room01")
parser.add_argument("--frames", type=int, default=150, help="frames per visit")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

import numpy as np
import omni.replicator.core as rep

sys.path.insert(0, os.path.dirname(__file__))
import scene

CAM_Z = scene.FLOOR_Z + 1.0
PITCH = np.radians(-15)
LOG = None


def log(msg: str) -> None:
    """Kit swallows stdout; progress goes to <out>/progress.log."""
    LOG.write(msg + "\n")
    LOG.flush()


USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # USD camera looks down -z with y up; OpenCV looks down +z, y down


def trajectory(visit: int, n: int):
    """(position, look-at) pairs. Visit 1: counter-clockwise loop; visit 2: clockwise, tighter."""
    cx, cy, rx, ry, sign = (0.0, 0.7, 2.7, 2.3, 1) if visit == 1 else (0.0, 0.8, 2.3, 1.9, -1)
    corner = np.array(scene.UNOBSERVED_CORNER)
    for i in range(n):
        th = sign * 2 * np.pi * i / n
        p = np.array([cx + rx * np.cos(th), cy + ry * np.sin(th)])
        yaw = th + np.pi + np.radians(50) * np.sin(3 * th)  # face across the room, sweeping
        if visit == 2:  # keep the corner at least 60 deg off the optical axis (half field of view ~30 deg)
            b = np.arctan2(*(corner - p)[::-1])
            d = np.angle(np.exp(1j * (yaw - b)))
            if abs(d) < np.radians(60):
                yaw = b + np.sign(d or 1) * np.radians(60)
        f = np.array([np.cos(yaw) * np.cos(PITCH), np.sin(yaw) * np.cos(PITCH), np.sin(PITCH)])
        pos = np.array([p[0], p[1], CAM_Z])
        yield pos, pos + f


def main() -> None:
    global LOG
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/progress.log", "w") as LOG:
        record()


def record() -> None:
    log("loading scene")
    scene.load(app)
    log("scene loaded")
    cam = rep.create.camera(position=(0, 0, CAM_Z), look_at=(1, 0, CAM_Z), focal_length=18.0)
    rp = rep.create.render_product(cam, (args.width, args.height))
    ann = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ("rgb", "distance_to_image_plane", "camera_params")}
    for a in ann.values():
        a.attach(rp)
    W, H = args.width, args.height
    depth, rgb, K, T_wc, video, src, gt = [], [], [], [], [], [], {}
    for visit in (1, 2):
        gt[visit] = scene.set_visit(app, visit)
        log(f"visit {visit} props placed: {gt[visit]}")
        for _ in range(10):
            app.update()
        for i, (pos, look) in enumerate(trajectory(visit, args.frames)):
            with cam:
                rep.modify.pose(position=tuple(pos), look_at=tuple(look))
            rep.orchestrator.step(rt_subframes=2)
            d = np.asarray(ann["distance_to_image_plane"].get_data(), np.float32)
            d[~np.isfinite(d)] = 0
            cp = ann["camera_params"].get_data()
            view = np.asarray(cp["cameraViewTransform"], np.float64).reshape(4, 4).T  # world -> USD camera
            proj = np.asarray(cp["cameraProjection"], np.float64).reshape(4, 4).T
            k = np.array([[proj[0, 0] * W / 2, 0, W / 2], [0, proj[1, 1] * H / 2, H / 2], [0, 0, 1]])
            depth.append(d)
            rgb.append(np.asarray(ann["rgb"].get_data())[..., :3])
            K.append(k)
            T_wc.append(np.linalg.inv(USD_TO_CV @ view))
            video.append(visit)
            src.append(i)
            if i % 25 == 0:
                log(f"visit {visit} frame {i}")
    depth, T_wc, K = np.stack(depth), np.stack(T_wc), np.stack(K)
    np.savez_compressed(f"{args.out}/visits.npz", depth=depth.astype(np.float16), conf=np.ones_like(depth, np.float16),
                        K=K.astype(np.float32), T_wc=T_wc.astype(np.float32), rgb=np.stack(rgb).astype(np.uint8),
                        video=np.array(video), src=np.array(src))
    truth = {name: {"change": change, "visit1": gt[1][name], "visit2": gt[2][name]}
             for name, (_, _, _, change) in scene.PROPS.items()}
    with open(f"{args.out}/gt.json", "w") as f:
        json.dump(truth, f, indent=2)
    # self-check: back-project one frame, the lowest points must sit on the floor
    i = 0
    v, u = np.nonzero(depth[i] > 0)
    z = depth[i][v, u]
    pc = np.stack([(u - K[i][0, 2]) * z / K[i][0, 0], (v - K[i][1, 2]) * z / K[i][1, 1], z, np.ones_like(z)])
    pw = (T_wc[i] @ pc)[:3]
    check = {"floor_z_true": scene.FLOOR_Z, "floor_z_from_depth": round(float(np.percentile(pw[2], 2)), 3),
             "fx": round(float(K[0][0, 0]), 1), "frames": len(depth)}
    with open(f"{args.out}/check.json", "w") as f:
        json.dump(check, f, indent=2)


try:
    main()
finally:
    app.close()
