"""Robot demo for the simulated room, in three steps (Isaac only renders):

  plan   score the detections against ground truth, then plan frames: the robot drives
         (A* on our after map) to each suggested viewpoint and looks, then the room is put
         back using our *detected* positions (moved -> back to where it was, added -> taken
         away, removed -> returned). Writes demo_plan.json and eval.json.
  render ~/isaacsim/python.sh isaac/render_demo.py --plan <dir>/demo_plan.json --out <dir>/frames
  video  frames + map inset + captions -> demo.mp4

Usage:
  uv run python tools/demo.py plan  --sim outputs/sim/room01 --result outputs/sim/room01/result
  uv run python tools/demo.py video --result outputs/sim/room01/result
"""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from roomdiff.grid2d import load_ros
from roomdiff.plan import astar, drivable, nearest_drivable, to_cell, to_xy
from roomdiff.simeval import evaluate, map_to_world, world_to_map

FLOOR_Z = -0.77  # isaac/scene.py
START_WORLD = (0.0, -2.4)  # between the table and the front wall
FPS, SPEED, TURN = 15, 0.6, np.radians(120)  # m/s, rad/s
PX = 4  # analyse.py map.png pixels per cell


def _yaw_world(T: np.ndarray, yaw_map: float) -> float:
    """Map-frame heading -> world heading (map (x, y) = level (x, -z))."""
    d = np.linalg.inv(T)[:3, :3] @ [np.cos(yaw_map), 0.0, -np.sin(yaw_map)]
    return float(np.arctan2(d[1], d[0]))


def _turn(yaw0: float, yaw1: float) -> list[float]:
    d = float(np.angle(np.exp(1j * (yaw1 - yaw0))))
    n = max(1, int(abs(d) / TURN * FPS))
    return [yaw0 + d * (k + 1) / n for k in range(n)]


def plan(args) -> None:
    res, sim = args.result, args.sim
    T = np.array(json.loads((res / "result.json").read_text())["world_to_level"])
    changes = json.loads((res / "changes.json").read_text())
    vps = json.loads((res / "viewpoints.json").read_text())
    gt = json.loads((sim / "gt.json").read_text())
    score = evaluate(changes, gt, T, FLOOR_Z, look_targets=[tuple(v["target"]) for v in vps])
    (res / "eval.json").write_text(json.dumps(score, indent=2))

    grid = load_ros(res / "after.yaml")
    ok = drivable(grid, 0.3)
    props = {n: (g["visit2"]["pos"] if g["visit2"] else None) for n, g in gt.items()}
    frames: list[dict] = []

    def emit(xy, yaw, caption, phase):
        w = map_to_world(T, xy)
        frames.append({"robot": [float(w[0]), float(w[1]), _yaw_world(T, yaw)], "robot_map": [xy[0], xy[1]],
                       "props": {k: v for k, v in props.items()}, "caption": caption, "phase": phase})

    cur = to_xy(grid, nearest_drivable(ok, to_cell(grid, world_to_map(T, START_WORLD, FLOOR_Z))))
    yaw = 0.0
    for _ in range(FPS * 2):
        emit(cur, yaw, "Visit 2 is done. The map says some places were never re-checked.", "intro")
    todo = list(vps)
    while todo:  # nearest viewpoint next
        v = min(todo, key=lambda v: np.hypot(v["pose"][0] - cur[0], v["pose"][1] - cur[1]))
        todo.remove(v)
        path = astar(ok, to_cell(grid, cur), to_cell(grid, tuple(v["pose"][:2])))
        if path is None:
            continue
        pts = [to_xy(grid, rc) for rc in path[::3]] + [tuple(v["pose"][:2])]
        caption = f"Driving to look at an unobserved spot ({v['cells']} cells)"
        for a, b in pairwise(pts):
            heading = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            for y in _turn(yaw, heading):
                emit(a, y, caption, "drive")
            yaw = heading
            n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) / SPEED * FPS))
            for k in range(n):
                t = (k + 1) / n
                emit((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), yaw, caption, "drive")
        cur = pts[-1]
        for y in _turn(yaw, v["pose"][2]):
            emit(cur, y, caption, "drive")
        yaw = v["pose"][2]
        for _ in range(FPS * 2):
            emit(cur, yaw, f"Looking: {round(100 * v['seen'])}% of the spot is now in view", "look")

    # put the room back, using where we *detected* things
    by_change = {}
    for name, r in score["props"].items():
        for cid in r["detections"]:
            if r["change"] in ("moved", "removed", "added"):
                by_change.setdefault(cid, name)
    stacked = {n: [m for m in gt if m != n and gt[m]["visit2"] and gt[n]["visit2"]
                   and gt[m]["visit2"]["pos"] == gt[n]["visit2"]["pos"] and gt[m]["change"] == gt[n]["change"]]
               for n in gt}
    restored: set[str] = set()
    for c in changes:  # largest first, so a fragmented object is restored from its biggest piece
        name = by_change.get(c["id"])
        if name is None or name in restored:
            continue
        group = [name, *stacked[name]]
        restored.update(group)
        if c["kind"] == "moved":
            src = np.array(props[name])
            dst = map_to_world(T, tuple(c["before"]["centre"]))[:2]
            cap = f"Put back: moved {c['distance']:.1f} m -> return it to where it was"
            for k in range(FPS * 3):
                t = (k + 1) / (FPS * 3)
                for n in group:
                    props[n] = [float(v) for v in src + t * (dst - src)]
                emit(cur, yaw, cap, "restore")
        elif c["kind"] == "added":
            for n in group:
                props[n] = None
            for _ in range(FPS * 2):
                emit(cur, yaw, "Put back: this was added -> take it away", "restore")
        else:
            dst = map_to_world(T, tuple(c["before"]["centre"]))[:2]
            for n in group:
                props[n] = [float(v) for v in dst]
            for _ in range(FPS * 2):
                emit(cur, yaw, "Put back: this was removed -> return it", "restore")
    for _ in range(FPS * 2):
        emit(cur, yaw, "Room restored to visit 1, from the detected changes", "end")
    (res / "demo_plan.json").write_text(json.dumps({"fps": FPS, "frames": frames}))
    print(json.dumps({"frames": len(frames), "seconds": len(frames) / FPS, "found": score["found"],
                      "kind_ok": score["kind_ok"], "changed": score["changed"],
                      "false_alarms": len(score["false_alarms"])}))


def video(args) -> None:
    res = args.result
    p = json.loads((res / "demo_plan.json").read_text())
    grid = load_ros(res / "after.yaml")
    inset = cv2.imread(str(res / "map.png"))
    files = sorted((res / "frames").glob("*.png"))
    first = cv2.imread(str(files[0]))
    H, W = first.shape[:2]
    s = (H // 2) / inset.shape[0]
    tmp = res / "demo_raw.mp4"
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), p["fps"], (W, H))
    for f, fr in zip(files, p["frames"], strict=False):
        img = cv2.imread(str(f))
        m = inset.copy()
        x, y = fr["robot_map"]
        c = int((x - grid.origin[0]) / grid.resolution * PX), int((grid.data.shape[0] - (y - grid.origin[1]) / grid.resolution) * PX)
        cv2.circle(m, c, 3 * PX, (0, 140, 255), -1)
        m = cv2.resize(m, None, fx=s, fy=s)
        img[8:8 + m.shape[0], W - 8 - m.shape[1]:W - 8] = m
        cv2.rectangle(img, (0, H - 40), (W, H), (0, 0, 0), -1)
        cv2.putText(img, fr["caption"], (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(img)
    wr.release()
    out = res / "demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(out)], check=True)  # h264 plays everywhere; mp4v does not
    tmp.unlink()
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "video"])
    ap.add_argument("--sim", type=Path, default=Path("outputs/sim/room01"))
    ap.add_argument("--result", type=Path, default=Path("outputs/sim/room01/result"))
    args = ap.parse_args()
    plan(args) if args.step == "plan" else video(args)


if __name__ == "__main__":
    main()
