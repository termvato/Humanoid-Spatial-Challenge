"""Two videos in, results out: the whole pipeline on this machine's GPU.

frames (ffmpeg) -> room check (S22: refuse two different rooms) -> pi3 on both videos (recon.npz)
-> 3D change map, SAM outlines, moved matching, robot maps (roomdiff.analyse) -> names and
before/after verdicts (Qwen2.5-VL, roomdiff.labelling) -> 3D viewer page (viewer.html).
`summary.json` holds the outcome; on refusal it holds the error and RoomMismatch is raised.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


class RoomMismatch(Exception):
    """The two videos do not show the same room (S22)."""


def room_score(frames1: list[Path], frames2: list[Path], n: int = 40) -> float:
    """S22 image score on up to `n` evenly spaced frames per video."""
    import cv2

    from roomdiff.roomcheck import frame_features, overlap_score

    def feats(fl):
        idx = np.linspace(0, len(fl) - 1, min(n, len(fl))).round().astype(int)
        return frame_features([cv2.cvtColor(cv2.imread(str(fl[i])), cv2.COLOR_BGR2RGB) for i in idx])
    return overlap_score(feats(frames1), feats(frames2))


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_pair(video1: Path, video2: Path, out: Path, labels: bool = True, stride: int = 10, max_frames: int = 0,
             title: str = "", progress: Callable[[str], None] = print, keep_frames: bool = False) -> dict:
    """Run everything on `video1` (before) and `video2` (after), writing into `out`. Returns summary.json."""
    from roomdiff.analyse import analyse
    from roomdiff.recon import reconstruct, video_frames
    from roomdiff.roomcheck import REFUSE_BELOW
    from roomdiff.view3d import write_page

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {"video1": Path(video1).name, "video2": Path(video2).name}

    def save():
        summary["seconds"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    progress("reading frames")
    fdir = out / "frames"
    f1, f2 = video_frames(video1, fdir / "video1"), video_frames(video2, fdir / "video2")
    progress("checking both videos show the same room")
    summary["room_score"] = round(room_score(f1, f2), 3)
    _free_gpu()
    if summary["room_score"] < REFUSE_BELOW:
        summary["error"] = (f"The two videos do not seem to show the same room (image match {summary['room_score']:.2f}, "
                            f"same-room pairs score >= 0.63, refused below {REFUSE_BELOW}). Record both videos in one room, "
                            "with overlapping views.")
        save()
        if not keep_frames:
            shutil.rmtree(fdir, ignore_errors=True)
        raise RoomMismatch(summary["error"])
    progress("3D reconstruction of both videos (pi3)")
    recon = reconstruct(f1, f2, out / "recon.npz", stride=stride, max_frames=max_frames)
    if not keep_frames:
        shutil.rmtree(fdir, ignore_errors=True)
    _free_gpu()
    res = analyse(recon, out, progress=progress)
    _free_gpu()
    if labels:
        from roomdiff.labelling import label_result
        from roomdiff.vlm import LocalVLM

        progress("naming the changes (VLM)")
        vlm = LocalVLM()
        label_result(out, vlm.ask)
        vlm.close()
        _free_gpu()
    progress("3D viewer page")
    write_page(out, title or f"{summary['video1']} -> {summary['video2']}")
    chs = json.loads((out / "changes.json").read_text())
    summary.update(overlap_3d=res.get("overlap_3d"), scale_m_per_unit=res["scale_m_per_unit"],
                   changes=[{k: c.get(k) for k in ("id", "kind", "label", "before", "after", "distance")}
                            | {"confident": c.get("confident", True)} for c in chs])  # unlabelled: all confident
    if "warning" in res:
        summary["warning"] = res["warning"]
    save()
    progress("done")
    return summary
