"""`roomdiff` command line.

  roomdiff run before.mp4 after.mp4 --out results/   everything on this machine's GPU (exit 3: different rooms)
  roomdiff run before.mp4 after.mp4 --out results/ --modal   the same on a Modal GPU (infra/modal_app.py deployed)
  roomdiff serve [--modal]                          the upload page on http://127.0.0.1:8000
  roomdiff view results/                            rebuild the 3D viewer page of a result
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EXIT_MISMATCH = 3


def _run(args) -> int:
    if args.modal:
        from roomdiff.remote import ModalJobs

        jobs = ModalJobs(max_frames=args.max_frames or 150)
        job = jobs.start(args.video1, args.video2)
        print(f"job {job} on Modal")
        last = None
        while True:
            s = jobs.status(job) or {}
            if s.get("stage") != last:
                last = s.get("stage")
                print(last, flush=True)
            if s.get("state") in ("done", "refused", "failed"):
                break
            time.sleep(3)
        jobs.download(job, args.out)
        if s["state"] == "refused":
            print(s["error"], file=sys.stderr)
            return EXIT_MISMATCH
        if s["state"] == "failed":
            print(s["error"], file=sys.stderr)
            return 1
        summary = s["summary"]
    else:
        from roomdiff.run import RoomMismatch, run_pair

        try:
            summary = run_pair(args.video1, args.video2, args.out, labels=not args.no_labels,
                               max_frames=args.max_frames, keep_frames=args.keep_frames)
        except RoomMismatch as e:
            print(e, file=sys.stderr)
            return EXIT_MISMATCH
    if "warning" in summary:
        print("WARNING:", summary["warning"], file=sys.stderr)
    for c in summary["changes"]:
        if c["confident"]:
            print(f"#{c['id']} {c['kind']}: {c['label'] or 'unnamed'}"
                  + (f", moved {c['distance']:.2f} m" if c["kind"] == "moved" else ""))
    print(f"open {args.out / 'viewer.html'}")
    return 0


def _serve(args) -> int:
    import uvicorn

    from roomdiff.web import LocalJobs, make_app

    if args.modal:
        from roomdiff.remote import ModalJobs

        jobs = ModalJobs()
    else:
        jobs = LocalJobs(args.out, labels=not args.no_labels, max_frames=args.max_frames)
    examples = Path(__file__).resolve().parents[2] / "examples"
    uvicorn.run(make_app(jobs, examples=examples), host=args.host, port=args.port)
    return 0


def _view(args) -> int:
    from roomdiff.view3d import write_page

    print(write_page(args.result, args.title or args.result.name))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roomdiff", description="What changed between two videos of a room, in 3D.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="two videos -> change list, robot maps and a 3D viewer page")
    r.add_argument("video1", type=Path, help="before")
    r.add_argument("video2", type=Path, help="after")
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("serve", help="upload page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--out", type=Path, default=Path("outputs/web"), help="where local jobs write results")
    for p in (r, s):
        p.add_argument("--modal", action="store_true", help="run on a Modal GPU instead of this machine")
        p.add_argument("--no-labels", action="store_true", help="skip the VLM (no names, every change confident)")
        p.add_argument("--max-frames", type=int, default=0,
                       help="cap frames per video for pi3 (0 = every 10th frame; ~150 fits 24 GB)")
    r.add_argument("--keep-frames", action="store_true", help="keep the extracted video frames")
    v = sub.add_parser("view", help="rebuild the 3D viewer page of a result folder")
    v.add_argument("result", type=Path)
    v.add_argument("--title", default="")
    args = ap.parse_args(argv)
    return {"run": _run, "serve": _serve, "view": _view}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
