"""The upload page: drop a before and an after video, press Run, watch progress, open the 3D viewer.

`make_app(jobs)` builds the FastAPI app around a job runner: `LocalJobs` runs roomdiff.run on this
machine's GPU (`roomdiff serve`); infra/modal_app.py plugs in a runner that spawns the same pipeline
on a Modal GPU (the public page). Only the standard library and FastAPI are imported at module level
(FastAPI resolves the upload parameters' annotations from module globals), so the web container stays small.
"""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

PAGE = Path(__file__).with_name("upload.html")
RESULT_FILES = {"viewer.html": "text/html", "summary.json": "application/json", "changes.json": "application/json",
                "map.png": "image/png", "changes.jpg": "image/jpeg"}
MAX_MB = 200


class Jobs(Protocol):
    def start(self, video1: Path, video2: Path) -> str: ...
    def status(self, job: str) -> dict | None: ...
    def file(self, job: str, name: str) -> bytes | None: ...


class LocalJobs:
    """Runs each job in a thread, one at a time (one GPU), results under `root/<job>`."""

    def __init__(self, root: Path, **run_kwargs):
        self.root, self.kwargs = root, run_kwargs
        self.state: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, video1: Path, video2: Path) -> str:
        job = uuid.uuid4().hex[:12]
        self.state[job] = {"state": "queued", "stage": "waiting for the GPU"}
        threading.Thread(target=self._run, args=(job, video1, video2), daemon=True).start()
        return job

    def _run(self, job: str, video1: Path, video2: Path) -> None:
        from roomdiff.run import RoomMismatch, run_pair

        st = self.state[job]
        with self.lock:
            st["state"] = "running"
            try:
                s = run_pair(video1, video2, self.root / job, progress=lambda m: st.update(stage=m), **self.kwargs)
                st.update(state="done", summary=s)
            except RoomMismatch as e:
                st.update(state="refused", error=str(e))
            except Exception as e:  # noqa: BLE001 (shown on the page; the server keeps running)
                st.update(state="failed", error=f"{type(e).__name__}: {e}")

    def status(self, job: str) -> dict | None:
        return self.state.get(job)

    def file(self, job: str, name: str) -> bytes | None:
        p = self.root / job / name
        return p.read_bytes() if p.exists() else None


def make_app(jobs: Jobs, examples: Path | None = None, note: str = ""):
    """FastAPI app: / (page), POST /api/run, GET /api/job/<id>, GET /r/<id>/<file>, /examples/<file>."""
    app = FastAPI(title="roomdiff")

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        ex = sorted(p.stem for p in examples.glob("*.html")) if examples and examples.exists() else []
        return PAGE.read_text().replace("/*CONFIG*/", json.dumps({"examples": ex, "note": note, "max_mb": MAX_MB}))

    @app.post("/api/run")
    async def run(video1: UploadFile, video2: UploadFile) -> dict:
        tmp = Path(tempfile.mkdtemp(prefix="roomdiff_"))
        paths = []
        for i, f in enumerate((video1, video2), 1):
            data = await f.read()
            if len(data) > MAX_MB * 1e6:
                raise HTTPException(413, f"video {i} is over {MAX_MB} MB")
            if not data:
                raise HTTPException(400, f"video {i} is empty")
            p = tmp / f"video{i}{Path(f.filename or '.mp4').suffix.lower() or '.mp4'}"
            p.write_bytes(data)
            paths.append(p)
        try:
            return {"job": jobs.start(*paths)}
        except RuntimeError as e:  # e.g. the public page's daily cap
            raise HTTPException(429, str(e)) from e

    @app.get("/api/job/{job}")
    def job_status(job: str) -> dict:
        s = jobs.status(job)
        if s is None:
            raise HTTPException(404, "unknown job")
        return s

    @app.get("/r/{job}/{name}")
    def result(job: str, name: str) -> Response:
        if name not in RESULT_FILES:
            raise HTTPException(404)
        data = jobs.file(job, name)
        if data is None:
            raise HTTPException(404)
        return Response(data, media_type=RESULT_FILES[name])

    @app.get("/examples/{name}", response_class=HTMLResponse)
    def example(name: str) -> str:
        p = (examples / f"{name}.html") if examples else None
        if not p or not p.exists() or p.parent != examples:
            raise HTTPException(404)
        return p.read_text()

    return app
