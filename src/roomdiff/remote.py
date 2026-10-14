"""Jobs on Modal: the same roomdiff.run pipeline on a cloud GPU (infra/modal_app.py, deployed once).

Videos go into the `roomdiff-results` volume under <job>/input/, the GPU function `pipeline` runs
roomdiff.run there and writes the result files next to them (and deletes the videos); progress is
kept in the `roomdiff-jobs` dict. Used by the public page and by `roomdiff run/serve --modal`.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path

APP = "roomdiff-app"
RESULTS = "roomdiff-results"
JOBS = "roomdiff-jobs"


class ModalJobs:
    def __init__(self, daily_cap: int = 0, max_frames: int = 150):
        import modal

        self.vol = modal.Volume.from_name(RESULTS, create_if_missing=True)
        self.jobs = modal.Dict.from_name(JOBS, create_if_missing=True)
        self.fn = modal.Function.from_name(APP, "pipeline")
        self.cap, self.max_frames = daily_cap, max_frames

    def start(self, video1: Path, video2: Path) -> str:
        if self.cap:
            key = f"count:{datetime.date.today().isoformat()}"
            n = self.jobs.get(key, 0)
            if n >= self.cap:
                raise RuntimeError(f"The public demo has reached today's limit of {self.cap} runs. "
                                   "Please try tomorrow, or run it yourself (see the README).")
            self.jobs[key] = n + 1
        job = uuid.uuid4().hex[:12]
        names = [f"video{i}{Path(v).suffix.lower() or '.mp4'}" for i, v in enumerate((video1, video2), 1)]
        with self.vol.batch_upload() as b:
            for v, n in zip((video1, video2), names, strict=True):
                b.put_file(str(v), f"{job}/input/{n}")
        self.jobs[job] = {"state": "queued", "stage": "waiting for a cloud GPU"}
        self.fn.spawn(job, names[0], names[1], self.max_frames)
        return job

    def status(self, job: str) -> dict | None:
        return self.jobs.get(job)

    def file(self, job: str, name: str) -> bytes | None:
        from modal.exception import NotFoundError

        try:
            return b"".join(self.vol.read_file(f"{job}/{name}"))
        except (FileNotFoundError, NotFoundError):  # not there (yet)
            return None

    def download(self, job: str, out: Path) -> None:
        """Copy every result file of `job` into `out`."""
        out.mkdir(parents=True, exist_ok=True)
        for e in self.vol.listdir(job, recursive=True):
            if e.type.name == "FILE":
                p = out / Path(e.path).relative_to(job)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"".join(self.vol.read_file(e.path)))
