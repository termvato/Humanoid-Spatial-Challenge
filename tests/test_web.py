from fastapi.testclient import TestClient

from roomdiff.web import make_app


class FakeJobs:
    def __init__(self):
        self.started = []

    def start(self, video1, video2):
        self.started.append((video1.read_bytes(), video2.read_bytes()))
        if len(self.started) > 1:
            raise RuntimeError("daily limit")
        return "job1"

    def status(self, job):
        return {"state": "done", "stage": "done"} if job == "job1" else None

    def file(self, job, name):
        return b"<html>3d</html>" if (job, name) == ("job1", "viewer.html") else None


def test_upload_run_status_result(tmp_path):
    (tmp_path / "pair01.html").write_text("<p>example</p>")
    jobs = FakeJobs()
    c = TestClient(make_app(jobs, examples=tmp_path))
    page = c.get("/").text
    assert "Before video" in page and '"pair01"' in page
    files = {"video1": ("a.mp4", b"one"), "video2": ("b.mov", b"two")}
    assert c.post("/api/run", files=files).json() == {"job": "job1"}
    assert jobs.started == [(b"one", b"two")]
    assert c.get("/api/job/job1").json()["state"] == "done"
    assert c.get("/api/job/nope").status_code == 404
    assert c.get("/r/job1/viewer.html").text == "<html>3d</html>"
    assert c.get("/r/job1/recon.npz").status_code == 404  # only listed result files
    assert c.post("/api/run", files=files).status_code == 429  # runner refused (cap)
    assert c.get("/examples/pair01").text == "<p>example</p>"
    assert c.get("/examples/..%2Fsecret").status_code == 404


def test_empty_upload_rejected():
    c = TestClient(make_app(FakeJobs()))
    assert c.post("/api/run", files={"video1": ("a.mp4", b""), "video2": ("b.mp4", b"x")}).status_code == 400
