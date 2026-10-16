import json

import numpy as np

from roomdiff.view3d import page_data, write_page


def _result(tmp_path):
    box = {"centre": [1.0, 2.0], "bottom": 0.0, "top": 0.5, "size": [0.3, 0.4]}
    chs = [{"id": 0, "kind": "added", "before": None, "after": box, "distance": None, "blobs": [5],
            "label": None, "confident": False},
           {"id": 1, "kind": "moved", "before": box, "after": box, "distance": 0.7, "blobs": [2, 3], "was": [2],
            "label": "suitcase", "confident": True}]
    (tmp_path / "changes.json").write_text(json.dumps(chs))
    xyz = np.array([[0.0, 0.0, 0.0], [1.2345, 0.5, -2.0]], np.float32)
    np.savez(tmp_path / "points.npz", before_xyz=xyz, before_rgb=np.zeros((2, 3), np.uint8),
             after_xyz=xyz, after_rgb=np.full((2, 3), 255, np.uint8))
    (tmp_path / "crops").mkdir()
    (tmp_path / "crops" / "1_2_0.jpg").write_bytes(b"jpg")
    return tmp_path


def test_page_data_orders_and_encodes(tmp_path):
    d = page_data(_result(tmp_path), "t")
    assert [c["id"] for c in d["changes"]] == [1, 0]  # confident first
    assert [t["side"] for t in d["changes"][0]["thumbs"]] == ["was"]  # the "now" crop does not exist
    import base64
    mm = np.frombuffer(base64.b64decode(d["points"]["before"]["xyz"]), "<i2")
    assert list(mm[3:]) == [1234, 500, -2000] or list(mm[3:]) == [1235, 500, -2000]
    assert d["subtitle"].startswith("1 moved, 0 added")


def test_write_page_embeds_data(tmp_path):
    p = write_page(_result(tmp_path), "t")
    html = p.read_text()
    assert "/*DATA*/" not in html and '"suitcase"' in html


def test_page_with_no_changes(tmp_path):
    r = _result(tmp_path)
    (r / "changes.json").write_text("[]")
    d = page_data(r, "t")
    assert d["changes"] == [] and d["subtitle"].startswith("0 moved, 0 added")
    assert "/*DATA*/" not in write_page(r, "t").read_text()
