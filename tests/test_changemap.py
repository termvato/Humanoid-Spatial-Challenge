"""End-to-end change map on a rendered synthetic room: floor, back wall, and a box that disappears."""

import numpy as np
from pycocotools import mask as rle

from roomdiff.changemap import blob_mask, build, to_scenediff
from roomdiff.frames import Frames, depth_edges

H, W = 60, 80
K = np.array([[60.0, 0, 40], [0, 60, 30], [0, 0, 1]])


def _render(T_wc: np.ndarray, box: bool) -> np.ndarray:
    """Ray-cast depth: back wall z = 4, floor y = 1, optional box x,y,z in [-0.4,0.4] x [0.2,1] x [2,2.8]."""
    v, u = np.mgrid[0:H, 0:W]
    dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    R, o = T_wc[:3, :3], T_wc[:3, 3]
    dirs = dirs_c @ R.T
    t = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tw = (4.0 - o[2]) / dirs[..., 2]
        t = np.where(tw > 0, np.minimum(t, tw), t)
        tf = (1.0 - o[1]) / dirs[..., 1]
        t = np.where(tf > 0, np.minimum(t, tf), t)
        if box:
            lo, hi = np.array([-0.4, 0.2, 2.0]), np.array([0.4, 1.0, 2.8])
            t0 = (lo - o) / dirs
            t1 = (hi - o) / dirs
            tn = np.minimum(t0, t1).max(-1)
            tx = np.maximum(t0, t1).min(-1)
            hit = (tx >= tn) & (tn > 0)
            t = np.where(hit, np.minimum(t, tn), t)
    return (t * dirs_c[..., 2]).astype(np.float32)  # z depth = t * (dir_c . z)


def _frames(box: bool, video: int) -> Frames:
    Ts = []
    for x in (-0.6, -0.3, 0.0, 0.3, 0.6):
        T = np.eye(4)
        T[:3, 3] = [x, 0.0, 0.0]
        Ts.append(T)
    depth = np.stack([_render(T, box) for T in Ts])
    valid = np.isfinite(depth) & (depth > 0)
    depth[~valid] = 0
    n = len(Ts)
    return Frames(depth, valid, np.tile(K, (n, 1, 1)), np.stack(Ts), np.full(n, video), np.arange(n) * 30)


def test_removed_box_is_one_blob_at_the_box():
    cm = build(_frames(True, 1), _frames(False, 2), voxel=0.1, min_voxels=5)
    removed = [b for b in cm.blobs if b.kind == "removed"]
    assert len(removed) == 1
    assert not [b for b in cm.blobs if b.kind == "added"]
    c = removed[0].centre
    assert abs(c[0]) < 0.2 and 0.2 < c[1] < 1.0 and 1.9 < c[2] < 2.9
    assert cm.counts()["unobserved"] > 0  # wall and floor behind the box were hidden before


def test_blob_projects_onto_box_pixels_and_exports():
    before = _frames(True, 1)
    cm = build(before, _frames(False, 2), voxel=0.1, min_voxels=5)
    b = cm.blobs[0]
    m = blob_mask(before, 2, cm.grid.centres_of(b.idx), cm.grid.size, cm.tau)
    box_px = before.depth[2] < 2.9
    iou = (m & box_px).sum() / (m | box_px).sum()
    assert iou > 0.5
    pred = to_scenediff(cm, before.transformed(np.eye(4)), resample=30)
    assert pred["H"] == H and pred["W"] == W
    entry = pred[b.id]["video_1"]
    assert set(entry) <= {0, 1, 2, 3, 4}
    assert rle.decode({**entry[2]["mask"], "counts": entry[2]["mask"]["counts"].encode()}).shape == (H, W)


def test_depth_edges_marks_discontinuity_only():
    d = np.ones((10, 10), np.float32)
    d[:, 5:] = 2.0
    e = depth_edges(d)
    assert e[:, 4:6].all() and not e[:, :3].any() and not e[:, 7:].any()


def test_conf_drop_keeps_same_share(tmp_path):
    from roomdiff.frames import from_recon_npz

    rng = np.random.default_rng(0)
    n, h, w = 2, 20, 20
    for conf in (rng.uniform(0.0, 1.0, (n, h, w)), rng.uniform(0.6, 0.8, (n, h, w))):
        p = tmp_path / "r.npz"
        np.savez(p, depth=np.ones((n, h, w)), conf=conf, K=np.tile(np.eye(3), (n, 1, 1)),
                 T_wc=np.tile(np.eye(4), (n, 1, 1)), video=np.array([1, 2]), src=np.array([0, 0]))
        f = from_recon_npz(p, conf_drop=0.2)
        assert abs(f.valid.mean() - 0.8) < 0.02
