import numpy as np

from roomdiff.recon import intrinsics, sample, target_size


def test_target_size_keeps_aspect_multiple_of_14():
    assert target_size(1920, 1080) == (518, 294)
    assert target_size(1080, 1920) == (294, 518)


def test_sample_stride_and_cap():
    assert sample(100, 10, 0) == list(range(0, 100, 10))
    idx = sample(1000, 10, 50)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 999


def test_intrinsics_recover_pinhole_focal():
    h, w, fx = 294, 518, 400.0
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 2.0 + 0.3 * np.sin(u / 50)  # any depth
    pts = np.stack([(u + 0.5 - w / 2) * z / fx, (v + 0.5 - h / 2) * z / fx, z], -1)[None]
    K = intrinsics(pts, np.ones((1, h, w)))
    assert abs(K[0, 0, 0] - fx) / fx < 0.02 and abs(K[0, 1, 1] - fx) / fx < 0.02
    assert K[0, 0, 2] == w / 2 and K[0, 1, 2] == h / 2
