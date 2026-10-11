import numpy as np
from scipy.spatial.transform import Rotation

from roomdiff.metrics import ate, depth_errors, umeyama


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = Rotation.random(random_state=1).as_matrix()
    dst = 2.5 * src @ R.T + [1, -2, 3]
    s, R_est, t = umeyama(src, dst)
    assert abs(s - 2.5) < 1e-9
    np.testing.assert_allclose(R_est, R, atol=1e-9)
    np.testing.assert_allclose(t, [1, -2, 3], atol=1e-9)


def test_ate_zero_for_scaled_copy_and_positive_for_noise():
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.normal(size=(40, 3)), axis=0)
    assert ate(ref * 0.3 + 5, ref)[0] < 1e-9
    noisy = ref + rng.normal(0, 0.1, ref.shape)
    rmse, s, _ = ate(noisy, ref)
    assert 0.05 < rmse < 0.2 and abs(s - 1) < 0.05


def test_depth_errors_median_vs_global_scale():
    ref = np.full((4, 4), 2.0)
    pred = np.full((4, 4), 1.0)
    m = np.ones((4, 4), bool)
    assert depth_errors(pred, ref, m)["abs_rel"] == 0.0  # median scaling fixes it
    e = depth_errors(pred, ref, m, scale=1.0)
    assert abs(e["abs_rel"] - 0.5) < 1e-12 and e["delta1"] == 0.0
