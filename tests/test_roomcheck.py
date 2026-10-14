import numpy as np

from roomdiff.roomcheck import overlap_score


def test_overlap_score_same_vs_different():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 8))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    assert overlap_score(a, a[::-1]) > 0.99
    b = rng.normal(size=(10, 8))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    assert overlap_score(a, b) < 0.9
    # one-sided overlap (video 2 sees only part of video 1) is limited by the weaker direction
    assert overlap_score(a, a[:2]) < overlap_score(a, a)
