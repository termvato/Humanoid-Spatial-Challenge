import numpy as np

from roomdiff.objects import choose_mask, iou, prompts_from_mask


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_prompts_box_and_points_inside():
    m = _rect(100, 100, 20, 60, 30, 80)
    box, pts = prompts_from_mask(m, n_points=3, pad=0.0)
    np.testing.assert_allclose(box, [30, 20, 79, 59])
    assert 1 <= len(pts) <= 3
    for x, y in pts:
        assert m[int(y), int(x)]


def test_choose_mask_picks_agreeing_candidate_or_none():
    coarse = _rect(50, 50, 10, 30, 10, 30)
    good = _rect(50, 50, 9, 31, 9, 31)
    whole = np.ones((50, 50), bool)
    tiny = _rect(50, 50, 0, 2, 0, 2)
    assert iou(choose_mask(np.stack([whole, good, tiny]), coarse), good) == 1.0
    assert choose_mask(np.stack([tiny, np.zeros_like(whole)]), coarse) is None
