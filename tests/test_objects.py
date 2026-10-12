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


def test_colour_descriptor_normalised_and_masked():
    from roomdiff.objects import colour_descriptor

    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:, :10] = (255, 0, 0)
    rgb[:, 10:] = (0, 0, 255)
    left = _rect(20, 20, 0, 20, 0, 10)
    d = colour_descriptor(rgb, left)
    assert abs(d.sum() - 1) < 1e-6
    assert np.count_nonzero(d) == 1  # only red pixels counted


def test_match_moved_greedy_one_to_one_with_size_gate():
    from roomdiff.objects import match_moved

    red, blue = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    removed = {0: {"desc": red, "voxels": 100}, 1: {"desc": blue, "voxels": 100}}
    added = {5: {"desc": red, "voxels": 90}, 6: {"desc": blue, "voxels": 10}}  # blue one too small
    pairs = match_moved(removed, added)
    assert [(r, a) for r, a, _ in pairs] == [(0, 5)]
