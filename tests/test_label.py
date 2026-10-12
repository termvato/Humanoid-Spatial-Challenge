import numpy as np

from roomdiff.label import best_views, changed, closest_poses, crop, parse


def test_parse():
    assert parse("Black suitcase.") == "black suitcase"
    assert parse("A printer") == "printer"
    assert parse("none") is None
    assert parse("None, it is a wall.") is None
    assert parse("") is None


def test_best_views_largest_and_spread():
    m = {f: np.zeros((10, 10), bool) for f in range(20)}
    for f, mask in m.items():
        mask[: f % 10 + 1] = True  # frames 9 and 19 largest, then 8 and 18
    assert best_views(m, k=2) == [9, 19]
    assert len(best_views(m, k=3)) == 3


def test_crop_contains_outline_and_upscales():
    rgb = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros((480, 640), bool)
    mask[200:220, 300:330] = True
    c = crop(rgb, mask)
    assert min(c.shape[:2]) >= 224
    assert (c[..., 0] == 255).any()  # red outline drawn


def test_changed():
    assert changed("Changed.") is True and changed("same") is False and changed("maybe") is None


def test_closest_poses_prefers_near_and_aligned():
    T = np.tile(np.eye(4), (4, 1, 1))
    T[1, 0, 3] = 0.1  # close, same direction
    T[2, 0, 3] = 2.0  # far
    T[3, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # same place, looking sideways
    assert list(closest_poses(T, 0, np.array([3, 2, 1]), scale=1.0)) == [1, 3, 2]
