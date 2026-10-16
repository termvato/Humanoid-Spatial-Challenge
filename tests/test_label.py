import numpy as np

from roomdiff.label import best_views, changed, closest_poses, crop, majority, parse


def test_parse():
    assert parse("Black suitcase.") == "black suitcase"
    assert parse("A printer") == "printer"
    assert parse("none") is None
    assert parse("None, it is a wall.") is None
    assert parse("") is None
    assert parse("White wall.") is None and parse("door frame") is None and parse("floor") is None
    assert parse("hat on shelf") == "hat on shelf" and parse("suitcase handle") == "suitcase handle"
    assert parse("wallet") == "wallet" and parse("printer in box") == "printer in box"


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
    assert (c[..., 0] == 255).any()  # red box drawn


def test_changed():
    assert changed("Yes.") is True and changed("no") is False and changed("maybe") is None


def test_closest_poses_prefers_near_and_aligned():
    T = np.tile(np.eye(4), (4, 1, 1))
    T[1, 0, 3] = 0.1  # close, same direction
    T[2, 0, 3] = 2.0  # far
    T[3, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # same place, looking sideways
    assert list(closest_poses(T, 0, np.array([3, 2, 1]), scale=1.0)) == [1, 3, 2]


def test_best_views_skip_border_and_blur():
    rng = np.random.default_rng(0)
    rgb = np.zeros((3, 40, 40, 3), np.uint8)
    rgb[0] = rng.integers(0, 255, (40, 40, 3))  # sharp, small mask
    rgb[1] = 128  # flat (blurry), large mask
    rgb[2] = rng.integers(0, 255, (40, 40, 3))  # sharp, largest mask but cut by the image border
    m = {f: np.zeros((40, 40), bool) for f in range(3)}
    m[0][10:20, 10:20] = True
    m[1][5:35, 5:35] = True
    m[2][0:38, 0:38] = True
    assert best_views(m, k=1) == [2]  # size only
    assert best_views(m, k=1, rgb=rgb) == [0]


def test_majority():
    assert majority([True, False, False]) is False
    assert majority([True, False]) is True  # tie counts as changed
    assert majority([None, True]) is True and majority([None]) is None and majority([]) is None



def test_label_result_with_no_changes_asks_nothing(tmp_path):
    from roomdiff.labelling import label_result

    (tmp_path / "changes.json").write_text("[]")

    def ask(items, prompt):
        raise AssertionError("no VLM call expected")
    assert label_result(tmp_path, ask) == []
    assert (tmp_path / "labels.json").exists()
