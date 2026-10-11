"""Tests for keyframe selection on synthetic images and voxel sets."""

import cv2
import numpy as np

from roomdiff.keyframes import finger_mask, quality, select_coverage, select_windows, sharpness


def _texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (640, 480, 3), dtype=np.uint8)


def test_blur_lowers_sharpness():
    img = _texture()
    assert sharpness(cv2.GaussianBlur(img, (21, 21), 6)) < 0.1 * sharpness(img)


def test_finger_mask_finds_blurry_skin_blob_only():
    img = _texture()
    finger = img.copy()
    cv2.circle(finger, (240, 640), 150, (120, 150, 220), -1)  # BGR skin tone, bottom centre
    finger = np.where(finger == (120, 150, 220), cv2.GaussianBlur(finger, (31, 31), 10), finger)
    m = finger_mask(finger)
    h = m.shape[0]
    assert m[-h // 8 :, m.shape[1] // 2 - 10 : m.shape[1] // 2 + 10].mean() > 0.8
    assert m[: h // 2].mean() < 0.01
    assert finger_mask(img).mean() < 0.01  # sharp random texture is never a finger


def test_quality_penalises_occlusion_and_uses_rank():
    q = quality(np.array([10.0, 1000.0, 50.0]), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(q, [0.0, 1.0, 0.5])
    q = quality(np.array([10.0, 1000.0, 50.0]), np.array([0.0, 0.2, 0.0]))
    assert q[1] == 0.0


def test_select_windows_one_per_window():
    q = np.zeros(100)
    q[[3, 40, 77]] = 1.0
    np.testing.assert_array_equal(select_windows(q, 3), [3, 40, 77])


def test_select_coverage_prefers_new_voxels_over_repeats():
    seen = [np.arange(0, 100)] * 20 + [np.arange(100, 150)] * 20
    q = np.ones(40)
    picks = select_coverage(seen, q, n=2, min_gap=1)
    assert picks[0] < 20 and picks[1] >= 20  # second pick covers the other half of the room


def test_select_coverage_respects_quality_and_gap():
    seen = [np.arange(0, 100)] * 10
    q = np.linspace(0.1, 1.0, 10)
    picks = select_coverage(seen, q, n=3, min_gap=3)
    assert picks[-1] == 9  # best quality first
    assert len(picks) == 1  # identical views add nothing after the first


def test_select_coverage_stops_when_gap_excludes_all():
    seen = [np.array([i]) for i in range(5)]
    picks = select_coverage(seen, np.ones(5), n=5, min_gap=10)
    assert len(picks) == 1
