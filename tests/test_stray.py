"""Tests for the Stray Scanner loader, on synthetic capture folders."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roomdiff.io.stray import arkit_to_opencv, backproject, load_stray, scale_intrinsics

HEADER_NEW = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"
HEADER_OLD = "timestamp, frame, x, y, z, qx, qy, qz, qw"
RGB = (1920, 1440)


def _write_capture(root: Path, header: str, rows: list[list[float]], K0: np.ndarray | None = None) -> None:
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    lines = [header] + [", ".join("" if v is None else str(v) for v in r) for r in rows]
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")
    if K0 is not None:
        np.savetxt(root / "camera_matrix.csv", K0, delimiter=",")


def test_identity_pose_flips_to_opencv_axes():
    """ARKit camera looks down -z; in OpenCV axes it must look down +z."""
    T = arkit_to_opencv(np.eye(4))
    forward_cv = T[:3, 2]  # camera z axis in world
    np.testing.assert_allclose(forward_cv, [0, 0, -1])  # world -z, same physical direction as ARKit forward
    np.testing.assert_allclose(T[:3, 1], [0, -1, 0])  # OpenCV y is down: world -y when ARKit y is up


def test_conversion_is_involution():
    rng = np.random.default_rng(0)
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.normal(size=3)
    np.testing.assert_allclose(arkit_to_opencv(arkit_to_opencv(T)), T, atol=1e-12)
    np.testing.assert_allclose(arkit_to_opencv(T)[:3, 3], T[:3, 3])  # position unchanged


def test_load_new_format_per_frame_intrinsics(tmp_path):
    q = Rotation.from_euler("y", 90, degrees=True).as_quat()  # x, y, z, w
    rows = [
        [0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 1, 1400, 1401, 960, 720, None, None],
        [0.1, 1, 1.0, 2.0, 3.0, *q, 1410, 1411, 961, 721, None, None],
    ]
    _write_capture(tmp_path, HEADER_NEW, rows)
    cap = load_stray(tmp_path, rgb_size=RGB)
    assert len(cap) == 2
    np.testing.assert_array_equal(cap.frames, [0, 1])
    np.testing.assert_allclose(cap.T_wc[1, :3, 3], [1, 2, 3])
    assert cap.K[1, 0, 0] == 1410 and cap.K[1, 1, 2] == 721
    # Stray poses are used as stored (already OpenCV camera axes)
    np.testing.assert_allclose(cap.T_wc[1, :3, :3], Rotation.from_quat(q).as_matrix(), atol=1e-12)


def test_load_old_format_falls_back_to_camera_matrix(tmp_path):
    K0 = np.array([[1500.0, 0, 950], [0, 1500, 710], [0, 0, 1]])
    _write_capture(tmp_path, HEADER_OLD, [[0.0, 0, 0, 0, 0, 0, 0, 0, 1]], K0=K0)
    cap = load_stray(tmp_path, rgb_size=RGB)
    np.testing.assert_allclose(cap.K[0], K0)


def test_scale_intrinsics_to_depth_resolution():
    K = np.array([[1440.0, 0, 960], [0, 1440, 720], [0, 0, 1]])
    Kd = scale_intrinsics(K, 256 / 1920, 192 / 1440)
    np.testing.assert_allclose(Kd, [[192, 0, 128], [0, 192, 96], [0, 0, 1]])


def test_depth_and_confidence_read_in_metres(tmp_path):
    _write_capture(tmp_path, HEADER_OLD, [[0.0, 7, 0, 0, 0, 0, 0, 0, 1]], K0=np.eye(3))
    mm = np.full((192, 256), 1234, np.uint16)
    cv2.imwrite(str(tmp_path / "depth" / "000007.png"), mm)
    cv2.imwrite(str(tmp_path / "confidence" / "000007.png"), np.full((192, 256), 2, np.uint8))
    cap = load_stray(tmp_path, rgb_size=RGB)
    assert cap.depth(0).dtype == np.float32
    np.testing.assert_allclose(cap.depth(0), 1.234, rtol=1e-6)
    assert cap.confidence(0).max() == 2


def test_missing_depth_raises(tmp_path):
    _write_capture(tmp_path, HEADER_OLD, [[0.0, 3, 0, 0, 0, 0, 0, 0, 1]], K0=np.eye(3))
    cap = load_stray(tmp_path, rgb_size=RGB)
    with pytest.raises(FileNotFoundError):
        cap.depth(0)


def test_backproject_flat_wall_lands_in_front_of_camera():
    """A wall 2 m ahead of an ARKit camera at the origin is at world z = -2."""
    K = np.array([[192.0, 0, 128], [0, 192, 96], [0, 0, 1]])
    depth = np.full((192, 256), 2.0, np.float32)
    T_wc = arkit_to_opencv(np.eye(4))
    pts = backproject(depth, K, T_wc)
    assert pts.shape == (192 * 256, 3)
    np.testing.assert_allclose(pts[:, 2], -2.0, atol=1e-6)
    # principal point pixel maps straight ahead
    centre = pts[96 * 256 + 128]
    np.testing.assert_allclose(centre, [0, 0, -2], atol=1e-6)
    # image top row (v=0) is physically up, i.e. world +y for an upright ARKit camera
    assert pts[0, 1] > 0


def test_backproject_filters_confidence_and_range():
    K = np.array([[192.0, 0, 128], [0, 192, 96], [0, 0, 1]])
    depth = np.full((192, 256), 1.0, np.float32)
    depth[0, 0] = 0.0  # invalid
    depth[0, 1] = 9.0  # too far
    conf = np.full((192, 256), 2, np.uint8)
    conf[0, 2] = 1  # low confidence
    pts = backproject(depth, K, np.eye(4), conf=conf, min_conf=2, max_depth=5.0)
    assert len(pts) == 192 * 256 - 3


PAIR01 = Path(__file__).parents[1] / "data" / "raw" / "pair01" / "before"


@pytest.mark.skipif(not PAIR01.exists(), reason="real capture not present (data/raw is untracked)")
def test_real_capture_frames_agree_in_world():
    """Regression for the pose convention: depth from frames 1 s apart must land on the same surfaces."""
    from scipy.spatial import cKDTree

    cap = load_stray(next(p for p in PAIR01.iterdir() if (p / "odometry.csv").exists()))
    d = []
    for i in (100, 500, 900):
        a, b = (backproject(cap.depth(k), cap.K_depth(k), cap.T_wc[k], cap.confidence(k), max_depth=3.0)
                for k in (i, i + 30))
        d.append(np.median(cKDTree(b).query(a[::10])[0]))
    assert np.median(d) < 0.05
