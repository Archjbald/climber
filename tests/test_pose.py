"""Tests for src.pose."""

from unittest.mock import MagicMock, patch

import numpy as np
from pytest import approx

from src.pose import clean_keypoint, extract_pose, fix_left_right_switches

LEFT = [5, 7, 9, 11, 13, 15]
RIGHT = [6, 8, 10, 12, 14, 16]

@patch("src.pose.Body")
@patch("src.pose.PoseTracker")
def test_get_pose_processing_loop(mock_pose_tracker_class, mock_body):
    mock_tracker_instance = MagicMock()
    mock_pose_tracker_class.return_value = mock_tracker_instance

    mock_tracker_instance.side_effect = [
        (np.array([[[10.0, 20.0]]]), np.array([[0.90]])),  # Frame 1
        (np.array([[[15.0, 25.0]]]), np.array([[0.80]])),  # Frame 2
    ]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 2.0

    mock_cap.read.side_effect = [
        (True, "mock_frame_1"),
        (True, "mock_frame_2"),
        (False, None),
    ]

    to_openpose = False
    keypoints, scores = extract_pose(mock_cap, openpose_skeleton=to_openpose)

    mock_pose_tracker_class.assert_called_once_with(
        mock_body,
        mode="balanced",
        det_frequency=1,
        backend="onnxruntime",
        device="cpu",
        to_openpose=to_openpose,
    )

    assert mock_tracker_instance.call_count == 2
    mock_tracker_instance.assert_any_call("mock_frame_1")
    mock_tracker_instance.assert_any_call("mock_frame_2")

    expected_keypoints = np.array([[[10.0, 20.0]], [[15.0, 25.0]]])
    expected_scores = np.array([[0.90],[ 0.80]])

    assert np.array_equal(keypoints, expected_keypoints)
    assert np.array_equal(scores, expected_scores)


# --- clean_keypoint --------------------------------------------------------


def test_clean_keypoint_preserves_shape():
    coords = np.random.rand(20, 17, 2) * 10.0
    assert clean_keypoint(coords).shape == (20, 17, 2)


def test_clean_keypoint_drops_a_velocity_outlier():
    coords = np.zeros((20, 17, 2))
    coords[:, :, 0] = np.linspace(0, 19, 20)[:, None]  # steady drift on every keypoint
    coords[10, 9, 0] = 500.0  # one impossible jump

    cleaned = clean_keypoint(coords, max_velocity=30.0, window_len=7, poly_order=2)

    assert cleaned[10, 9, 0] == approx(10.0, abs=1.0)


def test_clean_keypoint_zeros_a_fully_missing_keypoint():
    coords = np.zeros((15, 17, 2))
    coords[:, 3, :] = np.nan

    cleaned = clean_keypoint(coords, window_len=0)

    assert np.all(cleaned[:, 3] == 0.0)


# --- fix_left_right_switches ----------------------------------------------


def test_fix_left_right_switches_restores_a_swapped_frame():
    coords = np.zeros((10, 17, 2))
    for i, (left, right) in enumerate(zip(LEFT, RIGHT)):
        coords[:, left, 0] = 1.0 + i
        coords[:, right, 0] = 100.0 + i

    swapped = coords.copy()
    swapped[5, LEFT, 0] = coords[5, RIGHT, 0]
    swapped[5, RIGHT, 0] = coords[5, LEFT, 0]

    fixed = fix_left_right_switches(swapped)

    assert np.array_equal(fixed[5, LEFT, 0], coords[5, LEFT, 0])
    assert np.array_equal(fixed[5, RIGHT, 0], coords[5, RIGHT, 0])


def test_fix_left_right_switches_leaves_a_consistent_track_untouched():
    coords = np.zeros((8, 17, 2))
    for k in range(17):
        coords[:, k, 0] = np.arange(8) + k

    assert np.array_equal(fix_left_right_switches(coords), coords)
