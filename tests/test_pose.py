import numpy as np
from unittest.mock import patch, MagicMock
from src.pose import extract_pose

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
