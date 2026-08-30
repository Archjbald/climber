"""Tests for the src.main.process_vid pipeline."""

from unittest.mock import patch, MagicMock
from src.main import process_vid
from tests.helpers import make_pose


@patch("src.main.cv2.VideoCapture")
@patch("src.main.get_pose")
@patch("src.main.clean_keypoint")
@patch("src.main.analyse_climb")
@patch("src.main.cfg")
def test_process_vid_standard_execution(
    mock_cfg, mock_analyse, mock_clean, mock_get_pose, mock_VideoCapture
):
    mock_cfg.DEBUG = False
    mock_cfg.DRAW = False
    mock_cfg.USE_OPENPOSE = True

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 60.0
    mock_VideoCapture.return_value = mock_cap

    fake_keypoints,fake_scores = make_pose()
    mock_get_pose.return_value = (fake_keypoints, fake_scores)
    mock_clean.return_value = "cleaned_kps"
    mock_analyse.return_value = {"move_count": 5}

    result = process_vid("fake_video.mp4")


    assert result == {
        "video_metadata": {"duration_seconds": 2.0, "fps": 30.0},
        "climbing_metrics": {"move_count": 5},
    }

    mock_get_pose.assert_called_once_with(mock_cap, cache_file=None, use_openpose=True, pose_tracker=None)
    mock_clean.assert_called_once()
    mock_analyse.assert_called_once_with("cleaned_kps", 30.0)
    mock_cap.release.assert_called_once()
