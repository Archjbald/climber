"""Tests for the src.main.process_vid pipeline."""

from unittest.mock import patch

import numpy as np

from src.main import process_vid
from tests.helpers import make_climber_track


@patch("src.main.get_pose")
def test_process_vid_returns_metadata_and_metrics(mock_get_pose, sample_video_path):
    """process_vid wires real video I/O, cleaning and analysis into one result dict."""
    keypoints = make_climber_track(30)
    scores = np.ones((30, 17))
    mock_get_pose.return_value = (keypoints, scores)

    result = process_vid(sample_video_path)

    assert set(result) == {"video_metadata", "climbing_metrics"}
    assert result["video_metadata"] == {"duration_seconds": 1.0, "fps": 30.0}
    assert set(result["climbing_metrics"]) == {"move_count", "static_time"}


@patch("src.main.get_pose")
def test_process_vid_counts_no_move_for_a_still_climber(mock_get_pose, sample_video_path):
    """A motionless keypoint track yields zero moves and a non-zero static time."""
    keypoints = make_climber_track(30)
    scores = np.ones((30, 17))
    mock_get_pose.return_value = (keypoints, scores)

    metrics = process_vid(sample_video_path)["climbing_metrics"]

    assert metrics["move_count"] == 0
    assert metrics["static_time"] > 0


@patch("src.main.get_pose")
def test_process_vid_counts_one_move_per_reach(mock_get_pose, sample_video_path):
    """Each wrist reach in the keypoint track is counted as one move."""
    keypoints = make_climber_track(120, reaches=((30, 45), (80, 95)))
    scores = np.ones((120, 17))
    mock_get_pose.return_value = (keypoints, scores)

    metrics = process_vid(sample_video_path)["climbing_metrics"]

    assert metrics["move_count"] == 2
