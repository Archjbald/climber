"""Shared pytest fixtures for the test suite."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video_path(tmp_path):
    """Write a 1-second synthetic video to a temp path and return it."""
    video_file = tmp_path / "synthetic_test_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (640, 480))

    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + i * 5, 240), 30, (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return str(video_file)


@pytest.fixture
def sample_fake_path(tmp_path):
    """Write an empty (frameless) video file to a temp path and return it."""
    video_file = tmp_path / "empty_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (640, 480))
    out.release()
    return str(video_file)
