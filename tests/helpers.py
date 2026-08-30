"""Test fixtures and fake-data factories shared across the test suite."""

import pytest
import numpy as np
import cv2


@pytest.fixture
def sample_video_path(tmp_path):
    """Generates a temporary 1-second dummy video for testing."""
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
    """Generates a temporary 1-second dummy video for testing."""
    video_file = tmp_path / "empty_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (640, 480))

    out.release()
    return str(video_file)



def make_video(filename: str = "video.mp4") -> dict:
    """Build a multipart `files` dict for a fake video upload."""
    return {
        "file": (
            filename,
            b"dummy video bytes",
            "video/mp4",
        )
    }


def make_process_results() -> dict:
    """Return a sample `process_vid` result payload."""
    return {
        "video_metadata": {
            "duration_seconds": 10,
            "fps": 30,
        },
        "climbing_metrics": {"moves": 5},
    }

def make_pose(nb_frames: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return random (keypoints, scores) arrays shaped like pose-tracker output."""
    return np.random.rand(nb_frames, 1, 17, 2), np.random.rand(nb_frames, 1, 17)


def make_motions(nb_frames: int = 2, nb_kp: int = 1) -> np.ndarray:
    """Return a motion-state array that is static for the first half of the frames."""
    motions = np.ones((nb_kp, nb_frames), dtype=np.int8)

    motions[:, :nb_frames//2] = 0

    return motions