"""Test fixtures and fake-data factories shared across the test suite."""

import numpy as np


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