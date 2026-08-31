"""Fake-data factories shared across the test suite."""

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
            "duration_seconds": 10.0,
            "fps": 30.0,
        },
        "climbing_metrics": {
            "move_count": 5,
            "static_time": 3.5,
        },
    }

def make_pose(nb_frames: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return random (keypoints, scores) arrays shaped like pose-tracker output."""
    return np.random.rand(nb_frames, 1, 17, 2), np.random.rand(nb_frames, 1, 17)


def make_climber_track(
    nb_frames: int, reaches: tuple = (), step: float = 4.0, keypoint: int = 9
) -> np.ndarray:
    """Return a (frames, 17, 2) keypoint track with fixed shoulders and optional reaches.

    Every keypoint stays still except `keypoint`, which translates by `step` px/frame
    over each (start, stop) interval in `reaches` and then holds its position.
    """
    track = np.zeros((nb_frames, 17, 2), dtype=float)
    track[:, 5] = (0.0, 0.0)
    track[:, 6] = (10.0, 0.0)

    frames = np.arange(nb_frames)
    for start, stop in reaches:
        track[:, keypoint, 0] += np.clip(frames - start, 0, stop - start) * step

    return track


def make_motions(nb_frames: int = 2, nb_kp: int = 1) -> np.ndarray:
    """Return a motion-state array that is static for the first half of the frames."""
    motions = np.ones((nb_kp, nb_frames), dtype=np.int8)

    motions[:, :nb_frames//2] = 0

    return motions