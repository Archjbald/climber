"""End-to-end pipeline: pose extraction, keypoint cleaning, visualization, and climb analysis."""

import sys

import cv2

from src.pose import get_pose, clean_keypoint
from src.utils import vis_vid, check_vid, get_cache_path
from src.analyse import analyse_climb

from src.config import config as cfg


def process_vid(video_path: str, pose_tracker=None) -> dict:
    """Run the full pipeline on a video and return its metadata and climbing metrics."""
    cap = cv2.VideoCapture(video_path)

    cache = get_cache_path(video_path) if cfg.DEBUG and cfg.USE_CACHE else None
    keypoints, scores = get_pose(
        cap, cache_file=cache, use_openpose=cfg.USE_OPENPOSE, pose_tracker=pose_tracker
    )

    keypoints[scores < cfg.CONF_THRESH] = None
    scores[scores < cfg.CONF_THRESH] = 0
    cleaned_keypoints = clean_keypoint(
        keypoints, window_len=7, poly_order=3, max_velocity=50.0
    )

    if cfg.DRAW:
        vis_vid(cap, cleaned_keypoints, scores, mode="all")

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    cap_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap_duration = round(cap_frame_count / cap_fps, 3)

    analysis = analyse_climb(cleaned_keypoints, cap_fps)

    if cfg.DEBUG:
        print("Moves: ", analysis["move_count"])
        print("Static timme: ", analysis["static_time"], "s.")

    data = {
        "video_metadata": {
            "duration_seconds": cap_duration,
            "fps": cap_fps,
        },
        "climbing_metrics": analysis,
    }

    cap.release()
    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if check_vid(file_path):
            process_vid(file_path)
        else:
            raise FileNotFoundError("File not exist or is corrupted")
    else:
        print("Please provide a video path: python src/main.py <video path>")
