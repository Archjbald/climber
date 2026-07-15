import os
import cv2
import numpy as np

from src.pose import get_pose, clean_keypoint
from src.utils import vis_vid, plot_vals, check_vid
from src.analyse import analyse_climb, analyse_center, count_moves

from src.config import config as cfg


def process_vid(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if cfg.DEBUG and os.path.isfile(cfg.SAVE_FILE):
        npz_file = np.load(cfg.SAVE_FILE)
        keypoints = npz_file['keypoints']
        scores = npz_file['scores']
    else:
        keypoints, scores = get_pose(cap, cfg.USE_OPENPOSE)
        if cfg.DEBUG:
            np.savez(cfg.SAVE_FILE, keypoints=keypoints, scores=scores)

    keypoints[scores < 0.3] = None
    scores[scores < 0.3] = 0
    cleaned_keypoints = clean_keypoint(keypoints, window_len=7, poly_order=3, max_velocity=50.0)

    if cfg.DRAW:
        vis_vid(cap, cleaned_keypoints, scores, mode="all")

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    cap_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap_duration = round(cap_frame_count / cap_fps, 3)

    analysis = analyse_climb(cleaned_keypoints, cap_fps)

    data = {
        "video_metadata": {
            "duration_seconds": cap_duration,
            "fps": cap_fps,
        },
        "climbing_metrics": analysis,
    }


    cap.release()
    return data


if __name__ == '__main__':
    file_path = "data/test.mp4"
    if check_vid(file_path):
        process_vid("data/test.mp4")
    else:
        raise FileNotFoundError("File not exist or is corrupted")