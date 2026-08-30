import cv2

from src.pose import get_pose, clean_keypoint
from src.utils import vis_vid, check_vid
from src.analyse import analyse_climb

from src.config import config as cfg


def process_vid(video_path: str, pose_tracker=None) -> dict:
    cap = cv2.VideoCapture(video_path)

    cache = cfg.SAVE_FILE if cfg.DEBUG else None
    keypoints, scores = get_pose(
        cap, cache_file=cache, use_openpose=cfg.USE_OPENPOSE, pose_tracker=pose_tracker
    )

    keypoints[scores < 0.3] = None
    scores[scores < 0.3] = 0
    cleaned_keypoints = clean_keypoint(
        keypoints, window_len=7, poly_order=3, max_velocity=50.0
    )

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


if __name__ == "__main__":
    file_path = "data/test.mp4"
    if check_vid(file_path):
        process_vid("data/test.mp4")
    else:
        raise FileNotFoundError("File not exist or is corrupted")
