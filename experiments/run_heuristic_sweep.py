import mlflow
import cv2
from src.pose import clean_keypoint, get_pose
from src.analyse import analyse_climb
from experiments.utils_expe import log_com_trajectory_plot
from src.analyse import analyse_center


mlflow.set_experiment("climbing_heuristic_tuning")
cap = cv2.VideoCapture("data/test.mp4")
keypoints, scores = get_pose(cap, cache_file="keypoints.npz")
fps = cap.get(cv2.CAP_PROP_FPS)
keypoints[scores < 0.3] = None
scores[scores < 0.3] = 0

for window_len in [5, 7, 11]:
    for poly_order in [2, 3]:
        if poly_order >= window_len:
            continue
        for max_v in [30.0, 50.0, 80.0]:
            with mlflow.start_run(run_name=f"w{window_len}_p{poly_order}_v{max_v}"):
                mlflow.log_params({"window_len": window_len, "poly_order": poly_order, "max_velocity": max_v})
                cleaned = clean_keypoint(keypoints, window_len=window_len, poly_order=poly_order, max_velocity=max_v)
                analysis = analyse_climb(cleaned, fps)
                mlflow.log_metrics({k: v for k, v in analysis.items() if isinstance(v, (int, float))})

                centers, _ = analyse_center(cleaned, plot=False)
                log_com_trajectory_plot(centers[:, 0], centers[:, 1])