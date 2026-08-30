import time
import cv2
import numpy as np
import mlflow

from pathlib import Path
from src.pose import extract_pose
from src.config import config as cfg
from experiments.utils_expe import log_skeleton_overlay

mlflow.set_experiment("climbing_backend_sweep")
CLIPS = ["data/test.mp4"]
CONFIGS = [{"mode": "lightweight"}, {"mode": "balanced"}, {"mode": "performance"}]

for pose_cfg in CONFIGS:
    for clip in CLIPS:
        with mlflow.start_run(run_name=f"{pose_cfg['mode']}_{Path(clip).stem}"):
            mlflow.log_params({**pose_cfg, "clip": clip, "openpose": cfg.USE_OPENPOSE})
            cap = cv2.VideoCapture(clip)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            t0 = time.time()
            keypoints, scores = extract_pose(cap, cfg.USE_OPENPOSE, **pose_cfg)
            elapsed = time.time() - t0

            mid_frame_idx = len(keypoints) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ok, mid_frame = cap.read()
            if ok:
                log_skeleton_overlay(mid_frame, keypoints[mid_frame_idx])

            valid = scores[scores > 0]
            mlflow.log_metrics(
                {
                    "mean_confidence": float(np.mean(valid)) if len(valid) else 0.0,
                    "pct_valid_keypoints": float((scores >= 0.3).mean()),
                    "processing_fps": frame_count / elapsed,
                }
            )

            cap.release()
