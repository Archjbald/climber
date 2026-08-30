"""MLflow logging helpers for experiment artifacts (plots and skeleton overlays)."""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import mlflow


def log_com_trajectory_plot(
    com_x: np.ndarray, com_y: np.ndarray, tmp_path: str = "tmp_com_trajectory.png"
) -> None:
    """Plot the center-of-mass x/y trajectory and log it to MLflow."""
    fig, ax = plt.subplots()
    ax.plot(com_x, label="CoM x")
    ax.plot(com_y, label="CoM y")
    ax.set_xlabel("frame")
    ax.set_ylabel("pixel position")
    ax.legend()
    fig.savefig(tmp_path)
    mlflow.log_artifact(tmp_path, artifact_path="plots")
    plt.close(fig)


def log_skeleton_overlay(
    frame: np.ndarray, keypoints: np.ndarray, tmp_path: str = "tmp_skeleton_frame.png"
) -> None:
    """Draw keypoints on `frame` and log the resulting image to MLflow."""
    frame = frame.copy()
    for pt in keypoints:
        if pt is None or np.isnan(pt).any():
            continue
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    cv2.imwrite(tmp_path, frame)
    mlflow.log_artifact(tmp_path, artifact_path="frames")
