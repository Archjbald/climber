from rtmlib import Body, PoseTracker, Custom
from functools import partial
from tqdm import tqdm

from scipy.signal import savgol_filter
import cv2
import numpy as np


from config import *


def get_pose(cap, openpose_skeleton):
    device = 'cpu'
    backend = 'onnxruntime'

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_tracker = PoseTracker(Body,
                               mode='balanced',
                               det_frequency=1,  # detect every 10 frames
                               backend=backend,
                               device=device,
                               to_openpose=False)

    frame_idx = 0
    detection = [[], []]
    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1
            pbar.update(1)
            if not success:
                break

            keypoints, scores = pose_tracker(frame)
            detection[0].append(keypoints)
            detection[1].append(scores)

    print(len(detection[0]))
    keypoints = np.vstack(detection[0])
    scores = np.vstack(detection[1])
    return keypoints, scores


def clean_keypoint(coords, max_velocity=30.0, window_len=11, poly_order=2):
    cleaned = coords.copy().astype(float)
    num_frames, num_keypoints, _ = cleaned.shape

    # velocity thresh
    diffs = np.diff(cleaned, axis=0)
    velocities = np.sqrt(np.sum(diffs ** 2, axis=2))
    frame_idx, kp_idx = np.where(velocities > max_velocity)

    valid_mask = (frame_idx + 1) < num_frames
    cleaned[frame_idx[valid_mask] + 1, kp_idx[valid_mask]] = np.nan

    # interpolation
    frames = np.arange(num_frames)

    for kp in range(num_keypoints):
        for axis in range(2):  # 0 for X, 1 for Y
            signal = cleaned[:, kp, axis]
            nans = np.isnan(signal)

            # 2a. Fill Gaps
            if np.any(nans):
                if np.all(nans):
                    # If the entire video for this keypoint is NaN, fill with 0s to prevent crash
                    signal[:] = 0.0
                else:
                    signal[nans] = np.interp(frames[nans], frames[~nans], signal[~nans])

            # 2b. Smooth (Modifies the signal array in-place inside 'cleaned')
            if window_len > 3:
                cleaned[:, kp, axis] = savgol_filter(signal, window_length=window_len, polyorder=poly_order)

    return cleaned
