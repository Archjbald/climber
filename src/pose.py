"""Pose extraction and keypoint post-processing (caching, outlier removal, smoothing)."""

from __future__ import annotations

from rtmlib import Body, PoseTracker
from tqdm import tqdm

from scipy.signal import savgol_filter
import cv2
import numpy as np
import os

# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear, 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow, 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle


def get_pose_tracker(
    mode: str = "balanced",
    backend: str = "onnxruntime",
    device: str = "cpu",
    det_frequency: int = 1,
    openpose_skeleton: bool = False,
) -> PoseTracker:
    """Create an rtmlib PoseTracker for body keypoints."""
    pose_tracker = PoseTracker(
        Body,
        mode=mode,
        det_frequency=det_frequency,
        backend=backend,
        device=device,
        to_openpose=openpose_skeleton,
    )
    return pose_tracker


def get_pose(
    cap, cache_file: str | None = None, use_openpose: bool = False, **pose_kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Load keypoints/scores from `cache_file` if it exists, otherwise extract and optionally cache them."""
    if cache_file and os.path.isfile(cache_file):
        npz = np.load(cache_file)
        keypoints, scores = npz["keypoints"], npz["scores"]
    else:
        keypoints, scores = extract_pose(cap, use_openpose, **pose_kwargs)
        if cache_file:
            np.savez(cache_file, keypoints=keypoints, scores=scores)

    return keypoints, scores


def extract_pose(
    cap,
    openpose_skeleton: bool = False,
    pose_tracker=None,
    mode: str = "balanced",
    device: str = "cpu",
    backend: str = "onnxruntime",
) -> tuple[np.ndarray, np.ndarray]:
    """Run pose tracking over every frame of `cap` and return stacked keypoints and scores."""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if pose_tracker is None:
        pose_tracker = get_pose_tracker(
            mode, backend, device, openpose_skeleton=openpose_skeleton
        )

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
            detection[0].append(keypoints)  # NB_person, NB_kp, 2
            detection[1].append(scores)  # NB_person, NB_kp, 2

    print(len(detection[0]))
    keypoints = np.vstack(detection[0])
    scores = np.vstack(detection[1])
    return keypoints, scores


def clean_keypoint(
    coords: np.ndarray,
    max_velocity: float = 30.0,
    window_len: int = 11,
    poly_order: int = 2,
) -> np.ndarray:
    """Drop velocity outliers, fix left/right swaps, interpolate gaps, and smooth keypoint tracks."""
    cleaned = coords.copy().astype(float)
    num_frames, num_keypoints, _ = cleaned.shape

    # velocity thresh
    diffs = np.diff(cleaned, axis=0)
    velocities = np.sqrt(np.sum(diffs**2, axis=2))
    frame_idx, kp_idx = np.where(velocities > max_velocity)

    valid_mask = (frame_idx + 1) < num_frames
    cleaned[frame_idx[valid_mask] + 1, kp_idx[valid_mask]] = np.nan

    # fix non -voluntary switches
    cleaned = fix_left_right_switches(cleaned)

    # interpolation
    frames = np.arange(num_frames)

    for kp in range(num_keypoints):
        for axis in range(2):
            signal = cleaned[:, kp, axis]
            nans = np.isnan(signal)

            # Fill Gaps
            if np.any(nans):
                if np.all(nans):
                    signal[:] = 0.0
                else:
                    signal[nans] = np.interp(frames[nans], frames[~nans], signal[~nans])

            # Smooth
            if window_len > 3:
                cleaned[:, kp, axis] = savgol_filter(
                    signal, window_length=window_len, polyorder=poly_order
                )

    return cleaned


def fix_left_right_switches(coords: np.ndarray) -> np.ndarray:
    """Correct frames where left/right keypoints are swapped, using constant-velocity prediction."""
    left_indices = [5, 7, 9, 11, 13, 15]
    right_indices = [6, 8, 10, 12, 14, 16]
    fixed = coords.copy().astype(float)
    num_frames = len(fixed)

    for t in range(2, num_frames):
        prev1_l = fixed[t - 1, left_indices]  # t-1
        prev2_l = fixed[t - 2, left_indices]  # t-2

        prev1_r = fixed[t - 1, right_indices]
        prev2_r = fixed[t - 2, right_indices]

        curr_l = fixed[t, left_indices]
        curr_r = fixed[t, right_indices]

        # Compute speed
        v_l = prev1_l - prev2_l
        v_r = prev1_r - prev2_r

        v_l = np.where(np.isnan(v_l), 0.0, v_l)
        v_r = np.where(np.isnan(v_r), 0.0, v_r)

        # Predict pose
        pred_l = prev1_l + v_l
        pred_r = prev1_r + v_r

        # Compute metrics
        sq_dist_normal_l = (curr_l - pred_l) ** 2
        sq_dist_normal_r = (curr_r - pred_r) ** 2

        sq_dist_switched_l = (curr_r - pred_l) ** 2
        sq_dist_switched_r = (curr_l - pred_r) ** 2

        with np.errstate(all="ignore"):
            dist_normal = np.nanmean(sq_dist_normal_l) + np.nanmean(sq_dist_normal_r)
            dist_switched = np.nanmean(sq_dist_switched_l) + np.nanmean(
                sq_dist_switched_r
            )

        if np.isnan(dist_normal) or np.isnan(dist_switched):
            continue

        # 4. Correction dans le tableau principal
        if dist_switched < dist_normal:
            fixed[t, left_indices] = curr_r
            fixed[t, right_indices] = curr_l

    return fixed


def fix_left_right_switches_old(coords: np.ndarray) -> np.ndarray:
    """Deprecated: position-only variant of `fix_left_right_switches`."""
    fixed = coords.copy().astype(float)
    num_frames = len(fixed)

    left_indices = [5, 7, 9, 11, 13, 15]
    right_indices = [6, 8, 10, 12, 14, 16]

    for t in range(1, num_frames):
        prev_left = fixed[t - 1, left_indices]
        prev_right = fixed[t - 1, right_indices]

        curr_left = fixed[t, left_indices]
        curr_right = fixed[t, right_indices]

        sq_dist_normal_l = (curr_left - prev_left) ** 2
        sq_dist_normal_r = (curr_right - prev_right) ** 2

        sq_dist_switched_l = (curr_right - prev_left) ** 2
        sq_dist_switched_r = (curr_left - prev_right) ** 2

        with np.errstate(all="ignore"):
            dist_normal = np.nanmean(sq_dist_normal_l) + np.nanmean(sq_dist_normal_r)
            dist_switched = np.nanmean(sq_dist_switched_l) + np.nanmean(
                sq_dist_switched_r
            )

        if dist_switched < dist_normal:
            fixed[t, left_indices] = curr_right
            fixed[t, right_indices] = curr_left

    return fixed
