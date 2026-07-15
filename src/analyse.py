import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import vode

from src.utils import plot_vals, vis_vid

STATIC = 0
MOVING = 1


def compute_joint_speed(pose):
    velocity = np.diff(pose, axis=0)
    speed = np.linalg.norm(velocity, axis=2)
    speed = np.vstack([speed[0], speed])
    return speed


def shoulder_width(pose):
    left = pose[:, 5]
    right = pose[:, 6]
    return np.linalg.norm(left - right, axis=1)


def moving_average(x, window):
    if window <= 1:
        return x

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def compute_kp_state(speed, threshold, static_frames):
    state = STATIC
    counter = 0

    states = np.zeros(len(speed), dtype=np.uint8)

    sp_enter = 1.2 * threshold
    sp_exit = 0.8 * threshold

    for i in range(len(speed)):
        if state == STATIC:
            if speed[i] > sp_enter:
                state = MOVING
        else:
            if speed[i] < sp_exit:
                counter += 1
                if counter >= static_frames:
                    state = STATIC
                    counter = 0

        states[i] = state

    # Smooth state:
    for i in range(static_frames, len(states)):
        if states[i - static_frames] == states[i]:
            states[i - static_frames:i] = states[i]

    return states


def count_moves(
        pose,
        fps,
        speed_threshold=0.2,
        static_time=0.5,
        smooth_time=0.2,
):
    speed = compute_joint_speed(pose)

    body = np.median(shoulder_width(pose))

    smooth_frames = max(1, int(round(smooth_time * fps)))
    static_frames = max(1, int(round(static_time * fps)))

    # LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_ANKLE = 15, RIGHT_ANKLE = 16
    avg_speeds = [moving_average(speed[:, k] / body, smooth_frames) for k in (9, 10, 15, 16)]
    states = np.array([compute_kp_state(sp, speed_threshold, static_frames) for sp in avg_speeds], dtype=np.int8)

    diffs = np.diff(states, axis=1)

    moves = np.sum(np.any(diffs < 0, axis=0))

    return moves, states


def analyse_climb(poses, fps=30):
    moves, motions = count_moves(poses, fps=fps)
    # plot_vals(*motions)

    static_frame = sum(np.sum(motions, axis=0) == 0)
    static_time = static_frame / fps

    print("Moves: ", moves)
    print("Static Frames: ", static_time, "s.")

    analysis = {
        "move_count": moves.item(),
        "static_time": static_time.item(),
    }

    return analysis


def threshold_window(vals, thresh, wind, keep: "sup"):
    assert keep in ("sup", "inf")
    results = np.zeros_like(vals, dtype=bool)
    for i in range(len(vals)):
        min_t = max(0, i - wind // 2)
        max_t = min(len(vals) - 1, i + wind // 2)
        if (keep == "sup" and np.median(vals[min_t:max_t]) > thresh) or \
                (keep == "inf" and np.median(vals[min_t:max_t]) < thresh):
            results[i] = True

    return results


def analyse_center(poses):
    center = (poses[:, 11] + poses[:, 12]) / 2.0
    diff_center = np.diff(center, axis=0)
    velocities = np.sqrt(np.sum(diff_center ** 2, axis=-1))

    static = threshold_window(velocities, 2, 20, keep="inf")
    plot_vals(velocities, static)
    return center, velocities
