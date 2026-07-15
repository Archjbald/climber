import time

import cv2
import numpy as np
from rtmlib import draw_skeleton
from matplotlib import pyplot as plt

from config import *


def plot_vals(*args):
    x = np.linspace(0, 14, len(args[0]))
    plt.figure()
    for vals in args:
        plt.plot(x, vals)
    plt.show()


def vis_vid(cap, keypoints=None, scores=None, mode="all", save=True):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_len = 1 / fps

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if save:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    if mode == "one":
        keypoints_17 = np.zeros((len(keypoints), 1, 17, 2), dtype=float)
        keypoints_17[:, 0, 0] = keypoints
        scores_17 = np.zeros((len(keypoints), 1, 17), dtype=float)
        scores_17[:, 0, 0] = scores

    while cap.isOpened():
        now = time.time()
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break

        img_show = frame.copy()
        if keypoints is not None:
            if mode == "all":
                img_show = draw_skeleton(img_show,
                                         np.expand_dims(keypoints[frame_idx - 1], axis=0),
                                         np.expand_dims(scores[frame_idx - 1], axis=0),
                                         openpose_skeleton=USE_OPENPOSE,
                                         kpt_thr=0.25)
            elif mode == "climb":
                img_show = draw_skeleton(img_show,
                                         np.expand_dims(keypoints[frame_idx - 1], axis=0),
                                         np.expand_dims(scores[frame_idx - 1] *
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], axis=0),
                                         openpose_skeleton=USE_OPENPOSE,
                                         kpt_thr=0.25,
                                         radius=5
                                         )

            elif mode == "left":
                img_show = draw_skeleton(img_show,
                                         np.expand_dims(keypoints[frame_idx - 1], axis=0),
                                         np.expand_dims(scores[frame_idx - 1] *
                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], axis=0),
                                         openpose_skeleton=USE_OPENPOSE,
                                         kpt_thr=0.25,
                                         radius=5
                                         )

            elif mode == "one":
                img_show = draw_skeleton(img_show,
                                         keypoints_17[frame_idx - 1],
                                         scores_17[frame_idx - 1], openpose_skeleton=USE_OPENPOSE,
                                         kpt_thr=0.25,
                                         radius=5
                                         )
        if save:
            out.write(img_show)
        cv2.imshow('img', img_show)
        now2 = time.time()
        wait = max(1, round((frame_len - now2 + now) * 1000))
        cv2.waitKey(wait)

    if save:
        out.release()
