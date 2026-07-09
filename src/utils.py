import time

import cv2
import numpy as np
from rtmlib import draw_skeleton

from config import *


def vis_vid(cap, keypoints=None, scores=None, mode="all"):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_len = 1 / fps

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        cv2.imshow('img', img_show)
        now2 = time.time()
        wait = max(1, round((frame_len - now2 + now) * 1000))
        cv2.waitKey(wait)
