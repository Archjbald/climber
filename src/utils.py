"""Video integrity checks, debug plotting, and skeleton-overlay rendering helpers."""

from __future__ import annotations

import time

import cv2
import numpy as np
from rtmlib import draw_skeleton
from matplotlib import pyplot as plt
from PIL import Image

from src.config import config as cfg

import base64
import zlib


def check_vid(file_path: str) -> bool:
    """Return True if `file_path` can be opened and at least one frame read."""
    cap = cv2.VideoCapture(file_path)
    is_opened = cap.isOpened()
    has_frame, _ = cap.read() if is_opened else (False, None)
    cap.release()
    return is_opened and has_frame


def plot_vals(*args: np.ndarray) -> None:
    """Plot one or more signals on a shared x-axis (no-op unless DEBUG)."""
    if not cfg.DEBUG:
        return
    x = np.linspace(0, 14, len(args[0]))
    plt.figure()
    for vals in args:
        plt.plot(x, vals)
    plt.show()


def get_cache_path(file_path: str) -> str:
    normalized = file_path.replace("\\", "/").strip()
    compressed = zlib.compress(normalized.encode('utf-8'), level=9)
    uid = base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')
    return f'{uid}.npz'

def vis_vid(
    cap,
    keypoints: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    mode: str = "all",
    save: bool = True,
    gif: bool = False,
) -> None:
    """Render skeleton overlays over the video and display/save them (no-op unless DEBUG)."""
    if not cfg.DEBUG:
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_len = 1 / fps
    gif_fps = 10

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if save:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        pil_images = None
        out = None
        if gif:
            pil_images = []
        if not gif:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter("output.avi", fourcc, fps, (frame_width, frame_height))

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

        if gif:
            img_show = np.zeros_like(frame)
        else:
            img_show = frame.copy()

        if keypoints is not None:
            if mode == "all":
                img_show = draw_skeleton(
                    img_show,
                    np.expand_dims(keypoints[frame_idx - 1], axis=0),
                    np.expand_dims(scores[frame_idx - 1], axis=0),
                    openpose_skeleton=cfg.USE_OPENPOSE,
                    kpt_thr=0.25,
                )
            elif mode == "climb":
                img_show = draw_skeleton(
                    img_show,
                    np.expand_dims(keypoints[frame_idx - 1], axis=0),
                    np.expand_dims(
                        scores[frame_idx - 1]
                        * [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                        axis=0,
                    ),
                    openpose_skeleton=cfg.USE_OPENPOSE,
                    kpt_thr=0.25,
                    radius=5,
                )

            elif mode == "left":
                img_show = draw_skeleton(
                    img_show,
                    np.expand_dims(keypoints[frame_idx - 1], axis=0),
                    np.expand_dims(
                        scores[frame_idx - 1]
                        * [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                        axis=0,
                    ),
                    openpose_skeleton=cfg.USE_OPENPOSE,
                    kpt_thr=0.25,
                    radius=5,
                )

            elif mode == "one":
                img_show = draw_skeleton(
                    img_show,
                    keypoints_17[frame_idx - 1],
                    scores_17[frame_idx - 1],
                    openpose_skeleton=cfg.USE_OPENPOSE,
                    kpt_thr=0.25,
                    radius=5,
                )
        if save:
            if gif:
                if frame_idx % (fps // gif_fps) == 1:
                    pil_images.append(Image.fromarray(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)))
            else:
                out.write(img_show)
                
        cv2.imshow("img", img_show)
        now2 = time.time()
        wait = max(1, round((frame_len - now2 + now) * 1000))
        cv2.waitKey(wait)

    if save:
        if gif:
            pil_images[0].save(
                "output.gif",
                save_all=True,
                append_images=pil_images[1:],
                duration=1000 // gif_fps,  # duration per frame in milliseconds (1000 / fps)
                loop=0,  # 0 means infinite loop
            )
        else:
            out.release()
