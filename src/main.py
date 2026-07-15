import os
import cv2
import numpy as np

from src.pose import get_pose, clean_keypoint
from src.utils import vis_vid, plot_vals
from src.analyse import analyse_climb, analyse_center, count_moves

from config import *

cap = cv2.VideoCapture("data/test.mp4")

if os.path.isfile(SAVE_FILE):
    npz_file = np.load(SAVE_FILE)
    keypoints = npz_file['keypoints']
    scores = npz_file['scores']
else:
    keypoints, scores = get_pose(cap, USE_OPENPOSE)
    np.savez(SAVE_FILE, keypoints=keypoints, scores=scores)

keypoints[scores < 0.3] = None
scores[scores < 0.3] = 0
cleaned_keypoints = clean_keypoint(keypoints, window_len=7, poly_order=3, max_velocity=50.0)

if DRAW:
    vis_vid(cap, cleaned_keypoints, scores, mode="left")

analysis = analyse_climb(cleaned_keypoints, cap.get(cv2.CAP_PROP_FPS))
