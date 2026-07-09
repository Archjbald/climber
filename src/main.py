import os
import cv2
import numpy as np
from pose import get_pose, clean_keypoint
from utils import vis_vid

from config import *

# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear, 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow, 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

cap = cv2.VideoCapture("data/test.mp4")

if os.path.isfile(SAVE_FILE):
    npz_file = np.load(SAVE_FILE)
    keypoints = npz_file['keypoints']
    scores = npz_file['scores']
else:
    keypoints, scores = get_pose(cap, USE_OPENPOSE)
    np.savez(SAVE_FILE, keypoints=keypoints, scores=scores)

keypoints[scores < 0.3] = None
cleaned_keypoints = clean_keypoint(keypoints, window_len=7, poly_order=3, max_velocity=50.0)

if DRAW:
    vis_vid(cap, cleaned_keypoints, scores, mode="climb")
