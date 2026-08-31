"""Minimal rtmlib example: detect body keypoints on a single image and display them."""

import cv2
from rtmlib import Body, draw_skeleton

img = cv2.imread('../data/demo.jpg')

device = 'cpu'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False

model = Body(mode='lightweight',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
             backend=backend,
             device=device,
             to_openpose=openpose_skeleton)

keypoints, scores = model(img)

# visualize
# if you want to use black background instead of original image,
# img = np.zeros(img.shape, dtype=np.uint8)
img = draw_skeleton(img, keypoints, scores, kpt_thr=0.5, openpose_skeleton=openpose_skeleton)
cv2.imshow('img', img)
cv2.waitKey(0)
