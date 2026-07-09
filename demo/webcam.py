from rtmlib import Body, Custom, PoseTracker, draw_skeleton
import cv2

cap = cv2.VideoCapture(0)  # for video file instead of webcam, use cap = cv2.VideoCapture('./demo.mp4')

device = 'cpu'
backend = 'onnxruntime'
openpose_skeleton = False

pose_tracker = PoseTracker(Body,
                        mode='balanced',
                        det_frequency=10,  # detect every 10 frames
                        backend=backend, device=device,
                        to_openpose=False)

# # Or with a custom class
# from functools import partial
# custom = partial(Custom,
#                 det_class='YOLOX',
#                 det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
#                 det_input_size=(640, 640),
#                 pose_class='RTMPose',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
#                 pose_input_size=(192, 256))
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         backend=backend, device=device,
#                         to_openpose=False)

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1
    if not success:
        break

    keypoints, scores = pose_tracker(frame)

    img_show = frame.copy()
    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)
    cv2.imshow('img', img_show)
    cv2.waitKey(10)