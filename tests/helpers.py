import numpy as np

def make_video(filename="video.mp4"):
    return {
        "file": (
            filename,
            b"dummy video bytes",
            "video/mp4",
        )
    }


def make_process_results():
    return {
        "video_metadata": {
            "duration_seconds": 10,
            "fps": 30,
        },
        "climbing_metrics": {"moves": 5},
    }

def make_pose(nb_frames=2):
    return np.random.rand(nb_frames, 1, 17, 2), np.random.rand(nb_frames, 1, 17)

def make_motions(nb_frames=2, nb_kp=1):
    # return motion with half 0, half 1
    motions = np.ones((nb_kp, nb_frames), dtype=np.int8)

    motions[:, :nb_frames//2] = 0

    return motions