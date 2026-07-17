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