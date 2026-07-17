class AppConfig:
    def __init__(self):
        self.SAVE_FILE = "keypoints.npz"
        self.DRAW = False
        self.USE_OPENPOSE = False
        self.DEBUG = True


config = AppConfig()
