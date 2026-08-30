class AppConfig:
    """Runtime configuration flags for the pose pipeline."""

    def __init__(self):
        self.USE_CACHE = True
        self.DRAW = True
        self.USE_OPENPOSE = False
        self.DEBUG = True
        self.CONF_THRESH = 0.3


config = AppConfig()
