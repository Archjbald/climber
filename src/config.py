from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Runtime configuration flags for the pose pipeline."""

    DEBUG: bool = False
    USE_CACHE: bool = True
    DRAW: bool = True
    USE_OPENPOSE: bool = False
    CONF_THRESH: float = 0.3
    MAX_UPLOAD_MB: int = 200



config = AppConfig()
