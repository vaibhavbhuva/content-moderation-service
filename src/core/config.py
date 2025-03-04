from pathlib import Path
from pydantic_settings import BaseSettings

# Get the absolute path of the `.env` file in the current project
BASE_DIR = Path(__file__).resolve().parent.parent  # Gets `app/` folder
ENV_FILE_PATH = BASE_DIR / ".env"  # Looks for `.env` in the project root

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Moderation API"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    GOOGLE_APPLICATION_CREDENTIALS: str

    class Config:
        env_file = ".env"

settings = Settings()