"""
Configuration settings for the content moderation service.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

# Get the absolute path of the `.env` file in the current project
BASE_DIR = Path(__file__).resolve().parent.parent  # Gets `src/` folder
ENV_FILE_PATH = BASE_DIR.parent / ".env"  # Looks for `.env` in the project root


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Basic API settings
    PROJECT_NAME: str = "Content Moderation API"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    
    # Gemini API settings
    GEMINI_API_KEY: Optional[str] = None
    
    # Model settings
    MAX_TEXT_LENGTH: int = 500
    MIN_TEXT_LENGTH: int = 5
    LANGUAGE_DETECTION_SAMPLE_SIZE: int = 300  # characters to use for language detection
    
    # Text chunking settings
    CHUNKING_ENABLED: bool = True
    CHUNK_SIZE: int = 500          # tokens per chunk
    CHUNK_OVERLAP: int = 12        # overlap between chunks (tokens)
    MAX_CHUNKS_PER_TEXT: int = 10  # maximum chunks to process per text
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100  # requests per time window
    RATE_LIMIT_WINDOW: int = 60     # time window in seconds
    RATE_LIMIT_PER_ENDPOINT: bool = True  # separate limits per endpoint
    
    # Development settings
    DEBUG: bool = False

    class Config:
        """Pydantic configuration."""
        env_file = ENV_FILE_PATH
        case_sensitive = True


settings = Settings()
