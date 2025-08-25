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
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Kafka Settings
    KAFKA_ENABLED: bool = True
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_MODERATION_RESULTS_TOPIC: str = "dev.content.profanity"
    KAFKA_RETRIES: int = 3
    KAFKA_RETRY_BACKOFF_MS: int = 100

    # KAFKA_ACKS: str = "all"
    # KAFKA_BATCH_SIZE: int = 16384
    # KAFKA_LINGER_MS: int = 10
    # KAFKA_BUFFER_MEMORY: int = 33554432
    # KAFKA_MAX_REQUEST_SIZE: int = 1048576

    # Profanity - Transformer Models
    ENGLISH_TRANSFORMER_MODEL: str = "unitary/toxic-bert"
    INDIC_TRANSFORMER_MODEL: str = "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"
    LANGUAGE_DETECT_MODEL: str = "ZheYu03/xlm-r-langdetect-model"
    
    # Model settings
    MAX_TEXT_LENGTH: int = 500

    # General text length constraints (applies to multiple APIs)
    CONTENT_TEXT_MIN_LENGTH: int = 2
    CONTENT_TEXT_MAX_LENGTH: int = 3000

    # Language Detection settings
    LANGUAGE_DETECTION_SAMPLE_SIZE: int = 300  # characters to use for language detection
    
    # Text chunking settings
    CHUNKING_ENABLED: bool = True
    CHUNK_SIZE: int = 400          # tokens per chunk
    CHUNK_OVERLAP: int = 100       # overlap between chunks (tokens)
    MAX_CHUNKS_PER_TEXT: int = 10  # maximum chunks to process per text

    #  Hugging face settings
    HF_HUB_OFFLINE: int = 1 #To prevent HTTP calls to the Hub when loading a model.

    class Config:
        """Pydantic configuration."""
        env_file = ENV_FILE_PATH
        case_sensitive = True


settings = Settings()
