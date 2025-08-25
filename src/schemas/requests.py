"""
Request schemas for the content moderation service.
"""

from typing import Any, Dict, Optional
from .base import BaseModel


from typing import Optional
from pydantic import Field, validator
from .base import BaseModel
from src.core.config import settings

class ProfanityCheckRequest(BaseModel):
    """Request model for profanity check endpoints."""
    text: str = Field(..., min_length=settings.CONTENT_TEXT_MIN_LENGTH, max_length=settings.CONTENT_TEXT_MIN_LENGTH, description="Text to check for profanity")
    language: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="language hint for transformer models, using ISO 639-1 (e.g., 'hi', 'en')."
    )
    metadata: Optional[Dict[str, Any]] = {}

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


from typing import Optional
from pydantic import Field, validator
from .base import BaseModel

class LanguageDetectionRequest(BaseModel):
    """Request model for language detection endpoint."""
    text: str = Field(..., min_length=settings.CONTENT_TEXT_MIN_LENGTH, max_length=settings.CONTENT_TEXT_MAX_LENGTH, description="Text for language detection")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
