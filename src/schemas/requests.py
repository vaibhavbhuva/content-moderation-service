"""
Request schemas for the content moderation service.
"""

from typing import Optional
from .base import BaseModel


from typing import Optional
from pydantic import Field, validator
from .base import BaseModel

class ProfanityCheckRequest(BaseModel):
    """Request model for profanity check endpoints."""
    text: str = Field(..., min_length=1, max_length=15000, description="Text to check for profanity")
    language: Optional[str] = None  # Optional language hint for transformer models

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


from typing import Optional
from pydantic import Field, validator
from .base import BaseModel

class LanguageDetectionRequest(BaseModel):
    """Request model for language detection endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text for language detection")
    min_chars: Optional[int] = Field(
        5,
        ge=1,
        le=1000,
        description="Minimum characters required for detection"
    )

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
