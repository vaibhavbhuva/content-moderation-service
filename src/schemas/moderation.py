"""
Moderation schemas - Text moderation only.
"""

from typing import List
from .base import BaseModel as BaseSchema
class TextModerationRequest(BaseSchema):
    """Request model for text moderation."""
    text: str


class TextModerationResponse(BaseSchema):
    """Response model for text moderation."""
    category: str
    confidence: float