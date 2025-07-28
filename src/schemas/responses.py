"""
Response schemas for the content moderation service.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field
from .base import BaseModel


class ProfanityCheckData(BaseModel):
    """Profanity check response data model."""
    text: str = Field(..., description="The input text that was analyzed")
    isProfane: bool = Field(..., description="Whether profanity was detected")
    confidence: float = Field(..., description="Confidence score (0-100)")
    category: str = Field(..., description="Classification category")
    detected_language: Optional[str] = Field(None, description="Detected language code")
    
    # Additional fields for chunked responses
    text_length: Optional[int] = Field(None, description="Length of input text")
    chunking_used: Optional[bool] = Field(None, description="Whether text chunking was used")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks processed")
    profane_chunks: Optional[int] = Field(None, description="Number of profane chunks detected")
    clean_chunks: Optional[int] = Field(None, description="Number of clean chunks detected")
    aggregation_strategy: Optional[str] = Field(None, description="Strategy used for aggregating chunk results")
    chunk_statistics: Optional[Dict[str, Any]] = Field(None, description="Statistics about chunk processing")
    chunk_details: Optional[list] = Field(None, description="Detailed results for each chunk")

class ProfanityCheckResponse(BaseModel):
    """Profanity check response model."""
    status: str = Field(..., description="Response status (success/error)")
    message: str = Field(..., description="Response message")
    response_data: Optional[ProfanityCheckData] = Field(
        None,
        alias="responseData",
        description="Profanity check results"
    )


class LanguageDetectionData(BaseModel):
    """Language detection response data model."""
    language_code: Optional[str] = None
    language_name: Optional[str] = None
    confidence: Optional[float] = None


class LanguageDetectionResponse(BaseModel):
    """Language detection response model."""
    status: str
    message: str
    detected_language: Optional[str] = None
    language_name: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: Optional[List[LanguageDetectionData]] = None
