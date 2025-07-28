"""
Profanity Detection Controller

This module handles all profanity detection related endpoints.
"""

from fastapi import APIRouter, HTTPException
from src.core.logger import logger

from src.schemas.requests import LanguageDetectionRequest
from src.schemas.responses import LanguageDetectionResponse

from src.services.language_detection_service import detect_language_service



router = APIRouter(prefix="/language", tags=["Language"])

@router.post(
    "/detect",
    response_model=LanguageDetectionResponse,
    summary="Detect language of text",
    description="Detect if text is English, Indic, or other language using XLM-RoBERTa"
)
async def detect_text_language(request: LanguageDetectionRequest) -> LanguageDetectionResponse:
    """
    Detect the language of the provided text.
    
    This endpoint uses XLM-RoBERTa for accurate language detection
    with support for 100+ languages.
    """
    logger.info(f"Language detection request for text length: {len(request.text)} :: {request.text[:50]}")
    try:
        result = detect_language_service(request.text)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"Language detection response :: {result}")
        return LanguageDetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in language detection")
        raise HTTPException(status_code=500, detail="Internal server error")