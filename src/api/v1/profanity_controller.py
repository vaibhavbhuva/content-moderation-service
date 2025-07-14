"""
Profanity Detection Controller

This module handles all profanity detection related endpoints.
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from src.schemas.requests import ProfanityCheckRequest, LanguageDetectionRequest
from src.schemas.responses import ProfanityCheckResponse, LanguageDetectionResponse
from src.services.profanity_service import (
    check_profanity_llm, 
    check_profanity_transformer_chunked
)
from src.services.language_detection_service import detect_language_service

router = APIRouter(prefix="/profanity", tags=["Profanity Detection"])
logger = logging.getLogger("uvicorn.error")


@router.post(
    "/check-llm",
    response_model=ProfanityCheckResponse,
    summary="Check profanity using LLM",
    description="Detect profanity in text using Gemini LLM with contextual analysis"
)
async def check_profanity_with_llm(request: ProfanityCheckRequest) -> ProfanityCheckResponse:
    """
    Check for profanity using Large Language Model (Gemini).
    
    This endpoint uses advanced LLM capabilities to understand context
    and provide detailed reasoning for profanity detection.
    """
    logger.info(f"LLM profanity check request for text length: {len(request.text)}")
    
    try:
        result = check_profanity_llm(request.text)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return ProfanityCheckResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LLM profanity check: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/check-transformer", 
    response_model=ProfanityCheckResponse,
    summary="Check profanity using transformer models with automatic chunking",
    description="Detect profanity using specialized transformer models with automatic text chunking for long texts"
)
async def check_profanity_with_transformer(request: ProfanityCheckRequest) -> ProfanityCheckResponse:
    """
    Check for profanity using transformer models with automatic chunking.
    
    This endpoint automatically detects the language and uses the appropriate
    transformer model (English: toxic-bert, Indic: MuRIL) for profanity detection.
    
    **Features:**
    - Automatic language detection
    - Language-specific model selection  
    - Automatic chunking for long texts (>5000 characters)
    - Result aggregation using majority voting
    - Context preservation through sliding window overlap
    """
    logger.info(f"Transformer profanity check request for text length: {len(request.text)}")
    
    try:
        result = check_profanity_transformer_chunked(request.text, use_chunking=True)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return ProfanityCheckResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in transformer profanity check: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/detect-language",
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
    logger.info(f"Language detection request for text length: {len(request.text)}")
    
    try:
        result = detect_language_service(request.text, request.min_chars)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return LanguageDetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Legacy endpoint for backward compatibility
@router.post(
    "/profanity_validator",
    response_model=ProfanityCheckResponse,
    summary="[LEGACY] Check profanity using LLM",
    description="Legacy endpoint - use /check-llm instead",
    deprecated=True
)
async def legacy_profanity_check_llm(payload: ProfanityCheckRequest):
    """Legacy endpoint for backward compatibility. Use /check-llm instead."""
    logger.warning("Legacy endpoint /profanity_validator called - redirecting to /check-llm")
    return await check_profanity_with_llm(payload)


# Legacy endpoint for backward compatibility
@router.post(
    "/transformer",
    response_model=ProfanityCheckResponse,
    summary="[LEGACY] Check profanity using transformer",
    description="Legacy endpoint - use /check-transformer instead",
    deprecated=True
)
async def legacy_profanity_check_transformer(payload: ProfanityCheckRequest):
    """Legacy endpoint for backward compatibility. Use /check-transformer instead."""
    logger.warning("Legacy endpoint /transformer called - redirecting to /check-transformer")
    return await check_profanity_with_transformer(payload)


# Legacy endpoint for backward compatibility  
@router.post(
    "/detect_language",
    summary="[LEGACY] Detect language",
    description="Legacy endpoint - use /detect-language instead",
    deprecated=True
)
async def legacy_detect_language_endpoint(
    text: str = Body(..., embed=True, description="Text to detect language for")
):
    """Legacy endpoint for backward compatibility. Use /detect-language instead."""
    logger.warning("Legacy endpoint /detect_language called - redirecting to /detect-language")
    
    try:
        request = LanguageDetectionRequest(text=text)
        result = await detect_text_language(request)
        
        # Convert to legacy format
        legacy_result = {
            "status": result.status,
            "detected_language": result.detected_language,
            "raw": result.raw,
            "language_name": result.language_name,
            "confidence": result.confidence,
            "details": result.details
        }
        
        if result.status == "error":
            return JSONResponse(status_code=400, content=legacy_result)
        return legacy_result
        
    except Exception as e:
        logger.error(f"Error in legacy language detection: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Internal server error"
        })
