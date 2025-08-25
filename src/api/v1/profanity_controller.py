"""
Profanity Detection Controller

This module handles all profanity detection related endpoints.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Body
from src.core.logger import logger

from src.schemas.requests import ProfanityCheckRequest
from src.schemas.responses import ProfanityCheckResponse
from src.services.text_profanity_service import (
    check_profanity_transformer_chunked
)

from src.services.kafka_service import KafkaIntegrationService  


kafka_service = KafkaIntegrationService()

router = APIRouter(prefix="/moderation", tags=["Profanity Detection"])

@router.post(
    "/text", 
    response_model=ProfanityCheckResponse,
    summary="Check profanity using transformer models with automatic chunking",
    description="Detect profanity using specialized transformer models with automatic text chunking for long texts"
)
async def analyze_text_profanity(
    request: ProfanityCheckRequest,
    background_tasks: BackgroundTasks
    ) -> ProfanityCheckResponse:
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
    logger.info(f"Transformer profanity check request for text length: {len(request.text)} :: {request.text[:50]}")
    # print(request.dict())
    try:
        response = check_profanity_transformer_chunked(request.text, request.language)
        if response["status"] == "failed":
            background_tasks.add_task(
                kafka_service.send_moderation_result,
                request.model_dump_json(),
                response
            )
            raise HTTPException(status_code=400, detail=response["message"])
        logger.info("Profanity check completed successfully")
        background_tasks.add_task(
            kafka_service.send_moderation_result,
            request.model_dump_json(),
            response
        )
        return ProfanityCheckResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing moderation request:")
        background_tasks.add_task(
            kafka_service.send_moderation_result,
            request.model_dump_json(),
            {"status": "failed", "message": "Internal server error"}
        )
        raise HTTPException(status_code=500, detail="Internal server error")
