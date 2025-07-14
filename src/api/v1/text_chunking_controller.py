"""
Text Chunking Controller

Provides endpoints for testing and demonstrating text chunking functionality.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.services.text_chunking_service import chunk_text_service, get_text_chunker

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/chunking", tags=["Text Chunking"])


class ChunkingRequest(BaseModel):
    """Request model for text chunking."""
    text: str = Field(..., description="Text to chunk", min_length=1)
    strategy: str = Field(
        default="sliding_window", 
        description="Chunking strategy: 'sliding_window' or 'sentence_aware'"
    )


class ChunkInfo(BaseModel):
    """Information about a single chunk."""
    chunk_index: int
    text: str
    start_token: int
    end_token: int
    token_count: int
    character_count: int
    has_overlap_start: bool
    has_overlap_end: bool


class ChunkingResponse(BaseModel):
    """Response model for text chunking."""
    status: str
    message: str
    original_length: int
    total_tokens: int
    total_chunks: int
    strategy: str
    chunks: List[ChunkInfo]
    chunker_config: Dict[str, Any]


class ChunkingConfigResponse(BaseModel):
    """Response model for chunking configuration."""
    chunk_size: int
    overlap_size: int
    max_chunks: int
    step_size: int
    tokenizer_available: bool
    chunking_enabled: bool


@router.post(
    "/chunk-text",
    response_model=ChunkingResponse,
    summary="Chunk text with sliding window overlap",
    description="Break down long text into overlapping chunks for processing with language models"
)
async def chunk_text(request: ChunkingRequest) -> ChunkingResponse:
    """
    Chunk text using configurable sliding window strategy.
    
    This endpoint demonstrates how long texts can be broken down into
    overlapping chunks for processing with models that have token limits.
    
    **Chunking Strategies:**
    - `sliding_window`: Token-based chunking with configurable overlap
    - `sentence_aware`: Respects sentence boundaries while maintaining token limits
    
    **Use Cases:**
    - Processing long documents with transformer models
    - Maintaining context across chunk boundaries
    - Handling texts that exceed model token limits
    """
    logger.info(f"Chunking text of length {len(request.text)} using {request.strategy} strategy")
    
    try:
        # Validate strategy
        if request.strategy not in ["sliding_window", "sentence_aware"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid strategy. Use 'sliding_window' or 'sentence_aware'"
            )
        
        # Chunk the text
        result = chunk_text_service(request.text, request.strategy)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ChunkingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text chunking: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/config",
    response_model=ChunkingConfigResponse,
    summary="Get chunking configuration",
    description="Retrieve current text chunking configuration settings"
)
async def get_chunking_config() -> ChunkingConfigResponse:
    """
    Get the current text chunking configuration.
    
    This endpoint returns the current settings for text chunking including
    chunk size, overlap, and other parameters.
    """
    try:
        chunker = get_text_chunker()
        config = chunker.get_chunk_info()
        
        return ChunkingConfigResponse(**config)
        
    except Exception as e:
        logger.error(f"Error getting chunking config: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/demo",
    summary="Chunking demonstration",
    description="Interactive demo showing different chunking strategies"
)
async def chunking_demo():
    """
    Demonstration of text chunking with different strategies.
    
    This endpoint provides examples and explanations of how the
    chunking system works with different types of content.
    """
    demo_text = """
    This is a demonstration of the text chunking system. The chunking service 
    provides intelligent text segmentation with sliding window overlap to ensure 
    context preservation across chunk boundaries. This is particularly useful 
    when processing long documents with language models that have token limits.
    
    The system supports two main strategies: sliding window chunking which focuses 
    on token-based segmentation with configurable overlap, and sentence-aware 
    chunking which tries to respect sentence boundaries while staying within 
    token limits. Both strategies ensure that important context is preserved 
    across chunk boundaries through the overlap mechanism.
    
    With sliding window chunking, you can configure the chunk size (e.g., 512 tokens), 
    overlap size (e.g., 112 tokens), and maximum number of chunks to process. 
    This creates chunks like: Chunk 1 (tokens 0-511), Chunk 2 (tokens 400-911), 
    Chunk 3 (tokens 800-1311), and so on, with each chunk overlapping with the 
    previous one to maintain context continuity.
    """
    
    try:
        # Demonstrate both strategies
        sliding_result = chunk_text_service(demo_text, "sliding_window")
        sentence_result = chunk_text_service(demo_text, "sentence_aware")
        
        return {
            "message": "Text chunking demonstration",
            "demo_text": demo_text,
            "demo_text_length": len(demo_text),
            "strategies": {
                "sliding_window": {
                    "description": "Token-based chunking with configurable overlap",
                    "result": sliding_result
                },
                "sentence_aware": {
                    "description": "Sentence boundary-aware chunking with token limits", 
                    "result": sentence_result
                }
            },
            "use_cases": [
                "Processing long documents with transformer models",
                "Maintaining context across chunk boundaries",
                "Handling texts that exceed model token limits",
                "Profanity detection on large texts",
                "Language detection on lengthy content"
            ],
            "configuration_notes": [
                "Chunk size determines tokens per chunk (default: 512)",
                "Overlap size ensures context preservation (default: 112 tokens)",
                "Step size = chunk_size - overlap_size",
                "Maximum chunks prevents excessive processing (default: 10)"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in chunking demo: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
