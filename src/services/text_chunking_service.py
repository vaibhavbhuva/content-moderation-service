"""
Text Chunking Service with Sliding Window Strategy

This service provides intelligent text chunking with configurable overlap
for processing long texts that exceed model token limits.

Features:
- Sliding window overlap for context preservation
- Token-aware chunking using transformers tokenizer
- Configurable chunk size and overlap
- Smart sentence boundary detection
- Aggregation strategies for chunk results
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("Transformers library not available for precise token counting")

from src.core.config import settings

logger = logging.getLogger("uvicorn.error")


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_token: int
    end_token: int
    chunk_index: int
    total_chunks: int
    overlap_start: bool = False
    overlap_end: bool = False


@dataclass
class ChunkingResult:
    """Result of text chunking operation."""
    chunks: List[TextChunk]
    total_tokens: int
    original_length: int
    chunking_strategy: str


class TextChunker:
    """
    Advanced text chunker with sliding window overlap.
    
    Provides token-aware chunking with configurable overlap to ensure
    context preservation across chunk boundaries.
    """
    
    def __init__(self, 
                 chunk_size: int = None,
                 overlap_size: int = None,
                 max_chunks: int = None,
                 tokenizer_model: str = "bert-base-uncased"):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Tokens per chunk (default from settings)
            overlap_size: Overlap tokens between chunks (default from settings)
            max_chunks: Maximum chunks to create (default from settings)
            tokenizer_model: Tokenizer model for token counting
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap_size = overlap_size or settings.CHUNK_OVERLAP
        self.max_chunks = max_chunks or settings.MAX_CHUNKS_PER_TEXT
        
        # Validate configuration
        if self.overlap_size >= self.chunk_size:
            raise ValueError("Overlap size must be less than chunk size")
        
        # Log chunking configuration
        logger.info(f"TextChunker initialized: chunk_size={self.chunk_size}, "
                   f"overlap_size={self.overlap_size}, max_chunks={self.max_chunks}, "
                   f"step_size={self.chunk_size - self.overlap_size}")
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
                logger.info(f"Loaded tokenizer: {tokenizer_model}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {tokenizer_model}: {e}")
                self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the tokenizer or estimate.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed, using estimation: {e}")
        
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return max(1, len(text) // 4)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into list of tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.tokenize(text)
                return tokens
            except Exception as e:
                logger.warning(f"Tokenization failed, using word split: {e}")
        
        # Fallback: simple word splitting
        return text.split()
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed text
        """
        if self.tokenizer:
            try:
                return self.tokenizer.convert_tokens_to_string(tokens)
            except Exception as e:
                logger.warning(f"Detokenization failed, using join: {e}")
        
        # Fallback: simple joining
        return " ".join(tokens)
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries in text for better chunking.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence end positions
        """
        # Simple sentence boundary detection
        sentence_ends = []
        for match in re.finditer(r'[.!?]+\s+', text):
            sentence_ends.append(match.end())
        
        if not sentence_ends or sentence_ends[-1] < len(text):
            sentence_ends.append(len(text))
        
        return sentence_ends
    
    def chunk_by_tokens(self, text: str) -> ChunkingResult:
        """
        Chunk text by tokens with sliding window overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            ChunkingResult with all chunks and metadata
        """
        if not text or not text.strip():
            return ChunkingResult(
                chunks=[],
                total_tokens=0,
                original_length=0,
                chunking_strategy="empty_text"
            )
        
        total_tokens = self.count_tokens(text)
        
        # If text is small enough, return as single chunk
        if total_tokens <= self.chunk_size:
            chunk = TextChunk(
                text=text,
                start_token=0,
                end_token=total_tokens,
                chunk_index=0,
                total_chunks=1
            )
            return ChunkingResult(
                chunks=[chunk],
                total_tokens=total_tokens,
                original_length=len(text),
                chunking_strategy="single_chunk"
            )
        
        # Tokenize the text
        tokens = self.tokenize(text)
        chunks = []
        
        # Calculate step size (chunk_size - overlap_size)
        step_size = self.chunk_size - self.overlap_size
        
        chunk_index = 0
        start_token = 0
        
        while start_token < len(tokens) and chunk_index < self.max_chunks:
            # Calculate end token for this chunk
            end_token = min(start_token + self.chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.detokenize(chunk_tokens)
            
            # Create chunk with metadata
            chunk = TextChunk(
                text=chunk_text,
                start_token=start_token,
                end_token=end_token,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated after all chunks are created
                overlap_start=start_token > 0,
                overlap_end=end_token < len(tokens)
            )
            
            chunks.append(chunk)
            
            # Move to next chunk position
            start_token += step_size
            chunk_index += 1
            
            # Break if we've covered all tokens
            if end_token >= len(tokens):
                break
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            original_length=len(text),
            chunking_strategy="sliding_window"
        )
    
    def chunk_by_sentences(self, text: str) -> ChunkingResult:
        """
        Chunk text by sentences with token limits.
        
        This method tries to respect sentence boundaries while staying
        within token limits.
        
        Args:
            text: Input text to chunk
            
        Returns:
            ChunkingResult with sentence-aware chunks
        """
        if not text or not text.strip():
            return ChunkingResult(
                chunks=[],
                total_tokens=0,
                original_length=0,
                chunking_strategy="empty_text"
            )
        
        total_tokens = self.count_tokens(text)
        
        # If text is small enough, return as single chunk
        if total_tokens <= self.chunk_size:
            chunk = TextChunk(
                text=text,
                start_token=0,
                end_token=total_tokens,
                chunk_index=0,
                total_chunks=1
            )
            return ChunkingResult(
                chunks=[chunk],
                total_tokens=total_tokens,
                original_length=len(text),
                chunking_strategy="single_chunk"
            )
        
        # Find sentence boundaries
        sentence_boundaries = self.find_sentence_boundaries(text)
        chunks = []
        
        current_start = 0
        chunk_index = 0
        
        while current_start < len(text) and chunk_index < self.max_chunks:
            current_text = ""
            current_tokens = 0
            last_sentence_end = current_start
            
            # Add sentences until we reach token limit
            for boundary in sentence_boundaries:
                if boundary <= current_start:
                    continue
                
                candidate_text = text[current_start:boundary]
                candidate_tokens = self.count_tokens(candidate_text)
                
                if candidate_tokens <= self.chunk_size:
                    current_text = candidate_text
                    current_tokens = candidate_tokens
                    last_sentence_end = boundary
                else:
                    break
            
            # If no complete sentence fits, take what we can
            if not current_text:
                # Fall back to token-based chunking for this part
                remaining_text = text[current_start:]
                tokens = self.tokenize(remaining_text)
                chunk_tokens = tokens[:self.chunk_size]
                current_text = self.detokenize(chunk_tokens)
                current_tokens = len(chunk_tokens)
                last_sentence_end = current_start + len(current_text)
            
            # Create chunk
            chunk = TextChunk(
                text=current_text,
                start_token=self.count_tokens(text[:current_start]),
                end_token=self.count_tokens(text[:last_sentence_end]),
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                overlap_start=current_start > 0,
                overlap_end=last_sentence_end < len(text)
            )
            
            chunks.append(chunk)
            
            # Move to next position with overlap
            overlap_chars = min(len(current_text) // 4, len(current_text) - 1)
            current_start = max(last_sentence_end - overlap_chars, last_sentence_end)
            chunk_index += 1
            
            if last_sentence_end >= len(text):
                break
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            original_length=len(text),
            chunking_strategy="sentence_aware"
        )
    
    def chunk_text(self, text: str, strategy: str = "sliding_window") -> ChunkingResult:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy ("sliding_window" or "sentence_aware")
            
        Returns:
            ChunkingResult with chunks and metadata
        """
        if not settings.CHUNKING_ENABLED:
            # Return single chunk if chunking is disabled
            total_tokens = self.count_tokens(text)
            chunk = TextChunk(
                text=text,
                start_token=0,
                end_token=total_tokens,
                chunk_index=0,
                total_chunks=1
            )
            return ChunkingResult(
                chunks=[chunk],
                total_tokens=total_tokens,
                original_length=len(text),
                chunking_strategy="disabled"
            )
        
        if strategy == "sentence_aware":
            return self.chunk_by_sentences(text)
        else:
            return self.chunk_by_tokens(text)
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about chunker configuration."""
        return {
            "chunk_size": self.chunk_size,
            "overlap_size": self.overlap_size,
            "max_chunks": self.max_chunks,
            "step_size": self.chunk_size - self.overlap_size,
            "tokenizer_available": self.tokenizer is not None,
            "chunking_enabled": settings.CHUNKING_ENABLED
        }


# Global chunker instance
_text_chunker = None


def get_text_chunker() -> TextChunker:
    """Get or create the global text chunker instance."""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker


def chunk_text_service(text: str, strategy: str = "sliding_window") -> Dict[str, Any]:
    """
    Service function for text chunking.
    
    Args:
        text: Input text to chunk
        strategy: Chunking strategy
        
    Returns:
        Dictionary with chunking results
    """
    try:
        chunker = get_text_chunker()
        result = chunker.chunk_text(text, strategy)
        
        return {
            "status": "success",
            "message": f"Text chunked successfully using {result.chunking_strategy} strategy",
            "original_length": result.original_length,
            "total_tokens": result.total_tokens,
            "total_chunks": len(result.chunks),
            "strategy": result.chunking_strategy,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "start_token": chunk.start_token,
                    "end_token": chunk.end_token,
                    "token_count": chunk.end_token - chunk.start_token,
                    "character_count": len(chunk.text),
                    "has_overlap_start": chunk.overlap_start,
                    "has_overlap_end": chunk.overlap_end
                }
                for chunk in result.chunks
            ],
            "chunker_config": chunker.get_chunk_info()
        }
    
    except Exception as e:
        logger.error(f"Error in text chunking service: {str(e)}")
        return {
            "status": "error",
            "message": f"Text chunking error: {str(e)}",
            "chunks": []
        }


def aggregate_chunk_results(chunk_results: List[Dict[str, Any]], 
                          aggregation_strategy: str = "majority") -> Dict[str, Any]:
    """
    Aggregate results from multiple text chunks.
    
    Args:
        chunk_results: List of results from processing individual chunks
        aggregation_strategy: Strategy for aggregation ("majority", "average", "max_confidence")
        
    Returns:
        Aggregated result
    """
    if not chunk_results:
        return {
            "status": "error",
            "message": "No chunk results to aggregate"
        }
    
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    try:
        if aggregation_strategy == "majority":
            # Priority-based aggregation: if any chunk detects profanity, prioritize that
            profane_chunks = [result for result in chunk_results if result.get("category", "").lower() == "profane"]
            clean_chunks = [result for result in chunk_results if result.get("category", "").lower() in ["clean", "non-profane"]]
            
            if profane_chunks:
                # If profanity detected in any chunk, prioritize profane result
                # Use the highest confidence profane chunk
                best_profane = max(profane_chunks, key=lambda x: x.get("confidence", 0.0))
                
                logger.info(f"Aggregation (priority): Found {len(profane_chunks)} profane chunks out of {len(chunk_results)}. "
                           f"Using profane result with confidence {best_profane.get('confidence', 0.0):.3f} (0-1 range)")
                
                return {
                    "status": "success",
                    "category": "profane",
                    "confidence": best_profane.get("confidence", 0.0),
                    "chunk_count": len(chunk_results),
                    "profane_chunks": len(profane_chunks),
                    "clean_chunks": len(clean_chunks),
                    "aggregation_strategy": "priority_profane",
                    "best_profane_chunk": best_profane,
                    "individual_results": chunk_results
                }
            else:
                # All chunks are clean, use average confidence or highest confidence clean result
                if clean_chunks:
                    best_clean = max(clean_chunks, key=lambda x: x.get("confidence", 0.0))
                    avg_confidence = sum(result.get("confidence", 0.0) for result in clean_chunks) / len(clean_chunks)
                    
                    logger.info(f"Aggregation (priority): All {len(chunk_results)} chunks are clean. "
                               f"Using average confidence {avg_confidence:.3f} (0-1 range)")
                    
                    return {
                        "status": "success",
                        "category": "clean",
                        "confidence": avg_confidence,
                        "chunk_count": len(chunk_results),
                        "profane_chunks": 0,
                        "clean_chunks": len(clean_chunks),
                        "aggregation_strategy": "priority_clean",
                        "best_clean_chunk": best_clean,
                        "individual_results": chunk_results
                    }
                else:
                    # Fallback to traditional majority vote
                    categories = [result.get("category", "unknown") for result in chunk_results]
                    from collections import Counter
                    most_common = Counter(categories).most_common(1)[0]
                    
                    return {
                        "status": "success",
                        "category": most_common[0],
                        "confidence": most_common[1] / len(categories),
                        "chunk_count": len(chunk_results),
                        "aggregation_strategy": "majority_fallback",
                        "individual_results": chunk_results
                    }
        
        elif aggregation_strategy == "average":
            # Average confidence scores
            confidences = [result.get("confidence", 0.0) for result in chunk_results]
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                "status": "success",
                "confidence": avg_confidence,
                "chunk_count": len(chunk_results),
                "aggregation_strategy": "average",
                "individual_results": chunk_results
            }
        
        elif aggregation_strategy == "max_confidence":
            # Take result with highest confidence
            max_result = max(chunk_results, key=lambda x: x.get("confidence", 0.0))
            max_result["aggregation_strategy"] = "max_confidence"
            max_result["chunk_count"] = len(chunk_results)
            max_result["individual_results"] = chunk_results
            
            return max_result
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
    
    except Exception as e:
        logger.error(f"Error in result aggregation: {str(e)}")
        return {
            "status": "error",
            "message": f"Aggregation error: {str(e)}",
            "individual_results": chunk_results
        }
