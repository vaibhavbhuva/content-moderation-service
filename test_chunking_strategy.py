#!/usr/bin/env python3
"""
Test to verify chunking strategy with 500 tokens and 12 token overlap
"""

import sys
import os
sys.path.append('/home/alenkuriakose/Project Data/content-moderation-service')

from src.services.text_chunking_service import TextChunker, chunk_text_service
from src.core.config import settings

def test_chunking_strategy():
    """Test chunking strategy with the configured parameters"""
    
    print("Testing chunking strategy...")
    print(f"Configuration: CHUNK_SIZE={settings.CHUNK_SIZE}, CHUNK_OVERLAP={settings.CHUNK_OVERLAP}")
    
    # Create a long text (simulate ~1500 tokens worth of text)
    # Assuming ~4 characters per token on average
    sample_text = "This is a sample text for testing the chunking strategy. " * 100  # Repeat to make it long
    
    print(f"Sample text length: {len(sample_text)} characters")
    
    # Test the chunking service
    result = chunk_text_service(sample_text, "sliding_window")
    
    if result["status"] == "success":
        print(f"✅ Chunking successful!")
        print(f"Strategy: {result['strategy']}")
        print(f"Original length: {result['original_length']} characters")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Total chunks: {result['total_chunks']}")
        
        # Display chunker configuration
        chunker_config = result.get("chunker_config", {})
        print(f"\nChunker Configuration:")
        print(f"  Chunk size: {chunker_config.get('chunk_size')} tokens")
        print(f"  Overlap size: {chunker_config.get('overlap_size')} tokens")
        print(f"  Step size: {chunker_config.get('step_size')} tokens")
        print(f"  Max chunks: {chunker_config.get('max_chunks')}")
        print(f"  Tokenizer available: {chunker_config.get('tokenizer_available')}")
        
        # Verify the configuration matches expectations
        expected_chunk_size = 500
        expected_overlap = 12
        expected_step_size = expected_chunk_size - expected_overlap
        
        actual_chunk_size = chunker_config.get('chunk_size')
        actual_overlap = chunker_config.get('overlap_size')
        actual_step_size = chunker_config.get('step_size')
        
        print(f"\n📊 Configuration Verification:")
        print(f"  Expected chunk size: {expected_chunk_size}, Actual: {actual_chunk_size} {'✅' if actual_chunk_size == expected_chunk_size else '❌'}")
        print(f"  Expected overlap: {expected_overlap}, Actual: {actual_overlap} {'✅' if actual_overlap == expected_overlap else '❌'}")
        print(f"  Expected step size: {expected_step_size}, Actual: {actual_step_size} {'✅' if actual_step_size == expected_step_size else '❌'}")
        
        # Display chunk details
        print(f"\n📝 Chunk Details:")
        for i, chunk in enumerate(result["chunks"][:3]):  # Show first 3 chunks
            print(f"  Chunk {i+1}:")
            print(f"    Tokens: {chunk['start_token']}-{chunk['end_token']} ({chunk['token_count']} tokens)")
            print(f"    Characters: {chunk['character_count']}")
            print(f"    Overlap start: {chunk['has_overlap_start']}, Overlap end: {chunk['has_overlap_end']}")
            print(f"    Preview: {chunk['text'][:100]}...")
        
        if len(result["chunks"]) > 3:
            print(f"  ... and {len(result['chunks']) - 3} more chunks")
        
        # Verify overlap calculations
        if len(result["chunks"]) > 1:
            print(f"\n🔍 Overlap Verification:")
            for i in range(len(result["chunks"]) - 1):
                current_chunk = result["chunks"][i]
                next_chunk = result["chunks"][i + 1]
                
                expected_next_start = current_chunk["start_token"] + expected_step_size
                actual_next_start = next_chunk["start_token"]
                
                overlap_tokens = current_chunk["end_token"] - next_chunk["start_token"]
                
                print(f"  Chunk {i+1} -> Chunk {i+2}:")
                print(f"    Expected next start: {expected_next_start}, Actual: {actual_next_start} {'✅' if actual_next_start == expected_next_start else '❌'}")
                print(f"    Overlap tokens: {overlap_tokens} {'✅' if overlap_tokens == expected_overlap else '❌'}")
        
        return True
    else:
        print(f"❌ Chunking failed: {result['message']}")
        return False

if __name__ == "__main__":
    success = test_chunking_strategy()
    sys.exit(0 if success else 1)
