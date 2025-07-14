#!/usr/bin/env python3
"""
Quick test to verify the confidence aggregation fix
"""

import sys
import os
sys.path.append('/home/alenkuriakose/Project Data/content-moderation-service')

from src.services.text_chunking_service import aggregate_chunk_results

def test_confidence_aggregation():
    """Test confidence aggregation with sample chunk results"""
    
    # Sample chunk results with confidence values in 0-1 range (as fixed)
    chunk_results = [
        {
            "category": "profane",
            "confidence": 0.85,  # 85%
            "confidence_percentage": 85.0,
            "chunk_index": 0
        },
        {
            "category": "clean", 
            "confidence": 0.92,  # 92%
            "confidence_percentage": 92.0,
            "chunk_index": 1
        },
        {
            "category": "profane",
            "confidence": 0.78,  # 78%
            "confidence_percentage": 78.0,
            "chunk_index": 2
        }
    ]
    
    print("Testing confidence aggregation...")
    print(f"Input chunk results:")
    for result in chunk_results:
        print(f"  Chunk {result['chunk_index']}: {result['category']} (confidence: {result['confidence']:.3f})")
    
    # Test majority aggregation
    aggregated = aggregate_chunk_results(chunk_results, "majority")
    
    print(f"\nAggregated result:")
    print(f"  Status: {aggregated.get('status')}")
    print(f"  Category: {aggregated.get('category')}")
    print(f"  Confidence: {aggregated.get('confidence', 0.0):.3f} (0-1 range)")
    print(f"  Confidence as %: {aggregated.get('confidence', 0.0) * 100:.1f}%")
    print(f"  Strategy: {aggregated.get('aggregation_strategy')}")
    print(f"  Profane chunks: {aggregated.get('profane_chunks', 0)}")
    print(f"  Clean chunks: {aggregated.get('clean_chunks', 0)}")
    
    # Verify the confidence is not zero
    confidence = aggregated.get('confidence', 0.0)
    if confidence == 0.0:
        print("❌ ERROR: Confidence is still 0!")
        return False
    else:
        print(f"✅ SUCCESS: Confidence is {confidence:.3f} ({confidence * 100:.1f}%)")
        return True

if __name__ == "__main__":
    success = test_confidence_aggregation()
    sys.exit(0 if success else 1)
