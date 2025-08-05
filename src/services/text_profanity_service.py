"""
Profanity Detection Service

This service provides multiple approaches for profanity detection:
1. Transformer-based models (English/Indic)
3. Advanced language detection for code-mixed content
"""


import json
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.core.config import settings
from src.core.logger import logger
from .language_detection_service import get_language_group
from .text_chunking_service import chunk_text_service, aggregate_chunk_results

# Global instances for reuse
_transformer_models = {
    'english': None,
    'indic': None
}


def _load_english_model():
    """Load English profanity detection model"""
    if _transformer_models['english'] is not None:
        return _transformer_models['english']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(settings.ENGLISH_TRANSFORMER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(settings.ENGLISH_TRANSFORMER_MODEL).to(device)
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'NOT_TOXIC', 1: 'TOXIC'}
    _transformer_models['english'] = (tokenizer, model, id2label, device)
    return _transformer_models['english']


def _load_indic_model():
    """Load Indic profanity detection model"""
    if _transformer_models['indic'] is not None:
        return _transformer_models['indic']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = settings.INDIC_TRANSFORMER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # Official MuRIL model label mapping (from config.json)
    id2label = {0: 'Normal', 1: 'Abusive'}
    
    _transformer_models['indic'] = (tokenizer, model, id2label, device)
    return _transformer_models['indic']


def _process_english_model(text: str, detected_lang: str) -> Dict[str, Any]:
    """Process text using English toxic-bert model"""
    tokenizer, model, id2label, device = _load_english_model()
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    toxic_indices = [i for i, p in enumerate(probs) if p >= 0.4]
    toxic_labels = [id2label[i] for i in toxic_indices]
    toxic_confidences = [float(probs[i]) for i in toxic_indices]
    max_conf = float(max(probs)) if len(probs) > 0 else 0.0
    
    if toxic_labels:
        main_label = 'Profane'
        main_confidence = max(toxic_confidences)
    else:
        main_label = 'Non-Profane'
        main_confidence = 1.0 - max_conf
    
    # Confidence adjustment
    if main_label == 'Profane' and max_conf < 0.8:
        main_label = 'Non-Profane'
        main_confidence = 1.0 - max_conf
    
    if main_label == 'Non-Profane' and main_confidence < 0.8:
        main_label = 'Profane'
        main_confidence = 1.0 - main_confidence
    
    return {
        "status": "success",
        "message": "Profanity check completed",
        "responseData": {
            "text": text,
            "isProfane": main_label == 'Profane',
            "confidence": round(main_confidence*100, 2),
            "category": main_label,
            "detected_language": detected_lang
        }
    }


def _process_indic_model(text: str, detected_lang: str) -> Dict[str, Any]:
    """Process text using Indic MuRIL model"""
    tokenizer, model, _, device = _load_indic_model()
    encoding = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
    
    pred = predicted_class.cpu().item()
    conf = confidence.cpu().item()
    
    # Use official MuRIL labels: 0='Normal', 1='Abusive'
    if pred == 0:
        category = 'Clean'
        is_profane = False
    elif pred == 1:
        category = 'Profane/Abusive' 
        is_profane = True
    else:
        category = 'Processing Error'
        is_profane = False
    
    return {
        "status": "success",
        "message": "Profanity check completed",
        "responseData": {
            "text": text,
            "isProfane": is_profane,
            "confidence": round(conf*100, 2),
            "category": category,
            "detected_language": detected_lang
        }
    }


def check_profanity_transformer(text: str, language: str) -> Dict[str, Any]:
    """
    Detect profanity using transformer models (English/Indic) with clean language detection.
    Uses limited characters for language detection to improve performance.
    
    Args:
        text: Input text to check for profanity
        
    Returns:
        Dictionary with status, message, and responseData
    """
    
    try:
        language_group = get_language_group(language)
        if language_group == "english":
            profanity_res  = _process_english_model(text, language)
        elif language_group == "indic":
            profanity_res = _process_indic_model(text, language)
        else:
            logger.info(f"Using English model as fallback for language: {language}")
            profanity_res =  _process_english_model(text, language)
        return profanity_res
            
    except Exception as e:
        logger.error(f"Transformer profanity detection error: {str(e)}")
        return {
            "status": "error",
            "message": f"Transformer model error: {str(e)}",
            "responseData": None
        }


# Enhanced chunking-aware profanity detection functions

def check_profanity_transformer_chunked(text: str, language: str) -> Dict[str, Any]:
    """
    Check profanity using transformer models with optional text chunking.
    
    Args:
        text: Input text to check for profanity
        
    Returns:
        Dictionary with status, message, and responseData
    """
    try:
        # Check if chunking is needed and enabled
        if settings.CHUNKING_ENABLED and len(text) > settings.MAX_TEXT_LENGTH:
            logger.info(f"Text length {len(text)} exceeds limit ({settings.MAX_TEXT_LENGTH}), initiating chunking process")
            logger.info(f"Chunking configuration: chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}, max_chunks={settings.MAX_CHUNKS_PER_TEXT}")
            
            # Chunk the text
            chunking_result = chunk_text_service(text, "sliding_window")
            
            if chunking_result["status"] == "error":
                logger.warning(f"Chunking failed: {chunking_result['message']}")
                logger.info("Falling back to processing truncated text without chunking")
                # Fall back to processing without chunking (truncated)
                return check_profanity_transformer(text[:settings.MAX_TEXT_LENGTH], language)
            
            logger.info(f"Text successfully chunked into {len(chunking_result['chunks'])} chunks using {chunking_result.get('strategy', 'unknown')} strategy")
            logger.info(f"Chunking details: original_length={chunking_result.get('original_length', 0)}, "
                       f"total_tokens={chunking_result.get('total_tokens', 0)}, "
                       f"chunks_created={len(chunking_result['chunks'])}")
            
            # Process each chunk
            chunk_results = []
            for chunk_info in chunking_result["chunks"]:
                chunk_text = chunk_info["text"]
                chunk_index = chunk_info["chunk_index"]
                
                logger.info(f"Processing chunk {chunk_index + 1}/{len(chunking_result['chunks'])}: "
                           f"tokens {chunk_info['start_token']}-{chunk_info['end_token']} "
                           f"({chunk_info['token_count']} tokens, {len(chunk_text)} chars)")
                
                chunk_result = check_profanity_transformer(chunk_text, language)
                
                if chunk_result["status"] == "success" and chunk_result["responseData"]:
                    # Extract relevant data for aggregation
                    response_data = chunk_result["responseData"]
                    is_chunk_profane = response_data.get("isProfane", False)
                    chunk_confidence = response_data.get("confidence", 0.0)
                    chunk_category = response_data.get("category", "unknown")
                    
                    # Log detailed individual chunk result
                    logger.info(f"Chunk {chunk_index + 1} result: "
                               f"category='{chunk_category}', "
                               f"isProfane={is_chunk_profane}, "
                               f"confidence={chunk_confidence:.1f}%, "
                               f"model='{response_data.get('model_used', 'unknown')}', "
                               f"language='{response_data.get('detected_language', 'unknown')}', "
                               f"tokens={chunk_info.get('token_count', 0)}, "
                               f"chars={len(chunk_text)}")
                    
                    # Log chunk text preview for debugging
                    chunk_preview = chunk_text[:150].replace('\n', ' ').replace('\r', ' ')
                    if len(chunk_text) > 150:
                        chunk_preview += "..."
                    logger.debug(f"Chunk {chunk_index + 1} text preview: '{chunk_preview}'")
                    
                    # Log reasoning if available (for debugging profanity detection)
                    chunk_reasoning = response_data.get("reasoning", "")
                    if chunk_reasoning:
                        logger.debug(f"Chunk {chunk_index + 1} reasoning: {chunk_reasoning}")
                    
                    chunk_results.append({
                        "category": "profane" if is_chunk_profane else "clean",
                        "confidence": chunk_confidence / 100.0,  # Convert percentage to 0-1 range for aggregation
                        "confidence_percentage": chunk_confidence,  # Keep original percentage for reference
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                        "reasoning": response_data.get("reasoning", ""),
                        "model_used": response_data.get("model_used", "transformer"),
                        "detected_language": response_data.get("detected_language", "unknown"),
                        "chunk_tokens": chunk_info.get("token_count", 0),
                        "chunk_chars": len(chunk_text)
                    })
                else:
                    error_message = chunk_result.get('message', 'Unknown error')
                    logger.warning(f"Chunk {chunk_index + 1} processing failed: {error_message}")
                    logger.warning(f"Failed chunk details: tokens={chunk_info.get('token_count', 0)}, "
                                 f"chars={len(chunk_text)}, "
                                 f"start_token={chunk_info.get('start_token', 0)}, "
                                 f"end_token={chunk_info.get('end_token', 0)}")
                    
                    # Log a preview of the failed chunk for debugging
                    failed_chunk_preview = chunk_text[:100].replace('\n', ' ').replace('\r', ' ')
                    if len(chunk_text) > 100:
                        failed_chunk_preview += "..."
                    logger.debug(f"Failed chunk {chunk_index + 1} text preview: '{failed_chunk_preview}'")
            
            logger.info(f"Chunk processing completed: {len(chunk_results)}/{len(chunking_result['chunks'])} chunks processed successfully")
            
            # Log detailed summary of all chunk results
            if chunk_results:
                logger.info("=== CHUNK RESULTS SUMMARY ===")
                for i, result in enumerate(chunk_results):
                    logger.info(f"  Chunk {i+1}: {result['category']} "
                               f"(confidence: {result['confidence_percentage']:.1f}%, "
                               f"tokens: {result['chunk_tokens']}, "
                               f"chars: {result['chunk_chars']}, "
                               f"model: {result['model_used']}, "
                               f"lang: {result['detected_language']})")
                    
                    # Log reasoning for profane chunks
                    if result['category'] == 'profane' and result.get('reasoning'):
                        logger.info(f"    Profanity reasoning: {result['reasoning']}")
                
                logger.info("=== END CHUNK RESULTS ===")
            
            if not chunk_results:
                logger.error(f"No chunks were processed successfully out of {len(chunking_result['chunks'])} total chunks")
                return {
                    "status": "error",
                    "message": "Failed to process any chunks",
                    "responseData": None
                }
            
            # Log chunk statistics before aggregation
            profane_chunks = sum(1 for r in chunk_results if r["category"] == "profane")
            clean_chunks = len(chunk_results) - profane_chunks
            avg_confidence = sum(r["confidence_percentage"] for r in chunk_results) / len(chunk_results)  # Use percentage for logging
            
            logger.info(f"Chunk analysis summary: {profane_chunks} profane, {clean_chunks} clean, "
                       f"avg_confidence={avg_confidence:.3f}%")
            
            # Aggregate results with priority strategy
            logger.info(f"Aggregating results from {len(chunk_results)} chunks with priority strategy")
            aggregated = aggregate_chunk_results(chunk_results, "majority")
            
            # Log aggregation result (get counts from aggregated result if available)
            final_profane_count = aggregated.get("profane_chunks", profane_chunks)
            final_clean_count = aggregated.get("clean_chunks", clean_chunks)
            
            logger.info(f"Aggregation result: category='{aggregated.get('category', 'unknown')}', "
                       f"confidence={aggregated.get('confidence', 0.0):.3f} (0-1 range), "
                       f"strategy='{aggregated.get('aggregation_strategy', 'unknown')}'")
            
            # Format final response
            is_profane = aggregated.get("category", "clean") == "profane"
            confidence_decimal = aggregated.get("confidence", 0.0)  # This is in 0-1 range from aggregation
            confidence_percentage = round(confidence_decimal * 100, 2)  # Convert back to percentage
            
            # Log final aggregated result with chunk contribution details
            profane_chunk_indices = [r['chunk_index'] + 1 for r in chunk_results if r['category'] == 'profane']
            clean_chunk_indices = [r['chunk_index'] + 1 for r in chunk_results if r['category'] == 'clean']
            
            logger.info(f"Final aggregated result: isProfane={is_profane}, confidence={confidence_percentage:.2f}%, "
                       f"strategy='{aggregated.get('aggregation_strategy', 'priority_based')}'")
            
            if profane_chunk_indices:
                logger.info(f"Profane chunks detected: {profane_chunk_indices}")
                # Log the most confident profane chunk
                profane_chunks = [r for r in chunk_results if r['category'] == 'profane']
                if profane_chunks:
                    best_profane = max(profane_chunks, key=lambda x: x['confidence_percentage'])
                    logger.info(f"Highest confidence profane chunk: #{best_profane['chunk_index'] + 1} "
                               f"(confidence: {best_profane['confidence_percentage']:.1f}%, "
                               f"model: {best_profane['model_used']})")
            
            if clean_chunk_indices:
                logger.info(f"Clean chunks detected: {clean_chunk_indices}")
            
            logger.info(f"Aggregation rationale: {aggregated.get('aggregation_strategy', 'priority_based')} strategy - "
                       f"{'Profanity detected, using highest confidence profane chunk' if is_profane else 'All chunks clean, using average confidence'}")
            return {
                "status": "success",
                "message": f"Profanity check completed using chunking ({len(chunk_results)} chunks)",
                "responseData": {
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "text_length": len(text),
                    "isProfane": is_profane,
                    "confidence": confidence_percentage,  # Return as percentage to match API format
                    "detected_language": language,
                    "category": "profane" if is_profane else "clean",
                    "chunking_used": True,
                    "total_chunks": len(chunk_results),
                    "profane_chunks": final_profane_count,
                    "clean_chunks": final_clean_count,
                    "aggregation_strategy": aggregated.get("aggregation_strategy", "priority_based"),
                    "model_used": "transformer+chunking",
                    "chunk_statistics": {
                        "total_processed": len(chunk_results),
                        "total_attempted": len(chunking_result['chunks']),
                        "profane_detected": final_profane_count,
                        "clean_detected": final_clean_count,
                        "average_confidence": avg_confidence,
                        "best_chunk": aggregated.get("best_profane_chunk") or aggregated.get("best_clean_chunk", {}),
                        "aggregation_rationale": f"Priority-based: {'Profanity detected in any chunk takes priority' if is_profane else 'All chunks clean, using average confidence'}"
                    },
                    "chunk_details": chunk_results
                }
            }
        
        else:
            logger.info(f"Chunking disabled in settings, processing text normally")
            result = check_profanity_transformer(text, language)
            logger.info(f"Profanity response :: {json.dumps(result)}")
            return result
    
    except Exception as e:
        logger.error(f"Error in chunked transformer profanity check: {str(e)}")
        return {
            "status": "error",
            "message": f"Chunked profanity check error: {str(e)}",
            "responseData": None
        }
