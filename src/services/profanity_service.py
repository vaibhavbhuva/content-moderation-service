"""
Profanity Detection Service

This service provides multiple approaches for profanity detection:
1. Transformer-based models (English/Indic)
2. LLM-based detection using Gemini
3. Advanced language detection for code-mixed content
"""

import os
import logging
import pandas as pd
import torch
import numpy as np
import re
from typing import Dict, Any, Optional
from tqdm import tqdm

# Import settings
from src.core.config import settings

# Google Gemini imports
from google import genai
from google.genai import types

# Transformers imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Please install with: pip install transformers torch")

# Import the clean language detection service
from .language_detection_service import get_language_detector
from .text_chunking_service import chunk_text_service, aggregate_chunk_results

logger = logging.getLogger("uvicorn.error")

# Constants
CLEAN_LANG_DETECTOR_NAME = "Clean XLM-RoBERTa"
PROFANITY_CHECK_COMPLETED = "Profanity check completed"

# --- Advanced Language Detection (for backward compatibility) ---
class AdvancedLanguageDetector:
    """Advanced language detector that can identify code-mixed languages"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xlm_tokenizer = None
        self.xlm_model = None
        self.load_xlm_model()
        
        # Enhanced script ranges with more comprehensive coverage
        self.script_ranges = {
            'hindi': (0x0900, 0x097F),      # Devanagari
            'bengali': (0x0980, 0x09FF),    # Bengali
            'tamil': (0x0B80, 0x0BFF),      # Tamil
            'telugu': (0x0C00, 0x0C7F),     # Telugu
            'kannada': (0x0C80, 0x0CFF),    # Kannada
            'malayalam': (0x0D00, 0x0D7F),  # Malayalam
            'gujarati': (0x0A80, 0x0AFF),   # Gujarati
            'punjabi': (0x0A00, 0x0A7F),    # Gurmukhi
            'oriya': (0x0B00, 0x0B7F),      # Oriya
            'marathi': (0x0900, 0x097F),    # Devanagari (same as Hindi)
            'urdu': (0x0600, 0x06FF),       # Arabic script (for Urdu)
        }
        
        # Common English words that appear in code-mixed text
        self.english_indicators = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'a', 'an', 'some', 'any', 'all',
            'no', 'not', 'very', 'so', 'too', 'quite', 'really', 'just', 'only'
        }
        
        # Common Hinglish/code-mixed patterns
        self.code_mixed_patterns = [
            r'\b(kar|kar\w+|kya|hai|hain|tha|the|ho|hota|hoti|nahi|nahin)\b',
            r'\b(main|mein|me|tum|aap|woh|yeh|koi|kuch|sab|sabko)\b',
            r'\b(bhi|bhe|se|pe|ko|ka|ke|ki|mein|mai)\b',
            r'\b(good|bad|nice|cool|awesome|great|ok|okay)\b.*[\u0900-\u097F]',
            r'[\u0900-\u097F].*\b(good|bad|nice|cool|awesome|great|ok|okay)\b'
        ]
    
    def load_xlm_model(self):
        """Load XLM-RoBERTa model for Indic/English language detection"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available")
                
            logger.info("Loading XLM-RoBERTa for Indic/English language detection...")
            # Using a smaller, faster model for language detection
            model_name = "papluca/xlm-roberta-base-language-detection"
            self.xlm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.xlm_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            
            # Focus only on Indic languages and English
            self.xlm_id2lang = {
                4: 'english',
                7: 'hindi', 
                17: 'urdu'
            }
            
            # Map other languages to 'other' for filtering
            self.xlm_relevant_ids = {4, 7, 17}  # english, hindi, urdu
            
            logger.info("✅ XLM-RoBERTa loaded for Indic/English detection")
            self.use_xlm = True
        except Exception as e:
            logger.warning(f"⚠️ Could not load XLM-RoBERTa model: {e}")
            logger.info("Using rule-based detection only")
            self.use_xlm = False
    
    def detect_script_distribution(self, text):
        """Analyze the distribution of different scripts in the text"""
        if pd.isna(text):
            return {}
        
        text = str(text)
        char_counts = {lang: 0 for lang in self.script_ranges}
        english_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():  # Only count alphabetic characters
                char_code = ord(char)
                total_chars += 1
                
                # Check if it's English (basic Latin)
                if 0x0041 <= char_code <= 0x007A:  # A-Z, a-z
                    english_chars += 1
                    continue
                
                # Check Indic scripts
                for lang, (start, end) in self.script_ranges.items():
                    if start <= char_code <= end:
                        char_counts[lang] += 1
                        break
        
        if total_chars == 0:
            return {}
        
        # Calculate percentages
        distribution = {}
        for lang, count in char_counts.items():
            if count > 0:
                distribution[lang] = count / total_chars
        
        if english_chars > 0:
            distribution['english'] = english_chars / total_chars
        
        return distribution
    
    def detect_code_mixing_patterns(self, text):
        """Detect code-mixing patterns in the text"""
        if pd.isna(text):
            return False, []
        
        text = str(text).lower()
        found_patterns = []
        
        for pattern in self.code_mixed_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_patterns.append(pattern)
        
        # Check for English words mixed with non-Latin scripts
        words = text.split()
        english_words = [word for word in words if word.lower() in self.english_indicators]
        has_non_latin = any(ord(char) > 127 for char in text)
        
        is_code_mixed = (
            len(found_patterns) > 0 or 
            (len(english_words) > 0 and has_non_latin)
        )
        
        return is_code_mixed, found_patterns
    
    def detect_language_xlm(self, text):
        """Use XLM-RoBERTa for Indic/English language detection"""
        if not self.use_xlm or pd.isna(text) or str(text).strip() == "":
            return None, 0.0
        
        try:
            # Prepare input for XLM-RoBERTa
            inputs = self.xlm_tokenizer(
                str(text), 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.xlm_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Only return results for relevant languages (Indic + English)
            if predicted_class in self.xlm_relevant_ids:
                detected_lang = self.xlm_id2lang[predicted_class]
                return detected_lang, confidence
            else:
                # Non-relevant language detected, treat as unknown
                return 'unknown', confidence
            
        except Exception as e:
            logger.error(f"XLM-RoBERTa detection error: {e}")
            return None, 0.0
    
    def detect_language_advanced(self, text):
        """Simplified language detection focused on Indic languages, code-mixed, and English"""
        if pd.isna(text) or str(text).strip() == "":
            return "unknown", 0.0, {}
        
        text = str(text)
        
        # Step 1: Use XLM-RoBERTa for initial classification
        xlm_lang, xlm_conf = None, 0.0
        if self.use_xlm:
            xlm_lang, xlm_conf = self.detect_language_xlm(text)
        
        # Step 2: Get script distribution and code-mixing patterns
        script_dist = self.detect_script_distribution(text)
        is_code_mixed, patterns = self.detect_code_mixing_patterns(text)
        
        # Step 3: Simplified decision logic for Indic/English focus
        result_details = {
            "xlm_prediction": xlm_lang,
            "xlm_confidence": xlm_conf,
            "distribution": script_dist,
            "code_mixed": is_code_mixed,
            "patterns": patterns
        }
        
        # High confidence XLM-RoBERTa prediction
        if xlm_lang and xlm_conf > 0.8:
            if xlm_lang == 'english':
                if is_code_mixed:
                    # Check for Indic scripts to identify specific code-mixing
                    indic_scripts = [lang for lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam'] 
                                   if lang in script_dist and script_dist[lang] > 0.1]
                    if indic_scripts:
                        dominant_indic = max(indic_scripts, key=lambda x: script_dist[x])
                        return f"code_mixed_{dominant_indic}_english", xlm_conf, result_details
                    else:
                        return "mixed_english", xlm_conf, result_details
                return "english", xlm_conf, result_details
            elif xlm_lang in ['hindi', 'urdu']:
                return xlm_lang, xlm_conf, result_details
        
        # Medium confidence or no XLM - use script analysis
        if script_dist:
            sorted_scripts = sorted(script_dist.items(), key=lambda x: x[1], reverse=True)
            
            # Check for English + Indic combination (code-mixing)
            if 'english' in script_dist and any(lang in script_dist for lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam']):
                indic_langs = [lang for lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam'] if lang in script_dist]
                if indic_langs and (is_code_mixed or script_dist['english'] > 0.2):
                    dominant_indic = max(indic_langs, key=lambda x: script_dist[x])
                    confidence = script_dist['english'] + script_dist[dominant_indic]
                    return f"code_mixed_{dominant_indic}_english", confidence, result_details
            
            # Single dominant script
            if sorted_scripts:
                primary_lang, primary_conf = sorted_scripts[0]
                if primary_conf > 0.6:
                    if primary_lang in ['hindi', 'tamil', 'telugu', 'bengali', 'kannada', 'malayalam', 'urdu']:
                        return primary_lang, primary_conf, result_details
                    elif primary_lang == 'english':
                        if is_code_mixed:
                            return "mixed_english", primary_conf, result_details
                        return "english", primary_conf, result_details
        
        # Fallback: if we detect code-mixing patterns but unclear scripts
        if is_code_mixed:
            return "code_mixed_unknown", 0.5, result_details
        
        # Default to English if XLM suggested English with lower confidence
        if xlm_lang == 'english':
            return "english", xlm_conf, result_details
        
        # Final fallback
        return "unknown", 0.0, result_details


# Global instances for reuse
_advanced_language_detector = None
_transformer_models = {
    'english': None,
    'indic': None
}


def get_advanced_language_detector():
    """Get or create the advanced language detector instance"""
    global _advanced_language_detector
    if _advanced_language_detector is None:
        _advanced_language_detector = AdvancedLanguageDetector()
    return _advanced_language_detector


def _load_english_model():
    """Load English profanity detection model"""
    if _transformer_models['english'] is not None:
        return _transformer_models['english']
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'NOT_TOXIC', 1: 'TOXIC'}
    _transformer_models['english'] = (tokenizer, model, id2label, device)
    return _transformer_models['english']


def _load_indic_model():
    """Load Indic profanity detection model"""
    if _transformer_models['indic'] is not None:
        return _transformer_models['indic']
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # Official MuRIL model label mapping (from config.json)
    id2label = {0: 'Normal', 1: 'Abusive'}
    
    _transformer_models['indic'] = (tokenizer, model, id2label, device)
    return _transformer_models['indic']


def _process_english_model(text: str, detected_lang: str, lang_result: dict) -> Dict[str, Any]:
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
        "message": PROFANITY_CHECK_COMPLETED,
        "responseData": {
            "text": text,
            "isProfane": main_label == 'Profane',
            "confidence": round(main_confidence*100, 2),
            "category": main_label,
            "detected_language": detected_lang,
            "language_detection_sample_size": lang_result.get("sample_size", "unknown"),
            "model_used": "English (toxic-bert)"
        }
    }


def _process_indic_model(text: str, detected_lang: str, lang_result: dict) -> Dict[str, Any]:
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
        "message": PROFANITY_CHECK_COMPLETED,
        "responseData": {
            "text": text,
            "isProfane": is_profane,
            "confidence": round(conf*100, 2),
            "category": category,
            "detected_language": detected_lang,
            "language_detection_sample_size": lang_result.get("sample_size", "unknown"),
            "model_used": "Indic (MuRIL)"
        }
    }


def check_profanity_transformer(text: str) -> Dict[str, Any]:
    """
    Detect profanity using transformer models (English/Indic) with clean language detection.
    Uses limited characters for language detection to improve performance.
    
    Args:
        text: Input text to check for profanity
        
    Returns:
        Dictionary with status, message, and responseData
    """
    logger.info(f"Checking profanity (transformer) for text length: {len(text)}")
    
    if pd.isna(text) or str(text).strip() == "":
        return {
            "status": "error",
            "message": "Input text is empty",
            "responseData": None
        }
    
    try:
        # Use limited characters for language detection (configurable sample size)
        sample_size = settings.LANGUAGE_DETECTION_SAMPLE_SIZE
        lang_detection_sample = str(text)[:sample_size] if len(str(text)) > sample_size else str(text)
        logger.info(f"Using {len(lang_detection_sample)} characters for language detection (from {len(str(text))} total, sample_size={sample_size})")
        
        # Use clean language detection from new service
        detector = get_language_detector()
        lang_result = detector.detect(lang_detection_sample)
        
        if 'error' in lang_result:
            logger.warning(f"Language detection failed: {lang_result['error']}")
            # Fallback to English model if detection fails
            return _process_english_model(text, "unknown", {"model_used": CLEAN_LANG_DETECTOR_NAME, "fallback": True, "sample_size": len(lang_detection_sample)})
        
        # Add sample size information to lang_result
        lang_result["sample_size"] = len(lang_detection_sample)
        lang_result["total_text_length"] = len(str(text))
        
        detected_lang = lang_result.get('language_code', 'unknown')
        language_group = lang_result.get('language_group', 'other')
        confidence = lang_result.get('confidence', 0.0)
        
        logger.info(f"Language detection from sample: {detected_lang} ({language_group}), Confidence: {confidence:.3f}")
        
        # Model selection based on language group (but process full text)
        if language_group == "english":
            # Use English model for English content
            return _process_english_model(text, detected_lang, lang_result)
        elif language_group == "indic":
            # Use Indic model for Indic languages
            return _process_indic_model(text, detected_lang, lang_result)
        else:
            # Use English model as default for other/unknown languages
            logger.info(f"Using English model as fallback for language: {detected_lang}")
            return _process_english_model(text, detected_lang, lang_result)
            
    except Exception as e:
        logger.error(f"Transformer profanity detection error: {str(e)}")
        return {
            "status": "error",
            "message": f"Transformer model error: {str(e)}",
            "responseData": None
        }


def check_profanity_llm(text: str) -> Dict[str, Any]:
    """
    Check profanity using Gemini LLM.
    
    Args:
        text: Input text to check for profanity
        
    Returns:
        Dictionary with status, message, and responseData
    """
    logger.info(f"Checking profanity (LLM) for: {text}")
    
    if pd.isna(text) or str(text).strip() == "":
        return {
            "status": "error",
            "message": "Input text is empty",
            "responseData": None
        }
    
    try:
        # Check if Gemini API key is available
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            return {
                "status": "error",
                "message": "Gemini API key not configured. Please set GEMINI_API_KEY in your environment.",
                "responseData": None
            }
        
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash-preview-04-17"
        
        # Prepare the prompt and schema
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"Analyze the following text for profanity and respond with a JSON object containing:\n- 'contains_profanity': boolean (true if profanity is detected, false otherwise)\n- 'confidence': number between 0-100 (confidence percentage in your assessment)\n- 'reasoning': string (brief explanation of your decision, mentioning specific words or patterns if profanity is found)\n\nText to analyze: \"{text}\"\n\nRespond only with the JSON object, no additional text.")
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["contains_profanity", "confidence", "reasoning"],
                properties={
                    "contains_profanity": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                        description="Whether profanity was detected in the text",
                    ),
                    "confidence": genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                        description="Confidence percentage (0-100) in the profanity assessment",
                    ),
                    "reasoning": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Brief explanation of the decision, mentioning specific words or patterns if profanity is found",
                    ),
                },
            ),
            system_instruction=[
                types.Part.from_text(text="""Analyze the following text for profanity also keep the context of entire sentence in mind and respond with a JSON object containing:
                    - "contains_profanity": boolean (true if profanity is detected, false otherwise)
                    - "confidence": number between 0-100 (confidence percentage in your assessment)
                    - "reasoning": string (explanation of your decision, mentioning specific words or patterns if profanity is found and a brief explanation of your reasoning)

                    Text to analyze: "{text}"

                    Respond only with the JSON object, no additional text."""
                                     ),
            ],
        )
        
        output = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            output += chunk.text
        
        import json
        data = json.loads(output)
        is_profane = data.get("contains_profanity", False)
        confidence = data.get("confidence", 0)
        reasoning = data.get("reasoning", "")
        category = "profane" if is_profane else "clean"
        
        return {
            "status": "success",
            "message": PROFANITY_CHECK_COMPLETED,
            "responseData": {
                "text": text,
                "isProfane": is_profane,
                "confidence": confidence,
                "category": category,
                "reasoning": reasoning,
                "model_used": "Gemini LLM"
            }
        }
        
    except Exception as e:
        logger.error(f"Error during LLM profanity check: {e}")
        return {
            "status": "error",
            "message": f"LLM profanity check error: {str(e)}",
            "responseData": None
        }


# Enhanced chunking-aware profanity detection functions

def check_profanity_transformer_chunked(text: str, use_chunking: bool = True) -> Dict[str, Any]:
    """
    Check profanity using transformer models with optional text chunking.
    
    Args:
        text: Input text to check for profanity
        use_chunking: Whether to use text chunking for long texts
        
    Returns:
        Dictionary with status, message, and responseData
    """
    logger.info(f"Checking profanity (transformer+chunking) for text length: {len(text)}")
    
    if not text or not text.strip():
        return {
            "status": "error",
            "message": "Input text is empty",
            "responseData": None
        }
    
    try:
        # Check if chunking is needed and enabled
        print(f"use_chunking: {use_chunking}", settings.CHUNKING_ENABLED, len(text), settings.MAX_TEXT_LENGTH)
        if use_chunking and settings.CHUNKING_ENABLED and len(text) > settings.MAX_TEXT_LENGTH:
            logger.info(f"Text length {len(text)} exceeds limit ({settings.MAX_TEXT_LENGTH}), initiating chunking process")
            logger.info(f"Chunking configuration: chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}, max_chunks={settings.MAX_CHUNKS_PER_TEXT}")
            
            # Chunk the text
            chunking_result = chunk_text_service(text, "sliding_window")
            
            if chunking_result["status"] == "error":
                logger.warning(f"Chunking failed: {chunking_result['message']}")
                logger.info("Falling back to processing truncated text without chunking")
                # Fall back to processing without chunking (truncated)
                return check_profanity_transformer(text[:settings.MAX_TEXT_LENGTH])
            
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
                
                chunk_result = check_profanity_transformer(chunk_text)
                
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
            # Use regular processing for short texts
            if not use_chunking:
                logger.info(f"Chunking disabled by parameter, processing text normally (length: {len(text)})")
            elif not settings.CHUNKING_ENABLED:
                logger.info(f"Chunking disabled in settings, processing text normally (length: {len(text)})")
            else:
                logger.info(f"Text length {len(text)} <= {settings.MAX_TEXT_LENGTH}, processing without chunking")
            
            # Process without chunking and log the result
            result = check_profanity_transformer(text)
            if result.get("status") == "success" and result.get("responseData"):
                response_data = result["responseData"]
                logger.info(f"Single chunk processing result: "
                           f"isProfane={response_data.get('isProfane', False)}, "
                           f"confidence={response_data.get('confidence', 0.0):.1f}%, "
                           f"category='{response_data.get('category', 'unknown')}', "
                           f"model='{response_data.get('model_used', 'unknown')}', "
                           f"language='{response_data.get('detected_language', 'unknown')}'")
            
            return result
    
    except Exception as e:
        logger.error(f"Error in chunked transformer profanity check: {str(e)}")
        return {
            "status": "error",
            "message": f"Chunked profanity check error: {str(e)}",
            "responseData": None
        }
