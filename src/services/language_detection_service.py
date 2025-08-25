"""
Clean Language Detection System

XLM-RoBERTa based language detection with 97.9% accuracy across 100+ languages.

Usage:
    from src.services.language_detection_service import LanguageDetector
    
    detector = LanguageDetector()
    result = detector.detect("Hello, how are you?")
    print(f"Detected: {result['language']} ({result['confidence']:.2%})")
"""

import time
from typing import Dict, Any
from transformers import pipeline

from src.core.config import settings
from src.core.logger import logger

# Simplified language mappings (commonly used languages)
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'ar': 'Arabic', 'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
    'ur': '', 'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
    'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'cs': 'Czech', 'hu': 'Hungarian',
    'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sk': 'Slovak', 'sl': 'Slovenian',
    'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'el': 'Greek', 'he': 'Hebrew',
    'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Filipino',
    'sw': 'Swahili', 'af': 'Afrikaans', 'zu': 'Zulu', 'yo': 'Yoruba', 'ha': 'Hausa',
    'bn_rom': "Bengali", "hi_rom": "Hindi", "ta_rom": "Tamil", "te_rom": "Telugu",
    "ur_rom": "Urdu"
}

# Language groups for our profanity detection logic
INDIC_LANGUAGES = {'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur', 'hi_rom', 'bn_rom', 'ta_rom', 'te_rom'}
ENGLISH_LANGUAGES = {'en'}


def get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    return LANGUAGE_NAMES.get(code.lower(), f"Unknown ({code})")


def get_language_group(code: str) -> str:
    """Get language group for profanity detection routing."""
    code_lower = code.lower()
    if code_lower in ENGLISH_LANGUAGES:
        return "english"
    elif code_lower in INDIC_LANGUAGES:
        return "indic"
    else:
        return "other"


class LanguageDetector:
    """Simple XLM-RoBERTa based language detector."""

    def __init__(self):
        """Initialize the language detector."""
        try:
            self.classifier = pipeline(
                "text-classification",
                model=settings.LANGUAGE_DETECT_MODEL,
                top_k=5
            )
            logger.info(
                "Successfully loaded XLM-RoBERTa language detection model")
        except Exception as e:
            logger.exception(f"Failed to load language detection model")
            self.classifier = None

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the given text.

        Args:
            text (str): Text to analyze

        Returns:
            Dict with language detection results
        """
        if not self.classifier:
            return {
                'error': 'Language detection model not available',
                'language_code': None,
                'language_name': None,
                'confidence': 0.0
            }

        try:
            results = self.classifier(text)
            if isinstance(results, list) and len(results) > 0:
                predictions = results[0] if isinstance(
                    results[0], list) else results
            else:
                raise ValueError("Unexpected model output format")

            sorted_predictions = sorted(
                predictions, key=lambda x: x['score'], reverse=True)

            # Get top prediction
            top_pred = sorted_predictions[0]
            language_code = top_pred['label'].lower()
            confidence = top_pred['score']
            language_name = get_language_name(language_code)

            return {
                'language_code': language_code,
                'language_name': language_name,
                'confidence': confidence,
                'predictions': [
                    {
                        'language_code': pred['label'].lower(),
                        'language_name': get_language_name(pred['label'].lower()),
                        'confidence': pred['score']
                    }
                    for pred in sorted_predictions
                ]
            }

        except Exception as e:
            logger.exception(f"Error while language detection")
            return {
                'error': "An unexpected error occurred. Please try again later",
                'language_code': None,
                'language_name': None,
                'confidence': 0.0
            }

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        return {
            "name": "XLM-RoBERTa Language Detection",
            "model": settings.LANGUAGE_DETECT_MODEL,
            "accuracy": "97.9%",
            "languages": "100+ languages supported"
        }


# Global instance for reuse
_language_detector = None


def get_language_detector():
    """Get or create the language detector instance"""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector


def detect_language_service(text: str):
    """Enhanced language detection service using clean XLM-RoBERTa detection"""

    try:
        sample_size = settings.LANGUAGE_DETECTION_SAMPLE_SIZE
        lang_detection_sample = str(text)[:sample_size] if len(str(text)) > sample_size else str(text)
        detector = get_language_detector()
        result = detector.detect(lang_detection_sample)
        if 'error' in result:
            return {
                "status": "error",
                "message": result['error']
            }

        return {
            "status": "success",
            "message": "Language detected successfully",
            "detected_language": result['language_code'],
            "language_name": result['language_name'],
            "confidence": result['confidence'],
            "top_predictions": result.get('predictions', [])[:3]
        }
    except Exception as e:
        logger.exception(f"Error in language detection service")
        return {
            "status": "error",
            "message": f"Language detection failed. Please try again later"
        }
