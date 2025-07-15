"""
Configuration constants and settings for the content moderation service.
"""

from typing import Dict, List, Set

# Language mappings and constants
LANGUAGE_NAMES: Dict[str, str] = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'ar': 'Arabic', 'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
    'ur': 'Urdu', 'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
    'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'cs': 'Czech', 'hu': 'Hungarian',
    'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sk': 'Slovak', 'sl': 'Slovenian',
    'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'el': 'Greek', 'he': 'Hebrew',
    'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Filipino',
    'sw': 'Swahili', 'af': 'Afrikaans', 'zu': 'Zulu', 'yo': 'Yoruba', 'ha': 'Hausa'
}

# Language groups for profanity detection routing
INDIC_LANGUAGES: Set[str] = {'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur'}
ENGLISH_LANGUAGES: Set[str] = {'en'}

# Model configurations
MODEL_CONFIGS = {
    "language_detection": {
        "model_name": "ZheYu03/xlm-r-langdetect-model",
        "task": "text-classification",
        "accuracy": "97.9%",
        "languages_supported": "100+"
    },
    "english_profanity": {
        "model_name": "unitary/toxic-bert",
        "task": "text-classification",
        "threshold": 0.4
    },
    "indic_profanity": {
        "model_name": "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL",
        "task": "text-classification",
        "labels": {0: 'Normal', 1: 'Abusive'}
    },
    "gemini_llm": {
        "model_name": "gemini-2.5-flash-preview-04-17",
        "temperature": 0,
        "response_format": "application/json"
    }
}

# API response messages
API_MESSAGES = {
    "profanity_check_completed": "Profanity check completed",
    "language_detection_completed": "Language detection completed",
    "empty_text_error": "Input text is empty",
    "min_chars_error": "Input text must be at least {min_chars} characters",
    "model_not_available": "Language detection model not available",
    "transformer_error": "Transformers library not available. Please install with: pip install transformers torch"
}

# Error codes
ERROR_CODES = {
    "EMPTY_TEXT": "EMPTY_TEXT",
    "MIN_CHARS": "MIN_CHARS_NOT_MET", 
    "MODEL_UNAVAILABLE": "MODEL_UNAVAILABLE",
    "TRANSFORMER_MISSING": "TRANSFORMER_MISSING",
    "API_KEY_MISSING": "API_KEY_MISSING",
    "INTERNAL_ERROR": "INTERNAL_ERROR"
}

# Script ranges for language detection
SCRIPT_RANGES = {
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

# Common English indicators for code-mixed text detection
ENGLISH_INDICATORS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
    'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'this', 'that', 'these', 'those', 'a', 'an', 'some', 'any', 'all',
    'no', 'not', 'very', 'so', 'too', 'quite', 'really', 'just', 'only'
}

# Code-mixed patterns for detection
CODE_MIXED_PATTERNS = [
    r'\b(kar|kar\w+|kya|hai|hain|tha|the|ho|hota|hoti|nahi|nahin)\b',
    r'\b(main|mein|me|tum|aap|woh|yeh|koi|kuch|sab|sabko)\b',
    r'\b(bhi|bhe|se|pe|ko|ka|ke|ki|mein|mai)\b',
    r'\b(good|bad|nice|cool|awesome|great|ok|okay)\b.*[\u0900-\u097F]',
    r'[\u0900-\u097F].*\b(good|bad|nice|cool|awesome|great|ok|okay)\b'
]
