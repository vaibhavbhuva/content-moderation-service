"""
Common utility functions and helpers.
"""

import re
import logging
from typing import Dict, Any, Optional, Union
import pandas as pd

from src.data.constants import (
    LANGUAGE_NAMES, 
    INDIC_LANGUAGES, 
    ENGLISH_LANGUAGES,
    API_MESSAGES,
    ERROR_CODES
)

logger = logging.getLogger(__name__)


def get_language_name(code: str) -> str:
    """Get human-readable language name from language code."""
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


def validate_text_input(text: Union[str, None], min_chars: int = 5) -> Dict[str, Any]:
    """Validate text input for processing."""
    if pd.isna(text) or not text or str(text).strip() == "":
        return {
            "valid": False,
            "error_code": ERROR_CODES["EMPTY_TEXT"],
            "message": API_MESSAGES["empty_text_error"]
        }
    
    if len(str(text).strip()) < min_chars:
        return {
            "valid": False,
            "error_code": ERROR_CODES["MIN_CHARS"],
            "message": API_MESSAGES["min_chars_error"].format(min_chars=min_chars)
        }
    
    return {
        "valid": True,
        "message": "Text validation passed"
    }


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def create_error_response(
    message: str, 
    error_code: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "status": "error",
        "message": message,
        "error_code": error_code,
        "responseData": None
    }


def create_success_response(
    message: str,
    response_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized success response."""
    response = {
        "status": "success",
        "message": message
    }
    
    if response_data is not None:
        response["responseData"] = response_data
        
    return response


def sanitize_input(text: str, max_length: int = 5000) -> str:
    """Sanitize input text for security and processing."""
    if not text:
        return ""
    
    text = str(text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    # Remove potentially harmful characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text
