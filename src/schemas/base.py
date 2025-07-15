"""
Base schema models for the content moderation service.
"""

from pydantic import BaseModel as PydanticBaseModel
from typing import Any, Dict, Optional


class BaseModel(PydanticBaseModel):
    """Base model for all schemas."""
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        use_enum_values = True
        validate_assignment = True
        populate_by_name = True  # Allow both alias and field names


class ApiResponse(BaseModel):
    """Standard API response model."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
