import json
from datetime import datetime
from typing import Any, Optional

from src.core.logger  import logger

class MessageSerializer:
    @staticmethod
    def serialize_value(value: Any) -> bytes:
        """Serialize message value to bytes"""
        try:
            if isinstance(value, dict):
                return json.dumps(value, default=MessageSerializer._json_serializer).encode('utf-8')
            elif isinstance(value, str):
                return value.encode('utf-8')
            else:
                return json.dumps(value, default=MessageSerializer._json_serializer).encode('utf-8')
        except Exception as e:
            logger.exception(f"Error serializing message value: {e}")
            raise

    @staticmethod
    def serialize_key(key: Optional[str]) -> Optional[bytes]:
        """Serialize message key to bytes"""
        return key.encode('utf-8') if key else None

    @staticmethod
    def _json_serializer(obj):
        """JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
