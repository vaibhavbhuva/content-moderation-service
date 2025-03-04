from pydantic import BaseModel
from typing import List

class TextModerationRequest(BaseModel):
    text: str

class TextModerationResponse(BaseModel):
    category: str
    confidence: float

class ImageModerationResponse(BaseModel):
    adult: str
    medical: str
    spoofed: str
    violence: str
    racy: str