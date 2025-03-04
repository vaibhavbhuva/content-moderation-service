from typing import List
from fastapi import APIRouter, HTTPException
from src.services.language_service import LanguageService
from google.cloud import language_v1 as language
from src.schemas.moderation import TextModerationRequest, TextModerationResponse


router = APIRouter()
language_service = LanguageService()

@router.post("/moderate_text/", response_model=List[TextModerationResponse])
async def moderate_text_api(request: TextModerationRequest):
    """Moderates the given text using Google Cloud Natural Language API."""
    try:
        response = language_service.moderate_text(request.text)


        def confidence(category: language.ClassificationCategory) -> float:
            return category.confidence

        categories = response.moderation_categories
        sorted_categories = sorted(categories, key=confidence, reverse=True)
        data = [{"category": category.name, "confidence": category.confidence} for category in sorted_categories]
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during text moderation: {str(e)}")
