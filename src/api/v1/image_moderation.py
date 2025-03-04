from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.vision_service import VisionService
from src.schemas.moderation import ImageModerationResponse

router = APIRouter()
vision_service = VisionService()

@router.post("/moderate_image/", response_model=ImageModerationResponse)
async def moderate_image_api(file: UploadFile = File(...)):
    """Detects unsafe features in the uploaded image using Google Cloud Vision API."""
    try:
        content = await file.read()
        response = vision_service.detect_safe_search(content)
        print("----->",response)
        safe = response.safe_search_annotation

        likelihood_name = (
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
        )

        result = {
            "adult": likelihood_name[safe.adult],
            "medical": likelihood_name[safe.medical],
            "spoofed": likelihood_name[safe.spoof],
            "violence": likelihood_name[safe.violence],
            "racy": likelihood_name[safe.racy],
        }

        if response.error.message:
            raise HTTPException(status_code=500, detail=f"Vision API error: {response.error.message}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image moderation: {str(e)}")
