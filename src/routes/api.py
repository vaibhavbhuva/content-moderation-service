from fastapi import APIRouter
from src.api.v1 import image_moderation, text_moderation

api_router = APIRouter()
api_router.include_router(image_moderation.router)
api_router.include_router(text_moderation.router)