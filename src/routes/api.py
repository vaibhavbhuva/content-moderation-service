from fastapi import APIRouter
from src.api.v1 import profanity_controller, language_controller

api_router = APIRouter()
api_router.include_router(profanity_controller.router)
api_router.include_router(language_controller.router)