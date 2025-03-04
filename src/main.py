# Entry point for FastAPI.

from fastapi import FastAPI, status
from pydantic import BaseModel
from src.core.config import settings
from src.routes.api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Moderation API!"}

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get(
    "/health",
    tags=["Health Check APIs"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    include_in_schema=True,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")