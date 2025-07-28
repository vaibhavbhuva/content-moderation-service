"""
Main entry point for the Content Moderation API.

This FastAPI application provides content moderation capabilities including:
- Text profanity detection using LLM and transformer models
- Language detection with support for 100+ languages
- Configurable rate limiting for API protection
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from src.core.config import settings
from src.core.middleware import setup_cors
from src.routes.api import api_router
from src.core.logger import logger

from src.services.kafka_service import KafkaIntegrationService

# Global service instances
kafka_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    try:
        global kafka_service
        # Perform startup tasks here
        logger.info("Initializing application components...")
        
        if settings.KAFKA_ENABLED:
            kafka_service = KafkaIntegrationService()

        # Add your startup tasks:
        # - Model loading
        # - Database connections
        # - Cache warming
        # - External service health checks
        
        logger.info("✅ Application startup completed successfully")
        
    except Exception as e:
        logger.critical(f"❌ Application startup failed: {str(e)}", exc_info=True)
        raise
    
    yield

    logger.info("Shutting down Profanity Detection Service...")
    if kafka_service:
        # Send shutdown kafka service
        kafka_service.flush_messages()
        kafka_service.shutdown()
    
    # Shutdown
    logger.info("🛑 Service shutdown complete   ")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Content Moderation API with AI-powered text analysis",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Setup middleware
setup_cors(app)

class HealthCheck(BaseModel):
    """Response model for health check endpoint."""
    status: str = "OK"
    version: str = settings.PROJECT_VERSION

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Content Moderation API!",
        "version": settings.PROJECT_VERSION,
        "status": "running"
    }


@app.get(
    "/health",
    tags=["Health Check"],
    summary="Perform a Health Check",
    response_model=HealthCheck,
)
async def get_health() -> HealthCheck:
    """Health check endpoint."""
    logger.debug("Health check completed successfully")
    return HealthCheck()

# Include API router
logger.info("📡 Including API routes...")
app.include_router(api_router, prefix=settings.API_V1_STR)
logger.info("✅ API routes configured")