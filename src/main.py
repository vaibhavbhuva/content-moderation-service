"""
Main entry point for the Content Moderation API.

This FastAPI application provides content moderation capabilities including:
- Text profanity detection using LLM and transformer models
- Language detection with support for 100+ languages
- Configurable rate limiting for API protection
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, Request
from pydantic import BaseModel

from src.core.config import settings
from src.core.middleware import setup_cors
from src.core.rate_limiter import rate_limit_middleware, cleanup_rate_limiter, get_rate_limiter, get_rate_limit_key
from src.routes.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    if settings.RATE_LIMIT_ENABLED:
        # Start background cleanup task for rate limiter
        cleanup_task = asyncio.create_task(cleanup_rate_limiter())
        app.state.cleanup_task = cleanup_task
    
    yield
    
    # Shutdown
    if hasattr(app.state, 'cleanup_task'):
        app.state.cleanup_task.cancel()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="Content Moderation API with AI-powered text analysis",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Setup middleware
setup_cors(app)

# Add rate limiting middleware
if settings.RATE_LIMIT_ENABLED:
    app.middleware("http")(rate_limit_middleware)


class HealthCheck(BaseModel):
    """Response model for health check endpoint."""
    status: str = "OK"


class RateLimitStatus(BaseModel):
    """Response model for rate limit status endpoint."""
    enabled: bool
    limit: int
    window: int
    per_endpoint: bool
    remaining: int
    reset: int


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Content Moderation API!",
        "version": "2.0.0",
        "documentation": "/docs",
        "rate_limiting": {
            "enabled": settings.RATE_LIMIT_ENABLED,
            "limit": settings.RATE_LIMIT_REQUESTS if settings.RATE_LIMIT_ENABLED else None,
            "window": settings.RATE_LIMIT_WINDOW if settings.RATE_LIMIT_ENABLED else None,
            "per_endpoint": settings.RATE_LIMIT_PER_ENDPOINT if settings.RATE_LIMIT_ENABLED else None
        }
    }


@app.get(
    "/health",
    tags=["Health Check"],
    summary="Perform a Health Check",
    response_model=HealthCheck,
)
async def get_health() -> HealthCheck:
    """Health check endpoint."""
    return HealthCheck()


@app.get(
    "/rate-limit-status",
    tags=["Rate Limiting"],
    summary="Check Rate Limit Status",
    response_model=RateLimitStatus,
)
async def get_rate_limit_status(request: Request) -> RateLimitStatus:
    """
    Check the current rate limit status for the requesting client.
    
    This endpoint shows how many requests remain in the current window
    and when the rate limit will reset.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return RateLimitStatus(
            enabled=False,
            limit=0,
            window=0,
            per_endpoint=False,
            remaining=999999,
            reset=0
        )
    
    # Get rate limit key for this request
    rate_limit_key = get_rate_limit_key(request, settings.RATE_LIMIT_PER_ENDPOINT)
    
    # Check current status without consuming a request
    rate_limiter = get_rate_limiter()
    is_allowed, rate_info = rate_limiter.is_allowed(
        rate_limit_key,
        settings.RATE_LIMIT_REQUESTS,
        settings.RATE_LIMIT_WINDOW
    )
    
    return RateLimitStatus(
        enabled=True,
        limit=rate_info['limit'],
        window=rate_info['window'],
        per_endpoint=settings.RATE_LIMIT_PER_ENDPOINT,
        remaining=rate_info['remaining'],
        reset=rate_info['reset']
    )


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)