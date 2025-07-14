"""
Rate Limiting Middleware for Content Moderation API

Provides configurable rate limiting with in-memory storage.
For production, consider using Redis for distributed rate limiting.
"""

import time
import logging
from typing import Dict, Tuple, Optional
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio
from threading import Lock

from src.core.config import settings

logger = logging.getLogger("uvicorn.error")


class InMemoryRateLimiter:
    """
    Thread-safe in-memory rate limiter using sliding window algorithm.
    
    For production environments with multiple instances, consider using
    Redis-based rate limiting for distributed coordination.
    """
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = Lock()
    
    def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            key: Unique identifier for the client/endpoint
            limit: Maximum number of requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        cutoff_time = current_time - window
        
        with self.lock:
            # Clean old requests outside the window
            request_times = self.requests[key]
            while request_times and request_times[0] <= cutoff_time:
                request_times.popleft()
            
            # Check if limit is exceeded
            current_count = len(request_times)
            
            rate_limit_info = {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset": int(current_time + window),
                "window": window
            }
            
            if current_count < limit:
                # Allow request and record it
                request_times.append(current_time)
                rate_limit_info["remaining"] = limit - current_count - 1
                return True, rate_limit_info
            else:
                # Rate limit exceeded
                return False, rate_limit_info
    
    def clear_expired(self, max_age: int = 3600):
        """Clean up old entries to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        with self.lock:
            keys_to_remove = []
            for key, request_times in self.requests.items():
                # Remove old requests
                while request_times and request_times[0] <= cutoff_time:
                    request_times.popleft()
                
                # If no recent requests, mark key for removal
                if not request_times:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.requests[key]


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()


def get_client_id(request: Request) -> str:
    """
    Generate a unique client identifier for rate limiting.
    
    Uses X-Forwarded-For header if available (for proxy/load balancer),
    otherwise falls back to client IP.
    """
    # Check for X-Forwarded-For header (common with proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP if multiple are present
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        # Fall back to direct client IP
        client_ip = request.client.host if request.client else "unknown"
    
    return client_ip


def get_rate_limit_key(request: Request, per_endpoint: bool = True) -> str:
    """Generate rate limit key based on client and optionally endpoint."""
    client_id = get_client_id(request)
    
    if per_endpoint:
        # Include endpoint path for per-endpoint limiting
        endpoint = request.url.path
        return f"{client_id}:{endpoint}"
    else:
        # Global rate limit per client
        return client_id


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware for FastAPI.
    
    This middleware applies rate limiting based on the configuration settings.
    It can be configured to apply limits globally or per-endpoint.
    """
    # Skip rate limiting if disabled
    if not settings.RATE_LIMIT_ENABLED:
        response = await call_next(request)
        return response
    
    # Skip rate limiting for health check and docs
    if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
        response = await call_next(request)
        return response
    
    try:
        # Generate rate limit key
        rate_limit_key = get_rate_limit_key(request, settings.RATE_LIMIT_PER_ENDPOINT)
        
        # Check rate limit
        is_allowed, rate_info = _rate_limiter.is_allowed(
            rate_limit_key,
            settings.RATE_LIMIT_REQUESTS,
            settings.RATE_LIMIT_WINDOW
        )
        
        if not is_allowed:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for {get_client_id(request)} on {request.url.path}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_info['limit']} requests per {rate_info['window']} seconds",
                    "retry_after": rate_info['window'],
                    "rate_limit": rate_info
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info['limit']),
                    "X-RateLimit-Remaining": str(rate_info['remaining']),
                    "X-RateLimit-Reset": str(rate_info['reset']),
                    "X-RateLimit-Window": str(rate_info['window']),
                    "Retry-After": str(rate_info['window'])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(rate_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(rate_info['reset'])
        response.headers["X-RateLimit-Window"] = str(rate_info['window'])
        
        return response
        
    except Exception as e:
        logger.error(f"Error in rate limiting middleware: {str(e)}")
        # If rate limiting fails, allow the request to proceed
        response = await call_next(request)
        return response


async def cleanup_rate_limiter():
    """Periodic cleanup task to prevent memory leaks."""
    while True:
        try:
            _rate_limiter.clear_expired()
            # Run cleanup every hour
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in rate limiter cleanup: {str(e)}")
            await asyncio.sleep(3600)


def get_rate_limiter():
    """Get the global rate limiter instance."""
    return _rate_limiter
