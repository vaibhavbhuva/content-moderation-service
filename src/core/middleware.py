"""
Simple CORS setup for the content moderation service.
"""

from fastapi.middleware.cors import CORSMiddleware


def setup_cors(app):
    """Setup basic CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", ],  # Specific origins
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specific methods
        allow_headers=["Content-Type", "Authorization"],  # Specific headers
    )
