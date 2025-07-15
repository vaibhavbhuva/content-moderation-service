# Content Moderation Service

A comprehensive AI-powered content moderation service built with FastAPI, providing advanced text analysis capabilities with support for multiple languages and intelligent text processing.

## 🚀 Features

### Text Profanity Detection
- **LLM-Based Detection**: Advanced contextual analysis using Gemini 2.5 Flash
- **Transformer-Based Detection**: Specialized models for English (toxic-bert) and Indic languages (MuRIL)
- **Multi-Language Support**: 100+ languages with XLM-RoBERTa language detection
- **Code-Mixed Text**: Advanced handling of mixed-language content (e.g., Hinglish)
- **Automatic Text Chunking**: Intelligent chunking for long texts with sliding window overlap
- **Result Aggregation**: Majority voting and priority-based aggregation strategies

### Language Detection
- **High Accuracy**: 97.9% accuracy using XLM-RoBERTa
- **Real-time Processing**: Fast language identification for content routing
- **Code-Mixed Support**: Detection of mixed-language patterns
- **Script Analysis**: Character-based script distribution analysis

### Text Processing
- **Automatic Text Chunking**: Internal chunking for long texts with sliding window overlap
- **Token Management**: Configurable chunk sizes and overlap for optimal processing
- **Performance Optimization**: GPU acceleration support for transformer models
- **Rate Limiting**: Configurable rate limiting with per-endpoint controls

## 🏗️ Architecture

The application follows a clean architecture pattern with clear separation of concerns:

```
src/
├── api/v1/                 # API Controllers (Presentation Layer)
│   ├── profanity_controller.py    # Profanity detection endpoints
│   └── text_moderation.py         # Legacy text moderation endpoints
├── services/               # Business Logic (Service Layer)
│   ├── profanity_service.py       # Core profanity detection logic
│   ├── language_detection_service.py # Clean language detection
│   ├── text_chunking_service.py   # Text chunking algorithms (internal)
│   
├── data/                   # Data Layer
│   └── constants.py              # Language mappings and constants
├── schemas/                # Data Models
│   ├── base.py                   # Base Pydantic models
│   ├── requests.py               # Request schemas
│   ├── responses.py              # Response schemas
│   └── moderation.py             # Moderation-specific schemas
├── core/                   # Core Configuration
│   ├── config.py                 # Application settings
│   ├── logger.py                 # Logging configuration
│   ├── middleware.py             # CORS and other middleware
│   └── rate_limiter.py           # Rate limiting implementation
├── utils/                  # Utility Functions
│   └── helpers.py               # Common helper functions
└── main.py                 # Application Entry Point
```

## 📋 Prerequisites

- Python 3.8+
- Gemini API key (for LLM-based profanity detection)
- GPU support recommended for transformer models (optional)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/KB-iGOT/content-moderation-service.git
cd content-moderation-service
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the project root:
```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Configuration
DEBUG=false
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=500
CHUNKING_ENABLED=true
CHUNK_SIZE=500
CHUNK_OVERLAP=12
MAX_CHUNKS_PER_TEXT=10

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

5. **Run the application:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

Once the application is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔗 API Endpoints

### Core Profanity Detection

#### LLM-Based Detection
```http
POST /api/v1/profanity/check-llm
Content-Type: application/json

{
  "text": "Your text to analyze"
}
```

#### Transformer-Based Detection (with Automatic Chunking)
```http
POST /api/v1/profanity/check-transformer
Content-Type: application/json

{
  "text": "Your text to analyze"
}
```

### Language Detection
```http
POST /api/v1/profanity/detect-language
Content-Type: application/json

{
  "text": "Text to detect language for",
  "min_chars": 5
}
```

## 🎯 Usage Examples

### Python Client Examples

#### Basic Profanity Detection
```python
import requests

# LLM-based profanity detection with reasoning
response = requests.post(
    "http://localhost:8000/api/v1/profanity/check-llm",
    json={"text": "This is a sample text"}
)
result = response.json()
print(f"Is profane: {result['responseData']['isProfane']}")
print(f"Confidence: {result['responseData']['confidence']}%")
print(f"Reasoning: {result['responseData']['reasoning']}")

# Transformer-based detection with auto-chunking
response = requests.post(
    "http://localhost:8000/api/v1/profanity/check-transformer",
    json={"text": "This is a very long text that will be automatically chunked..."}
)
result = response.json()
print(f"Is profane: {result['responseData']['isProfane']}")
print(f"Chunking used: {result['responseData']['chunking_used']}")
print(f"Total chunks: {result['responseData']['total_chunks']}")
```

#### Language Detection
```python
# Language detection
response = requests.post(
    "http://localhost:8000/api/v1/profanity/detect-language", 
    json={"text": "Hello, how are you?"}
)
result = response.json()
print(f"Detected language: {result['language_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Language group: {result['details']['language_group']}")
```

## 🧪 Model Information

### Language Detection
- **Model**: XLM-RoBERTa (ZheYu03/xlm-r-langdetect-model)
- **Accuracy**: 97.9%
- **Languages**: 100+ supported languages
- **Special Features**: Code-mixed language detection (e.g., Hinglish)
- **Use Case**: Language identification and routing to appropriate models

### English Profanity Detection
- **Model**: toxic-bert (unitary/toxic-bert)
- **Purpose**: English text toxicity detection
- **Categories**: Toxic, severe toxic, obscene, threat, insult, identity hate
- **Threshold**: 0.4 (configurable)

### Indic Profanity Detection  
- **Model**: MuRIL (Hate-speech-CNERG/indic-abusive-allInOne-MuRIL)
- **Languages**: Hindi, Tamil, Telugu, Bengali, Gujarati, Punjabi, and more
- **Labels**: Normal (0), Abusive (1)
- **Script Support**: Devanagari, Tamil, Telugu, Bengali, and other Indic scripts

### LLM-Based Detection
- **Model**: Gemini 2.5 Flash
- **Features**: 
  - Contextual analysis with reasoning
  - Multilingual support
  - JSON structured responses
  - Temperature 0 for consistent results
- **Output**: Confidence score, boolean result, and reasoning explanation

### Automatic Text Chunking (Internal)
- **Strategy**: Sliding window with overlap (automatically applied to long texts)
- **Token Management**: Configurable chunk size (default: 500 tokens)
- **Overlap**: Configurable overlap (default: 12 tokens) for context preservation
- **Aggregation**: Priority-based and majority voting strategies for chunk results

## 🔧 Configuration

Key configuration options in `src/core/config.py`:

```python
class Settings(BaseSettings):
    # Basic API settings
    PROJECT_NAME: str = "Content Moderation API"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    
    # Gemini API settings
    GEMINI_API_KEY: Optional[str] = None
    
    # Model settings
    MAX_TEXT_LENGTH: int = 500           # Maximum text length before chunking
    MIN_TEXT_LENGTH: int = 5             # Minimum text length for processing
    LANGUAGE_DETECTION_SAMPLE_SIZE: int = 300  # Characters for language detection
    
    # Text chunking settings
    CHUNKING_ENABLED: bool = True        # Enable automatic chunking
    CHUNK_SIZE: int = 500               # Tokens per chunk
    CHUNK_OVERLAP: int = 12             # Overlap between chunks (tokens)
    MAX_CHUNKS_PER_TEXT: int = 10       # Maximum chunks per text
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = True     # Enable rate limiting
    RATE_LIMIT_REQUESTS: int = 100      # Requests per time window
    RATE_LIMIT_WINDOW: int = 60         # Time window in seconds
    RATE_LIMIT_PER_ENDPOINT: bool = True # Separate limits per endpoint
    
    # Development settings
    DEBUG: bool = False
```

### Environment Variables

All settings can be overridden using environment variables in your `.env` file:

```env
# API Configuration
PROJECT_NAME="My Content Moderation API"
LOG_LEVEL=DEBUG

# Gemini Configuration
GEMINI_API_KEY=your_api_key_here

# Processing Limits
MAX_TEXT_LENGTH=1000
LANGUAGE_DETECTION_SAMPLE_SIZE=500

# Chunking Configuration
CHUNKING_ENABLED=true
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CHUNKS_PER_TEXT=20

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=50
RATE_LIMIT_WINDOW=60
```

## 📊 Response Formats

### Profanity Detection Response
```json
{
  "status": "success",
  "message": "Profanity check completed",
  "responseData": {
    "text": "Input text",
    "isProfane": false,
    "confidence": 95.23,
    "category": "clean",
    "detected_language": "english",
    "model_used": "English (toxic-bert)",
    "chunking_used": false,
    "total_chunks": 1
  }
}
```

### Chunked Profanity Detection Response
```json
{
  "status": "success",
  "message": "Profanity check completed using chunking (3 chunks)",
  "responseData": {
    "text": "Long input text...",
    "text_length": 1500,
    "isProfane": true,
    "confidence": 87.5,
    "category": "profane",
    "chunking_used": true,
    "total_chunks": 3,
    "profane_chunks": 1,
    "clean_chunks": 2,
    "aggregation_strategy": "priority_based",
    "model_used": "transformer+chunking",
    "chunk_statistics": {
      "total_processed": 3,
      "profane_detected": 1,
      "clean_detected": 2,
      "average_confidence": 75.2,
      "aggregation_rationale": "Profanity detected in any chunk takes priority"
    }
  }
}
```

### Language Detection Response
```json
{
  "status": "success",
  "detected_language": "english",
  "raw": "en",
  "language_name": "English", 
  "confidence": 0.9876,
  "details": {
    "model_used": "XLM-RoBERTa",
    "language_group": "english",
    "text_length": 25,
    "sample_used": 25,
    "code_mixed": false
  }
}
```

## 🚨 Error Handling

The API provides consistent error responses across all endpoints:

```json
{
  "status": "error",
  "message": "Descriptive error message",
  "error_code": "ERROR_CODE",
  "details": {
    "additional": "context"
  }
}
```

### Common Error Scenarios

#### Text Too Short
```json
{
  "status": "error",
  "message": "Input text is too short (minimum 5 characters required)",
  "responseData": null
}
```

#### Missing API Key
```json
{
  "status": "error", 
  "message": "Gemini API key not configured",
  "responseData": null
}
```

#### Chunking Failure
```json
{
  "status": "error",
  "message": "Failed to process any chunks",
  "responseData": null
}
```

## 🔍 Monitoring & Logging

The application includes comprehensive logging and monitoring:

- **Request/Response Logging**: All API calls with performance metrics
- **Chunk Processing**: Detailed logging of chunking operations
- **Model Performance**: Individual model prediction logging
- **Error Tracking**: Detailed error logging with context
- **Health Checks**: Basic health endpoint at `/health`

### Log Levels
- **INFO**: General operation logs, API requests, processing results
- **DEBUG**: Detailed chunk processing, model predictions, text previews
- **WARNING**: Fallback scenarios, deprecated endpoint usage
- **ERROR**: Failed operations, API errors, model failures

## 🐳 Docker Support

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t content-moderation-service .

# Run the container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -e LOG_LEVEL=INFO \
  content-moderation-service
```

### Docker Compose (Development)

```yaml
version: '3.8'
services:
  content-moderation:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - LOG_LEVEL=DEBUG
      - DEBUG=true
    volumes:
      - .:/app
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```



### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `python -m pytest`
6. Run linting: `flake8 src/`
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request



