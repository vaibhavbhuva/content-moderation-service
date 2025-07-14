# Content Moderation Service

A comprehensive AI-powered content moderation service built with FastAPI, providing advanced text and image moderation capabilities with support for multiple languages and AI models.

## 🚀 Features

### Text Moderation
- **LLM-Based Profanity Detection**: Advanced contextual analysis using Gemini LLM
- **Transformer-Based Detection**: Specialized models for English (toxic-bert) and Indic languages (MuRIL)
- **Multi-Language Support**: 100+ languages with XLM-RoBERTa language detection
- **Code-Mixed Text**: Advanced handling of mixed-language content (e.g., Hinglish)

### Image Moderation
- **Google Cloud Vision**: Comprehensive image content analysis
- **Multiple Categories**: Adult, medical, spoofed, violence, and racy content detection

### Competency Framework
- **Role Mapping**: AI-powered job role to competency framework mapping
- **LLM Analysis**: Intelligent competency selection based on role requirements

### Language Detection
- **High Accuracy**: 97.9% accuracy using XLM-RoBERTa
- **Real-time Processing**: Fast language identification for content routing
- **Code-Mixed Support**: Detection of mixed-language patterns

## 🏗️ Architecture

The application follows a clean architecture pattern with clear separation of concerns:

```
src/
├── api/v1/                 # API Controllers (Presentation Layer)
│   ├── profanity_controller.py
│   ├── competency_controller.py
│   ├── text_moderation.py
│   └── image_moderation.py
├── services/               # Business Logic (Service Layer)
│   ├── profanity_service.py
│   ├── language_detection_service.py
│   ├── language_service.py
│   ├── vision_service.py
│   └── role_mapping_service.py
├── data/                   # Data Layer
│   └── constants.py
├── schemas/                # Data Models
│   ├── base.py
│   ├── requests.py
│   ├── responses.py
│   └── moderation.py
├── core/                   # Core Configuration
│   ├── config.py
│   ├── logger.py
│   └── middleware.py
├── utils/                  # Utility Functions
│   └── helpers.py
└── main.py                 # Application Entry Point
```

## 📋 Prerequisites

- Python 3.8+
- Google Cloud credentials (for Vision and Natural Language APIs)
- Gemini API key (for LLM-based profanity detection)
- GPU support recommended for transformer models

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd content-moderation-service
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```bash
# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key

# Optional Configuration
DEBUG=false
LOG_LEVEL=INFO
USE_GPU=true
MAX_TEXT_LENGTH=5000
```

4. **Run the application:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

Once the application is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔗 API Endpoints

### Profanity Detection

#### LLM-Based Detection
```http
POST /api/v1/profanity/check-llm
Content-Type: application/json

{
  "text": "Your text to analyze"
}
```

#### Transformer-Based Detection
```http
POST /api/v1/profanity/check-transformer
Content-Type: application/json

{
  "text": "Your text to analyze",
  "language": "english"  // optional
}
```

### Language Detection
```http
POST /api/v1/profanity/detect-language
Content-Type: application/json

{
  "text": "Text to detect language for",
  "min_chars": 5  // optional
}
```

### Competency Mapping
```http
POST /api/v1/competency/map-role
Content-Type: application/json

{
  "organization": "Tech Corp",
  "role_title": "Senior Software Engineer",
  "department": "Engineering"  // optional
}
```

### Text Moderation (Google Cloud)
```http
POST /api/v1/moderate_text/
Content-Type: application/json

{
  "text": "Text to moderate"
}
```

### Image Moderation (Google Cloud)
```http
POST /api/v1/moderate_image/
Content-Type: multipart/form-data

image: <image_file>
```

## 🎯 Usage Examples

### Python Client Example

```python
import requests

# Profanity detection with LLM
response = requests.post(
    "http://localhost:8000/api/v1/profanity/check-llm",
    json={"text": "This is a sample text"}
)
result = response.json()
print(f"Is profane: {result['responseData']['isProfane']}")
print(f"Confidence: {result['responseData']['confidence']}%")

# Language detection
response = requests.post(
    "http://localhost:8000/api/v1/profanity/detect-language", 
    json={"text": "Hello, how are you?"}
)
result = response.json()
print(f"Detected language: {result['language_name']}")
print(f"Confidence: {result['confidence']:.2%}")

# Role mapping
response = requests.post(
    "http://localhost:8000/api/v1/competency/map-role",
    json={
        "organization": "Tech Innovations Inc",
        "role_title": "Data Scientist",
        "department": "AI Research"
    }
)
result = response.json()
print(f"Mapped competencies: {len(result['data']['mapped_competencies'])}")
```

## 🧪 Model Information

### Language Detection
- **Model**: XLM-RoBERTa (ZheYu03/xlm-r-langdetect-model)
- **Accuracy**: 97.9%
- **Languages**: 100+ supported languages
- **Use Case**: Language identification and content routing

### English Profanity Detection
- **Model**: toxic-bert (unitary/toxic-bert)
- **Purpose**: English text toxicity detection
- **Threshold**: 0.4 (configurable)

### Indic Profanity Detection  
- **Model**: MuRIL (Hate-speech-CNERG/indic-abusive-allInOne-MuRIL)
- **Languages**: Hindi, Tamil, Telugu, Bengali, and more
- **Labels**: Normal, Abusive

### LLM-Based Detection
- **Model**: Gemini 2.5 Flash
- **Features**: Contextual analysis, reasoning, multilingual support
- **Output**: JSON with confidence and reasoning

## 🔧 Configuration

Key configuration options in `src/core/config.py`:

```python
class Settings(BaseSettings):
    PROJECT_NAME: str = "Content Moderation API"
    API_V1_STR: str = "/api/v1"
    
    # Model settings
    USE_GPU: bool = True
    MAX_TEXT_LENGTH: int = 5000
    MIN_TEXT_LENGTH: int = 5
    
    # API settings  
    RATE_LIMIT_PER_MINUTE: int = 100
    DEBUG: bool = False
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
    "category": "Clean",
    "detected_language": "en",
    "model_used": "English (toxic-bert)"
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
    "text_length": 25,
    "top_predictions": [...]
  }
}
```

## 🚨 Error Handling

The API provides consistent error responses:

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

## 🔍 Monitoring & Logging

The application includes comprehensive logging and monitoring:

- **Request/Response Logging**: All API calls are logged
- **Performance Metrics**: Response times tracked
- **Error Tracking**: Detailed error logging with stack traces
- **Health Checks**: `/health` endpoint for service monitoring
* Virtual environment (recommended).

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/KB-iGOT/profanity-moderation.git
    cd profanity-moderation
    ```

2.  **Create a virtual environment:**

    ```bashs
    python3.12 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install uv && uv pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**

    * Create a `.env` file in the root directory of the project.
    * Add the following line, replacing the path with your service account key file path:

        ```text
        LOG_LEVEL=INFO
        GOOGLE_APPLICATION_CREDENTIALS=<YOUR_SERVICE_ACCOUNT_KEY_FILE_PATH>
        ```

5.  **Running the Application:**

    ```bash
    uvicorn src.main:app --reload
    ```

6.  **Access the API:**

    * The API will be available at `http://127.0.0.1:8000/docs`.

### API Endpoints

* **`/moderate_text/` (POST):** Moderates text.
    * Request body: `{"text": "Your text here"}`
    * Response: JSON array of category and confidence scores.
* **`/moderate_image/` (POST):** Detects unsafe features in an image.
    * Request: Upload an image file.
    * Response: JSON object with safe search results.

### Docker

1.  **Build the Docker image:**

    ```bash
    docker build -t moderation-api .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 -e LOG_LEVEL=INFO -e GOOGLE_APPLICATION_CREDENTIALS=/app/prj-demo-fe2334234.json -v $(pwd)/prj-demo-fe2334234.json:/app/prj-demo-fe2334234.json moderation-api
    ```

    * **Important:** Replace `prj-demo-fe2334234.json` with your actual service account key file name.
    * The `-v` flag mounts the service account key file into the container.
    * Or, use a docker secret or other method to securely provide your credentials.

### Error Handling

* The API returns HTTP 500 errors with detailed error messages in case of exceptions.
* Vision API errors are also handled and returned as part of the response.

### Using UV

This project now utilizes the `uv` package for faster dependency installation. `uv` is significantly faster than `pip` and is recommended for production deployments.

### .env file

The .env file is used to store environment variables, such as the path to your service account key file. This allows you to keep sensitive information out of your code.