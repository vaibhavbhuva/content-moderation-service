# 🛡️ Content Moderation Service

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg) ![Kafka](https://img.shields.io/badge/Kafka-Enabled-orange.svg)

**A comprehensive AI-powered content moderation service built with FastAPI, providing advanced text analysis capabilities with support for multiple languages and intelligent processing.**

**Quick Navigation:** [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-api-documentation) • [🔧 Configuration](#-configuration) • [🐳 Docker](#-docker-deployment)

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#-architecture)
- [📋 Prerequisites](#-prerequisites)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#-configuration)
- [📖 API Documentation](#-api-documentation)
- [🧪 AI Models](#-ai-models)
- [📊 API Examples](#-api-examples)
- [📡 Kafka Integration](#-kafka-integration)
- [🔍 Monitoring](#-monitoring)
- [🐳 Docker Deployment](#-docker-deployment)
- [🤝 Contributing](#-contributing)

---

## ✨ Features

### 🎯 Text Profanity Detection
- **Transformer-Based Models**: English (toxic-bert) & Indic (MuRIL)
- **100+ Languages**: XLM-RoBERTa language detection
- **Code-Mixed Support**: Hinglish, Spanglish, etc.
- **Smart Chunking**: Sliding window for long texts
- **Result Aggregation**: Priority-based & majority voting

### 🔍 Language Detection
- **97.9% Accuracy**: XLM-RoBERTa powered
- **Real-time Processing**: Fast language identification
- **Mixed Languages**: Code-switched text support
- **Script Analysis**: Character distribution analysis

### ⚙️ Advanced Processing
- **Automatic Chunking**: Intelligent text segmentation
- **Token Management**: Configurable sizes & overlap
- **GPU Acceleration**: Transformer model optimization
- **Rate Limiting**: Per-endpoint controls

### 📡 Event Streaming
- **Kafka Integration**: Real-time event emission
- **Custom Topics**: Configurable event routing
- **Retry Logic**: Reliable message delivery
- **Monitoring**: Built-in observability

---

## 🏗️ Architecture

### 📁 Project Structure

```
src/
├── api/
│   └── v1/                           # API Controllers
│       ├── __init__.py
│       ├── language_controller.py    # Language detection endpoints
│       └── profanity_controller.py   # Profanity detection endpoints
├── services/                         # Business Logic
│   ├── kafka/                        # Kafka Integration
│   │   ├── __init__.py
│   │   └── producer.py               # Kafka producer and messaging
│   ├── __init__.py
│   ├── kafka_service.py              # Kafka Integration service
│   ├── language_detection_service.py # Language identification
│   ├── text_chunking_service.py      # Text segmentation
│   └── text_profanity_service.py     # Core profanity detection
├── data/                             # Data Layer
│   └── constants.py                  # Language mappings
├── schemas/                          # Data Models
│   ├── base.py                       # Base Pydantic models
│   ├── requests.py                   # Request schemas
│   ├── responses.py                  # Response schemas
│   └── moderation.py                 # Moderation schemas
├── core/                            # Configuration
│   ├── config.py                     # App settings
│   ├── logger.py                     # Logging setup
│   └── middleware.py                 # CORS & middleware
├── utils/                           # Utilities
│   └── helpers.py                    # Helper functions
└── main.py                          # Application entry
```

---

## 📋 Prerequisites

- **[Python 3.12+](https://www.python.org/)**
- **[UV Package Manager](https://github.com/astral-sh/uv)**
- **[Apache Kafka](https://kafka.apache.org/)** (optional - only needed if `KAFKA_ENABLED=true`)

### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup
```bash
# Clone repository
git clone https://github.com/KB-iGOT/content-moderation-service.git
cd content-moderation-service

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2️⃣ Configure Environment
Create `.env` file:
```bash
cp .env.example .env
# Edit configuration as needed
```

### 3️⃣ Start Service
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ Test API
```bash
curl -X POST "http://localhost:8000/api/v1/moderation/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "language": "en"}'
```

Service running at: http://localhost:8000

---

## ⚙️ Configuration

### 📊 Environment Variables

| Group | Variable | Type | Default | Description |
|-------|----------|------|---------|-------------|
| **API** | `PROJECT_NAME` | `string` | `"Content Moderation API"` | Application display name |
| | `PROJECT_VERSION` | `string` | `"1.0.0"` | API version identifier |
| | `API_V1_STR` | `string` | `"/api/v1"` | API version prefix |
| | `LOG_LEVEL` | `string` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| **AI Models** | `ENGLISH_TRANSFORMER_MODEL` | `string` | `"unitary/toxic-bert"` | English profanity detection model |
| | `INDIC_TRANSFORMER_MODEL` | `string` | `"Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"` | Indic languages model |
| | `LANGUAGE_DETECT_MODEL` | `string` | `"ZheYu03/xlm-r-langdetect-model"` | Language detection model |
| | `HF_HUB_OFFLINE` | `integer` | `1` | Prevent Hugging Face Hub calls |
| **Text Processing** | `MAX_TEXT_LENGTH` | `integer` | `500` | Max length before chunking |
| | `CHUNKING_ENABLED` | `boolean` | `true` | Enable automatic chunking |
| | `CHUNK_SIZE` | `integer` | `500` | Tokens per chunk |
| | `CHUNK_OVERLAP` | `integer` | `100` | Overlap between chunks |
| | `MAX_CHUNKS_PER_TEXT` | `integer` | `10` | Maximum chunks per text |
| **Kafka** | `KAFKA_ENABLED` | `boolean` | `true` | Enable Kafka streaming |
| | `KAFKA_BOOTSTRAP_SERVERS` | `string` | `"localhost:9092"` | Kafka broker connection |
| | `KAFKA_MODERATION_RESULTS_TOPIC` | `string` | `"dev.content.profanity"` | Results topic name |
| | `KAFKA_RETRIES` | `integer` | `3` | Retry attempts |
| | `KAFKA_RETRY_BACKOFF_MS` | `integer` | `100` | Retry delay (ms) |

### 📄 Example Configuration

```env
# API Settings
PROJECT_NAME="My Content Moderation API"
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=1000

# Chunking Settings
CHUNKING_ENABLED=true
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_CHUNKS_PER_TEXT=20

# Kafka Settings (Optional)
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_MODERATION_RESULTS_TOPIC=content.moderation.results

# Hugging Face
HF_HUB_OFFLINE=1
```

---

## 📖 API Documentation

### 📚 Documentation Links

| Documentation | URL | Description |
|------------------|--------|----------------|
| **Interactive Docs** | `/docs` | Swagger UI with live testing |
| **ReDoc** | `/redoc` | Clean documentation format |
| **OpenAPI Schema** | `/api/v1/openapi.json` | Raw OpenAPI specification |

### 🎯 Core Endpoints & Examples

#### 🛡️ Profanity Detection
```http
POST /api/v1/moderation/text
```
**Purpose:** Analyze text for profanity content

**Request:**
```json
{
  "text": "It was a pleasure to grade this!",
  "language": "en",
  "metadata": {
    "user_id": "user123",
    "content_id": "post456",
    "source": "web_app"
  }
}
```

### 📝 **Request Body Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | `string` | ✅ **Yes** | The actual text content to analyze for profanity |
| `language` | `string` | ✅ **Yes** | Language code (e.g., "en", "hi", "es") |
| `metadata` | `object` | ❌ Optional | Additional context information for tracking/logging |


**Response:**
```json
{
  "status": "success",
  "message": "Profanity check completed",
  "responseData": {
    "text": "It was a pleasure to grade this!",
    "isProfane": false,
    "confidence": 99.92,
    "category": "Non-Profane",
    "detected_language": "en",
    "text_length": null,
    "chunking_used": null,
    "total_chunks": null,
    "profane_chunks": null,
    "clean_chunks": null,
    "aggregation_strategy": null,
    "chunk_statistics": null,
    "chunk_details": null
  }
}
```

**Chunked Analysis (Long Text):**
```json
{
  "status": "success",
  "message": "Profanity check completed using chunking (2 chunks)",
  "responseData": {
    "text": "Artificial Intelligence (AI) is no longer a futuristic concept—it’s part of our daily routines. From personalized recommendations on streaming platforms to advanced healthcare diagnostics, AI is resha...",
    "isProfane": false,
    "confidence": 99.94,
    "category": "clean",
    "detected_language": "en",
    "text_length": 2851,
    "chunking_used": true,
    "total_chunks": 2,
    "profane_chunks": 0,
    "clean_chunks": 2,
    "aggregation_strategy": "priority_clean",
    "chunk_statistics": {
      "total_processed": 2,
      "total_attempted": 2,
      "profane_detected": 0,
      "clean_detected": 2,
      "average_confidence": 99.94,
      "best_chunk": {
        "category": "clean",
        "confidence": 0.9995,
        "confidence_percentage": 99.95,
        "chunk_index": 1,
        "chunk_text": "efficient. in this article, we explore how ai is enhancing user experiences and creating opportuniti...",
        "reasoning": "",
        "model_used": "transformer",
        "detected_language": "en",
        "chunk_tokens": 104,
        "chunk_chars": 599
      },
      "aggregation_rationale": "Priority-based: All chunks clean, using average confidence"
    },
    "chunk_details": [
      {
        "category": "clean",
        "confidence": 0.9993000000000001,
        "confidence_percentage": 99.93,
        "chunk_index": 0,
        "chunk_text": "artificial intelligence ( ai ) is no longer a futuristic concept — it ’ s part of our daily routines...",
        "reasoning": "",
        "model_used": "transformer",
        "detected_language": "en",
        "chunk_tokens": 500,
        "chunk_chars": 2855
      },
      {
        "category": "clean",
        "confidence": 0.9995,
        "confidence_percentage": 99.95,
        "chunk_index": 1,
        "chunk_text": "efficient. in this article, we explore how ai is enhancing user experiences and creating opportuniti...",
        "reasoning": "",
        "model_used": "transformer",
        "detected_language": "en",
        "chunk_tokens": 104,
        "chunk_chars": 599
      }
    ]
  }
}
```

#### 🔍 Language Detection
```http
POST /api/v1/language/detect
```
**Purpose:** Identify text language

**Request:**
```json
{
  "text": "Hello, how are you today?"
}
```

### 📝 **Request Structure**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | `string` | ✅ **Yes** | The text you want to identify the language for |


**Response:**
```json
{
  "status": "success",
  "message": "Language detected successfully",
  "detected_language": "en",
  "language_name": "English",
  "confidence": 0.9999,
  "top_predictions": [
    {
      "language_code": "en",
      "language_name": "English",
      "confidence": 0.9999
    },
    {
      "language_code": "de",
      "language_name": "German",
      "confidence": 0.0001
    }
  ]
}
```

---

## 📡 Kafka Integration

### 📨 Event Structure

When `KAFKA_ENABLED=true`, the service emits events after each profanity check:

```json
{
  "request_data": {
    "text": "It was a pleasure to grade this!",
    "language": "en",
    "metadata": {
      "user_id": "user123",
      "content_id": "post456",
      "source": "web_app"
    }
  },
  "response_data": {
    "status": "success",
    "message": "Profanity check completed",
    "responseData": {
      "text": "It was a pleasure to grade this!",
      "isProfane": false,
      "confidence": 99.92,
      "category": "Non-Profane",
      "detected_language": "en",
      "text_length": null,
      "chunking_used": null,
      "total_chunks": null,
      "profane_chunks": null,
      "clean_chunks": null,
      "aggregation_strategy": null,
      "chunk_statistics": null,
      "chunk_details": null
    }
  }
}
```

### 🎯 Topic Configuration

| Topic | Purpose | Retention | Partitions |
|-------|---------|-----------|------------|
| `dev.content.profanity` | Profanity results | 7 days | 3 |

---

## 🔍 Monitoring

### ❤️ Health Monitoring

#### Health Check Endpoint
```http
GET /health
```

**Response:**
```json
{
  "status": "OK",
  "version": "1.0.0"
}
```

---

## 🐳 Docker Deployment

### 🚀 Quick Deploy

#### Build Image
```bash
docker build -t content-moderation-service .
```

#### Run Container
```bash
docker run -p 8000:8000 \
  -e KAFKA_ENABLED=false \
  -e LOG_LEVEL=INFO \
  content-moderation-service
```

#### With Environment File
```bash
docker run -p 8000:8000 \
  --env-file .env \
  content-moderation-service
```

#### Check Status
```bash
curl http://localhost:8000/health
```

---

## 🤝 Contributing

### 🌟 We Welcome Contributions!

| Issues | Features | Documentation | Testing |
|-----------|-------------|------------------|------------|
| Bug reports | New features | Improve docs | Add tests |
| Performance issues | Model improvements | API examples | Coverage increase |

### 📋 Contribution Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Update** documentation
6. **Ensure** all tests pass (`pytest`)
7. **Submit** a pull request

---

**Star this repository if you find it helpful!**

[⬆️ Back to Top](#-content-moderation-service)