# SigLIP Embedding Microservice

A FastAPI microservice that provides vector embeddings for text and images using Google's SigLIP model.

## Features

- **Text Embeddings**: Generate 1152-dimensional embeddings from text
- **Image Embeddings**: Generate 1152-dimensional embeddings from images via file upload
- **CPU Optimized**: Runs efficiently on CPU without GPU requirements
- **Efficient**: Model loaded once at startup, not on every request
- **Simple API**: Clean REST endpoints with automatic interactive documentation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Configuration

The service can be configured using environment variables or a `.env` file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your settings
nano .env
```

Available configuration options:
- `HOST` - Host to bind to (default: `127.0.0.1`)
- `PORT` - Port to listen on (default: `8000`)

## Usage

### Start the Server

**Default (localhost only - recommended for development):**
```bash
python app.py
```

The server will start on `http://localhost:8000` by default (only accessible from your machine).

**Using .env file (recommended):**
```bash
# Create your configuration file
cp .env.example .env
# Edit .env with your settings, then run:
python app.py
```

**Override with environment variables:**
```bash
# Listen on all interfaces (for production deployment behind firewall/proxy)
HOST=0.0.0.0 PORT=8000 python app.py

# Custom port on localhost
PORT=8080 python app.py
```

**Configuration priority:** Environment variables > `.env` file > defaults

**Alternatively, use uvicorn directly:**
```bash
# Localhost only
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# All interfaces
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Reference

### Overview

All endpoints return JSON responses. Embeddings are L2-normalized 1152-dimensional vectors suitable for cosine similarity calculations.

### Endpoints

#### Service Information

**`GET /`** - Get service information

Returns metadata about the service, model, and available endpoints.

**Response** (200 OK):
```json
{
  "message": "SigLIP Embedding Service",
  "version": "1.0.0",
  "model": "google/siglip-so400m-patch14-384",
  "embedding_dimension": 1152,
  "device": "cpu",
  "endpoints": {
    "text": "/embed/text",
    "image": "/embed/image",
    "image_upload": "/embed/image/upload",
    "health": "/health"
  }
}
```

**Example**:
```bash
curl http://localhost:8000/
```

---

#### Health Check

**`GET /health`** - Health check endpoint

Returns the service health status and model loading state.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

#### Text Embedding

**`POST /embed/text`** - Generate embedding from text

Generate a 1152-dimensional embedding vector from input text.

**Request Body**:
```json
{
  "text": "string (required)"
}
```

**Response** (200 OK):
```json
{
  "embedding": [0.123, -0.456, ...],  // Array of 1152 floats
  "shape": [1152]
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "detail": "Error generating text embedding: <error message>"
}
```

**Examples**:

```bash
# curl
curl -X POST "http://localhost:8000/embed/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'
```

```python
# Python
import requests

response = requests.post(
    "http://localhost:8000/embed/text",
    json={"text": "a photo of a cat"}
)
result = response.json()
embedding = result["embedding"]  # List of 1152 floats
```

---

#### Image Embedding

**`POST /embed/image/upload`** - Generate embedding from uploaded image file

Generate a 1152-dimensional embedding vector from a multipart file upload.

**Request**: `multipart/form-data`
- `file`: Image file (required)

**Supported formats**: JPEG, PNG, BMP, GIF, and other PIL-supported formats

**Response** (200 OK):
```json
{
  "embedding": [0.123, -0.456, ...],  // Array of 1152 floats
  "shape": [1152]
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "detail": "Error generating image embedding: <error message>"
}
```

**Examples**:

```bash
# curl
curl -X POST "http://localhost:8000/embed/image/upload" \
  -F "file=@image.jpg"
```

```python
# Python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/embed/image/upload",
        files={"file": f}
    )
result = response.json()
embedding = result["embedding"]  # List of 1152 floats
```

---

### Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer with try-it-out functionality
- **ReDoc**: `http://localhost:8000/redoc` - Clean, readable API documentation

### Error Handling

All endpoints return standard HTTP status codes:

- **200 OK**: Request successful
- **422 Unprocessable Entity**: Invalid request body (missing required fields, wrong types)
- **500 Internal Server Error**: Server error (model inference failed, invalid image format, etc.)

Error responses include a `detail` field with a descriptive error message.

---

## Working with Embeddings

### Comparing Embeddings

To calculate similarity between text and image embeddings:

```python
import numpy as np

# Get embeddings
text_embedding = np.array(text_response["embedding"])
image_embedding = np.array(image_response["embedding"])

# Calculate cosine similarity (dot product of normalized vectors)
similarity = np.dot(text_embedding, image_embedding) * 100

print(f"Similarity: {similarity:.2f}%")
# 30-40 = very strong match
# 15-25 = good/decent match
```

### Testing the API

Use the provided test client to verify all endpoints:

```bash
python test_client.py
```

This will test text and image embedding endpoints and calculate similarity scores between them.

---

## Technical Details

### Model Information

- **Model**: google/siglip-so400m-patch14-384
- **Embedding Dimension**: 1152
- **Normalization**: L2 normalized (ready for cosine similarity)
- **Model Size**: ~1.5 GB
- **Device**: CPU (optimized for broad compatibility)

### Performance

- **Startup time**: 5-10 seconds (model loading)
- **Inference time (CPU)**: ~1-2 seconds per request
- **Memory usage**: ~2-3 GB RAM

The model is loaded once at startup and kept in memory for fast inference.

### Implementation Notes

- All embeddings are L2-normalized vectors suitable for cosine similarity calculations
- Text embeddings use `max_length` padding as required by SigLIP
- Image preprocessing handles various formats (JPEG, PNG, BMP, GIF, etc.)
- Thread-safe model inference with PyTorch's `no_grad()` context
- Embeddings are returned as flat arrays of 1152 floats
