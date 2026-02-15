# API Reference

## Overview

All endpoints return JSON responses. Embeddings are L2-normalized 1152-dimensional vectors suitable for cosine similarity calculations.

**Base URL:** `http://localhost:8000` (development)

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Endpoints

### Service Information

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
    "image": "/embed/image/upload",
    "health": "/health"
  }
}
```

**Example**:
```bash
curl http://localhost:8000/
```

---

### Health Check

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

### Text Embedding

**`POST /embed/text`** - Generate embedding from text

Generate a 1152-dimensional embedding vector from input text.

**Request Body**:
```json
{
  "text": "string (required, 1-5000 characters)"
}
```

**Validation Rules**:
- Text must be between 1 and 5000 characters
- Cannot be empty or contain only whitespace
- Cannot contain null bytes

**Response** (200 OK):
```json
{
  "embedding": [0.123, -0.456, ...],  // Array of 1152 floats
  "shape": [1152]
}
```

**Error Response** (422 Unprocessable Entity):
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "Text cannot be empty or contain only whitespace",
      "type": "value_error"
    }
  ]
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

### Image Embedding

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

## Error Handling

All endpoints return standard HTTP status codes:

- **200 OK**: Request successful
- **422 Unprocessable Entity**: Invalid request body (missing required fields, wrong types)
- **500 Internal Server Error**: Server error (model inference failed, invalid image format, etc.)

Error responses include a `detail` field with a descriptive error message.
