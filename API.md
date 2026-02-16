# API Reference

## Overview

All endpoints return JSON responses. Embeddings are L2-normalized 1152-dimensional vectors suitable for cosine similarity calculations.

**Embedding Format:** Embeddings are returned as **base64-encoded binary data** (IEEE 754 float32, little-endian) for bandwidth efficiency. See the [Binary Format](#binary-format) section for decoding instructions.

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
  "embedding": "AAAAvwAAgD8AAIC/...",  // Base64-encoded binary (6144 chars)
  "shape": [1152],
  "size_bytes": 4608
}
```

**Note:** The `embedding` field contains base64-encoded binary data (float32 array). See [Binary Format](#binary-format) section for decoding.

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
import base64
import numpy as np

response = requests.post(
    "http://localhost:8000/embed/text",
    json={"text": "a photo of a cat"}
)
result = response.json()

# Decode binary embedding
b64_string = result["embedding"]
binary_data = base64.b64decode(b64_string)
embedding = np.frombuffer(binary_data, dtype=np.float32)  # shape: (1152,)
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
  "embedding": "AAAAvwAAgD8AAIC/...",  // Base64-encoded binary (6144 chars)
  "shape": [1152],
  "size_bytes": 4608
}
```

**Note:** The `embedding` field contains base64-encoded binary data (float32 array). See [Binary Format](#binary-format) section for decoding.

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
import base64
import numpy as np

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/embed/image/upload",
        files={"file": f}
    )
result = response.json()

# Decode binary embedding
b64_string = result["embedding"]
binary_data = base64.b64decode(b64_string)
embedding = np.frombuffer(binary_data, dtype=np.float32)  # shape: (1152,)
```

---

## Binary Format

Embedding endpoints return embeddings in a bandwidth-efficient binary format.

### Response Structure

```json
{
  "embedding": "base64-encoded-string",
  "shape": [1152],
  "size_bytes": 4608
}
```

**Fields:**
- `embedding`: Base64-encoded binary data (6,144 characters)
- `shape`: Array dimensions (always `[1152]` for this model)
- `size_bytes`: Size of binary data in bytes (4,608 bytes = 1152 floats × 4 bytes)

### Binary Specification

The `embedding` field contains:
1. **Raw format**: Array of IEEE 754 binary32 (float32) values
2. **Byte order**: Little-endian
3. **Encoding**: Base64 (for JSON string compatibility)
4. **Size**: 1152 floats × 4 bytes = 4,608 bytes raw (6,144 characters base64)

### Cosine Similarity

Since embeddings are already L2-normalized by the model, cosine similarity is simply the dot product:

```python
# Python
similarity = np.dot(embedding1, embedding2)
```

```javascript
// JavaScript
function cosineSimilarity(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}
```

### Size Comparison

| Format | Size | Savings |
|--------|------|---------|
| JSON array of floats | ~17 KB | baseline |
| Binary (base64) | ~6 KB | **64%** |
| Binary (raw) | ~4.6 KB | 73% |

---

## Error Handling

All endpoints return standard HTTP status codes:

- **200 OK**: Request successful
- **422 Unprocessable Entity**: Invalid request body (missing required fields, wrong types)
- **500 Internal Server Error**: Server error (model inference failed, invalid image format, etc.)

Error responses include a `detail` field with a descriptive error message.
