# SigLIP Embedding Microservice

A FastAPI microservice that provides vector embeddings for text and images using Google's SigLIP model.

## Features

- **Text Embeddings**: Generate 1152-dimensional embeddings from text
- **Image Embeddings**: Generate 1152-dimensional embeddings from images
- **GPU Acceleration**: Automatic GPU detection with 10-20x speedup
- **Efficient**: Model loaded once at startup, not on every request
- **Multiple Input Methods**: Support for base64-encoded images or direct file uploads

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

**GPU Support**:
- Ensure you have CUDA installed (CUDA 11.8+ recommended)
- Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
- See [GPU_SETUP.md](GPU_SETUP.md) for detailed GPU setup instructions

## Usage

### Start the Server

**CPU (default)**:
```bash
python app.py
```

**GPU**:
```bash
DEVICE=cuda python app.py
```

**Explicit CPU**:
```bash
DEVICE=cpu python app.py
```

The server will start on `http://localhost:8000`

Alternatively, use uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Text Embedding

**Endpoint**: `POST /embed/text`

**Request**:
```json
{
  "text": "a photo of a cat"
}
```

**Response**:
```json
{
  "embedding": [0.123, -0.456, ...],
  "shape": [1152]
}
```

**Example with curl**:
```bash
curl -X POST "http://localhost:8000/embed/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "a photo of a cat"}'
```

**Example with Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/embed/text",
    json={"text": "a photo of a cat"}
)
embedding = response.json()["embedding"]
```

#### 2. Image Embedding (Base64)

**Endpoint**: `POST /embed/image`

**Request**:
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg..."
}
```

**Response**:
```json
{
  "embedding": [0.123, -0.456, ...],
  "shape": [1152]
}
```

**Example with Python**:
```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/embed/image",
    json={"image_base64": image_b64}
)
embedding = response.json()["embedding"]
```

#### 3. Image Embedding (File Upload)

**Endpoint**: `POST /embed/image/upload`

**Example with curl**:
```bash
curl -X POST "http://localhost:8000/embed/image/upload" \
  -F "file=@image.jpg"
```

**Example with Python**:
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/embed/image/upload",
        files={"file": f}
    )
embedding = response.json()["embedding"]
```

### Interactive Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Testing

Use the provided test client to verify the API:

```bash
python test_client.py
```

This will test both text and image embedding endpoints and calculate similarity scores.

## Comparing Embeddings

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

## Model Information

- **Model**: google/siglip-so400m-patch14-384
- **Embedding Dimension**: 1152
- **Normalization**: L2 normalized (ready for cosine similarity)
- **Model Size**: ~1.5 GB

## GPU Configuration

The service automatically detects and uses GPU if available. You can control this with the `DEVICE` environment variable:

- `auto` (default): Use GPU if available, otherwise CPU
- `cuda`: Force GPU usage (fails if no GPU available)
- `cpu`: Force CPU usage

**Examples**:
```bash
# Auto-detect
python app.py

# Force GPU
DEVICE=cuda python app.py

# Force CPU
DEVICE=cpu python app.py
```

**GPU Memory Requirements**:
- Model: ~1.5 GB
- Runtime: ~2-3 GB total
- Recommended: 4GB+ VRAM

**Performance**:
- CPU: ~1-2 seconds per request
- GPU: ~50-200ms per request (10-20x faster)

## Notes

- The model is loaded once at startup, making subsequent requests fast
- All embeddings are L2-normalized, suitable for cosine similarity calculations
- Text embeddings use max_length padding as required by SigLIP
- GPU usage is automatically detected and configured
- Check `/health` endpoint to verify device being used
