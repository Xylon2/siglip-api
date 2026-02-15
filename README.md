# SigLIP Embedding Microservice

A FastAPI microservice that provides vector embeddings for text and images using Google's SigLIP model.

## Features

- **Text Embeddings**: Generate 1152-dimensional embeddings from text
- **Image Embeddings**: Generate 1152-dimensional embeddings from images via file upload
- **CPU Optimized**: Runs efficiently on CPU without GPU requirements
- **Efficient**: Model loaded once at startup, not on every request
- **Simple API**: Clean REST endpoints with automatic interactive documentation

## Installation

### Development

```bash
# Install as a package with dependencies
pip install -e .
```

### Building for Deployment (PEX)

Build a single executable file containing all dependencies:

```bash
# Install PEX builder
pip install pex

# Build the PEX file
pex . --entry-point app:main --output-file dist/siglip-service.pex --python-shebang='/usr/bin/env python3'

# Or use make
make build
```

This creates `dist/siglip-service.pex` - a single executable file containing your app and all dependencies (similar to a Clojure uberjar).

Deploy the PEX file to your server and run it:

```bash
# On server - no pip install needed!
./siglip-service.pex

# With environment variables
HOST=0.0.0.0 PORT=8000 WORKERS=4 ./siglip-service.pex

# Or use python directly
python siglip-service.pex
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

### Quick Start

Start the development server:
```bash
uvicorn app:app --reload
```

The server will start on `http://localhost:8000` and automatically reload on code changes.

### Development

For development, use uvicorn with the `--reload` flag for automatic code reloading:

```bash
# Default (localhost:8000 with auto-reload)
uvicorn app:app --reload

# Custom host and port
uvicorn app:app --host 127.0.0.1 --port 8080 --reload

# With detailed logging
uvicorn app:app --reload --log-level debug

# All interfaces (for testing across network)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Development best practices:**
- Use `--reload` for automatic code reloading during development
- Use `--log-level debug` for detailed logging when troubleshooting
- Default `127.0.0.1` keeps the service local to your machine
- Use `0.0.0.0` only when you need to access from other devices on your network

### Production

For production deployments, use uvicorn without `--reload` and with multiple workers:

```bash
# Production with 4 workers
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# With access logs disabled (better performance)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4 --no-access-log

# Behind a reverse proxy (recommended)
uvicorn app:app --host 127.0.0.1 --port 8000 --workers 4 --proxy-headers
```

**Production deployment recommendations:**

1. **Workers**: Use multiple workers for better performance
   - Formula: `(2 * CPU_cores) + 1`
   - Example: 4-core CPU = 9 workers

2. **Reverse Proxy**: Run behind nginx or similar
   - Bind to `127.0.0.1` and let the proxy handle external traffic
   - Use `--proxy-headers` to trust forwarded headers

3. **Process Manager**: Use systemd, supervisor, or docker
   - Ensures service restarts on failure
   - Manages logs and monitoring

4. **Environment Variables**: Use `.env` file or system environment
   ```bash
   # Example with environment variables
   HOST=0.0.0.0 PORT=8000 uvicorn app:app --workers 4
   ```

### Using Make Commands

Convenience commands are available via the Makefile:

```bash
# Development server with auto-reload
make dev

# Production server with workers
make prod

# Run tests
make test

# Show all commands
make help
```

### Alternative: Direct Python Execution

You can also run the app directly with Python (uses uvicorn internally):

```bash
# Using .env file configuration
python app.py

# With environment variables
HOST=0.0.0.0 PORT=8080 python app.py
```

**Note:** This method uses settings from your `.env` file but doesn't support advanced uvicorn options like workers or reload.

### Production Deployment with Systemd

Deploy the PEX file to your server (e.g., `/opt/siglip/siglip-service.pex`) and create a systemd service file at `/etc/systemd/system/siglip-embedding.service`:

```ini
[Unit]
Description=SigLIP Embedding Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/siglip
Environment="HOST=127.0.0.1"
Environment="PORT=8000"
Environment="WORKERS=4"
ExecStart=/opt/siglip/siglip-service.pex
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable siglip-embedding
sudo systemctl start siglip-embedding
sudo systemctl status siglip-embedding
```

View logs:
```bash
sudo journalctl -u siglip-embedding -f
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
