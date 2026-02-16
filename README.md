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
HOST=0.0.0.0 PORT=8000 ./siglip-service.pex

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

### Development

```bash
# Quick start - localhost with auto-reload
make dev
# or: uvicorn app:app --reload

# Custom options
uvicorn app:app --host 127.0.0.1 --port 8080 --reload --log-level debug
```

### Production

```bash
# Using make (binds to 0.0.0.0:8000)
make prod

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000

# Behind reverse proxy (recommended)
uvicorn app:app --host 127.0.0.1 --port 8000 --proxy-headers

# Using environment variables
HOST=0.0.0.0 PORT=8000 python app.py
```

**Production tips:** Use a reverse proxy (nginx) + process manager (systemd). See systemd example below.

### Make Commands

```bash
make dev      # Development with auto-reload
make prod     # Production server
make test     # Run test client
make build    # Build PEX executable
make help     # Show all commands
```

### Systemd Service Example

Create `/etc/systemd/system/siglip-embedding.service`:

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
ExecStart=/opt/siglip/siglip-service.pex
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now siglip-embedding
sudo journalctl -u siglip-embedding -f  # view logs
```

## API Reference

See **[API.md](API.md)** for complete endpoint documentation.

**Quick reference:**
- `GET /` - Service info
- `GET /health` - Health check
- `POST /embed/text` - Text embedding
- `POST /embed/image/upload` - Image embedding

**Interactive docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Testing the API

Use the provided test client to verify all endpoints:

```bash
python test_client.py
```

This will test text and image embedding endpoints and calculate similarity scores between them.

---

### Response Structure

All embedding endpoints return JSON with the following structure:

```json
{
  "embedding": "base64-encoded-binary-data...",
  "shape": [1152],
  "size_bytes": 4608
}
```

### Binary Format Specification

- **Encoding**: Base64 string (for JSON compatibility)
- **Underlying Format**: IEEE 754 binary32 (float32)
- **Byte Order**: Little-endian
- **Dimensions**: 1152 floats × 4 bytes each = 4,608 bytes
- **Base64 Size**: 6,144 characters (4,608 bytes × 4/3 base64 overhead)

### Cosine Similarity

The embeddings are already L2-normalized by the model, so cosine similarity is simply the dot product:

```python
similarity = np.dot(embedding1, embedding2)
```

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
- **Response size**: ~6 KB per embedding (binary format)

The model is loaded once at startup and kept in memory for fast inference.

### Implementation Notes

- All embeddings are L2-normalized vectors suitable for cosine similarity calculations
- Text embeddings use `max_length` padding as required by SigLIP
- Image preprocessing handles various formats (JPEG, PNG, BMP, GIF, etc.)
- Thread-safe model inference with PyTorch's `no_grad()` context
- Embeddings returned as base64-encoded binary (float32, little-endian)
- Binary format provides 64% bandwidth reduction vs JSON arrays
