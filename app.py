from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, field_validator
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from io import BytesIO
import uvicorn
import os
from dotenv import load_dotenv
import base64
import numpy as np

# Load environment variables from .env file if it exists
load_dotenv()

app = FastAPI(title="SigLIP Embedding Service")

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to embed (1-5000 characters)")

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        # Strip whitespace and check if empty
        stripped = v.strip()
        if not stripped:
            raise ValueError("Text cannot be empty or contain only whitespace")

        # Check for null bytes
        if '\x00' in v:
            raise ValueError("Text cannot contain null bytes")

        return v

class EmbeddingResponse(BaseModel):
    embedding: str = Field(..., description="Base64-encoded binary float32 array")
    shape: list[int] = Field(..., description="Shape of the array")
    size_bytes: int = Field(..., description="Size of binary data in bytes")

def encode_embedding_binary(embedding: list[float]) -> dict:
    """
    Encode embedding as base64 binary float32 format.

    Args:
        embedding: List of floats from model

    Returns:
        Dictionary with base64 string and metadata
    """
    arr = np.array(embedding, dtype=np.float32)
    binary_data = arr.tobytes()
    b64_string = base64.b64encode(binary_data).decode('ascii')

    return {
        "embedding": b64_string,
        "shape": list(arr.shape),
        "size_bytes": len(binary_data)
    }

class ModelService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def load_model(self):
        if self.model is None:
            # Use CPU only
            self.device = torch.device("cpu")
            print("Using CPU")

            print("Loading SigLIP model... (this may take a moment)")
            self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
            self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=True)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully on CPU!")

    def get_text_embedding(self, text: str) -> list[float]:
        text_inputs = self.processor(
            text=[text],
            return_tensors="pt"
        )

        # Move inputs to device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # Get text model outputs
            text_outputs = self.model.text_model(**text_inputs)

            # Extract embeddings - use pooler_output if available
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_features = text_outputs.pooler_output
            else:
                # Fall back to last hidden state at position 0 (CLS token)
                text_features = text_outputs.last_hidden_state[:, 0, :]

        # Normalize (text_features should be [batch_size, embedding_dim])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Return first batch item as flat list
        return text_features[0].cpu().numpy().tolist()

    def get_image_embedding(self, image: Image.Image) -> list[float]:
        image_inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to device
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.model.vision_model(**image_inputs)

            # Extract embeddings - use pooler_output if available
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                image_features = vision_outputs.pooler_output
            else:
                # Fall back to last hidden state at position 0 (CLS token)
                image_features = vision_outputs.last_hidden_state[:, 0, :]

        # Normalize (image_features should be [batch_size, embedding_dim])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Return first batch item as flat list
        return image_features[0].cpu().numpy().tolist()

model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    model_service.load_model()

@app.get("/")
async def root():
    return {
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

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "device": "cpu"
    }

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    """
    Generate text embedding in binary format (base64-encoded float32).

    Returns ~6 KB for 1152-dimensional embeddings.
    """
    try:
        embedding = model_service.get_text_embedding(request.text)
        return EmbeddingResponse(**encode_embedding_binary(embedding))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text embedding: {str(e)}")

@app.post("/embed/image/upload", response_model=EmbeddingResponse)
async def embed_image_upload(file: UploadFile = File(...)):
    """
    Generate image embedding in binary format (base64-encoded float32).

    Returns ~6 KB for 1152-dimensional embeddings.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        embedding = model_service.get_image_embedding(image)
        return EmbeddingResponse(**encode_embedding_binary(embedding))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image embedding: {str(e)}")

def main():
    """Entry point for running the service."""
    # Default to localhost for security, but allow override via HOST env var
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=os.getenv("LOG_LEVEL", "info")
    )

if __name__ == "__main__":
    main()
