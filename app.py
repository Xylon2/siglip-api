from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, field_validator
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from io import BytesIO
import uvicorn
import os
from dotenv import load_dotenv

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
    embedding: list[float]
    shape: list[int]

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
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move inputs to device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)

        # Normalize - squeeze to remove any extra dimensions
        text_vector = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
        text_vector = text_vector.squeeze()

        # Convert to flat list - ensure 1D array
        embedding_array = text_vector.cpu().numpy()
        if embedding_array.ndim > 1:
            embedding_array = embedding_array.flatten()

        return embedding_array.tolist()

    def get_image_embedding(self, image: Image.Image) -> list[float]:
        image_inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to device
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)

        # Normalize - squeeze to remove any extra dimensions
        image_vector = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)
        image_vector = image_vector.squeeze()

        # Convert to flat list - ensure 1D array
        embedding_array = image_vector.cpu().numpy()
        if embedding_array.ndim > 1:
            embedding_array = embedding_array.flatten()

        return embedding_array.tolist()

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
    try:
        embedding = model_service.get_text_embedding(request.text)
        return EmbeddingResponse(
            embedding=embedding,
            shape=[len(embedding)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text embedding: {str(e)}")

@app.post("/embed/image/upload", response_model=EmbeddingResponse)
async def embed_image_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        embedding = model_service.get_image_embedding(image)
        return EmbeddingResponse(
            embedding=embedding,
            shape=[len(embedding)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image embedding: {str(e)}")

if __name__ == "__main__":
    # Default to localhost for security, but allow override via HOST env var
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
