from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import base64
from io import BytesIO
import uvicorn

app = FastAPI(title="SigLIP Embedding Service")

class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_base64: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    shape: list[int]

class ModelService:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is None:
            print("Loading SigLIP model... (this may take a moment)")
            self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
            self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=True)
            self.model.eval()
            print("Model loaded successfully!")

    def get_text_embedding(self, text: str) -> tuple[torch.Tensor, tuple]:
        text_inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)

        text_vector = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
        return text_vector, text_vector.shape

    def get_image_embedding(self, image: Image.Image) -> tuple[torch.Tensor, tuple]:
        image_inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)

        image_vector = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)
        return image_vector, image_vector.shape

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
        "endpoints": {
            "text": "/embed/text",
            "image": "/embed/image",
            "image_upload": "/embed/image/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None
    }

@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    try:
        embedding, shape = model_service.get_text_embedding(request.text)
        return EmbeddingResponse(
            embedding=embedding.numpy().tolist(),
            shape=list(shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text embedding: {str(e)}")

@app.post("/embed/image", response_model=EmbeddingResponse)
async def embed_image(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data))

        embedding, shape = model_service.get_image_embedding(image)
        return EmbeddingResponse(
            embedding=embedding.numpy().tolist(),
            shape=list(shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image embedding: {str(e)}")

@app.post("/embed/image/upload", response_model=EmbeddingResponse)
async def embed_image_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        embedding, shape = model_service.get_image_embedding(image)
        return EmbeddingResponse(
            embedding=embedding.numpy().tolist(),
            shape=list(shape)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image embedding: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
