from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import base64
from io import BytesIO
import uvicorn
import os

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
        self.device = None

    def load_model(self):
        if self.model is None:
            # Determine device (GPU if available, otherwise CPU)
            device_env = os.getenv("DEVICE", "auto").lower()

            if device_env == "cuda" or (device_env == "auto" and torch.cuda.is_available()):
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            elif device_env == "cpu" or device_env == "auto":
                self.device = torch.device("cpu")
                print("Using CPU")
            else:
                raise ValueError(f"Invalid DEVICE value: {device_env}. Use 'auto', 'cuda', or 'cpu'")

            print("Loading SigLIP model... (this may take a moment)")
            self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
            self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=True)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded successfully on {self.device}!")

    def get_text_embedding(self, text: str) -> tuple[torch.Tensor, tuple]:
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

        # Move back to CPU for numpy conversion
        text_vector = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
        text_vector = text_vector.cpu()
        return text_vector, text_vector.shape

    def get_image_embedding(self, image: Image.Image) -> tuple[torch.Tensor, tuple]:
        image_inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to device
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)

        # Move back to CPU for numpy conversion
        image_vector = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)
        image_vector = image_vector.cpu()
        return image_vector, image_vector.shape

model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    model_service.load_model()

@app.get("/")
async def root():
    device_info = {
        "device": str(model_service.device) if model_service.device else "not loaded",
        "cuda_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available() and model_service.device and model_service.device.type == "cuda":
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

    return {
        "message": "SigLIP Embedding Service",
        "version": "1.0.0",
        "model": "google/siglip-so400m-patch14-384",
        "embedding_dimension": 1152,
        "device": device_info,
        "endpoints": {
            "text": "/embed/text",
            "image": "/embed/image",
            "image_upload": "/embed/image/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    response = {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "device": str(model_service.device) if model_service.device else None
    }

    if torch.cuda.is_available() and model_service.device and model_service.device.type == "cuda":
        response["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        response["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)

    return response

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
