import requests
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_text_embedding():
    print("Testing text embedding...")
    response = requests.post(
        f"{BASE_URL}/embed/text",
        json={"text": "a photo of a forest"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Text embedding shape: {result['shape']}")
        print(f"  First 5 values: {result['embedding'][:5]}")
        return np.array(result['embedding'])
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")
        return None

def test_image_embedding(image_path):
    print(f"\nTesting image embedding with {image_path}...")

    if not Path(image_path).exists():
        print(f"✗ Image file not found: {image_path}")
        return None

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/embed/image/upload",
            files={"file": f}
        )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Image embedding shape: {result['shape']}")
        print(f"  First 5 values: {result['embedding'][:5]}")
        return np.array(result['embedding'])
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")
        return None

def calculate_similarity(embedding1, embedding2, label1="Vector 1", label2="Vector 2"):
    if embedding1 is None or embedding2 is None:
        return

    similarity = np.dot(embedding1, embedding2) * 100
    print(f"\n{'='*50}")
    print(f"Similarity between {label1} and {label2}:")
    print(f"  {similarity:.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("SigLIP Embedding API Test Client")
    print("="*50)

    # Check server status
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            info = response.json()
            print(f"\nServer Info:")
            print(f"  Version: {info.get('version')}")
            print(f"  Model: {info.get('model')}")
            print(f"  Device: {info.get('device')}")
            print("="*50)
    except Exception as e:
        print(f"Warning: Could not get server info: {e}")
        print("="*50)

    # Test text embedding
    text_embedding = test_text_embedding()

    # Test image embedding with default image if it exists
    image_path = "images/larisa-k-autumn-6708984_1280.jpg"
    image_embedding = test_image_embedding(image_path)

    # Calculate similarity between text and image
    if text_embedding is not None and image_embedding is not None:
        calculate_similarity(text_embedding, image_embedding,
                           "Text: 'a photo of a forest'",
                           f"Image: {image_path}")
