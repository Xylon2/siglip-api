import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import sys

args_as_string = " ".join(sys.argv[1:])

# 1. Setup: Load Model & Processor
print("Loading model... (this takes time only on first run)")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=True)

# ---------------------------------------------------------
# PART A: Get Vector from TEXT
# ---------------------------------------------------------
text_query = args_as_string
print(text_query)

# Tokenize the text (convert words to ID numbers)
# padding="max_length" is CRITICAL for SigLIP to work correctly
text_inputs = processor(
    text=[text_query], 
    padding="max_length", 
    truncation=True,
    return_tensors="pt"
)

# Extract the Text Vector
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

# Normalize the vector (Standard practice for cosine similarity)
text_vector = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)

print(f"\n--- Text Vector for '{text_query}' ---")
print(f"Shape: {text_vector.shape}") # Should be (1152,)
print(f"First 5 numbers: {text_vector[:5].numpy()}")


# ---------------------------------------------------------
# PART B: Get Vector from IMAGE (Just to compare)
# ---------------------------------------------------------
image = Image.open("images/woods.avif")

image_inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)

# Normalize image vector too
image_vector = image_features[0] / image_features[0].norm(dim=-1, keepdim=True)

print(f"\n--- Image Vector (Cats) ---")
print(f"Shape: {image_vector.shape}")
print(f"First 5 numbers: {image_vector[:5].numpy()}")

# ---------------------------------------------------------
# PART C: The Magic (Compare them)
# ---------------------------------------------------------
# We calculate the "Dot Product" (Similarity Score)
similarity = (text_vector @ image_vector) * 100

print(f"\nSimilarity Score: {similarity.item():.2f}%")
# -100 to +100
#
# 30 - 40 = very strong match
# 15 - 25 = A good/decent match
