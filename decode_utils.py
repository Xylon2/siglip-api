"""
Utility functions for decoding binary embeddings from the SigLIP API.
"""
import base64
import numpy as np


def decode_embedding(response: dict) -> np.ndarray:
    """
    Decode a binary embedding response from the API.

    Args:
        response: Dictionary from API containing 'embedding', 'shape', and 'size_bytes'

    Returns:
        numpy array of float32 values

    Example:
        >>> import requests
        >>> resp = requests.post("http://localhost:8000/embed/text",
        ...                      json={"text": "hello world"})
        >>> data = resp.json()
        >>> embedding = decode_embedding(data)
        >>> embedding.shape
        (1152,)
        >>> embedding.dtype
        dtype('float32')
    """
    b64_string = response['embedding']
    binary_data = base64.b64decode(b64_string)
    arr = np.frombuffer(binary_data, dtype=np.float32)

    # Reshape if needed (currently always 1D)
    if 'shape' in response:
        arr = arr.reshape(response['shape'])

    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        a: First embedding (already normalized by the model)
        b: Second embedding (already normalized by the model)

    Returns:
        Cosine similarity score between -1 and 1

    Note:
        SigLIP embeddings are already L2-normalized, so we can just
        use dot product for cosine similarity.
    """
    return np.dot(a, b)
