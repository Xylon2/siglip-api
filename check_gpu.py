#!/usr/bin/env python3
"""
Simple script to check GPU availability for SigLIP service.
"""

import torch

print("GPU Availability Check")
print("=" * 60)

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print()

    # Check current memory usage
    print("Current GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print()

    print("✓ GPU is available and ready to use!")
    print("  Run with: DEVICE=cuda python app.py")
else:
    print()
    print("✗ No GPU available. Will run on CPU.")
    print()
    print("To enable GPU support:")
    print("  1. Install NVIDIA GPU drivers")
    print("  2. Install CUDA toolkit (11.8 or later)")
    print("  3. Install PyTorch with CUDA:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("=" * 60)
