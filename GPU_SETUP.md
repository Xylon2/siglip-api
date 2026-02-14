# GPU Setup Guide

This guide helps you set up GPU acceleration for the SigLIP Embedding Service.

## Table of Contents

- [Local GPU Setup](#local-gpu-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Local GPU Setup

### Prerequisites

1. **NVIDIA GPU** with compute capability 3.5 or higher
2. **NVIDIA GPU Drivers** installed
3. **CUDA Toolkit** (11.8 or later recommended)

### Step 1: Check GPU Availability

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Should show your GPU information
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify GPU in Python

```bash
python check_gpu.py
```

### Step 4: Run the Service with GPU

```bash
DEVICE=cuda python app.py
```

## Verification

### Check if GPU is Being Used

1. **Via API**:
```bash
curl http://localhost:8000/ | jq '.device'
```

Expected output:
```json
{
  "device": "cuda:0",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "gpu_memory_total_gb": 24.0
}
```

2. **Via Health Endpoint**:
```bash
curl http://localhost:8000/health | jq
```

3. **Monitor GPU Usage**:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Then make requests to the API and observe GPU memory usage increase.

## Troubleshooting

### Issue: "CUDA not available"

**Solution**:
```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Check if PyTorch can see CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "CUDA out of memory"

**Solution**:
- Your GPU doesn't have enough VRAM
- Close other applications using GPU
- Minimum 4GB VRAM recommended
- Use CPU mode: `DEVICE=cpu python app.py`

### Issue: "RuntimeError: No HIP GPUs are available"

This error appears on AMD GPUs. This service requires NVIDIA GPUs with CUDA support.

**Solution**:
- Use CPU mode: `DEVICE=cpu python app.py`
- Or use a system with NVIDIA GPU

## Performance Comparison

### Typical Performance

| Device | Cold Start | Inference Time |
|--------|-----------|----------------|
| CPU (Intel i7) | 5-10s | 1-2s per request |
| GPU (RTX 3090) | 5-10s | 50-200ms per request |
| GPU (T4) | 5-10s | 100-300ms per request |

**Note**: Cold start time is the same because model loading happens on CPU. The difference is in inference speed.

### Benchmark Script

```python
import time
import requests

url = "http://localhost:8000/embed/text"
data = {"text": "a photo of a cat"}

# Warmup
requests.post(url, json=data)

# Benchmark
times = []
for i in range(10):
    start = time.time()
    requests.post(url, json=data)
    times.append(time.time() - start)

print(f"Average time: {sum(times)/len(times):.3f}s")
print(f"Min time: {min(times):.3f}s")
print(f"Max time: {max(times):.3f}s")
```

## Cloud GPU Options

If you don't have a local GPU, consider these cloud options:

- **AWS**: EC2 instances (g4dn, p3, p4d)
- **Google Cloud**: Compute Engine with GPU
- **Azure**: NC-series VMs
- **Lambda Labs**: GPU cloud instances
- **Paperspace**: Gradient notebooks/deployments
- **RunPod**: Affordable GPU rentals

For production deployments with auto-scaling, consider:
- **AWS SageMaker**
- **Google Cloud Run with GPUs**
- **Azure ML**
- **Hugging Face Inference Endpoints**
