.PHONY: help install check-gpu run run-gpu run-cpu docker-build docker-run docker-gpu test clean

help:
	@echo "SigLIP Embedding Service - Available Commands"
	@echo "=============================================="
	@echo "install        - Install dependencies"
	@echo "check-gpu      - Check GPU availability"
	@echo "run            - Run service (auto-detect GPU)"
	@echo "run-gpu        - Run service with GPU"
	@echo "run-cpu        - Run service with CPU"
	@echo "docker-build   - Build Docker image (CPU)"
	@echo "docker-run     - Run Docker container (CPU)"
	@echo "docker-gpu     - Run Docker container (GPU)"
	@echo "test           - Run test client"
	@echo "clean          - Remove cache and build files"

install:
	pip install -r requirements.txt

check-gpu:
	python check_gpu.py

run:
	python app.py

run-gpu:
	DEVICE=cuda python app.py

run-cpu:
	DEVICE=cpu python app.py

docker-build:
	docker build -t siglip-api .

docker-build-gpu:
	docker build -f Dockerfile.gpu -t siglip-api-gpu .

docker-run:
	docker run -p 8000:8000 siglip-api

docker-run-gpu:
	docker run --gpus all -p 8000:8000 -e DEVICE=cuda siglip-api-gpu

docker-compose-up:
	docker-compose up -d

docker-compose-gpu:
	docker-compose -f docker-compose.gpu.yml up -d

docker-compose-down:
	docker-compose down
	docker-compose -f docker-compose.gpu.yml down

test:
	python test_client.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
