.PHONY: help install dev prod run test clean

help:
	@echo "SigLIP Embedding Service - Available Commands"
	@echo "=============================================="
	@echo "install  - Install dependencies"
	@echo "dev      - Run development server (auto-reload)"
	@echo "prod     - Run production server (4 workers)"
	@echo "run      - Run service (basic, for compatibility)"
	@echo "test     - Run test client"
	@echo "clean    - Remove cache and build files"

install:
	pip install -e .

dev:
	uvicorn app:app --reload --log-level info

prod:
	uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

run:
	uvicorn app:app

test:
	python test_client.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
