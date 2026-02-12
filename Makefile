.PHONY: help install run test clean

help:
	@echo "SigLIP Embedding Service - Available Commands"
	@echo "=============================================="
	@echo "install  - Install dependencies"
	@echo "run      - Run service (CPU)"
	@echo "test     - Run test client"
	@echo "clean    - Remove cache and build files"

install:
	pip install -r requirements.txt

run:
	python app.py

test:
	python test_client.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
