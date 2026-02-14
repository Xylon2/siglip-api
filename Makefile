.PHONY: help install check-gpu run run-gpu run-cpu test clean

help:
	@echo "SigLIP Embedding Service - Available Commands"
	@echo "=============================================="
	@echo "install        - Install dependencies"
	@echo "check-gpu      - Check GPU availability"
	@echo "run            - Run service (auto-detect GPU)"
	@echo "run-gpu        - Run service with GPU"
	@echo "run-cpu        - Run service with CPU"
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

test:
	python test_client.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
