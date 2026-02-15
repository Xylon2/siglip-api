.PHONY: help dev prod test build clean

help:
	@echo "Available commands:"
	@echo "  make dev     - Run development server with auto-reload"
	@echo "  make prod    - Run production server with workers"
	@echo "  make test    - Run tests"
	@echo "  make build   - Build PEX executable"
	@echo "  make clean   - Remove build artifacts"

dev:
	uvicorn app:app --reload

prod:
	uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

test:
	python test_client.py

build:
	mkdir -p dist
	pex . --entry-point app:main --output-file dist/siglip-service.pex --python-shebang='/usr/bin/env python3'

clean:
	rm -rf dist/ build/ *.egg-info __pycache__
