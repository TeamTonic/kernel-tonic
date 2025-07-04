#!/bin/bash
# Complete training pipeline for Kernel Tonic model

set -e

echo "=== Kernel Tonic Training Pipeline ==="

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set your Hugging Face token:"
    echo "export HF_TOKEN=your_token_here"
    exit 1
fi

# Create directories
mkdir -p data models logs checkpoints

echo "=== Building Docker Image ==="
docker build -t kernel-tonic:latest .

echo "=== Starting Training ==="
docker-compose up kernel-tonic-train

echo "=== Exporting Model for vLLM ==="
docker-compose up kernel-tonic-export

echo "=== Starting vLLM Inference Server ==="
docker-compose up -d kernel-tonic-vllm

echo "=== Training Pipeline Complete ==="
echo "vLLM server is running on http://localhost:8000"
echo "You can now make inference requests to the API" 