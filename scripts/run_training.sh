#!/bin/bash
# Simple training pipeline for Kernel Tonic model (no Docker)

set -e

echo "=== Kernel Tonic Training Pipeline (No Docker) ==="

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set your Hugging Face token:"
    echo "export HF_TOKEN=your_token_here"
    exit 1
fi

# Create directories
mkdir -p data models logs checkpoints

echo "=== Starting Training ==="
python scripts/train.py --config small --batch-size 2 --num-epochs 4

echo "=== Training Pipeline Complete ==="
echo "Checkpoints and logs are available in ./checkpoints and ./logs." 