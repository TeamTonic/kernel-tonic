# Dockerfile for Kernel Tonic - OLMo-based model optimized for AMD MI300X
# Based on AMD ROCm and optimized for vLLM inference

FROM amd/rocm:6.0.2-ubuntu-22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCR_VISIBLE_DEVICES=0
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV HIP_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch, TorchVision, and Torchaudio for ROCm 6.x
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Copy requirements.txt and install all other dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install the project in development mode
RUN pip3 install -e .

# Create directories for data and models
RUN mkdir -p /workspace/data /workspace/models /workspace/logs

# Set environment variables for training
ENV HF_TOKEN=""
ENV CUDA_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0

# Expose port for vLLM API
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "train" ]; then\n\
    echo "Starting training..."\n\
    python scripts/train.py "${@:2}"\n\
elif [ "$1" = "export" ]; then\n\
    echo "Exporting model for vLLM..."\n\
    python scripts/export_vllm.py "${@:2}"\n\
elif [ "$1" = "vllm" ]; then\n\
    echo "Starting vLLM server..."\n\
    python -m vllm.entrypoints.openai.api_server \\\n        --model /workspace/models/kernel-tonic \\\n        --tensor-parallel-size 1 \\\n        --gpu-memory-utilization 0.9 \\\n        --max-model-len 8192 \\\n        --dtype fp8 \\\n        --quantization fp8 \\\n        --host 0.0.0.0 \\\n        --port 8000\n\
else\n\
    echo "Usage: docker run <image> [train|export|vllm] [args...]"\n\
    echo "  train: Start training the model"\n\
    echo "  export: Export model for vLLM"\n\
    echo "  vllm: Start vLLM inference server"\n\
fi' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]

# Default command
CMD ["help"] 