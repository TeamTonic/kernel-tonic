# Kernel Tonic: OLMo-based Model for MI300X

A simplified and optimized transformer model architecture based on OLMo (Open Language Model), specifically designed for AMD MI300X accelerators with FP8 quantization support and vLLM integration.

## Architecture Overview

### Model Specifications
- **Base Architecture**: Simplified OLMo transformer decoder
- **Target Hardware**: AMD Instinct MI300X
- **Precision**: FP8 quantization for inference
- **Framework**: PyTorch with ROCm optimizations
- **Inference Engine**: vLLM integration
- **Custom Kernels**: Hugging Face kernel-builder integration

### Key Features
- Optimized kernels for MI300X compute units
- FP8 quantization for memory efficiency
- Flash Attention 3.0 integration
- Transformer Engine optimizations
- Multi-dataset training support (up to 6 datasets)
- Custom HIP kernels via kernel-builder
- Docker containerization for easy deployment

## Project Structure

```
kernel_tonic/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── architecture.py      # Simplified OLMo architecture
│   │   ├── layers.py           # Custom layers and kernels
│   │   ├── config.py           # Model configuration
│   │   └── kernel_integration.py # Custom kernel integration
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── attention.py        # Optimized attention kernels
│   │   ├── linear.py           # Optimized linear kernels
│   │   └── activation.py       # Optimized activation kernels
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── fp8.py             # FP8 quantization
│   │   └── kernels.py         # Quantized kernels
│   └── training/
│       ├── __init__.py
│       ├── trainer.py          # Training loop
│       └── datasets.py         # Multi-dataset handling
├── kernels/
│   └── monolithic/
│       ├── kernel_builder.yaml # Kernel builder config
│       ├── kernel.hip          # Custom HIP kernels
│       └── __init__.py         # Python interface
├── configs/
│   ├── model_config.yaml      # Model configuration
│   └── training_config.yaml    # Training parameters
├── scripts/
│   ├── train.py               # Training script
│   ├── export_vllm.py         # vLLM export script
│   ├── run_training.sh        # Complete training pipeline
│   └── test_inference.py      # Inference test script
├── tests/
│   ├── test_model.py
│   ├── test_kernels.py
│   └── test_quantization.py
├── Dockerfile                 # Docker image for MI300X
├── docker-compose.yml         # Docker Compose setup
├── requirements.txt
├── setup.py
└── README.md
```

## MI300X Optimizations

### Hardware-Specific Features
- **Compute Units**: Optimized for MI300X CDNA3 architecture
- **Memory Hierarchy**: Leverages HBM3 memory bandwidth
- **Tensor Cores**: FP8 tensor operations
- **Multi-GPU**: Support for multi-MI300X configurations

### Kernel Optimizations
- Custom attention kernels using HIP
- Optimized linear layer implementations
- Flash Attention 3.0 integration
- Memory-efficient activation functions
- Monolithic kernel-builder integration

## Quick Start

### Prerequisites
- AMD MI300X GPU
- ROCm 5.7.3+
- Docker and Docker Compose
- Hugging Face token for dataset access

### 1. Set Environment Variables
```bash
export HF_TOKEN=your_huggingface_token_here
```

### 2. Run Complete Training Pipeline
```bash
# Make script executable
chmod +x scripts/run_training.sh

# Run complete pipeline
./scripts/run_training.sh
```

This will:
1. Build the Docker image
2. Train the model on Colossal OSCAR 1.0 (all languages)
3. Export the model for vLLM
4. Start the vLLM inference server

### 3. Test Inference
```bash
python scripts/test_inference.py
```

## Manual Usage

### Training
```bash
# Build Docker image
docker build -t kernel-tonic:latest .

# Run training
docker run --gpus all -e HF_TOKEN=$HF_TOKEN \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  kernel-tonic:latest train --config small --batch-size 4
```

### Export for vLLM
```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/models:/workspace/models \
  kernel-tonic:latest export \
  --checkpoint /workspace/checkpoints/best_model.pt \
  --output-dir /workspace/models/kernel-tonic \
  --config small
```

### vLLM Inference
```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/workspace/models \
  kernel-tonic:latest vllm
```

## Docker Compose Usage

### Start Training
```bash
docker-compose up kernel-tonic-train
```

### Start vLLM Server
```bash
docker-compose up -d kernel-tonic-vllm
```

### Export Model
```bash
docker-compose up kernel-tonic-export
```

## Custom Kernel Development

### Kernel Builder Integration
The project includes a monolithic kernel-builder setup in `kernels/monolithic/`:

```bash
# Kernel builder configuration
kernels/monolithic/kernel_builder.yaml

# Custom HIP kernels
kernels/monolithic/kernel.hip

# Python interface
kernels/monolithic/__init__.py
```

### Adding New Kernels
1. Add kernel function to `kernel.hip`
2. Update `kernel_builder.yaml`
3. Add Python interface in `__init__.py`
4. Integrate in `src/model/kernel_integration.py`

## Dataset: Colossal OSCAR 1.0

The training uses the [Colossal OSCAR 1.0](https://huggingface.co/datasets/oscar-corpus/colossal-oscar-1.0) dataset with all languages. The dataset is automatically downloaded and streamed during training.

## Model Configurations

- **Small**: 125M parameters (768 hidden, 12 layers)
- **Medium**: 1.3B parameters (1536 hidden, 24 layers)
- **Large**: 7B parameters (4096 hidden, 32 layers)
- **XLarge**: 13B parameters (5120 hidden, 40 layers)

## Performance Targets

Based on AMD ROCm documentation:
- **Throughput**: Optimized for high token generation rates
- **Latency**: Low inference latency for real-time applications
- **Memory Efficiency**: FP8 quantization for reduced memory footprint
- **Scalability**: Multi-GPU support for larger models

## API Usage

Once the vLLM server is running, you can make requests:

```python
import requests

# Text completion
response = requests.post("http://localhost:8000/v1/completions", json={
    "model": "kernel-tonic",
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
})

# Chat completion
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "kernel-tonic",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
})
```

## Troubleshooting

### Common Issues

1. **HF_TOKEN not set**
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **GPU not detected**
   ```bash
   # Check ROCm installation
   rocm-smi
   ```

3. **vLLM server not starting**
   ```bash
   # Check logs
   docker-compose logs kernel-tonic-vllm
   ```

4. **Out of memory**
   - Reduce batch size
   - Use smaller model configuration
   - Enable gradient checkpointing

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Training on RunPod MI300X Cloud

1. **Launch an MI300X instance on [RunPod.io](https://www.runpod.io/).**
2. **SSH into your instance and clone your repo.**
   ```bash
   git clone <your-repo-url>
   cd kernel_tonic
   ```
3. **Set your Hugging Face token:**
   ```bash
   export HF_TOKEN=your_hf_token_here
   ```
4. **Run the training pipeline:**
   ```bash
   chmod +x scripts/run_training.sh
   ./scripts/run_training.sh
   ```

## Push Model and Kernel to Hugging Face Hub

**Push Model:**
```bash
python scripts/push_model_to_hub.py
```

**Push Kernel:**
```bash
python scripts/push_kernel_to_hub.py
```

## Deploy Trained Model with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t kernel-tonic:latest .
   ```
2. **Run the vLLM server:**
   ```bash
   docker run --gpus all -p 8000:8000 \
     -v $(pwd)/models:/workspace/models \
     kernel-tonic:latest vllm
   ```
3. **Test inference:**
   ```bash
   python scripts/test_inference.py
   ``` 