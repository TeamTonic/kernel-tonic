#!/usr/bin/env python3
"""
Export trained Kernel Tonic model for vLLM inference on MI300X.
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model.config import KernelTonicConfig
from model.architecture import KernelTonicForCausalLM
from model.kernel_integration import dequantize_model_fp8


def export_for_vllm(checkpoint_path: str, output_dir: str, config: KernelTonicConfig):
    """Export model for vLLM inference."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = KernelTonicForCausalLM(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Dequantize if needed (for compatibility)
    model = dequantize_model_fp8(model, config)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in vLLM-compatible format
    print(f"Saving model to {output_dir}")
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    # Save config
    config_dict = config.to_dict()
    config_dict['model_type'] = 'kernel_tonic'
    config_dict['architectures'] = ['KernelTonicForCausalLM']
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save tokenizer config
    tokenizer_config = {
        "model_type": "kernel_tonic",
        "tokenizer_class": "AutoTokenizer",
        "pad_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>"
    }
    
    with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print("Model exported successfully for vLLM!")


def create_vllm_launch_script(output_dir: str, model_name: str):
    """Create a vLLM launch script for MI300X."""
    script_content = f'''#!/bin/bash
# vLLM launch script for Kernel Tonic model on MI300X

export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Launch vLLM server
python -m vllm.entrypoints.openai.api_server \\
    --model {output_dir} \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192 \\
    --dtype fp8 \\
    --quantization fp8 \\
    --host 0.0.0.0 \\
    --port 8000
'''
    
    script_path = os.path.join(output_dir, 'launch_vllm.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"vLLM launch script created: {script_path}")


def main():
    parser = argparse.ArgumentParser(description='Export model for vLLM inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for exported model')
    parser.add_argument('--config', type=str, default='small',
                       choices=['small', 'medium', 'large', 'xlarge'],
                       help='Model configuration')
    parser.add_argument('--model-name', type=str, default='kernel-tonic',
                       help='Model name for vLLM')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config == 'small':
        from model.config import get_small_config
        config = get_small_config()
    elif args.config == 'medium':
        from model.config import get_medium_config
        config = get_medium_config()
    elif args.config == 'large':
        from model.config import get_large_config
        config = get_large_config()
    else:
        from model.config import get_xlarge_config
        config = get_xlarge_config()
    
    # Export model
    export_for_vllm(args.checkpoint, args.output_dir, config)
    
    # Create vLLM launch script
    create_vllm_launch_script(args.output_dir, args.model_name)
    
    print(f"""
Export completed successfully!

To run inference with vLLM:
1. cd {args.output_dir}
2. ./launch_vllm.sh

Or manually:
python -m vllm.entrypoints.openai.api_server \\
    --model {args.output_dir} \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192 \\
    --dtype fp8 \\
    --quantization fp8 \\
    --host 0.0.0.0 \\
    --port 8000
""")


if __name__ == "__main__":
    main() 