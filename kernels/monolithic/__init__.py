"""
Python interface for the monolithic HIP kernel (OLMo/MI300X, FP8) via kernel-builder.
"""

from kernel_builder import load_kernel
import os

# Load the monolithic kernel
KERNEL_PATH = os.path.join(os.path.dirname(__file__), 'kernel.hip')
kernel = load_kernel(KERNEL_PATH, entry_point='monolithic_kernel')

# Kernel function stubs
def attention_fp8(*args, **kwargs):
    return kernel.attention_fp8(*args, **kwargs)

def mlp_fp8(*args, **kwargs):
    return kernel.mlp_fp8(*args, **kwargs)

def rmsnorm_fp8(*args, **kwargs):
    return kernel.rmsnorm_fp8(*args, **kwargs)

def embedding_fp8(*args, **kwargs):
    return kernel.embedding_fp8(*args, **kwargs)

def quantize_fp8(*args, **kwargs):
    return kernel.quantize_fp8(*args, **kwargs)

def dequantize_fp8(*args, **kwargs):
    return kernel.dequantize_fp8(*args, **kwargs) 