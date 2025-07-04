"""
Model configuration for Kernel Tonic - Simplified OLMo architecture optimized for MI300X.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch


@dataclass
class KernelTonicConfig:
    """
    Configuration for the simplified OLMo-based model optimized for MI300X.
    
    Based on OLMo architecture but simplified and optimized for:
    - AMD MI300X CDNA3 architecture
    - FP8 quantization for inference
    - vLLM integration
    - Multi-dataset training support
    """
    
    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # For grouped-query attention
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Architecture specific
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    activation_function: str = "swiglu"  # OLMo uses SwiGLU
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    
    # MI300X optimizations
    use_flash_attention: bool = True
    use_fused_mlp: bool = True
    use_fused_attention: bool = True
    use_fp8_quantization: bool = True
    use_tensor_cores: bool = True
    
    # Quantization settings
    quantize_linear: bool = True
    quantize_attention: bool = True
    fp8_format: str = "e4m3"  # e4m3 or e5m2
    
    # Training optimizations
    gradient_checkpointing: bool = False
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 1
    
    # Multi-dataset support
    max_datasets: int = 6
    dataset_weights: Optional[List[float]] = None
    
    # vLLM integration
    use_vllm: bool = True
    vllm_max_model_len: int = 8192
    vllm_gpu_memory_utilization: float = 0.9
    
    # Performance tuning
    use_hip_kernels: bool = True
    use_optimized_kernels: bool = True
    memory_efficient_attention: bool = True
    
    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
            
        if self.dataset_weights is None:
            self.dataset_weights = [1.0] * self.max_datasets
            
        # Validate configuration
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
            
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot be greater than num_attention_heads")
            
        if len(self.dataset_weights) != self.max_datasets:
            raise ValueError("dataset_weights length must match max_datasets")
    
    @property
    def head_dim(self) -> int:
        """Get the dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_parameters(self) -> int:
        """Calculate the total number of parameters."""
        # Simplified calculation for transformer decoder
        embedding_params = self.vocab_size * self.hidden_size
        layer_params = self.num_hidden_layers * (
            # Self-attention
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O projections
            4 * self.hidden_size +  # Biases
            
            # MLP
            2 * self.hidden_size * self.intermediate_size +  # Up projection
            self.intermediate_size * self.hidden_size +  # Down projection
            2 * self.intermediate_size + self.hidden_size +  # Biases
            
            # Layer norms
            2 * self.hidden_size * 2  # Two layer norms per layer
        )
        final_norm_params = self.hidden_size * 2
        
        return embedding_params + layer_params + final_norm_params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "activation_function": self.activation_function,
            "layer_norm_eps": self.layer_norm_eps,
            "use_cache": self.use_cache,
            "use_flash_attention": self.use_flash_attention,
            "use_fused_mlp": self.use_fused_mlp,
            "use_fused_attention": self.use_fused_attention,
            "use_fp8_quantization": self.use_fp8_quantization,
            "use_tensor_cores": self.use_tensor_cores,
            "quantize_linear": self.quantize_linear,
            "quantize_attention": self.quantize_attention,
            "fp8_format": self.fp8_format,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_gradient_accumulation": self.use_gradient_accumulation,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_datasets": self.max_datasets,
            "dataset_weights": self.dataset_weights,
            "use_vllm": self.use_vllm,
            "vllm_max_model_len": self.vllm_max_model_len,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "use_hip_kernels": self.use_hip_kernels,
            "use_optimized_kernels": self.use_optimized_kernels,
            "memory_efficient_attention": self.memory_efficient_attention,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "KernelTonicConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Predefined model configurations
def get_small_config() -> KernelTonicConfig:
    """Get small model configuration (125M parameters)."""
    return KernelTonicConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )


def get_medium_config() -> KernelTonicConfig:
    """Get medium model configuration (1.3B parameters)."""
    return KernelTonicConfig(
        vocab_size=32000,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=24,
        max_position_embeddings=4096,
    )


def get_large_config() -> KernelTonicConfig:
    """Get large model configuration (7B parameters)."""
    return KernelTonicConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=8192,
    )


def get_xlarge_config() -> KernelTonicConfig:
    """Get extra large model configuration (13B parameters)."""
    return KernelTonicConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=8192,
    ) 