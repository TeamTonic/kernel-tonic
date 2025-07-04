"""
Kernel integration for OLMo-based model - replaces PyTorch ops with custom HIP kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor

try:
    from kernels.monolithic import (
        attention_fp8, mlp_fp8, rmsnorm_fp8, 
        embedding_fp8, quantize_fp8, dequantize_fp8
    )
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False
    print("Warning: Custom kernels not available, falling back to PyTorch operations")

from .config import KernelTonicConfig


class KernelOptimizedAttention(nn.Module):
    """Attention layer using custom HIP kernels for MI300X optimization."""
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # Projections (will use custom kernels if available)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        # Use custom kernels if available
        self.use_custom_kernels = KERNELS_AVAILABLE and config.use_optimized_kernels
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Handle past key/value states
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat key/value heads if using grouped-query attention
        if self.num_key_value_heads != self.num_heads:
            key_states = self._repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = self._repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Use custom attention kernel if available
        if self.use_custom_kernels:
            attn_output = attention_fp8(
                query_states, key_states, value_states,
                attention_mask, self.scaling, self.attn_dropout.p
            )
        else:
            # Fallback to PyTorch attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Project output
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


class KernelOptimizedMLP(nn.Module):
    """MLP layer using custom HIP kernels for MI300X optimization."""
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        
        # Gate and up projections (for SwiGLU)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        # Down projection
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Use custom kernels if available
        self.use_custom_kernels = KERNELS_AVAILABLE and config.use_optimized_kernels
    
    def forward(self, x: Tensor) -> Tensor:
        if self.use_custom_kernels:
            # Use custom MLP kernel
            return mlp_fp8(x, self.gate_proj.weight, self.up_proj.weight, 
                          self.down_proj.weight, self.config.intermediate_size)
        else:
            # Fallback to PyTorch MLP
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            
            # SwiGLU activation
            gate = F.silu(gate)
            gate = gate * up
            
            return self.down_proj(gate)


class KernelOptimizedRMSNorm(nn.Module):
    """RMSNorm using custom HIP kernels for MI300X optimization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, config: KernelTonicConfig = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.config = config
        
        # Use custom kernels if available
        self.use_custom_kernels = KERNELS_AVAILABLE and config and config.use_optimized_kernels
    
    def forward(self, x: Tensor) -> Tensor:
        if self.use_custom_kernels:
            return rmsnorm_fp8(x, self.weight, self.eps)
        else:
            # Fallback to PyTorch RMSNorm
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight


class KernelOptimizedEmbedding(nn.Module):
    """Embedding layer using custom HIP kernels for MI300X optimization."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, config: KernelTonicConfig = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.config = config
        
        # Use custom kernels if available
        self.use_custom_kernels = KERNELS_AVAILABLE and config and config.use_optimized_kernels
        
        # Initialize weights
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        if self.use_custom_kernels:
            return embedding_fp8(input_ids, self.weight)
        else:
            # Fallback to PyTorch embedding
            return F.embedding(input_ids, self.weight)


class KernelOptimizedTransformerBlock(nn.Module):
    """Transformer block using custom HIP kernels for MI300X optimization."""
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.self_attn = KernelOptimizedAttention(config)
        
        # MLP layer
        self.mlp = KernelOptimizedMLP(config)
        
        # Layer normalization
        self.input_layernorm = KernelOptimizedRMSNorm(config.hidden_size, config.layer_norm_eps, config)
        self.post_attention_layernorm = KernelOptimizedRMSNorm(config.hidden_size, config.layer_norm_eps, config)
        
        # Dropout
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self.hidden_dropout(hidden_states)
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.hidden_dropout(hidden_states)
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


def quantize_model_fp8(model: nn.Module, config: KernelTonicConfig) -> nn.Module:
    """Quantize model weights to FP8 using custom kernels."""
    if not KERNELS_AVAILABLE:
        print("Warning: Custom kernels not available, skipping FP8 quantization")
        return model
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Quantize linear layer weights
            if hasattr(module, 'weight'):
                module.weight.data = quantize_fp8(module.weight.data)
        elif isinstance(module, nn.Embedding):
            # Quantize embedding weights
            if hasattr(module, 'weight'):
                module.weight.data = quantize_fp8(module.weight.data)
    
    return model


def dequantize_model_fp8(model: nn.Module, config: KernelTonicConfig) -> nn.Module:
    """Dequantize model weights from FP8 using custom kernels."""
    if not KERNELS_AVAILABLE:
        print("Warning: Custom kernels not available, skipping FP8 dequantization")
        return model
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Dequantize linear layer weights
            if hasattr(module, 'weight'):
                module.weight.data = dequantize_fp8(module.weight.data)
        elif isinstance(module, nn.Embedding):
            # Dequantize embedding weights
            if hasattr(module, 'weight'):
                module.weight.data = dequantize_fp8(module.weight.data)
    
    return model 