"""
Optimized layers for Kernel Tonic - OLMo-based model with MI300X optimizations.
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

from .config import KernelTonicConfig


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as used in OLMo.
    Optimized for MI300X with potential FP8 quantization.
    """
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        
    def forward(self, x: Tensor) -> Tensor:
        """
        SwiGLU activation: x * SiLU(Wx) * Vx
        where W and V are learned parameters.
        """
        # Split input into two halves for gate and value
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply SiLU (Swish) activation to first half
        x1 = F.silu(x1)
        
        # Multiply with second half
        return x1 * x2


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than standard LayerNorm for MI300X.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x: Tensor) -> Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for positional encoding.
    Optimized for MI300X tensor cores.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Pre-compute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            seq_len: Sequence length
            
        Returns:
            Tensor with rotary embeddings applied
        """
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = emb.cos()
        sin = emb.sin()
        
        # Reshape for broadcasting
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)
        
        # Apply rotation to input
        x_rot = torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
        return x * cos + x_rot * sin


class OptimizedLinear(nn.Module):
    """
    Optimized linear layer with FP8 quantization support for MI300X.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: KernelTonicConfig = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Use Transformer Engine if available and FP8 is enabled
        if (TRANSFORMER_ENGINE_AVAILABLE and config and 
            config.use_fp8_quantization and config.quantize_linear):
            self.use_te = True
            self.linear = te.Linear(
                in_features, out_features, bias=bias,
                params_dtype=torch.float16
            )
        else:
            self.use_te = False
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.use_te:
            return self.linear(x)
        else:
            return F.linear(x, self.weight, self.bias)


class OptimizedAttention(nn.Module):
    """
    Optimized multi-head attention with MI300X-specific optimizations.
    Supports Flash Attention, FP8 quantization, and grouped-query attention.
    """
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = OptimizedLinear(self.hidden_size, self.num_heads * self.head_dim, config=config)
        self.k_proj = OptimizedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, config=config)
        self.v_proj = OptimizedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, config=config)
        self.o_proj = OptimizedLinear(self.num_heads * self.head_dim, self.hidden_size, config=config)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            config.max_position_embeddings,
            config.rope_theta
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        # Flash Attention support
        self.use_flash_attention = (FLASH_ATTENTION_AVAILABLE and 
                                  config.use_flash_attention)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with optimized attention computation.
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            query_states = self.rotary_emb(query_states, seq_len)
            key_states = self.rotary_emb(key_states, seq_len)
        
        # Handle past key/value states
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat key/value heads if using grouped-query attention
        if self.num_key_value_heads != self.num_heads:
            key_states = self._repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = self._repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Compute attention
        if self.use_flash_attention and attention_mask is None:
            # Use Flash Attention for better performance
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states
            )
        else:
            # Standard attention computation
            attn_output = self._standard_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
        
        # Project output
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat key/value heads for grouped-query attention."""
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    
    def _flash_attention_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward pass using Flash Attention."""
        # Reshape for Flash Attention
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply Flash Attention
        output = flash_attn.flash_attn_func(q, k, v, dropout_p=self.config.attention_dropout)
        return output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
    
    def _standard_attention_forward(self, q: Tensor, k: Tensor, v: Tensor, 
                                  attention_mask: Optional[Tensor]) -> Tensor:
        """Standard attention computation."""
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        return attn_output


class OptimizedMLP(nn.Module):
    """
    Optimized MLP with SwiGLU activation and FP8 quantization support.
    """
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        
        # Use Transformer Engine if available
        if (TRANSFORMER_ENGINE_AVAILABLE and config.use_fused_mlp and 
            config.use_fp8_quantization):
            self.use_te = True
            self.mlp = te.LayerNormMLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size,
                bias=True,
                activation="swiglu",
                params_dtype=torch.float16
            )
        else:
            self.use_te = False
            # Gate and up projections (for SwiGLU)
            self.gate_proj = OptimizedLinear(config.hidden_size, config.intermediate_size, config=config)
            self.up_proj = OptimizedLinear(config.hidden_size, config.intermediate_size, config=config)
            # Down projection
            self.down_proj = OptimizedLinear(config.intermediate_size, config.hidden_size, config=config)
            # Activation
            self.act_fn = SwiGLU(config)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.use_te:
            return self.mlp(x)
        else:
            # SwiGLU: x * SiLU(Wx) * Vx
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            
            # Apply SwiGLU activation
            gate = self.act_fn(torch.cat([gate, up], dim=-1))
            
            # Down projection
            return self.down_proj(gate)


class TransformerBlock(nn.Module):
    """
    Single transformer block with optimized layers for MI300X.
    """
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.self_attn = OptimizedAttention(config)
        
        # MLP layer
        self.mlp = OptimizedMLP(config)
        
        # Layer normalization
        self.input_layernorm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        
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
        """
        Forward pass of transformer block with residual connections.
        """
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