"""
Main model architecture for Kernel Tonic - Simplified OLMo transformer optimized for MI300X.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import KernelTonicConfig
from .layers import TransformerBlock, RMSNorm, OptimizedLinear


class KernelTonicModel(nn.Module):
    """
    Simplified OLMo-based transformer model optimized for AMD MI300X.
    
    Key features:
    - Simplified OLMo architecture
    - MI300X-specific optimizations
    - FP8 quantization support
    - Flash Attention integration
    - Multi-dataset training support
    """
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embedding (learned)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        
        # Output projection
        self.lm_head = OptimizedLinear(config.hidden_size, config.vocab_size, bias=False, config=config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using OLMo-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings layer."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module):
        """Set the input embeddings layer."""
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key/value states for caching
            inputs_embeds: Pre-computed input embeddings
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return as dictionary
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Get position embeddings
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.embed_positions(position_ids)
        
        # Combine token and position embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        
        # Convert attention mask to causal mask
        causal_mask = self._prepare_causal_mask(attention_mask, hidden_states.dtype)
        
        # Initialize outputs
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        
        # Forward through transformer layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_forward(
                    layer, hidden_states, causal_mask, position_ids, past_key_value,
                    output_attentions, use_cache
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add final hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Prepare outputs
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions]
                if v is not None
            )
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }
    
    def _prepare_causal_mask(self, attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
        """Prepare causal attention mask."""
        batch_size, seq_length = attention_mask.shape
        causal_mask = torch.full(
            (batch_size, seq_length, seq_length),
            float("-inf"),
            dtype=dtype,
            device=attention_mask.device,
        )
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_length, seq_length, device=attention_mask.device), diagonal=1)
        causal_mask.masked_fill_(mask.bool(), float("-inf"))
        
        # Apply attention mask
        causal_mask = causal_mask.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        
        return causal_mask
    
    def _gradient_checkpointing_forward(
        self,
        layer: TransformerBlock,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_value: Optional[Tuple[Tensor, Tensor]],
        output_attentions: bool,
        use_cache: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass with gradient checkpointing."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        layer_outputs = torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer),
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
        return layer_outputs


class KernelTonicForCausalLM(nn.Module):
    """
    Kernel Tonic model for causal language modeling.
    """
    
    def __init__(self, config: KernelTonicConfig):
        super().__init__()
        self.config = config
        self.model = KernelTonicModel(config)
        self.lm_head = self.model.lm_head
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings layer."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module):
        """Set the input embeddings layer."""
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self) -> nn.Module:
        """Get the output embeddings layer."""
        return self.lm_head
    
    def set_output_embeddings(self, value: nn.Module):
        """Set the output embeddings layer."""
        self.lm_head = value
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for causal language modeling.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
        }
    
    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
        }
    
    def _reorder_cache(self, past_key_values: List[Tuple[Tensor, Tensor]], beam_idx: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past 