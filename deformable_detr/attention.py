import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Optional
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import MultiScaleDeformableAttention


class TemporalMSDA(nn.Module):
    def __init__(self, config: DeformableDetrConfig, num_heads: int, n_points: int):
        super().__init__()
        self.attn = MultiScaleDeformableAttention()

        if config.d_model % num_heads != 0:
            raise ValueError(f'embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}')
        
        dim_per_head = config.d_model // num_heads
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0): # Check if dim_per_head is power of 2
            warnings.warn(
                "You'd better set embed_dim (d_model) in TemporalMSDA to make the dimension of each "
                "attention head a power of 2 which is more efficient in the authors' CUDA implementation."
            )

        self.im2col_step = 64
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets  = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj  = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.disable_custom_kernels = config.disable_custom_kernels
        self._reset_parameters()
        
        
    def _reset_parameters(self): # Adapted from DeformableDetrPreTrainedModel
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        default_dtype = torch.get_default_dtype()
        thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * 2.0 * math.pi / self.n_heads
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
        )[..., 0].repeat(1, self.n_levels, self.n_points)
        
        for i in range(self.n_points): grid_init[:, :, i] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)
        

    def forward(
        self, hidden_states: Tensor, 
        encoder_hidden_states=None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tensor] = None,
        reference_points=None, temporal_shapes=None, level_start_index=None,
    ):
        # Add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None: hidden_states += position_embeddings
        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if temporal_shapes.sum() != sequence_length:
            raise ValueError('Make sure to align the temporal shapes with the sequence length of the encoder hidden states.')

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None: # We invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
            
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(batch_size, num_queries, self.n_heads, self.n_levels, self.n_points)
        attention_weights = self.attention_weights(hidden_states).view(batch_size, num_queries, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(batch_size, num_queries, self.n_heads, self.n_levels, self.n_points)
        
        sampling_locations = reference_points[:, :, None, :, None, 0] # (batch_size, num_queries, 1, n_levels, 1)
        if reference_points.shape[-1] == 1:
            sampling_locations += sampling_offsets / temporal_shapes[None, None, None, :, None]
        elif reference_points.shape[-1] == 2:
            sampling_locations += sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 1] * 0.5
        else: raise ValueError(f'Last dim of reference_points must be 1 or 2, but got {reference_points.shape[-1]}')

        # For 1D temporal data, we set the 'height' dimension to be always 0.5 (the middle of the only row)
        # Similarly, the H dimension in temporal_shapes is always 1 for temporal data
        sampling_locations = torch.stack((sampling_locations, 0.5 * sampling_locations.new_ones(sampling_locations.shape)), -1)
        temporal_shapes = torch.stack([temporal_shapes.new_ones(temporal_shapes.shape), temporal_shapes], -1)
        output = self.output_proj(self.attn(
            value,
            None, # The `value_spatial_shapes` arg here does nothing in HF implementation, so I set it to None
            temporal_shapes, # and use this arg as both spatial_shapes and spatial_shapes_list
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        ))
        return output, attention_weights