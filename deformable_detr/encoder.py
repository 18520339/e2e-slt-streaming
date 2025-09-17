import torch
import torch.nn.functional as F
from torch import nn, Tensor, FloatTensor, LongTensor
from typing import Optional
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    ACT2FN, GradientCheckpointingLayer, 
    DeformableDetrPreTrainedModel, BaseModelOutput
)
from attention import TemporalMSDA


class DeformableDetrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = TemporalMSDA(config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self, hidden_states: Tensor,  
        attention_mask: Tensor, position_embeddings: Optional[Tensor] = None, 
        reference_points=None, temporal_shapes=None, level_start_index=None, 
        output_attentions: bool = False,
    ):
        ''' 
        Args:
            hidden_states (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`): Input to the layer.
            attention_mask (`FloatTensor` of shape `(batch_size, sequence_length)`): Attention mask.
            position_embeddings (`FloatTensor`, *optional*): Position embeddings, to be added to `hidden_states`.
            reference_points (`FloatTensor`, *optional*): Reference points.
            temporal_shapes (`LongTensor`, *optional*): Temporal shapes of the backbone feature maps.
            level_start_index (`LongTensor`, *optional*): Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. 
                See `attentions` under returned tensors for more detail.
        '''
        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            temporal_shapes=temporal_shapes,
            level_start_index=level_start_index,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.final_layer_norm(hidden_states)
        
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions: outputs += (attn_weights,)
        return outputs
    
    
class DeformableDetrEncoder(DeformableDetrPreTrainedModel):
    '''
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a [`DeformableDetrEncoderLayer`]. 
    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: DeformableDetrConfig
    '''

    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.dropout = config.dropout
        self.layers = nn.ModuleList([DeformableDetrEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.post_init() # Initialize weights and apply final processing


    @staticmethod
    def get_reference_points(temporal_shapes, valid_ratios, device):
        ''' Get reference points for each feature map. Used in decoder.

        Args:
            temporal_shapes (`LongTensor` of shape `(num_feature_levels,)`): Temporal shapes of each feature map.
            valid_ratios (`FloatTensor` of shape `(batch_size, num_feature_levels,)`): Valid ratios of each feature map.
            device (`torch.device`): Device on which to create the tensors.
        Returns:
            `FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 1)`
        '''
        reference_points_list = []
        for level, width in enumerate(temporal_shapes):
            ref = torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device)
            ref = ref.reshape(-1)[None] / (valid_ratios[:, None, level] * width)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # (B, num_queries, num_feature_levels)
        return reference_points[:, :, :, None] # Add an extra dim for 1D case because HF logic expects 2D references


    def forward(
        self, inputs_embeds=None,
        attention_mask=None, position_embeddings=None,
        temporal_shapes=None, level_start_index=None, valid_ratios=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        '''
        Args:
            inputs_embeds (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            temporal_shapes (`LongTensor` of shape `(num_feature_levels, 2)`):
                Temporal shapes of each feature map.
            level_start_index (`LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. 
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        '''
        if output_attentions is None: output_attentions = self.config.output_attentions
        if output_hidden_states is None: output_hidden_states = self.config.output_hidden_states
        if return_dict is None: return_dict = self.config.use_return_dict

        # temporal_shapes = tuple(temporal_shapes)
        reference_points = self.get_reference_points(temporal_shapes, valid_ratios, device=inputs_embeds.device)
        hidden_states = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states: encoder_states += (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                temporal_shapes=temporal_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions: all_attentions += (layer_outputs[1],)

        if output_hidden_states: encoder_states += (hidden_states,)
        if not return_dict: return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)