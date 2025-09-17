import torch
import torch.nn.functional as F
from torch import nn, Tensor, FloatTensor, LongTensor
from typing import Optional
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    ACT2FN, GradientCheckpointingLayer, 
    DeformableDetrPreTrainedModel, DeformableDetrDecoderOutput,
    BaseModelOutput, inverse_sigmoid
)
from attention import TemporalMSDA


class DeformableDetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeformableDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, batch_first=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Cross-Attention
        self.encoder_attn = TemporalMSDA(config, num_heads=config.decoder_attention_heads, n_points=config.decoder_n_points)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        

    def forward(
        self, hidden_states: Tensor, 
        position_embeddings: Optional[Tensor] = None, self_attn_mask: Optional[Tensor] = None, 
        reference_points=None, temporal_shapes=None, level_start_index=None,
        encoder_hidden_states: Optional[Tensor] = None, encoder_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        '''
        Args:
            hidden_states (`FloatTensor`): Input to the layer of shape `(batch_size, num_queries, hidden_size)`.
            position_embeddings (`FloatTensor`, *optional*): Position embeddings that are added to queries and keys in self-attention layer.
            reference_points (`FloatTensor`, *optional*): Reference points.
            temporal_shapes (`LongTensor`, *optional*): Temporal shapes.
            level_start_index (`LongTensor`, *optional*): Level start index.
            encoder_hidden_states (`FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding 
                elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. 
                See `attentions` under returned tensors for more detail.
        '''
        # Self Attention
        residual = hidden_states
        q = k = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        hidden_states, self_attn_weights  = self.self_attn(q, k, hidden_states, key_padding_mask=~self_attn_mask)
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention
        residual = hidden_states
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            temporal_shapes=temporal_shapes,
            level_start_index=level_start_index,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights, cross_attn_weights)
        return outputs
    
    
class DeformableDetrDecoder(DeformableDetrPreTrainedModel):
    '''
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DeformableDetrDecoderLayer`].
    The decoder updates the query embeddings through multiple self-attention and cross-attention layers. Some tweaks for Deformable DETR:
    - `position_embeddings`, `reference_points`, `temporal_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.
    '''
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList([DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.post_init() # Initialize weights and apply final processing


    def forward(
        self, inputs_embeds=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        position_embeddings=None, self_attn_mask=None,
        reference_points=None, temporal_shapes=None,
        level_start_index=None, valid_ratios=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        '''
        Args:
            inputs_embeds (`FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            encoder_attention_mask (`LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            self_attn_mask (`FloatTensor` of shape `(num_queries, num_queries)`, *optional*):
                Mask to avoid performing attention on subsequent positions. Mask values selected in `[0, 1]`
            reference_points (`FloatTensor` of shape `(batch_size, num_queries, 1 or 2)`, *optional*):
                The normalized reference points used in the cross-attention, used to compute the sampling offsets.
            temporal_shapes (`FloatTensor` of shape `(num_feature_levels, 2)`):
                Temporal shapes of the feature maps.
            level_start_index (`LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        '''
        if output_attentions is None: output_attentions = self.config.output_attentions
        if output_hidden_states is None: output_hidden_states = self.config.output_hidden_states
        if return_dict is None: return_dict = self.config.use_return_dict
        if inputs_embeds is not None: hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate, intermediate_reference_points = (), ()

        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 2:
                layer_reference_points = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            elif reference_points.shape[-1] == 1:
                layer_reference_points = reference_points[:, :, None] * valid_ratios[:, None, :, None]
            else: raise ValueError("Reference points' last dimension must be of size 2")

            if output_hidden_states: all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings,
                self_attn_mask,
                layer_reference_points,
                temporal_shapes,
                level_start_index,
                encoder_hidden_states,  # As a positional argument for gradient checkpointing
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if self.bbox_embed is not None: # Hack implementation for iterative bounding box refinement
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 2:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                elif reference_points.shape[-1] == 1:
                    new_reference_points = tmp
                    new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else: raise ValueError(f'Last dim of reference_points must be 1 or 2, but got {reference_points.shape[-1]}')
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # Add hidden states from the last decoder layer
        if output_hidden_states: all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                intermediate,
                intermediate_reference_points,
                all_hidden_states,
                all_self_attns,
                all_cross_attentions,
            ] if v is not None)
            
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )