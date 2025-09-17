import torch
import torch.nn.functional as F
from torch import nn, Tensor, FloatTensor
from typing import Union, List, Optional, Tuple
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConvModel, DeformableDetrPreTrainedModel,
    ModelOutput, BaseModelOutput, build_position_encoding
)
from encoder import DeformableDetrEncoder
from decoder import DeformableDetrDecoder


class DeformableDetrModelOutput(ModelOutput):
    init_reference_points: Optional[FloatTensor] = None
    last_hidden_state: Optional[FloatTensor] = None
    intermediate_hidden_states: Optional[FloatTensor] = None
    intermediate_reference_points: Optional[FloatTensor] = None
    decoder_hidden_states: Optional[tuple[FloatTensor]] = None
    decoder_attentions: Optional[tuple[FloatTensor]] = None
    cross_attentions: Optional[tuple[FloatTensor]] = None
    encoder_last_hidden_state: Optional[FloatTensor] = None
    encoder_hidden_states: Optional[tuple[FloatTensor]] = None
    encoder_attentions: Optional[tuple[FloatTensor]] = None
    enc_outputs_class: Optional[FloatTensor] = None
    enc_outputs_coord_logits: Optional[FloatTensor] = None
    mask_flatten: Optional[FloatTensor] = None
    temporal_shapes: Optional[FloatTensor] = None
    level_start_index: Optional[FloatTensor] = None
    valid_ratios: Optional[FloatTensor] = None
    self_attn_mask: Optional[FloatTensor] = None
    

class DeformableDetrModel(DeformableDetrPreTrainedModel): # Re-wired for 1D features
    '''
    A temporal variant of HF transformers.DeformableDetrModel:
    - Expects pixel_values_1d: (B, C, T), pixel_mask_1d: (B, T)
    - Builds multi-scale features as (B, C, 1, T_l)
    - Keeps the same encoder/decoder, temporal_shapes, valid_ratios logic (with height=1)
    '''
    def __init__(self, config: DeformableDetrConfig, temporal_channels: Optional[List[int]] = None):
        super().__init__(config)
        if self.config.two_stage:
            raise NotImplementedError('two_stage=True is not supported in the temporal variant')
        elif temporal_channels is None:
            temporal_channels = [config.d_model * 2 ** l for l in range(config.num_feature_levels)]
            
        # Temporal backbone + positional encoding
        backbone = TemporalConvEncoder(
            in_channels=config.num_channels,
            channels=temporal_channels,
            strides=[2] * len(temporal_channels),
            kernel_sizes=[5] + [3] * (len(temporal_channels) - 1),
            norm=True,
        )
        position_embeddings = build_position_encoding(config)
        self.backbone = DeformableDetrConvModel(backbone, position_embeddings)

        # Input projection: map each backbone level to d_model with 1x1 Conv2d
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for level in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[level]
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, config.d_model, kernel_size=1), 
                    nn.GroupNorm(32, config.d_model)
                ))
            for _ in range(config.num_feature_levels - num_backbone_outs): # Extra levels from last backbone level
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, config.d_model),
                ))
                in_channels = config.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv1d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                nn.GroupNorm(32, config.d_model),
            )])

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoder(config)
        self.level_embed = nn.Parameter(Tensor(config.num_feature_levels, config.d_model)) # For multi-scale features
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)
        self.reference_points = nn.Linear(config.d_model, 1) # Predict (center) in [0,1] for each query via sigmoid
        self._reset_parameters()
        self.post_init()
        
        
    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
            
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)


    def forward(
        self,
        pixel_values: Tensor,  # (B, C, T)
        pixel_mask: Optional[Tensor] = None,  # (B, T) 1=valid, 0=pad
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[FloatTensor], DeformableDetrModelOutput]:
        assert pixel_values.dim() == 3, 'Temporal model expects pixel_values of shape (B, C, T)'
        if output_attentions is None: output_attentions = self.config.output_attentions
        if output_hidden_states is None: output_hidden_states = self.config.output_hidden_states
        if return_dict is None: return_dict = self.config.use_return_dict
        
        B, C, T = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None: pixel_mask = torch.ones((B, T), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples (feature_map, mask) for each level
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)  # list of levels
        
        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources, masks = [], [] # Shape: (L, B, C_l, T_l), (L, B, T_l)
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None: raise ValueError('No attention mask was provided')

        # Add extra pyramid levels if the current number of levels in the backbone is less than num_feature_levels
        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources: # First extra level from last backbone feature map
                    source = self.input_proj[level](features[-1][0]) 
                    base_mask = features[-1][1]  # (B, T_last)
                else: # Further levels from previous extra level
                    source = self.input_proj[level](sources[-1])
                    base_mask = masks[-1]
                    
                # Resize mask to new temporal size
                mask = F.interpolate(base_mask.float(), size=source.shape[-2:]).to(torch.bool)
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Prepare encoder inputs (by flattening)
        source_flatten, mask_flatten, lvl_pos_embed_flatten = [], [], []
        temporal_shapes: List[Tuple[int, int]] = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, _, width = source.shape
            temporal_shapes.append(width)
            
            source = source.flatten(2).transpose(1, 2)        # (B, H*W, C) = (B, W, C)
            mask = mask.flatten(1)                            # (B, H*W)    = (B, W)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (B, H*W, C) = (B, W, C)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            
            source_flatten.append(source)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
        source_flatten = torch.cat(source_flatten, 1)               # (B, sum(W_l), C)
        mask_flatten = torch.cat(mask_flatten, 1)                   # (B, sum(W_l))
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # (B, sum(W_l), C)

        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=source_flatten.device)       # (L, 2)
        level_start_index = torch.cat((temporal_shapes.new_zeros((1,)), temporal_shapes.prod(1).cumsum(0)[:-1])) # (L,)
        valid_ratios = torch.stack([torch.sum(~m, 1).to(source_flatten.dtype) / m.shape[1] for m in masks], 1) # (B, L, 1)

        # Send source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide temporal_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                position_embeddings=lvl_pos_embed_flatten,
                temporal_shapes=temporal_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Prepare decoder inputs (query embeddings and reference points)
        batch_size, _, num_channels = encoder_outputs[0].shape
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        query_pos_embed = self.query_position_embeddings.weight
        self_attn_mask = torch.ones(batch_size, query_pos_embed.shape[0], device=query_pos_embed.device).bool()
        
        # query_pos_embed, target = torch.chunk(query_pos_embed, 2, dim=1)
        query_pos_embed, target = torch.split(query_pos_embed, num_channels, dim=1)
        query_pos_embed = query_pos_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Q, C)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)            # (B, Q, C)
        reference_points = self.reference_points(query_pos_embed).sigmoid()    # (B, Q, 1)
        init_reference_points = reference_points
        
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            position_embeddings=query_pos_embed,
            self_attn_mask=self_attn_mask, # For nn.MultiHeadAttention
            reference_points=reference_points,
            temporal_shapes=temporal_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
            return (init_reference_points,) + decoder_outputs + encoder_outputs + enc_outputs

        return DeformableDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points, # (num_layers, B, num_queries, 2)
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            mask_flatten=mask_flatten,
            temporal_shapes=temporal_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            self_attn_mask=self_attn_mask,
        )