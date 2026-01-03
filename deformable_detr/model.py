import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor, FloatTensor, LongTensor

from dataclasses import dataclass
from typing import Union, List, Optional, Tuple
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    BaseModelOutput, ModelOutput, 
    DeformableDetrPreTrainedModel,
    inverse_sigmoid
)
from backbones import CoSign1s
from .position_encoding import PositionEmbeddingSine
from .encoder import DeformableDetrEncoder
from .decoder import DeformableDetrDecoder


@dataclass
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
    mask_flatten: Optional[FloatTensor] = None
    temporal_shapes: Optional[FloatTensor] = None
    level_start_index: Optional[FloatTensor] = None
    valid_ratios: Optional[FloatTensor] = None
    

class DeformableDetrModel(DeformableDetrPreTrainedModel): # Re-wired for 1D features
    '''
    A temporal variant of HF transformers.DeformableDetrModel:
    - Expects pixel_values_1d: (B, C, T), pixel_mask_1d: (B, T)
    - Builds multi-scale features as (B, C, 1, T_l)
    - Keeps the same encoder/decoder, temporal_shapes, valid_ratios logic (with height=1)
    '''
    def __init__(self, config: DeformableDetrConfig, temporal_kernel=5, contrastive_mode=False):
        super().__init__(config)
        self.contrastive_mode = contrastive_mode
        # Positional encoding: d_model//2 for temporal pos, d_model//2 for duration pos
        self.position_embeddings = PositionEmbeddingSine(config.d_model // 2, normalize=True)
        self.backbone = CoSign1s(temporal_kernel=temporal_kernel, hidden_size=config.d_model, contrastive_mode=contrastive_mode)

        # Input projection: map each backbone level to d_model with 1x1 Conv2d
        if config.num_feature_levels > 1:
            input_proj_list = []
            in_channels = config.d_model
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, config.d_model, kernel_size=1), 
                nn.GroupNorm(32, config.d_model)
            ))
            for _ in range(config.num_feature_levels - 1): # Extra levels from last backbone level
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, config.d_model),
                ))
                in_channels = config.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv1d(config.d_model, config.d_model, kernel_size=1),
                nn.GroupNorm(32, config.d_model),
            )])

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DeformableDetrDecoder(config)
        self.level_embed = nn.Parameter(Tensor(config.num_feature_levels, config.d_model)) # For multi-scale features
        
        if config.with_box_refine:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)
            self.reference_points = nn.Linear(config.d_model, 2) # Predict (center, width) in [0,1] for each query via sigmoid
        else:
            self.pos_trans = nn.Linear(config.d_model, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
        self._reset_parameters()
        self.post_init()
        
        
    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
            
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)


    def get_proposal_pos_embed(self, proposals):  # Get the position embedding of the proposals
        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=proposals.dtype, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        proposals = proposals.sigmoid() * scale   # (B, num_queries, 2)
        pos = proposals[:, :, :, None] / dim_t    # (B, num_queries, 2, num_pos_feats)
        
        # (B, num_queries, 2, num_pos_feats // 2, 2) -> (B, num_queries, num_pos_feats * 2)
        return torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
    

    def forward(
        self, pixel_values: FloatTensor,          # [B(N), T, 77(K), 3(C)] Channel-last for CoSign backbone
        pixel_mask: Optional[LongTensor] = None,  # (B, T) 1=valid, 0=pad
        labels: Optional[list[dict]] = None,      # {'class_labels': LongTensor (N_i, ), 'boxes': FloatTensor (N_i, 2), 'seq_tokens': LongTensor (N_i, L)}
        encoder_outputs: Optional[FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[FloatTensor], DeformableDetrModelOutput]:
        assert pixel_values.dim() == 4 and pixel_values.shape[-1] == 3, 'Expected (B, T, K, 3)'
        if output_attentions is None: output_attentions = self.config.output_attentions
        if output_hidden_states is None: output_hidden_states = self.config.output_hidden_states
        if return_dict is None: return_dict = self.config.use_return_dict
        
        # Handle contrastive mode where backbone returns two views
        backbone_out = self.backbone(pixel_values)
        if self.training and self.contrastive_mode:
            view1, view2 = backbone_out  # Both [B, T, hidden_size]
            pixel_values = view1.permute(0, 2, 1)  # [B, d_model, T] - use view1 for main forward
            self._contrastive_views = (view1, view2)  # Store for loss computation
        else:
            pixel_values = backbone_out.permute(0, 2, 1)  # [B, d_model, T]
            self._contrastive_views = None
            
        B, C, T = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None: pixel_mask = torch.ones((B, T), dtype=torch.long, device=device)
        
        pos_level0 = self.position_embeddings(pixel_values, pixel_mask, durations=torch.sum(pixel_mask, 1)) # (B, d_model, T)
        source_level0 = self.input_proj[0](pixel_values) # (B, d_model, T)
        mask_level0 = pixel_mask.to(torch.bool)          # (B, T) 1=valid, 0=pad
        sources, masks, position_embeddings_list = [source_level0], [mask_level0], [pos_level0]
        if pixel_mask is None: raise ValueError('No attention mask was provided')

        # Add extra pyramid levels if the current number of levels in the backbone is less than num_feature_levels
        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        for level in range(1, self.config.num_feature_levels):
            if level == 1: # First extra level from last backbone feature map
                source = self.input_proj[level](pixel_values) # (B, d_model, T/2)
                base_mask = pixel_mask
            else: # Further levels from previous extra level
                source = self.input_proj[level](sources[-1])  # (B, d_model, T/4), (B, d_model, T/8), ...
                base_mask = masks[-1]

            # Resize mask to new temporal size (1, B, T) -> (1, B, T_l) -> (B, T_l)
            mask = F.interpolate(base_mask[None].float(), size=source.shape[-1:], mode='nearest').to(torch.bool)[0]
            pos_l = self.position_embeddings(source, mask, durations=torch.sum(mask, 1)).to(source.dtype)
            sources.append(source)
            masks.append(mask) # (B, T/2), (B, T/4), ...
            position_embeddings_list.append(pos_l)

        # Prepare encoder inputs (by flattening)
        source_flatten, mask_flatten, lvl_pos_embed_flatten = [], [], []
        temporal_shapes: List[int] = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, _, width = source.shape
            temporal_shapes.append(width)
            
            source = source.transpose(1, 2)                          # (B, T, C) 
            pos_embed = pos_embed.transpose(1, 2)                    # (B, T, C)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            
            source_flatten.append(source)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        source_flatten = torch.cat(source_flatten, 1)                # (B, \sum{T_l}, C)
        mask_flatten = torch.cat(mask_flatten, 1)                    # (B, \sum{T_l})
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (B, \sum{T_l}, C)

        # 1D temporal shapes and start indices
        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=source_flatten.device)     # (L,)
        level_start_index = torch.cat([temporal_shapes.new_zeros((1,)), temporal_shapes.cumsum(0)[:-1]])       # (L,)
        valid_ratios = torch.stack([torch.sum(m, 1).to(source_flatten.dtype) / m.shape[1] for m in masks], 1)  # (B, L) 

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
        if self.config.with_box_refine:
            query_pos_embed = self.query_position_embeddings.weight                    # (Q, 2*C)
            query_pos_embed, target = torch.split(query_pos_embed, num_channels, dim=1)
            query_pos_embed = query_pos_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Q, C)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)                    # (B, Q, C)
            reference_points = self.reference_points(query_pos_embed).sigmoid()        # (B, Q, 2)
            init_reference_points = reference_points
        else:
            max_caption_num = max([len(l['boxes']) for l in labels]) if labels is not None else self.config.num_queries
            gt_reference_points = torch.zeros(batch_size, max_caption_num, 2, device=device) # (B, max_N, 2)
            for i in range(batch_size): # Iterate over each window in the batch
                gt_reference_points[i, :len(labels[i]['boxes'])] = labels[i]['boxes']  # (center, width) in [0,1]
                
            topk_coords_logits = inverse_sigmoid(gt_reference_points)                  # (B, max_N, 2)
            reference_points = gt_reference_points                                     # (B, max_N, 2)
            init_reference_points = reference_points                                   # (B, max_N, 2)
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_logits)))
            query_pos_embed, target = torch.split(pos_trans_out, num_channels, dim=2)  # (B, max_N, C), (B, max_N, C)
            
        self_attn_mask = torch.ones(batch_size, query_pos_embed.shape[1], device=query_pos_embed.device).bool()
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
            return (init_reference_points,) + decoder_outputs + encoder_outputs + \
                   (mask_flatten, temporal_shapes, level_start_index, valid_ratios)

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
            mask_flatten=mask_flatten,
            temporal_shapes=temporal_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )