import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from copy import deepcopy
from typing import Union, Dict, List, Optional
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead,
    DeformableDetrObjectDetectionOutput,
    inverse_sigmoid
)
from deformable_detr import TemporalDeformableDetrModel
from loss import TemporalDeformableDetrForObjectDetectionLoss
from utils import ensure_cw_format


class TemporalDeformableDetrForObjectDetection(DeformableDetrPreTrainedModel):
    ''' Temporal Deformable DETR for 1D event localization.
    
    Temporal detection head on top of TemporalDeformableDetrModel.
    - class head: (d_model -> num_classes) with last class 'no-object'
    - bbox head: (d_model -> 2) predicts (center, length) normalized to [0,1]
    We do NOT use iterative box refinement (with_box_refine=False) to keep length independent of reference.
    
    Inputs:
    - pixel_values: (B, C, T)
    - pixel_mask:   (B, T) where 1=valid, 0=pad
    - labels: list of dicts per batch item with {
        'class_labels': LongTensor (N_i, )  foreground classes (no "no-object" here)
        'boxes': FloatTensor (N_i, 2)       (center, width) normalized to [0,1]
    }
    '''
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"class_embed\.[1-9]\d*"] # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _no_split_modules = None # We can't initialize the model on meta device as some weights are modified during the initialization
    
    def __init__(self, config: DeformableDetrConfig, captioner, temporal_channels: Optional[List[int]] = None):
        super().__init__(config)
        self.model = TemporalDeformableDetrModel(config, temporal_channels=temporal_channels) # Deformable DETR encoder-decoder model
        
        # Detection heads on top: class + 2D temporal box (center, width)
        self.count_embed = nn.Linear(config.d_model, self.config.num_queries + 1)  # Predict count of events in [0, num_queries]; last index for 0 events
        self.class_embed = nn.Linear(config.d_model, config.num_labels)  # num_labels should include background
        self.bbox_embed = DeformableDetrMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=2, num_layers=3)
        self.captioner = captioner
        
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        num_pred = config.decoder_layers  # two_stage False in our temporal model
        if config.with_box_refine:
            self.count_embed = nn.ModuleList([deepcopy(self.count_embed) for _ in range(num_pred)])
            self.class_embed = nn.ModuleList([deepcopy(self.class_embed) for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([deepcopy(self.bbox_embed) for _ in range(num_pred)])
            self.captioner = nn.ModuleList([deepcopy(self.captioner) for _ in range(num_pred)])
            self.model.decoder.bbox_embed = self.bbox_embed # Hack implementation for iterative bounding box refinement
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[1:], -2)
            self.count_embed = nn.ModuleList([self.count_embed for _ in range(num_pred)])
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.captioner = nn.ModuleList([self.captioner for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None
            
        self.loss_function = TemporalDeformableDetrForObjectDetectionLoss
        self.post_init()


    def forward(
        self,
        pixel_values: torch.FloatTensor, # (B,C,T)
        pixel_mask: Optional[torch.LongTensor] = None, # (B,T)
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], DeformableDetrObjectDetectionOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dic

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Decoder intermediate states: shape (B, L, Q, D) -> we need per-layer lists
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]        # (B, L, Q, D)
        init_reference = outputs.init_reference_points if return_dict else outputs[0]            # (B, Q, 2)
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]  # (B, L, Q, 2)

        # We only refine the center (dim 0) using decoder reference; length is predicted as-is and sigmoided
        outputs_counts, outputs_classes, outputs_coords = [], [], []
        outputs_captions_probs, outputs_captions_seq = [], []
        
        for level in range(hidden_states.shape[1]):
            # Count head: (B, num_queries, num_queries+1) logits for 0 to num_queries events
            outputs_count = self.count_embed[level](
                # Max is to get the most confident prediction across queries
                torch.max(hidden_states[:, level], dim=1, keepdim=False).values
            )
            outputs_counts.append(outputs_count)
            
            # Class head: (B, num_queries, num_classes) logits
            outputs_class = self.class_embed[level](hidden_states[:, level])  # (B, Q, C)
            outputs_classes.append(outputs_class)
            
            # Box head: (B, num_queries, 2) (center, width) in [0,1]
            reference = init_reference if level == 0 else inter_references[:, level - 1] # (B, Q, 2) if level != 0, we 
            reference = inverse_sigmoid(reference)                                       # (B, Q, 2)
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])                 # (B, Q, 2)
            delta_bbox[..., :2] += reference  # Consistent with HF logic for 2D references
            # delta_bbox[..., 0] += reference[..., 0] # Refine only the center. Leave length independent of reference
            outputs_coords.append(delta_bbox.sigmoid()) # (B, Q, 2) in (center,width) normalized [0,1]

            # Caption head: (B, num_queries, max_len, vocab_size) logits
            if level != hidden_states.shape[1] - 1: # No captioning at the last layer
                cap_probs, seq = self.captioner[level](
                    hidden_states[:, level], labels, reference, others, 
                    'teacher_forcing' if self.training else 'greedy'
                ) # (B, Q, max_len, vocab_size), (B, Q, max_len)
                outputs_captions_probs.append(cap_probs)
                outputs_captions_seq.append(seq)

        outputs_counts = torch.stack(outputs_counts)  # (L, B, Q, num_queries+1)
        outputs_class = torch.stack(outputs_classes)  # (L, B, Q, C)
        outputs_coord = torch.stack(outputs_coords)   # (L, B, Q, 2)
        logits = outputs_class[-1]                    # (B, Q, C)
        pred_boxes = outputs_coord[-1]                # (B, Q, 2) (center,width) normalized
        pred_counts = outputs_counts[-1]              # (B, Q, num_queries+1)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None: # Training mode
            fixed_targets: List[Dict[str, Tensor]] = [ 
                # Ensure targets provide (center,width). If start/end (s<e & min>=0) provided, convert here
                {'class_labels': t['class_labels'], 'boxes': ensure_cw_format(t['boxes'])}
                for t in labels
            ]
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, pred_counts,
                self.config, outputs_class, outputs_coord,
            )

        if not return_dict:
            output = (logits, pred_boxes) + outputs
            if auxiliary_outputs is not None: output += auxiliary_outputs
            return ((loss, loss_dict) + output) if loss is not None else output
        
        return DeformableDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )