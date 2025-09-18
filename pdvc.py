import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor, FloatTensor, LongTensor

from copy import deepcopy
from typing import Union, Dict, List, Optional
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead,
    DeformableDetrObjectDetectionOutput,
    DeformableDetrPreTrainedModel,
    inverse_sigmoid
)
from deformable_detr import DeformableDetrModel
from captioner import DeformableCaptioner
from loss import DeformableDetrForObjectDetectionLoss
from utils import ensure_cw_format


class PDVCOutput(DeformableDetrObjectDetectionOutput):
    pred_counts: Optional[FloatTensor] = None
    pred_cap_logits: Optional[FloatTensor] = None
    pred_tokens: Optional[LongTensor] = None
    

class DeformableDetrForObjectDetection(DeformableDetrPreTrainedModel):
    ''' Temporal Deformable DETR for 1D event localization.
    
    Temporal detection head on top of DeformableDetrModel.
    - class head: (d_model -> num_classes) with last class 'no-object'
    - bbox head: (d_model -> 2) predicts (center, length) normalized to [0,1]
    We do NOT use iterative box refinement (with_box_refine=False) to keep length independent of reference.
    
    Inputs:
    - pixel_values: (B, C, T)
    - pixel_mask:   (B, T) where 1=valid, 0=pad
    - labels: list of dicts per batch item with {
        'class_labels': LongTensor (N_i, )  foreground classes (no 'no-object' here)
        'boxes': FloatTensor (N_i, 2)       (center, width) normalized to [0,1]
    }
    '''
    _tied_weights_keys = [r"bbox_head\.[1-9]\d*", r"class_head\.[1-9]\d*"] # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _no_split_modules = None # We can't initialize the model on meta device as some weights are modified during the initialization
    
    def __init__(self, config: DeformableDetrConfig, temporal_channels: Optional[List[int]] = None):
        super().__init__(config)
        self.transformer = DeformableDetrModel(config, temporal_channels=temporal_channels) # Deformable DETR encoder-decoder model
        
        # Detection heads on top: class + 2D temporal box (center, width)
        self.count_head = nn.Linear(config.d_model, config.num_queries + 1)  # Predict count of events in [0, num_queries]; last index for 0 events
        self.class_head = nn.Linear(config.d_model, config.num_labels)  # num of foreground classes, no 'no-object' here
        self.bbox_head = DeformableDetrMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=2, num_layers=3)
        self.caption_head = DeformableCaptioner(config, vocab_size=config.vocab_size, rnn_num_layers=1, dropout_rate=0.1, max_caption_len=20)
        
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.class_head.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = config.decoder_layers  # two_stage False in our temporal model
        if config.with_box_refine:
            self.count_head   = nn.ModuleList([deepcopy(self.count_head)   for _ in range(num_pred)])
            self.class_head   = nn.ModuleList([deepcopy(self.class_head)   for _ in range(num_pred)])
            self.bbox_head    = nn.ModuleList([deepcopy(self.bbox_head)    for _ in range(num_pred)])
            self.caption_head = nn.ModuleList([deepcopy(self.caption_head) for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = self.bbox_head # Hack implementation for iterative bounding box refinement
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.count_head   = nn.ModuleList([self.count_head   for _ in range(num_pred)])
            self.class_head   = nn.ModuleList([self.class_head   for _ in range(num_pred)])
            self.bbox_head    = nn.ModuleList([self.bbox_head    for _ in range(num_pred)])
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None
            
        # self.loss_function = DeformableDetrForObjectDetectionLoss
        self.post_init()


    def forward(
        self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[list[dict]] = None, # {'class_labels': LongTensor (N_i, ), 'boxes': FloatTensor (N_i, 2), 'seq_tokens': LongTensor (N_i, L)}
        encoder_outputs: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], PDVCOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Send images through DETR base model to obtain encoder + decoder outputs
        outputs = self.transformer(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Decoder intermediate states: shape (B, L, Q, D) -> we need per-layer lists (D is the d_model)
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]        # (B, L, Q, D)
        init_reference = outputs.init_reference_points if return_dict else outputs[0]            # (B, Q, 2)
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]  # (B, L, Q, 2)

        # We only refine the center (dim 0) using decoder reference; length is predicted as-is and sigmoided
        outputs_counts, outputs_classes, outputs_coords = [], [], []
        outputs_cap_probs, outputs_cap_tokens = [], []
        
        for layer in range(hidden_states.shape[1]):
            layer_hidden_states = hidden_states[:, layer]  # (B, Q, D)
            
            # Count head: (B, num_queries, num_queries+1) logits for 0 to num_queries events
            # Max is to get the most confident prediction across queries
            outputs_count = self.count_head[layer](torch.max(layer_hidden_states, dim=1, keepdim=False).values)
            outputs_counts.append(outputs_count)
            
            # Box head: (B, num_queries, 2) (center, width) in [0,1]
            reference = init_reference if layer == 0 else inter_references[:, layer - 1] # (B, Q, 2) if layer != 0, we use previous layer
            reference = inverse_sigmoid(reference)                                       # (B, Q, 2)
            delta_bbox = self.bbox_head[layer](layer_hidden_states)                      # (B, Q, 2)
            if reference.shape[-1] == 2: delta_bbox += reference
            elif reference.shape[-1] == 1: delta_bbox[..., :1] += reference
            # elif reference.shape[-1] == 1: delta_bbox[..., :2] += reference
            outputs_coords.append(delta_bbox.sigmoid())                  # (B, Q, 2) in (center,width) normalized [0,1]
            
            # Class head: (B, num_queries, num_classes) logits
            outputs_class = self.class_head[layer](layer_hidden_states)  # (B, Q, C)
            outputs_classes.append(outputs_class)
            
            # Caption head: (B, Q, max_caption_len) logits over vocab
            if labels is not None: # Teacher forcing during training
                cap_probs = self.caption_head[layer](labels['seq_tokens'], layer_hidden_states, reference, transformer_outputs)
                cap_tokens = labels['seq_tokens'] # In training, we just return the input tokens
            
            else: # Greedy or multinomial sampling during inference
                if layer == hidden_states.shape[1] - 1: # Only predict captions for last layer
                    cap_probs, cap_tokens = self.caption_head[layer].sample(layer_hidden_states, reference, transformer_outputs)
                else: # For other layers, just append empty tensors
                    cap_probs = torch.zeros(
                        layer_hidden_states.shape[0], layer_hidden_states.shape[1],
                        self.caption_head[layer].max_caption_len, self.caption_head[layer].vocab_size + 1,
                        device=layer_hidden_states.device
                    )
                    cap_tokens = torch.zeros(
                        layer_hidden_states.shape[0], layer_hidden_states.shape[1],
                        self.caption_head[layer].max_caption_len,
                        device=layer_hidden_states.device, dtype=torch.long
                    )
            outputs_cap_probs.append(cap_probs)                 # (B, Q, length, vocab_size + 1)
            outputs_cap_tokens.append(cap_tokens)               # (B, Q, length)

        outputs_classes    = torch.stack(outputs_classes)       # (L, B, Q, C)
        outputs_coords     = torch.stack(outputs_coords)        # (L, B, Q, 2)
        outputs_counts     = torch.stack(outputs_counts)        # (L, B, Q, num_queries+1)
        outputs_cap_probs  = torch.stack(outputs_cap_probs)     # (L, B, Q, length, vocab_size + 1)
        outputs_cap_tokens = torch.stack(outputs_cap_tokens)    # (L, B, Q, length)
        
        logits          = outputs_classes[-1]                   # (B, Q, C)
        pred_boxes      = outputs_coords[-1]                    # (B, Q, 2) (center,width) normalized
        pred_counts     = outputs_counts[-1]                    # (B, Q, num_queries + 1)
        pred_cap_logits = outputs_cap_probs[-1]                 # (B, Q, length, vocab_size)
        pred_cap_tokens = outputs_cap_tokens[-1]                # (B, Q, length)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None: # Training mode
            # Ensure targets provide (center,width). If start/end (s<e & min>=0) provided, convert here
            labels: List[Dict[str, Tensor]] = [{
                'class_labels': t['class_labels'], 
                'boxes': ensure_cw_format(t['boxes']),
                'seq_tokens': t['seq_tokens']
            } for t in labels]
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                labels, logits, pred_boxes, pred_counts, pred_cap_logits,
                outputs_counts, outputs_classes, outputs_coords, outputs_cap_probs
            )

        if not return_dict:
            out = (logits, pred_boxes) + outputs
            if auxiliary_outputs is not None: out += auxiliary_outputs
            return ((loss, loss_dict) + out) if loss is not None else out
        
        return PDVCOutput(
            loss=loss, loss_dict=loss_dict, logits=logits, 
            pred_boxes=pred_boxes, pred_counts=pred_counts,
            pred_cap_logits=pred_cap_logits, pred_tokens=pred_cap_tokens,
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