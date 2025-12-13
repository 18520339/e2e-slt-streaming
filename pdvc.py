import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Optional

import torch
from torch import nn, FloatTensor, LongTensor
from transformers import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead,
    DeformableDetrPreTrainedModel,
    ModelOutput,
    inverse_sigmoid
)
from deformable_detr import DeformableDetrModel
from captioners import LSTMCaptioner, MBartDecoderCaptioner
from loss import DeformableDetrHungarianMatcher, DeformableDetrForObjectDetectionLoss
from utils import ensure_cw_format


@dataclass
class DeformableDetrObjectDetectionOutput(ModelOutput):
    loss: Optional[FloatTensor] = None
    loss_dict: Optional[dict] = None
    logits: Optional[FloatTensor] = None
    pred_boxes: Optional[FloatTensor] = None
    pred_counts: Optional[FloatTensor] = None
    pred_cap_logits: Optional[FloatTensor] = None
    pred_cap_tokens: Optional[LongTensor] = None
    auxiliary_outputs: Optional[list[dict]] = None
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
    

class DeformableDetrForObjectDetection(DeformableDetrPreTrainedModel):
    ''' Temporal Deformable DETR for 1D event localization.
    
    Temporal detection head on top of DeformableDetrModel.
    - class head: (d_model -> num_classes) with last class 'no-object'
    - bbox head: (d_model -> 2) predicts (center, length) normalized to [0,1]
    
    Inputs:
    - pixel_values: (B, C, T)
    - pixel_mask:   (B, T) where 1=valid, 0=pad
    - labels: list of dicts per batch item with {
        'class_labels': LongTensor (N_i, )  foreground classes (no 'no-object' here)
        'boxes': FloatTensor (N_i, 2)       (center, width) normalized to [0,1]
    }
    '''
    _tied_weights_keys = [ # When using clones, all layers > 0 will be clones, but layer 0 *is* required
        r"bbox_head\.[1-9]\d*", r"class_head\.[1-9]\d*",
        r"count_head\.[1-9]\d*", r"caption_head\.[1-9]\d*",
    ]  # Include all cloned heads for proper weight tying with Trainer
    _no_split_modules = None # We can't initialize the model on meta device as some weights are modified during the initialization
    
    def __init__(
        self, config: DeformableDetrConfig, captioner_class, vocab_size: int, 
        bos_token_id: int, eos_token_id: int, pad_token_id: int, decoder_start_token_id: int = None,
        temporal_kernel=5, num_cap_layers=1, cap_dropout_rate=0.1, max_event_tokens=20,
        weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_counter': 0.5, 'loss_caption': 2}
    ):
        super().__init__(config)
        self.transformer = DeformableDetrModel(config, temporal_kernel=temporal_kernel) # Deformable DETR encoder-decoder model
        self.matcher = DeformableDetrHungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
        
        # Detection heads on top: class + 2D temporal box (center, width)
        self.count_head = nn.Linear(config.d_model, config.num_queries + 1)  # Predict count of events in [0, num_queries]
        self.class_head = nn.Linear(config.d_model, config.num_labels)       # Num of foreground classes, no 'no-object' here
        self.bbox_head = DeformableDetrMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=2, num_layers=3)
        self.caption_head = captioner_class(
            config, vocab_size=vocab_size, 
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id, max_event_tokens=max_event_tokens,
            dropout_rate=cap_dropout_rate, num_layers=num_cap_layers, 
        )
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.class_head.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        # num_pred = (config.decoder_layers + 1) if config.two_stage else config.decoder_layers
        if config.with_box_refine:
            num_pred = config.decoder_layers
            self.count_head   = nn.ModuleList([deepcopy(self.count_head)   for _ in range(num_pred)])
            self.class_head   = nn.ModuleList([deepcopy(self.class_head)   for _ in range(num_pred)])
            self.bbox_head    = nn.ModuleList([deepcopy(self.bbox_head)    for _ in range(num_pred)])
            self.caption_head = nn.ModuleList([deepcopy(self.caption_head) for _ in range(num_pred)])
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            self.transformer.decoder.bbox_embed = self.bbox_head # Hack implementation for iterative bounding box refinement
        else:
            num_pred = config.decoder_layers + 1
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.count_head   = nn.ModuleList([self.count_head   for _ in range(num_pred)])
            self.class_head   = nn.ModuleList([self.class_head   for _ in range(num_pred)])
            self.bbox_head    = nn.ModuleList([self.bbox_head    for _ in range(num_pred)])
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
            
        self.loss_function = DeformableDetrForObjectDetectionLoss(config, pad_token_id=pad_token_id, weight_dict=weight_dict)
        self.post_init()


    def forward(
        self, pixel_values: FloatTensor,         # [B(N), T, 77(K), 3(C)] Channel-last for CoSign backbone
        pixel_mask: Optional[LongTensor] = None, # (B, T) 1=valid, 0=pad
        labels: Optional[list[dict]] = None,     # {'class_labels': LongTensor (N_i, ), 'boxes': FloatTensor (N_i, 2), 'seq_tokens': LongTensor (N_i, L)}
        encoder_outputs: Optional[FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[FloatTensor], DeformableDetrObjectDetectionOutput]:
        # Ensure targets provide (center,width). If start/end (s<e & min>=0) provided, convert here
        # if labels is not None: 
        #     labels: list[dict] = [{
        #         'class_labels': l['class_labels'], 
        #         'boxes': ensure_cw_format(l['boxes']),
        #         'seq_tokens': l['seq_tokens']
        #     } for l in labels]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Send images through DETR base model to obtain encoder + decoder outputs
        transformer_outputs = self.transformer(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            if output_attentions and output_hidden_states:
                encoder_last_hidden_state = transformer_outputs[7]
            elif output_attentions and not output_hidden_states:
                encoder_last_hidden_state = transformer_outputs[6]
            elif not output_attentions and output_hidden_states:
                encoder_last_hidden_state = transformer_outputs[5]
            else:
                encoder_last_hidden_state = transformer_outputs[4]
            
            transformer_outputs_for_captioner = {
                'encoder_last_hidden_state': encoder_last_hidden_state,
                'mask_flatten': transformer_outputs[-4],
                'temporal_shapes': transformer_outputs[-3],
                'level_start_index': transformer_outputs[-2],
                'valid_ratios': transformer_outputs[-1],
            }
        else:
            transformer_outputs_for_captioner = {
                'encoder_last_hidden_state': transformer_outputs.encoder_last_hidden_state,
                'mask_flatten': transformer_outputs.mask_flatten,
                'temporal_shapes': transformer_outputs.temporal_shapes,
                'level_start_index': transformer_outputs.level_start_index,
                'valid_ratios': transformer_outputs.valid_ratios,
            }

        # Decoder intermediate states: shape (B, L, Q, D) -> we need per-layer lists (D is the d_model)
        hidden_states = transformer_outputs.intermediate_hidden_states if return_dict else transformer_outputs[2]        # (B, L, Q, D)
        init_reference = transformer_outputs.init_reference_points if return_dict else transformer_outputs[0]            # (B, Q, 2)
        inter_references = transformer_outputs.intermediate_reference_points if return_dict else transformer_outputs[3]  # (B, L, Q, 2)

        # We only refine the center (dim 0) using decoder reference; length is predicted as-is and sigmoided
        outputs_counts, outputs_classes, outputs_coords = [], [], []
        outputs_cap_probs, outputs_cap_tokens = [], []
        
        for layer in range(hidden_states.shape[1]):
            layer_hidden_states = hidden_states[:, layer]  # (B, Q, D)
            B, Q, _ = layer_hidden_states.shape
            device = layer_hidden_states.device
            
            # Count head: (B, num_queries, num_queries+1) logits for 0 to num_queries events
            # Max is to get the most confident prediction across queries
            outputs_count = self.count_head[layer](torch.max(layer_hidden_states, dim=1, keepdim=False).values)
            outputs_counts.append(outputs_count)
            
            # Class head: (B, num_queries, num_classes) logits
            outputs_class = self.class_head[layer](layer_hidden_states)  # (B, Q, C)
            outputs_classes.append(outputs_class)
            
            # Box head: (B, num_queries, 2) (center, width) in [0, 1]
            reference = init_reference if layer == 0 else inter_references[:, layer - 1]   # (B, Q, 2) if layer != 0, we use previous layer
            if self.config.with_box_refine:
                delta_bbox = self.bbox_head[layer](layer_hidden_states)                    # (B, Q, 2)
                bbox_reference = inverse_sigmoid(reference)                                # (B, Q, 2) Revert to unnormalized space for box regression
                if bbox_reference.shape[-1] == 2: delta_bbox += bbox_reference
                elif bbox_reference.shape[-1] == 1: delta_bbox[..., :1] += bbox_reference
                outputs_coords.append(delta_bbox.sigmoid())                                # (B, Q, 2) in (center, width) normalized [0,1]
            else: outputs_coords.append(reference) # No box refinement, just output the reference points as boxes
            
            # Caption head: align seq_tokens to queries using Hungarian matching before teacher forcing
            if labels is not None: # Teacher forcing during training
                # Match current layer predictions to targets to align per-query tokens
                match_indices = self.matcher({'logits': outputs_class, 'pred_boxes': outputs_coords[-1]}, labels)
                
                # Align target seq_tokens to query order for teacher forcing (shape: B x Q x L)
                max_len = self.caption_head[layer].max_event_tokens
                aligned_tokens = torch.zeros(B, Q, max_len, dtype=torch.long, device=device)
                
                for b, (src_idx, tgt_idx) in enumerate(match_indices):
                    if src_idx.numel() == 0: continue
                    src_idx, tgt_idx = src_idx.to(device), tgt_idx.to(device)
                    tgt_tokens = labels[b]['seq_tokens'][tgt_idx].to(device)  # (M, L_label)
                    L = min(max_len, tgt_tokens.shape[-1])
                    aligned_tokens[b, src_idx, :L] = tgt_tokens[:, :L]
                
                cap_probs = self.caption_head[layer](aligned_tokens, layer_hidden_states, reference, transformer_outputs_for_captioner)
                outputs_cap_probs.append(cap_probs)               # (B, Q, Length - 1, vocab_size)
                outputs_cap_tokens.append(aligned_tokens)         # (B, Q, Length - 1)
                
        outputs_classes = torch.stack(outputs_classes)            # (L, B, Q, C)
        outputs_coords  = torch.stack(outputs_coords)             # (L, B, Q, 2)
        outputs_counts  = torch.stack(outputs_counts)             # (L, B, Q, num_queries + 1)
        logits          = outputs_classes[-1]                     # (B, Q, C)
        pred_boxes      = outputs_coords[-1]                      # (B, Q, 2) (center, width) normalized
        pred_counts     = outputs_counts[-1]                      # (B, num_queries + 1)
        
        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_cap_probs  = torch.stack(outputs_cap_probs)   # (L, B, Q, Length - 1, vocab_size)
            outputs_cap_tokens = torch.stack(outputs_cap_tokens)  # (L, B, Q, Length - 1)
            pred_cap_logits = outputs_cap_probs[-1]               # (B, Q, Length - 1, vocab_size)
            pred_cap_tokens = outputs_cap_tokens[-1]              # (B, Q, Length - 1)
            
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                labels, logits, pred_boxes, pred_counts, pred_cap_logits,
                outputs_classes, outputs_coords, outputs_counts, outputs_cap_probs
            )
            
        if not self.training: # Greedy or multinomial sampling for last layer during inference (B, Q, Length - 1)
            pred_cap_logits, pred_cap_tokens = self.caption_head[-1].sample(layer_hidden_states, reference, transformer_outputs_for_captioner)

        if not return_dict:
            out = (logits, pred_boxes, pred_counts, pred_cap_logits, pred_cap_tokens)
            if auxiliary_outputs is not None: out += (auxiliary_outputs,) + transformer_outputs[:-4]
            else: out += transformer_outputs[:-4] # Exclude mask_flatten, temporal_shapes, level_start_index, valid_ratios
            return ((loss, loss_dict) + out) if loss is not None else out
        
        return DeformableDetrObjectDetectionOutput(
            loss=loss, loss_dict=loss_dict, 
            logits=logits, pred_boxes=pred_boxes, pred_counts=pred_counts,
            pred_cap_logits=pred_cap_logits, pred_cap_tokens=pred_cap_tokens,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=transformer_outputs.last_hidden_state,
            decoder_hidden_states=transformer_outputs.decoder_hidden_states,
            decoder_attentions=transformer_outputs.decoder_attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            encoder_last_hidden_state=transformer_outputs.encoder_last_hidden_state,
            encoder_hidden_states=transformer_outputs.encoder_hidden_states,
            encoder_attentions=transformer_outputs.encoder_attentions,
            intermediate_hidden_states=transformer_outputs.intermediate_hidden_states,
            intermediate_reference_points=transformer_outputs.intermediate_reference_points,
            init_reference_points=transformer_outputs.init_reference_points,
        )


if __name__ == '__main__':
    from loader import get_streaming_loader
    from transformers import AutoTokenizer
    from postprocess import post_process_object_detection

    # Fetch 1 batch from Data loader
    max_event_tokens = 12
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', src_lang='en_XX', tgt_lang='en_XX', use_fast=True)
    train_loader = get_streaming_loader(split='train', batch_size=4, max_event_tokens=max_event_tokens, tokenizer=tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = next(iter(train_loader))
    video_ids, start_frames, end_frames = batch['video_ids'], batch['window_start_frames'], batch['window_end_frames']
    pixel_values = batch['pixel_values'].to(device)
    pixel_mask = batch['pixel_mask'].to(device)
    labels = [{k: v.to(device) for k, v in label.items()} for label in batch['labels']]
    
    print('Batch poses shape: ', pixel_values.shape)
    for video_id, start_frame, end_frame, events in zip(video_ids, start_frames, end_frames, labels):
        print(f'\nVIDEO ID: {video_id}, Start Frame: {start_frame}, End Frame: {end_frame}')
        for i, (box, event_tokens) in enumerate(zip(events['boxes'], events['seq_tokens'])):
            print(f'[Event {i + 1}] center={box[0]:.3f}, width={box[1]:.3f}, caption length={event_tokens.shape}:\n'
                  f'- Tokens: {event_tokens.tolist()}\n- Text: {tokenizer.decode(event_tokens)}')
    
    # Model config
    config = DeformableDetrConfig(
        d_model=256,
        encoder_layers=3,
        decoder_layers=3,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_n_points=4,
        decoder_n_points=4,
        num_feature_levels=3,
        num_queries=4,
        num_labels=1,  # Single foreground class for caption
        auxiliary_loss=True,
        # Loss hyper-params used by our PDVC loss
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        focal_alpha=0.25,
        with_box_refine=True, # Learnt (True) or Ground truth proposals (False) 
    )
    model = DeformableDetrForObjectDetection(
        config=config,
        captioner_class=MBartDecoderCaptioner,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.lang_code_to_id['en_XX'],
        max_event_tokens=max_event_tokens,
        cap_dropout_rate=0.1,
        num_cap_layers=3,
        weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_counter': 0.5, 'loss_caption': 2}
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized with {total_params / 1e6:.2f}M parameters')

    # Test Training and Inference step
    model.train()
    with torch.enable_grad():
        print('\n--- Training step ---')
        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, return_dict=True)
        print('- loss:', out.loss)
        print('- logits:', out.logits.shape)
        print('- pred_boxes:', out.pred_boxes.shape)
        print('- pred_counts:', out.pred_counts.shape)
        print('- pred_cap_logits:', out.pred_cap_logits.shape)
        print('- pred_cap_tokens:', out.pred_cap_tokens.shape)
        out.loss.backward()
        
    model.eval()
    with torch.no_grad():
        if config.with_box_refine:
            print('\n--- Inference step with learnt proposals (with_box_refine=True) ---')
            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True)
            print('- logits:', out.logits.shape)
            print('- pred_boxes:', out.pred_boxes.shape)
            print('- pred_counts:', out.pred_counts.shape)
            print('- pred_cap_logits:', out.pred_cap_logits.shape)
            print('- pred_cap_tokens:', out.pred_cap_tokens.shape)
            top_k, threshold = 10, 0.5 
        else:
            print('\n--- Inference step with ground truth proposals (with_box_refine=False) ---')
            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, return_dict=True)
            print('- logits:', out.logits.shape)
            print('- pred_cap_logits:', out.pred_cap_logits.shape)
            print('- pred_cap_tokens:', out.pred_cap_tokens.shape)
            top_k, threshold = None, 0.0

        results = post_process_object_detection( # Convert raw outputs to final events and captions
            outputs=out, top_k=top_k, threshold=threshold,
            target_lengths=pixel_mask.sum(dim=1).to(out.pred_boxes.dtype), # (B,)
            tokenizer=tokenizer,
        )
        print('\n--- Post-process results ---')
        for i, r in enumerate(results):
            num_events_kept = r['event_scores'].numel()
            print(f'[Window {i}] {num_events_kept} events kept (threshold={threshold}):')
            if num_events_kept == 0: continue
            for j, (score, label, event, cap_score, caption) in enumerate(zip(
                r['event_scores'], r['event_labels'], r['event_ranges'],
                r['event_caption_scores'], r['event_captions']
            )):
                start, end = event[0], event[1]
                print(f'- Event {j + 1}: score={score:.3f}; label={label}; span=({start:.3f}; {end:.3f}); '
                      f'cap_score={cap_score:.3f}; text="{caption}"')