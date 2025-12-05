import numpy as np
import torch
from torch import Tensor, TensorType
from transformers import AutoTokenizer
from typing import Union
from utils import cw_to_se


@torch.no_grad()
def post_process_object_detection(
    outputs, top_k: int = 10, threshold: float = 0.5, 
    target_lengths: Union[TensorType, list[int]] = None, 
    tokenizer: AutoTokenizer = None,
):
    '''
    Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (start, end) format. 

    Args:
        outputs ([`DetrObjectDetectionOutput`]): Raw outputs of the model.
        top_k (`int`, *optional*, defaults to 10):
            Keep only top k bounding boxes before filtering by thresholding.
        threshold (`float`, *optional*): 
            Score threshold to keep object detection predictions.
        target_lengths (`torch.Tensor` or `list[int]`, *optional*):
            Tensor of shape `(batch_size)` or list of integers containing the target length
            of each clip in the batch. If left to None, predictions will not be resized.
        tokenizer (`AutoTokenizer`, *optional*):
            Tokenizer used to decode caption tokens into strings.

    Returns:
        `list[Dict]`: A list of dictionaries, each dictionary containing the following keys:
        - `event_scores` (`torch.Tensor`): Scores of the kept predictions. 
            The scores are for the foreground class (object) and not for the no-object class.
            Shape `(num_kept_predictions,)`.
        - `event_labels` (`torch.Tensor`): Labels of the kept predictions, 
            always 1 (foreground class) since there is only one class. 
            Shape `(num_kept_predictions,)`.
        - `event_ranges` (`torch.Tensor`): Bounding boxes of the kept predictions in (start, end) format.
            Shape `(num_kept_predictions, 2)`.
        - `event_caption_scores` (`list[float]`): Caption scores of the kept predictions.
            Shape `(num_kept_predictions,)`.
        - `event_captions` (`list[str]`): Decoded caption tokens of the kept predictions.
            Shape `(num_kept_predictions,)`.
    '''
    out_logits, out_bbox = outputs.logits, outputs.pred_boxes
    pred_cap_logits, pred_cap_tokens = outputs.pred_cap_logits, outputs.pred_cap_tokens
    
    if target_lengths is not None:
        if len(out_logits) != len(target_lengths):
            raise ValueError('Make sure that you pass in as many target lengths as the batch dimension of the logits')

    prob = out_logits.sigmoid()                                                      # (batch_size, num_queries, num_classes)
    prob = prob.view(out_logits.shape[0], -1)                                        # (batch_size, num_queries * num_classes)
    k_value = min(top_k, prob.size(1)) if top_k else prob.size(1)                    # Ensure k_value does not exceed total predictions
    scores, topk_indexes = torch.topk(prob, k_value, dim=1)                          # (batch_size, k_value)

    topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='floor') # (batch_size, k_value)
    labels = topk_indexes % out_logits.shape[2]                                      # (batch_size, k_value)
    boxes = cw_to_se(out_bbox) # Convert (center, width) to (start, end) format, shape (batch_size, num_queries, 2)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))         # (batch_size, k_value, 2)

    # And from relative [0, 1] to absolute [0, target_length] coordinates
    if target_lengths is not None:
        scale_fct = torch.stack([target_lengths, target_lengths], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]
        
    if len(pred_cap_tokens):
        mask = (pred_cap_tokens != tokenizer.pad_token_id) & (pred_cap_tokens != tokenizer.eos_token_id) # (batch_size, num_queries, max_tokens_len)
        caption_scores = (pred_cap_logits * mask).sum(dim=-1).cpu().numpy()                              # (batch_size, num_queries)
        caption_scores = [caption_scores[i][topk_boxes[i]] for i in range(caption_scores.shape[0])]      # (batch_size, k_value)
        
        pred_cap_tokens = pred_cap_tokens.detach().cpu().numpy()                                         # (batch_size, num_queries, max_tokens_len)
        pred_cap_tokens = [pred_cap_tokens[i][topk_boxes[i]] for i in range(pred_cap_tokens.shape[0])]   # (batch_size, k_value, max_tokens_len)
        pred_captions = [tokenizer.batch_decode(                                                         # (batch_size, k_value) 
            np.where(topk_cap_tokens == -100, tokenizer.pad_token_id, topk_cap_tokens),                  # Replace -100 (used by HF) with pad token id
            skip_special_tokens=True, clean_up_tokenization_spaces=True
        ) for topk_cap_tokens in pred_cap_tokens]                                                     
    else: # No caption tokens predicted, so fill with empty strings and very low scores
        caption_scores = [[-1e5] * k_value] * out_logits.shape[0]                                        # (batch_size, k_value)
        pred_captions = [[''] * k_value] * out_logits.shape[0]                                           # (batch_size, k_value)
        
    return [{
        'event_scores': s[s > threshold], 
        'event_labels': l[s > threshold], 
        'event_ranges': b[s > threshold], 
        'event_caption_scores': [c[i] for i in range(len(c)) if s[i] > threshold],
        'event_captions': [t[i] for i in range(len(t)) if s[i] > threshold],
    } for s, l, b, c, t in zip(scores, labels, boxes, caption_scores, pred_captions)]