import torch
from torch import Tensor, TensorType
from transformers import AutoTokenizer
from typing import Union
from utils import cw_to_se


@torch.no_grad()
def post_process_object_detection(
    self, outputs, threshold: float = 0.5, 
    target_lengths: Union[TensorType, list[int]] = None, 
    top_k: int = 10, tokenizer: AutoTokenizer = None,
):
    '''
    Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (start, end) format. 

    Args:
        outputs ([`DetrObjectDetectionOutput`]): Raw outputs of the model.
        threshold (`float`, *optional*): Score threshold to keep object detection predictions.
        target_lengths (`torch.Tensor` or `list[int]`, *optional*):
            Tensor of shape `(batch_size)` or list of integers containing the target length
            of each clip in the batch. If left to None, predictions will not be resized.
        top_k (`int`, *optional*, defaults to 10):
            Keep only top k bounding boxes before filtering by thresholding.

    Returns:
        `list[Dict]`: A list of dictionaries, each dictionary containing the following keys:
        - `scores` (`torch.Tensor`): Scores of the kept predictions. 
            The scores are for the foreground class (object) and not for the no-object class.
            Shape `(num_kept_predictions,)`.
        - `labels` (`torch.Tensor`): Labels of the kept predictions, 
            always 1 (foreground class) since there is only one class. 
            Shape `(num_kept_predictions,)`.
        - `boxes` (`torch.Tensor`): Bounding boxes of the kept predictions in (start, end) format.
            Shape `(num_kept_predictions, 2)`.
        - `cap_scores` (`list[float]`): Caption scores of the kept predictions.
            Shape `(num_kept_predictions,)`.
        - `cap_tokens` (`list[str]`): Decoded caption tokens of the kept predictions.
            Shape `(num_kept_predictions,)`.
    '''
    out_logits, out_bbox = outputs.logits, outputs.pred_boxes
    pred_cap_tokens, pred_cap_logits = outputs.pred_cap_tokens, outputs.pred_cap_logits
    pred_counts = outputs.pred_counts.argmax(dim=-1).clamp(min=1)
    
    if target_lengths is not None:
        if len(out_logits) != len(target_lengths):
            raise ValueError('Make sure that you pass in as many target lengths as the batch dimension of the logits')

    prob = out_logits.sigmoid()               # (batch_size, num_queries, num_classes)
    prob = prob.view(out_logits.shape[0], -1) # (batch_size, num_queries * num_classes)
    k_value = min(top_k, prob.size(1))        # Ensure k_value does not exceed total predictions
    scores, topk_indexes = torch.topk(prob, k_value, dim=1) # (batch_size, k_value)

    topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='floor') # (batch_size, k_value)
    labels = topk_indexes % out_logits.shape[2]                                      # (batch_size, k_value)
    boxes = cw_to_se(out_bbox) # Convert (center, width) to (start, end) format, shape (batch_size, num_queries, 2)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))         # (batch_size, k_value, 2)

    # And from relative [0, 1] to absolute [0, height] coordinates
    if target_lengths is not None:
        scale_fct = torch.stack([target_lengths, target_lengths], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]
        
    if len(pred_cap_tokens):
        mask = pred_cap_tokens != tokenizer.eos_token_id                # (batch_size, num_queries, max_caption_len)
        cap_scores = (pred_cap_logits * mask).sum(dim=-1).cpu().numpy() # (batch_size, num_queries)
        cap_scores = [cap_scores[i][topk_boxes[i]] for i in range(cap_scores.shape[0])] # (batch_size, k_value)
        
        pred_cap_tokens = pred_cap_tokens.detach().cpu().numpy()        # (batch_size, num_queries, max_caption_len)
        pred_cap_tokens = [pred_cap_tokens[i][topk_boxes[i]] for i in range(pred_cap_tokens.shape[0])] # (batch_size, k_value, max_caption_len)
        pred_cap_tokens = [[
            tokenizer.decode(cap_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
            for cap_tokens in batch_cap_tokens
        ] for batch_cap_tokens in pred_cap_tokens]                      # (batch_size, k_value)
    else: # No caption tokens predicted, so fill with empty strings and very low scores
        cap_scores = [[-1e5] * k_value] * out_logits.shape[0]           # (batch_size, k_value)
        pred_cap_tokens = [[''] * k_value] * out_logits.shape[0]        # (batch_size, k_value)
        
    return [{
        'scores': s[s > threshold], 
        'labels': l[s > threshold], 
        'boxes': b[s > threshold], 
        'cap_scores': [c[i] for i in range(len(c)) if s[i] > threshold],
        'cap_tokens': [t[i] for i in range(len(t)) if s[i] > threshold],
    } for s, l, b, c, t in zip(scores, labels, boxes, cap_scores, pred_cap_tokens)]