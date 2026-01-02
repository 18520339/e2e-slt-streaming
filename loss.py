import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

from transformers.loss.loss_for_object_detection import (
    HungarianMatcher, ImageLoss, 
    _set_aux_loss, sigmoid_focal_loss,
    is_accelerate_available
) 
from utils import box_iou, generalized_box_iou, cw_to_se

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce
    
    
class DeformableDetrHungarianMatcher(HungarianMatcher):
    '''
    This class computes an assignment between the targets and the predictions of the network. 
    For efficiency reasons, the targets don't include the no_object. In general, there are more predictions than targets. 
    In this case, we do a 1-to-1 matching of the best predictions, while the others are un-matched (and thus treated as non-objects).
    cost = λ_cls * class_cost + λ_L1 * L1(cw_pred, cw_tgt) + λ_GIoU * GIoU(pred_se, tgt_se)
    
    Args:
        class_cost: The relative weight of the classification error in the matching cost.
        bbox_cost: The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost: The relative weight of the GIoU loss of the bounding box in the matching cost.
    '''
    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        '''
        outputs (`dict`):
            A dictionary that contains at least these entries:
            * 'logits': Tensor of dim [batch_size, num_queries, num_classes] with the classification logits (+1 for no-object at some index).
            * 'pred_boxes': Tensor of dim [batch_size, num_queries, 2] with the predicted box coordinates (center, width) in [0,1].
        targets (`list[dict]`):
            A list of targets (len(targets) = batch_size), where each target is a dict containing:
            * 'class_labels': Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                ground-truth objects in the target) containing the class labels (no no-object here).
            * 'boxes': Tensor of dim [num_target_boxes, 2] containing the target box coordinates.
                in [0,1], (center,width) or (start,end). This matcher expects (center,width)
        
        Returns:
            `list[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            
        For example: 
            We have a batch of 3 elements, with each element having 4, 3, and 5 target boxes respectively.
            If we set num_queries to 4, this function might return:
            [ (tensor([0, 1, 2, 3]), tensor([2, 0, 1, 2])),  # for the 1st batch element: 4 predictions, 4 targets
              (tensor([0, 1, 2]),    tensor([1, 2, 0])),     # for the 2nd batch element: 3 predictions, 3 targets
              (tensor([0, 1, 2, 3]), tensor([4, 0, 1, 2])) ] # for the 3rd batch element: 4 predictions, 4 targets
        '''
        batch_size, num_queries = outputs['logits'].shape[:2]
        out_prob = outputs['logits'].flatten(0, 1).sigmoid()
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v['class_labels'] for v in targets])
        target_bbox = torch.cat([v['boxes'] for v in targets])

        # Compute the classification cost
        alpha, gamma = 0.25, 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 and giou cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        giou_cost = -generalized_box_iou(cw_to_se(out_bbox), cw_to_se(target_bbox))

        # Final cost matrix
        cost_matrix = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class PDVCLoss(ImageLoss):
    ''' This class computes the loss for DETR. The process happens in 2 steps:
    1) We compute Hungarian assignment between ground truth boxes and the outputs of the model
    2) We supervise each pair of matched ground-truth / prediction (supervise class and box)
    '''
    def __init__(self, matcher, num_classes, focal_alpha, pad_token_id, losses):
        ''' Create the criterion
        num_classes: number of object categories, omitting the special no-object category
        matcher: module able to compute a matching between targets and proposals
        focal_alpha: alpha in Focal Loss
        pad_token_id: The padding token id for captions
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        '''
        nn.Module.__init__(self)
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.pad_token_id = pad_token_id
        self.losses = losses
        

    # Removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        '''
        Classification loss (Binary focal loss), where targets is a list of dicts, each must contain
        a key 'class_labels' containing a tensor of dim [num_target_boxes]
        '''
        if 'logits' not in outputs: raise KeyError('No logits were found in the outputs')
        idx = self._get_source_permutation_idx(indices)
        source_logits = outputs['logits']
        
        target_classes = torch.full(source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device)
        target_classes[idx] = torch.cat([t['class_labels'][i] for t, (b, i) in zip(targets, indices)])
        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            source_logits, target_classes_onehot, 
            num_boxes, alpha=self.focal_alpha, gamma=2
        ) * source_logits.shape[1]

        pred_counts = outputs['pred_counts']
        target_counts = torch.tensor(
            [min(len(target['boxes']), pred_counts.shape[1] - 1) for target in targets],  # Use min to clip
            device=source_logits.device, dtype=torch.long
        )
        target_counts_onehot = torch.zeros_like(pred_counts)
        target_counts_onehot.scatter_(1, target_counts.unsqueeze(-1), 1)
        loss_counter = F.binary_cross_entropy_with_logits(pred_counts, target_counts_onehot)
        return {'loss_ce': loss_ce, 'loss_counter': loss_counter}
    
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        '''
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
        Targets is a list of dicts, each must contain a key 'boxes' containing a tensor of dim [num_target_boxes, 2]. 
        The target boxes are expected in format (center, width/length), normalized by the image size.
        '''
        if 'pred_boxes' not in outputs: raise KeyError('No predicted boxes found in outputs')
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs['pred_boxes'][idx] # [batch_size, num_matched, 2]
        target_boxes = torch.cat([t['boxes'][i] for t, (b, i) in zip(targets, indices)], dim=0) # [batch_size, num_matched, 2]
        loss_bbox = F.l1_loss(source_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(cw_to_se(source_boxes), cw_to_se(target_boxes)))

        # Compute the self IoU, which is the average IoU between all pairs of predicted boxes in a batch
        # self_iou = torch.triu(box_iou(cw_to_se(source_boxes), cw_to_se(source_boxes))[0], diagonal=1)
        # sizes = [len(v) for v in indices]
        # self_iou = sum([ # Formula: sum of IoUs / (0.5 * n * (n-1)) for each batch element
        #     # 1/2 for upper triangle & n - 1 because we don't compare box with itself 
        #     c.split(sizes, -2)[i].sum() / (0.5 * sizes[i] * (sizes[i] - 1)) 
        #     for i, c in enumerate(self_iou.split(sizes, -1))
        # ])
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
            # 'loss_self_iou': self_iou
        }
        
    
    def loss_captions(self, outputs, targets, indices, num_boxes): 
        '''
        Compute the captioning loss, which is the cross-entropy loss of the predicted captions.
        Targets is a list of dicts, each must contain a key 'seq_tokens' containing a tensor of dim [num_target_boxes, max_len].
        The captions are expected to be tokenized and padded with 0 (the padding index).
        '''
        if 'pred_cap_logits' not in outputs: raise KeyError('No caption logits found in outputs')
        idx = self._get_source_permutation_idx(indices)
        source_logits = outputs['pred_cap_logits'][idx]          # [batch_size * num_matched, L, vocab_size]
        
        target_tokens = torch.cat([t['seq_tokens'][i] for t, (b, i) in zip(targets, indices)], dim=0)  # [batch_size * num_matched, max_len]
        if target_tokens.shape[1] > source_logits.shape[1]:      # Remove the start token for targets if it exists
            target_tokens = target_tokens[:, 1:source_logits.shape[1] + 1]  
        
        loss_caption = F.nll_loss(
            source_logits.reshape(-1, source_logits.shape[-1]),  # [batch_size * num_matched * L, vocab_size]
            target_tokens.reshape(-1),                           # [batch_size * num_matched * L]
            ignore_index=self.pad_token_id, 
        )
        return {'loss_caption': loss_caption}
        
            
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'captions': self.loss_captions,
            'masks': self.loss_masks,
            'cardinality': self.loss_cardinality,
        }
        if loss not in loss_map: raise ValueError(f'Loss {loss} not supported')
        return loss_map[loss](outputs, targets, indices, num_boxes)
    
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'auxiliary_outputs'}
        last_indices = self.matcher(outputs_without_aux, targets) # Retrieve the matching between outputs of last layer and targets

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t['class_labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                # accelerate.utils.reduce defaults to reduction='mean'. We want the global SUM here
                # and then explicitly divide by world_size to get the per-process average.
                num_boxes = reduce(num_boxes, reduction='sum')
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'auxiliary_outputs' in outputs:
            for i, auxiliary_outputs in enumerate(outputs['auxiliary_outputs']):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks': continue # Intermediate masks losses are too costly to compute, we ignore them
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    losses.update({f'{k}_{i}': v for k, v in l_dict.items()})
        return losses, last_indices
        
        
class DeformableDetrForObjectDetectionLoss:
    def __init__(
        self, config, pad_token_id=0,
        weight_dict={'loss_ce': 2, 'loss_bbox': 0, 'loss_giou': 4, 'loss_counter': 2, 'loss_caption': 2}
    ):
        super().__init__()
        self.config = config
        self.auxiliary_outputs = None
        self.criterion = PDVCLoss(
            matcher=DeformableDetrHungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost), 
            num_classes=config.num_labels, 
            focal_alpha=config.focal_alpha, 
            pad_token_id=pad_token_id,
            losses=['labels', 'boxes', 'cardinality', 'captions']
        )
        self.base_weight_dict = weight_dict.copy()  # Store original weights
        self.weight_dict = self.base_weight_dict.copy()
        
        if self.config.auxiliary_loss: # Pre-compute the full weight dict including auxiliary losses
            for layer in range(self.config.decoder_layers - 1):
                self.weight_dict.update({f'{k}_{layer}': v for k, v in self.base_weight_dict.items()})
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_classes, outputs_coords, outputs_counts, outputs_cap_probs):
        # This is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such as a dict having both a Tensor and a list.
        return [
            {'logits': a, 'pred_boxes': b, 'pred_counts': c, 'pred_cap_logits': d}
            for a, b, c, d in zip(outputs_classes[:-1], outputs_coords[:-1], outputs_counts[:-1], outputs_cap_probs[:-1])
        ]

    def __call__(
        self, labels, logits, pred_boxes, pred_counts, pred_cap_logits, 
        outputs_classes, outputs_coords, outputs_counts, outputs_cap_probs
    ):
        outputs = {'logits': logits, 'pred_boxes': pred_boxes, 'pred_counts': pred_counts, 'pred_cap_logits': pred_cap_logits}
        if self.config.auxiliary_loss:
            self.auxiliary_outputs = self._set_aux_loss(outputs_classes, outputs_coords, outputs_counts, outputs_cap_probs)
            outputs['auxiliary_outputs'] = self.auxiliary_outputs
        loss_dict, last_indices = self.criterion(outputs, labels) # Compute the losses, based on outputs and labels
        
        weight_dict = self.weight_dict # Use a local copy for weight adjustment to avoid mutation
        if not self.config.with_box_refine: # No loss on class and box if using ground truth proposals
            weight_dict = self.weight_dict.copy()
            for key in ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_counter']: 
                weight_dict[key] = 0 # We only need to pay attention to captioning performance
        
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return loss, loss_dict, self.auxiliary_outputs