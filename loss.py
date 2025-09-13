import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from transformers.loss.loss_for_object_detection import sigmoid_focal_loss, ImageLoss, HungarianMatcher
from utils import box_iou, generalized_box_iou, cw_to_se


class TemporalHungarianMatcher(HungarianMatcher):
    '''
    Hungarian matcher for 1D temporal boxes/segments using SciPy's linear_sum_assignment with class, L1, and IoU costs.
    cost = class_cost + bbox_cost * L1(cw_pred, cw_tgt) + giou_cost * (1 - IoU(pred_se, tgt_se))
    '''
    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        '''
        outputs:
        - 'logits': (B, Q, C) raw class logits with C = num_classes (+1 for no-object at some index)
        - 'pred_boxes': (B, Q, 2) predicted (center, length) in [0,1]
          
        targets: list of length B, each dict has:
        - 'class_labels': (N_i,) int64 in [0..C-1], foreground classes only (no no-object here)
        - 'boxes':        (N_i, 2) in [0,1], (center,width) or (start,end). This matcher expects (center,width).
        Returns: list of (predicted indices, target indices) per batch element
        '''
        batch_size, num_queries = outputs['logits'].shape[:2]
        out_prob = outputs['logits'].flatten(0, 1).sigmoid()
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 4]

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
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TemporalDeformableDetrImageLoss(ImageLoss):
    ''' This class computes the loss for DETR. The process happens in 2 steps:
    1) We compute Hungarian assignment between ground truth boxes and the outputs of the model
    2) We supervise each pair of matched ground-truth / prediction (supervise class and box)
    '''
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        ''' Create the criterion
        num_classes: number of object categories, omitting the special no-object category
        matcher: module able to compute a matching between targets and proposals
        focal_alpha: alpha in Focal Loss
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        '''
        nn.Module.__init__(self)
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses
        

    # Removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        '''
        Classification loss (Binary focal loss) targets dicts must contain
        the key 'class_labels' containing a tensor of dim [nb_target_boxes]
        '''
        if 'logits' not in outputs: raise KeyError('No logits were found in the outputs')
        source_logits = outputs['logits']
        idx = self._get_source_permutation_idx(indices)
        
        target_classes = torch.full(source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device)
        target_classes[idx] = torch.cat([t['class_labels'][J] for t, (_, J) in zip(targets, indices)])
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
        max_length = pred_counts.shape[1] - 1
        target_counters = torch.tensor(
            [len(target['boxes']) for target in targets if len(target['boxes']) < max_length else max_length], 
            device=source_logits.device, dtype=torch.long
        )
        target_counters_onehot = torch.zeros_like(pred_counts)
        target_counters_onehot.scatter_(1, target_counters.unsqueeze(-1), 1)
        loss_counter = F.binary_cross_entropy_with_logits(pred_counts, target_counters_onehot, reduction='none').mean(1).mean()
        return {'loss_ce': loss_ce, 'loss_counter': loss_counter}
    
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        '''
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key 'boxes' containing a tensor of dim [nb_target_boxes, 2]. 
        The target boxes are expected in format (center, width/length), normalized by the image size.
        '''
        if 'pred_boxes' not in outputs: raise KeyError('No predicted boxes found in outputs')
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(source_boxes, target_boxes, reduction='none')
        loss_giou = 1 - generalized_box_iou(cw_to_se(source_boxes), cw_to_se(target_boxes)).diag()

        # Compute the self IoU, which is the average IoU between all pairs of predicted boxes in a batch
        self_iou = torch.triu(box_iou(cw_to_se(source_boxes), cw_to_se(source_boxes))[0], diagonal=1)
        sizes = [len(v) for v in indices]
        self_iou = sum([ # Formula: sum of IoUs / (0.5 * n * (n-1)) for each batch element
            # 1/2 for upper triangle & n - 1 because we don't compare box with itself 
            c.split(sizes, -2)[i].sum() / (0.5 * sizes[i] * (sizes[i] - 1)) 
            for i, c in enumerate(self_iou.split(sizes, -1))
        ])
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_iou': loss_giou.sum() / num_boxes,
            'loss_self_iou': self_iou
        }
        
        
def TemporalDeformableDetrForObjectDetectionLoss(logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None, **kwargs):
    matcher = TemporalHungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
    criterion = TemporalDeformableDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=['labels', 'boxes', 'cardinality'],
    )
    criterion.to(device)
    
    # Compute the losses, based on outputs and labels
    outputs_loss = {}
    outputs_loss['logits'] = logits
    outputs_loss['pred_boxes'] = pred_boxes
    auxiliary_outputs = None
    
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss['auxiliary_outputs'] = auxiliary_outputs
    loss_dict = criterion(outputs_loss, labels)
    
    # Compute total loss, as a weighted sum of the various losses
    weight_dict = {'loss_ce': 1, 'loss_bbox': config.bbox_loss_coefficient}
    weight_dict['loss_giou'] = config.giou_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({f'{k}_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    return loss, loss_dict, auxiliary_outputs