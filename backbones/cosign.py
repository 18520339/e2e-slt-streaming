import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .gcn_utils import Graph
from .st_gcn_block import get_stgcn_chain
from config import *


def generate_mask(shape, ratio, dim):
    B, T, C = shape
    window_size_frames = int(WINDOW_DURATION_SECONDS * FPS)
    num_windows = T // window_size_frames
    random_mask = np.random.rand(B, num_windows, len(KPS_MODULES)) > (1 - 2 * ratio) # Shape: (B, num_windows, parts)
    mask_q, mask_k = np.zeros_like(random_mask), np.zeros_like(random_mask)
    position = np.where(random_mask) # Shape: (3, N) where N is number of True in random_mask and 3 means (B, num_windows, parts)
    num_true = len(position[0])

    index = np.random.choice(num_true, size=num_true // 2, replace=False).tolist()
    for i in range(num_true):
        if i in index: mask_q[position[0][i], position[1][i], position[2][i]] = 1
        else: mask_k[position[0][i], position[1][i], position[2][i]] = 1
        
    mask_q = mask_q.astype(np.bool8)
    mask_k = mask_k.astype(np.bool8)
    mask_cat_q = torch.ones(shape)
    mask_cat_k = torch.ones(shape)
    
    for i in range(B):
        for k in range(num_windows):
            start_frame = window_size_frames * k
            for j in range(mask_q.shape[2]):
                end_frame = T if k >= num_windows - 1 else window_size_frames * (k + 1)
                if mask_q[i, k, j]: mask_cat_q[i, start_frame:end_frame, dim * j: dim * (j + 1)] = 0
                if mask_k[i, k, j]: mask_cat_k[i, start_frame:end_frame, dim * j: dim * (j + 1)] = 0
    return mask_cat_q, mask_cat_k


class CoSign1s(nn.Module):
    def __init__(self, temporal_kernel, hidden_size, level='spatial', adaptive=True, mask_ratio=0.3):
        super().__init__()
        self.graph, A = {}, {}
        self.gcn_modules = {}
        self.mask_ratio = mask_ratio # Portion of keypoint groups to mask
        self.linear = nn.Sequential(nn.Linear(3, 64), nn.ReLU(inplace=True))

        for module in KPS_MODULES.keys():
            self.graph[module] = Graph(layout=f'{module}', strategy='distance', max_hop=1)
            A[module] = torch.tensor(self.graph[module].A, dtype=torch.float32, requires_grad=False)
            spatial_kernel_size = A[module].size(0)
            self.gcn_modules[module], final_dim = get_stgcn_chain(
                64, level, (temporal_kernel, spatial_kernel_size),
                A[module].clone(), adaptive
            )

        self.pool_func = F.avg_pool2d
        self.gcn_modules = nn.ModuleDict(self.gcn_modules)
        self.fusion = nn.Sequential(nn.Linear(final_dim * len(KPS_MODULES), hidden_size), nn.ReLU(inplace=True))
        self.out_size = hidden_size
        self.final_dim = final_dim
    
    
    def process_part_features(self, features):
        feat_list = []
        for module, kps_info in KPS_MODULES.items():
            kps_rng = kps_info['kps_rel_range']
            part_feat = self.gcn_modules[module](features[..., kps_rng[0]: kps_rng[1]])
            pooled_feat = self.pool_func(part_feat, (1, kps_rng[1] - kps_rng[0])).squeeze(-1)
            feat_list.append(pooled_feat)
        return torch.cat(feat_list, dim=1)
    
    
    def forward(self, x):
        # linear stage x.shape: [B(N), T, 77(K), 3(C)]
        static = self.linear(x).permute(0, 3, 1, 2) # [B, C, T, K]
        cat_feat = self.process_part_features(static).transpose(1, 2) # [B, T, C]

        if self.training:
            mask_view1, mask_view2 = generate_mask(cat_feat.shape, self.mask_ratio, self.final_dim)
            view1, view2 = mask_view1.to(cat_feat.device) * cat_feat, mask_view2.to(cat_feat.device) * cat_feat
            return {'view1': self.fusion(view1)} # [B, T, hidden]
        return {'cat': cat_feat, 'fusion': self.fusion(cat_feat)}