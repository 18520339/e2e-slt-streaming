import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Tuple


class TemporalConvEncoder(nn.Module):
    '''
    A minimal temporal encoder that produces a multi-scale 1D feature pyramid shaped as (B, C, 1, T_l),
    plus a correspondingly downsampled mask (B, 1, T_l) for each level, where True means "valid".

    Args:
        in_channels: input channels (e.g., RGB=3 if using raw frames or some precomputed channels)
        channels: list of output channels for each level
        strides: list of temporal strides for each level (must have same length as channels)
        kernel_sizes: list of kernel sizes for each level (same length as channels)
        norm: whether to use GroupNorm on each level
    '''

    def __init__(
        self, 
        in_channels: int, 
        channels: List[int] = (128, 256, 256),
        strides: List[int] = (2, 2, 2),
        kernel_sizes: List[int] = (5, 3, 3),
        norm: bool = True
    ):
        super().__init__()
        assert len(channels) == len(strides) == len(kernel_sizes)
        layers = []
        curr_in = in_channels
            
        for c, s, k in zip(channels, strides, kernel_sizes):
            pad = k // 2
            layers.append(nn.Sequential(
                nn.Conv1d(curr_in, c, kernel_size=k, stride=s, padding=pad),
                nn.GroupNorm(32, c) if norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ))
            curr_in = c

        self.stages = nn.ModuleList(layers)
        self.intermediate_channel_sizes = [s[0].out_channels for s in self.stages]
            

    def forward(self, pixel_values_1d: Tensor, pixel_mask_1d: Tensor) -> List[Tuple[Tensor, Tensor]]:
        '''
        pixel_values_1d: (B, C, T)
        pixel_mask_1d:   (B, T) with 1 for valid positions, 0 for padding
        Returns: list of tuples (feature_map 2d, mask 2d) for each level:
            - feature_map 2d: (B, C_l, 1, T_l)
            - mask 2d:        (B, 1, T_l) boolean, True for valid
        '''
        assert pixel_values_1d.dim() == 3, 'Expected (B, C, T)'
        assert pixel_mask_1d.dim() == 2, 'Expected (B, T)'
        B, _, _ = pixel_values_1d.shape
        x, m = pixel_values_1d, pixel_mask_1d
        outputs = []

        for stage in self.stages:
            x = stage(x)  # (B, C_l, T_l)
            T_l = x.shape[-1]
            
            # Downsample/resize mask to T_l via nearest interpolation in 'width' dimension
            mask_2d = m[:, None, None, :] # (B, 1, 1, T_l)
            mask_2d = F.interpolate(mask_2d.float(), size=(1, T_l), mode='nearest').to(torch.bool)  # (B, 1, 1, T_l)
            outputs.append((x.unsqueeze(2), mask_2d[:, 0]))  # (B, C_l, 1, T_l), (B, 1, T_l)
            m = mask_2d[:, 0, 0, :]  # (B, T_l) update mask for next stage based on stride (nearest is fine)
        return outputs