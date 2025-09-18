import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    '''
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    '''
    def __init__(self, embedding_dim=256, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is not None and normalize is False: raise ValueError('Normalize should be True if scale is passed')
        if scale is None: scale = 2 * math.pi
        self.scale = scale
        self.duration_embed_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        

    def forward(self, pixel_values, pixel_mask, durations):
        '''
        pixel_values 1d: (B, T, C)
        pixel_mask 1d:   (B, T) with 1 for valid positions, 0 for padding
        durations:       (B,) duration (in frames) of each sequence in the batch
        Returns: 
            pos: (B, T, C) position embeddings
        '''
        if pixel_mask is None: raise ValueError('No pixel mask provided')
        x_embed = pixel_mask.cumsum(1, dtype=torch.float32) # (B, T)
        
        if self.normalize:
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=pixel_values.dtype, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.embedding_dim)
        pos_x = x_embed[:, :, None] / dim_t # (B, T, embedding_dim)
        pos_x = torch.stack((
            pos_x[:, :, 0::2].sin(), # (B, T, embedding_dim/2)
            pos_x[:, :, 1::2].cos()  # (B, T, embedding_dim/2)
        ), dim=3).flatten(2)         # (B, T, embedding_dim)

        out = torch.zeros(len(durations), self.embedding_dim, device=durations.device) # (B, embedding_dim)
        for b, duration in enumerate(durations): out[b, :duration] = 1
        dur_embed = self.duration_embed_layer(out) # (B, embedding_dim)
        dur_embed = dur_embed.reshape(-1, 1, self.embedding_dim).expand_as(pos_x) # (B, T, embedding_dim)
        return torch.cat((pos_x, dur_embed), dim=2).permute(0, 2, 1) # (B, 2*embedding_dim, T)