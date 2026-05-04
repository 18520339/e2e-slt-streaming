'''MSKA pose-encoder backbone (gloss-free path).

Vendored from MSKA's recognition.py — DSTA / STAttentionBlock / PositionalEncoding only,
WITHOUT the gloss VisualHead, CTC loss, or translation network. Loads the pose-encoder
weights from the released MSKA SLT checkpoint by filtering keys with the prefix
`recognition_network.visual_backbone_keypoint.`.

Why gloss-free safe: MSKA's checkpoint was trained with gloss CTC supervision. We import only
the pose representation (no gloss vocab, no CTC head, no gloss-conditioned translator).
Gloss is never seen at training time of StreamSLST or at inference. Same legitimacy logic
as ImageNet pretraining for non-classification downstreams.

Input contract: (B, T, 133, 3) — full COCO-WholeBody-133 in pixel coords.
                We normalize to MSKA's expected range internally (x/=W, (h-y)/h, then [-1,1]).
Output contract: (B, T, hidden_size) — temporally upsampled back to T to match CoSign1s
                contract used by gfslt_models.py / pdvc.py.

Usage (drop-in replacement for CoSign1s in PoseFeatureExtractor):
    from backbones.mska_backbone import MSKABackbone
    backbone = MSKABackbone(hidden_size=1024, ckpt_path='checkpoints/mska.pth', canvas_w=210, canvas_h=260, num_frame=400)
    feats = backbone(poses)   # poses: (B, T, 133, 3) raw pixel coords

Reference:
  Wang et al., "Multi-stream keypoint attention network for sign language recognition and
  translation", arXiv:2405.05672. https://github.com/sutwangyan/MSKA
'''
import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- MSKA per-stream index arrays (identical for PHOENIX-2014T and CSL-Daily configs) -----
# Indices into a (T, 133, 3) COCO-WholeBody tensor.
MSKA_BODY = [0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 
             107, 108, 109, 110, 111, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 
             122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 23, 26, 29, 33, 36, 39, 41, 43, 46, 
             48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81]     # 79 nodes
MSKA_LEFT = [0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
             105, 106, 107, 108, 109, 110, 111]                                      # 27 nodes
MSKA_RIGHT = [0, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
             124, 125, 126, 127, 128, 129, 130, 131, 132]                            # 27 nodes
MSKA_FACE = [23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74,
             75, 76, 77, 79, 80, 81]                                                 # 26 nodes

# Channel progression for the 8-block STAttention stack: [in, out, inter, t_kernel, stride].
# Total temporal stride = 2 × 2 = 4. Final per-stream out-channels = 256.
MSKA_DSTA_NET = [[64, 64, 16, 7, 2], [64, 64, 16, 3, 1], [64, 128, 32, 3, 1], [128, 128, 32, 3, 1],
                 [128, 256, 64, 3, 2], [256, 256, 64, 3, 1], [256, 256, 64, 3, 1], [256, 256, 64, 3, 1]]
MSKA_FUSE_DIM = 4 * 256  # = 1024 after concat([left, face, right, body], dim=-1)
MSKA_DSTA_TOTAL_STRIDE = 4


# =============================================================================
# Vendored MSKA encoder modules (from MSKA/recognition.py, gloss components stripped)
# =============================================================================

class _PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len, domain):
        super().__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.domain = domain
        if domain == 'temporal': pos_list = [t for t in range(time_len) for _ in range(joint_num)]
        elif domain == 'spatial': pos_list = [j for _ in range(time_len) for j in range(joint_num)]
        else: raise ValueError(domain)
        
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(time_len * joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]


class _STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
                 kernel_size=1, stride=1, t_kernel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0., use_pes=True, use_pet=False):
        super().__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.glo_reg_t = glo_reg_t
        self.att_s = att_s
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            self.register_buffer("atts", torch.zeros((1, num_subset, num_node, num_node)))
            self.pes = _PositionalEncoding(in_channels, num_node, num_frame, "spatial")
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)
                
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else: self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            if use_spatial_att: self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att: self.downt1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att: self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)


    def forward(self, x):
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            y = self.pes(x) if self.use_pes else x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)
                ) * self.alphas
                
            if self.glo_reg_s: attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous().view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)
        z = self.out_nett(y)
        return self.relu(self.downt2(y) + z)


class _DSTA(nn.Module):
    '''Multi-stream Decoupled Spatial-Temporal Attention encoder.

    Mirrors MSKA/recognition.py:DSTA verbatim except the input-tensor `.cuda()` call has been
    replaced with device-aware indexing (we rely on the user to put the module + input on the
    correct device — same as every other module in this project).
    '''

    def __init__(self, num_frame=400, num_subset=6, dropout=0.1, num_channel=3, net_cfg=None, 
                 body_idx=None, left_idx=None, right_idx=None, face_idx=None, 
                 glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1, 
                 use_pet=False, use_pes=True):
        super().__init__()
        config = net_cfg
        self.out_channels = config[-1][1]
        in_channels_first = config[0][0]
        self.num_frame = num_frame
        self.body_idx = body_idx
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.face_idx = face_idx
        param = dict(num_subset=num_subset, glo_reg_s=glo_reg_s, att_s=att_s, glo_reg_t=glo_reg_t, att_t=att_t,
                     use_spatial_att=use_spatial_att, use_temporal_att=use_temporal_att,
                     use_pet=use_pet, use_pes=use_pes, attentiondrop=attentiondrop)
        # Per-stream input projections C → in_channels_first
        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels_first, 1),
            nn.BatchNorm2d(in_channels_first),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels_first, 1),
            nn.BatchNorm2d(in_channels_first),
            nn.LeakyReLU(0.1),
        )
        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels_first, 1),
            nn.BatchNorm2d(in_channels_first),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels_first, 1),
            nn.BatchNorm2d(in_channels_first),
            nn.LeakyReLU(0.1),
        )
        # Per-stream layer stacks. Node counts hardcoded per MSKA configs.
        self.face_graph_layers = nn.ModuleList()
        nf = num_frame
        for in_c, out_c, inter_c, tk, stride in config:
            self.face_graph_layers.append(_STAttentionBlock(
                in_c, out_c, inter_c, stride=stride, 
                t_kernel=tk, num_node=26, num_frame=nf, **param
            ))
            nf = int(nf / stride + 0.5)
            
        nf = num_frame
        self.left_graph_layers = nn.ModuleList()
        for in_c, out_c, inter_c, tk, stride in config:
            self.left_graph_layers.append(_STAttentionBlock(
                in_c, out_c, inter_c, stride=stride,
                num_node=27, t_kernel=tk, num_frame=nf, **param
            ))
            nf = int(nf / stride + 0.5)
            
        nf = num_frame
        self.right_graph_layers = nn.ModuleList()
        for in_c, out_c, inter_c, tk, stride in config:
            self.right_graph_layers.append(_STAttentionBlock(
                in_c, out_c, inter_c, stride=stride,
                num_node=27, t_kernel=tk, num_frame=nf, **param
            ))
            nf = int(nf / stride + 0.5)
            
        nf = num_frame
        self.body_graph_layers = nn.ModuleList()
        for in_c, out_c, inter_c, tk, stride in config:
            self.body_graph_layers.append(_STAttentionBlock(
                in_c, out_c, inter_c, stride=stride,
                num_node=79, t_kernel=tk, num_frame=nf, **param
            ))
            nf = int(nf / stride + 0.5)
        self.drop_out = nn.Dropout(dropout)
        

    def forward(self, x): 
        device = x.device # x: (B, C, T, 133)
        body_idx = torch.as_tensor(self.body_idx, device=device, dtype=torch.long)
        left_idx = torch.as_tensor(self.left_idx, device=device, dtype=torch.long)
        right_idx = torch.as_tensor(self.right_idx, device=device, dtype=torch.long)
        face_idx = torch.as_tensor(self.face_idx, device=device, dtype=torch.long)

        left = self.left_input_map(x.index_select(3, left_idx))
        right = self.right_input_map(x.index_select(3, right_idx))
        face = self.face_input_map(x.index_select(3, face_idx))
        body = self.body_input_map(x.index_select(3, body_idx))

        for m in self.face_graph_layers: face = m(face)
        for m in self.left_graph_layers: left = m(left)
        for m in self.right_graph_layers: right = m(right)
        for m in self.body_graph_layers: body = m(body)

        # (B, C, T_out, V) → (B, T_out, C, V) → mean over V → (B, T_out, C)
        left = left.permute(0, 2, 1, 3).contiguous().mean(3)
        right = right.permute(0, 2, 1, 3).contiguous().mean(3)
        face = face.permute(0, 2, 1, 3).contiguous().mean(3)
        body = body.permute(0, 2, 1, 3).contiguous().mean(3)
        # Concat: (B, T_out, 4*C) — matches MSKA fuse output. C=256 → fuse=1024.
        return torch.cat([left, face, right, body], dim=-1)


# =============================================================================
# Public wrapper: drop-in replacement for CoSign1s
# =============================================================================

class MSKABackbone(nn.Module):
    '''MSKA pose encoder wrapped to match CoSign1s I/O contract.

    Args:
        hidden_size: target output dim. Must equal 1024 (MSKA fuse dim) — assertion.
        ckpt_path: optional path to MSKA SLT .pth checkpoint. Loads only
                   `recognition_network.visual_backbone_keypoint.*` keys; gloss heads,
                   translation_network, and VLMapper are silently ignored.
        canvas_w, canvas_h: native pixel canvas of the dataset (210×260 PHOENIX, 512×512 CSL,
                   1280×720 H2S, 444×444 BOBSL). Used to normalize raw coords to [-1, 1].
        num_frame: max input window size in frames. Used to size positional encodings; can
                   exceed training W. Default 400 matches MSKA configs.
        contrastive_mode: kept for API compatibility with CoSign1s. Returns (feat, feat) pair
                   when True (no extra masked view — MSKA backbone has no built-in masking).
                   If you need true contrastive 2-views, wrap externally.

    Forward input: pose tensor of shape (B, T, 133, 3) in NATIVE pixel coords.
    Forward output:
        (B, T, hidden_size=1024) — temporally upsampled from MSKA's T/4 internal resolution
        via linear interp, so segmentation/translation heads see the same temporal
        granularity they get from CoSign1s.
        If contrastive_mode=True: tuple (feats, feats) — same tensor twice for compat.
    '''

    def __init__(
        self, hidden_size: int = 1024, ckpt_path: Optional[str] = None,
        canvas_w: int = 210, canvas_h: int = 260, num_frame: int = 400,
        contrastive_mode: bool = False, strict_ckpt: bool = False,
    ):
        super().__init__()
        if hidden_size != MSKA_FUSE_DIM:
            raise ValueError(f'MSKABackbone fuse dim is {MSKA_FUSE_DIM} (=4×256). '
                             f'Got hidden_size={hidden_size}. Set d_model accordingly or add a projection layer.')
        self.hidden_size = hidden_size
        self.canvas_w = float(canvas_w)
        self.canvas_h = float(canvas_h)
        self.contrastive_mode = contrastive_mode
        self.dsta = _DSTA(
            num_frame=num_frame, net_cfg=MSKA_DSTA_NET,
            body_idx=MSKA_BODY, left_idx=MSKA_LEFT, right_idx=MSKA_RIGHT, face_idx=MSKA_FACE,
        )
        if ckpt_path is not None: self.load_pretrained(ckpt_path, strict=strict_ckpt)

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        '''Load only the pose-encoder weights from MSKA's released SLT .pth.

        MSKA checkpoint layout: top-level dict with key 'model' wrapping the full state dict;
        keys are prefixed with `recognition_network.visual_backbone_keypoint.`. We strip that
        prefix and load into self.dsta. Gloss heads (`recognition_network.fuse_visual_head`,
        `*_visual_head`), translation_network, and mapper are skipped.
        '''
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        prefix = 'recognition_network.visual_backbone_keypoint.'
        encoder_sd = {}
        skipped_gloss = 0
        for k, v in sd.items():
            if k.startswith(prefix): encoder_sd[k[len(prefix):]] = v
            elif k.startswith('recognition_network.') and 'visual_head' in k:
                skipped_gloss += 1  # explicitly skip gloss VisualHead weights
                
        miss, unexp = self.dsta.load_state_dict(encoder_sd, strict=strict)
        n_loaded = len(encoder_sd) - len(unexp)
        print(f'[MSKABackbone] loaded {n_loaded}/{len(encoder_sd)} encoder weights from {ckpt_path}')
        print(f'[MSKABackbone] skipped {skipped_gloss} gloss VisualHead keys')
        if miss:  print(f'[MSKABackbone] missing keys ({len(miss)}): example={miss[:3]}')
        if unexp: print(f'[MSKABackbone] unexpected keys ({len(unexp)}): example={unexp[:3]}')


    def _normalize(self, p: torch.Tensor) -> torch.Tensor:
        '''Match MSKA datasets.py preprocessing: (B,T,V,3) pixel → [-1,1] x/y, conf untouched.

        Steps from MSKA dataset: x /= W; y = (h - y) / h;  then x = (x - 0.5) / 0.5,  y = (y - 0.5) / 0.5.
        Confidence channel is left as-is (kept inside C=3, so `num_channel=3` in DSTA).
        '''
        out = p.clone()
        out[..., 0] = out[..., 0] / self.canvas_w
        out[..., 1] = (self.canvas_h - out[..., 1]) / self.canvas_h
        out[..., 0] = (out[..., 0] - 0.5) / 0.5
        out[..., 1] = (out[..., 1] - 0.5) / 0.5
        return out
    

    def forward(self, poses: torch.Tensor):
        # poses: (B, T, 133, 3) NATIVE pixel coords. Returns (B, T, 1024)
        if poses.dim() != 4 or poses.shape[-2] != 133 or poses.shape[-1] != 3:
            raise ValueError(f'MSKABackbone expects (B, T, 133, 3); got {tuple(poses.shape)}. '
                             'If you currently feed (B, T, 77, 3) selected keypoints, route the raw '
                             '(T, 133, 3) data straight from the loader instead (skip ALL_SELECTED_IDS slicing).')

        B, T, V, C = poses.shape
        x = self._normalize(poses)               # (B, T, 133, 3)
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, C=3, T, V=133)
        feat = self.dsta(x)                      # (B, T_out, 1024); T_out = T // 4
        
        # Upsample T_out → T to match CoSign1s output contract. (Linear is fine; segmentation
        # head reads normalized [0,1] centers, so resampling is consistent.)
        # IMPORTANT: align_corners=True is required here. With align_corners=False the
        # endpoints are shifted by half a sample, creating ~1-frame systematic drift between
        # the predicted boundary positions and the GT — at strict tIoU thresholds (e.g. 0.9)
        # every prediction misses, collapsing localization recall AND translation metrics
        # (which are evaluated only on tIoU-matched events) to exactly 0.0.
        if feat.shape[1] != T: feat = F.interpolate(
            feat.transpose(1, 2),                # (B, D, T_out)
            size=T, mode='linear', 
            align_corners=True,
        ).transpose(1, 2)                        # (B, T, D)
        if self.contrastive_mode: return feat, feat  # API compat with CoSign1s 2-view return
        return feat