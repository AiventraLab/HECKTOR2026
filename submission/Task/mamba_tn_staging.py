import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List


try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    import warnings; warnings.warn("mamba_ssm not found, using GRU fallback")


class _FallbackMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(out)


def _make_mamba(d_model):
    if MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    return _FallbackMamba(d_model)


class MambaTNStagingModel(nn.Module):
    def __init__(self, seg_feat_dim: int, clin_feat_dim: int,
                 d_model: int = 64, num_t_classes: int = 4, num_n_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.seg_proj = nn.Sequential(
            nn.Linear(seg_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.pos_emb = nn.Embedding(8, d_model)
        self.mamba = _make_mamba(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.t_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_t_classes)
        )
        self.n_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_n_classes)
        )

    def forward(self, seg_feats: torch.Tensor, clin_feats: torch.Tensor):
        B = seg_feats.size(0)
        seg_tokens = self.seg_proj(seg_feats).unsqueeze(1)
        clin_tokens = self.clin_proj(clin_feats).unsqueeze(1)
        seq = torch.cat([seg_tokens, clin_tokens], dim=1)
        pos = self.pos_emb(torch.arange(seq.size(1), device=seq.device)).unsqueeze(0)
        seq = seq + pos
        seq = self.mamba(seq)
        seq = self.norm(seq)
        pooled = seq.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.t_head(pooled), self.n_head(pooled)


class FeatureGroupMamba(nn.Module):
    """Mamba TN staging over grouped tabular features as a token sequence.

    Groups 122+ features into logical tokens (geometric, radiomics_p,
    radiomics_n, clinical, rules) so Mamba models cross-group interactions.

    Args:
        group_dims: dict mapping group name -> feature count.
            E.g. {"geometric": 31, "rad_p": 30, "rad_n": 30, "clinical": 25, "rules": 10}
        d_model: Mamba hidden dim
        num_t_classes: T-stage classes (default 4)
        num_n_classes: N-stage classes (default 4)
        dropout: dropout rate
    """
    def __init__(self, group_dims: Dict[str, int], d_model: int = 64,
                 num_t_classes: int = 4, num_n_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.group_names = list(group_dims.keys())
        self.n_groups = len(self.group_names)
        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU()
            )
            for name, dim in group_dims.items()
        })
        self.pos_emb = nn.Embedding(self.n_groups + 4, d_model)
        self.mamba = _make_mamba(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.t_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_t_classes)
        )
        self.n_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_n_classes)
        )

    def forward(self, feats_dict: Dict[str, torch.Tensor]):
        B = next(iter(feats_dict.values())).size(0)
        tokens = []
        for i, name in enumerate(self.group_names):
            x = self.proj[name](feats_dict[name])
            tokens.append(x.unsqueeze(1))
        seq = torch.cat(tokens, dim=1)
        pos = self.pos_emb(torch.arange(seq.size(1), device=seq.device)).unsqueeze(0)
        seq = seq + pos
        seq = self.mamba(seq)
        seq = self.norm(seq)
        pooled = seq.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.t_head(pooled), self.n_head(pooled)


def extract_seg_features(pred_mask: np.ndarray, pet_volume: np.ndarray,
                          voxel_spacing_mm: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    voxel_vol = float(np.prod(voxel_spacing_mm))

    gtvp_mask = (pred_mask == 1).astype(np.uint8)
    gtvn_mask = (pred_mask == 2).astype(np.uint8)

    gtvp_vol = gtvp_mask.sum() * voxel_vol
    gtvn_vol = gtvn_mask.sum() * voxel_vol

    if gtvp_mask.sum() > 0:
        gtvp_suvmax = float(pet_volume[gtvp_mask > 0].max())
        coords = np.argwhere(gtvp_mask > 0)
        centroid = (coords.mean(axis=0) * voxel_spacing_mm).tolist()
    else:
        gtvp_suvmax = 0.0
        centroid = [0.0, 0.0, 0.0]

    if gtvn_mask.sum() > 0:
        gtvn_suvmax = float(pet_volume[gtvn_mask > 0].max())
        _, n_comps = ndimage.label(gtvn_mask)
    else:
        gtvn_suvmax = 0.0
        n_comps = 0

    feats = np.array([
        gtvp_vol, gtvn_vol,
        gtvp_suvmax, gtvn_suvmax,
        centroid[0], centroid[1], centroid[2],
        float(n_comps),
    ], dtype=np.float32)
    return feats