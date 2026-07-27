"""
Deep Supervision for HECKTOR 2026 Task 1.

Evidence: Used by NVAUTO (winner 2022, DSC 0.788), SegResNet auto-encoder
variant, and all top-5 HECKTOR 2022-2025 teams.

How it works: auxiliary segmentation heads at decoder levels 1, 2, 3.
Losses weighted [1.0, 0.5, 0.25] — full resolution gets highest weight.
Gradients flow to early decoder layers, preventing vanishing gradient
in deep 3D networks.

Reference: Myronenko et al. arXiv:2209.10809; Isensee nnU-Net 2021.
"""

import torch
import torch.nn as nn


class DeepSupervisionLoss(nn.Module):
    """
    Wraps any base loss with deep supervision.

    Usage in train loop:
        ds_loss = DeepSupervisionLoss(base_loss=get_hecktor2026_loss())
        # model must return list of logits at different scales
        loss = ds_loss(outputs, target)  # outputs = [full, half, quarter]
    """

    def __init__(self, base_loss, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.base_loss = base_loss
        self.weights   = weights

    def forward(self, outputs, target):
        """
        Args:
            outputs: list of tensors [B, C, D, H, W] at decreasing resolutions
                     OR single tensor (falls back to normal loss)
            target:  ground truth [B, 1, D, H, W] or [B, D, H, W]
        """
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target)

        total = 0.0
        for i, (out, w) in enumerate(zip(outputs, self.weights)):
            # Downsample target to match this output scale
            if out.shape[2:] != target.shape[-3:]:
                tgt = nn.functional.interpolate(
                    target.float().unsqueeze(1) if target.dim() == 4 else target.float(),
                    size=out.shape[2:],
                    mode="nearest"
                ).long().squeeze(1)
            else:
                tgt = target
            total = total + w * self.base_loss(out, tgt)

        return total / sum(self.weights[:len(outputs)])
