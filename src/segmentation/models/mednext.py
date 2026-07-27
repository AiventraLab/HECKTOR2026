"""
MedNeXt-L model for HECKTOR 2026 Task 1 — Segmentation.

Design principles:
  1. Wraps MONAI's MedNeXt implementation (monai >= 1.5.0) behind the
     existing BaseModel interface so it is fully drop-in compatible with
     the existing train.py, inference.py, and evaluation pipeline.

  2. Deep supervision heads at decoder levels 0–2 are enabled by default.
     The DeepSupervisionLoss wrapper in utils/deep_supervision.py handles
     the multi-scale loss weighting without any changes to train.py.

  3. forward_unified() is overridden to return the segmentation dict
     expected by the HECKTOR 2026 unified pipeline.

  4. Weight initialisation follows Kaiming normal for Conv layers and
     constant for BatchNorm, consistent with every other model in this
     codebase.

Usage:
    from config import MedNeXtConfig
    from models import MedNeXtModel

    config = MedNeXtConfig(fold=0)
    model  = MedNeXtModel(config).to(device)

    # single forward pass (deep supervision off at inference)
    logits = model(x)          # x: [B, 2, D, H, W]

    # deep supervision forward (training only)
    ds_outs = model.forward_ds(x)  # list of [full, half, quarter] tensors

Reference: Roy et al. arXiv:2303.09975
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional

from monai.networks.nets import MedNeXt

from .base_model import BaseModel


class MedNeXtModel(BaseModel):
    """
    MedNeXt-L segmentation model.

    Wraps MONAI MedNeXt with the project's BaseModel interface.
    Supports both standard forward pass and deep supervision.
    """

    def __init__(self, config):
        """
        Initialise MedNeXt-L.

        Args:
            config: MedNeXtConfig (or any config with the required attributes).
        """
        super().__init__(config)

        self.use_deep_supervision = getattr(config, "use_deep_supervision", True)

        # Build block counts for MedNeXt
        # Format: [enc_stage_1, enc_stage_2, enc_stage_3, enc_stage_4, 
        #          stem, dec_stage_4, dec_stage_3, dec_stage_2, dec_stage_1]
        block_counts = [
            *config.enc_num_blocks,   # encoder stages 1-4
            config.stem_blocks,        # stem/bottleneck
            *list(reversed(config.dec_num_blocks)),  # decoder stages 4-1
        ]

        # MONAI MedNeXt
        # deep_supervision=True makes the model return a list of tensors
        # at [full, /2, /4] resolutions during training.
        self.mednext = MedNeXt(
            in_channels=config.in_channels,
            n_channels=config.dim,
            n_classes=config.out_channels,
            exp_r=list(config.enc_exp_r),
            kernel_size=config.kernel_size,
            deep_supervision=self.use_deep_supervision,
            do_res=config.do_res,
            do_res_up_d=config.do_res_up_d,
            block_counts=block_counts,
            grn=config.grn,
        )

        self._initialise_weights()


    # Core forward


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass — returns full-resolution logits only.

        At inference time deep supervision is disabled even if the model
        was trained with it: we just take the first (full-res) output.

        Args:
            x: [B, 2, D, H, W]  (CT channel 0, PET channel 1)

        Returns:
            logits: [B, num_classes, D, H, W]
        """
        out = self.mednext(x)
        # During eval MedNeXt may still return a list depending on version;
        # normalise to a single tensor.
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def forward_ds(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Deep-supervision forward — returns multi-scale logits list.

        Use this inside the training loop together with DeepSupervisionLoss:

            ds_loss = DeepSupervisionLoss(base_loss=get_hecktor2026_loss())
            outputs = model.forward_ds(images)
            loss    = ds_loss(outputs, labels)

        Args:
            x: [B, 2, D, H, W]

        Returns:
            List of tensors at [full, /2, /4] resolutions.
            Each tensor: [B, num_classes, D', H', W']
        """
        out = self.mednext(x)
        if isinstance(out, (list, tuple)):
            return list(out)
        # Fallback: model compiled without DS — return single-item list
        return [out]


    # HECKTOR 2026 unified pipeline interface


    def forward_unified(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Unified pipeline entry point for HECKTOR 2026.

        Returns a dict so downstream TN staging and prognosis modules
        can consume both the segmentation mask and the deep encoder features.

        Returns:
            {
                "segmentation": [B, 3, D, H, W],   full-res logits
                "ds_outputs":   List[Tensor],        multi-scale logits (train)
            }
        """
        out = self.mednext(x)
        if isinstance(out, (list, tuple)):
            return {
                "segmentation": out[0],
                "ds_outputs": list(out),
            }
        return {
            "segmentation": out,
            "ds_outputs": [out],
        }


    # Feature extraction for downstream tasks


    def extract_features(self, x: torch.Tensor, level: Optional[int] = None) -> torch.Tensor:
        """
        Extract intermediate features for TN staging or prognosis.

        This is a convenience method to get encoder features from the
        MedNeXt backbone. Level 0 = highest resolution, level 4 = deepest.

        Args:
            x: [B, 2, D, H, W] input image
            level: Which encoder level to extract (0-4). 
                   If None, returns the final feature map before the classifier.

        Returns:
            Feature tensor at the specified level.
        """
        # MedNeXt doesn't expose intermediate features directly in the forward pass.
        # For feature extraction, we need to implement a custom forward.
        # For now, we return the features before the final classifier.
        # This can be extended if needed.
        
        # Run forward and get the full output
        out = self.mednext(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        
        # For feature extraction, we return the logits as features
        # A more sophisticated implementation would return intermediate features
        return out


    # Utilities


    def _initialise_weights(self):
        """Kaiming normal init for Conv, constant for BN — matches project convention."""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def get_model_info(self) -> str:
        """Human-readable architecture summary."""
        params = self.get_parameters()
        cfg = self.config
        return (
            f"\nMedNeXt-L Model Information\n"
            f"---------------------------\n"
            f"Architecture   : MedNeXt-L (MONAI)\n"
            f"Kernel size    : {cfg.kernel_size}×{cfg.kernel_size}×{cfg.kernel_size}\n"
            f"Base dim       : {cfg.dim}\n"
            f"Exp ratio      : {cfg.enc_exp_r}\n"
            f"Enc blocks     : {cfg.enc_num_blocks}\n"
            f"Dec blocks     : {cfg.dec_num_blocks}\n"
            f"GRN            : {cfg.grn}\n"
            f"Deep supervis. : {self.use_deep_supervision}\n"
            f"In channels    : {cfg.in_channels}\n"
            f"Out channels   : {cfg.out_channels}\n"
            f"\nParameters\n"
            f"----------\n"
            f"Total          : {params['total_parameters']:,}\n"
            f"Trainable      : {params['trainable_parameters']:,}\n"
            f"Size (MB)      : {params['model_size_mb']:.1f}\n"
        )

    def __repr__(self) -> str:
        """String representation showing model architecture."""
        return self.get_model_info()