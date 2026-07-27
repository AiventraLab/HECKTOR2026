"""
MedNeXt-L configuration for Segmentation. 

Architecture rationale:
    MedNeXt-L with 5x5x5 kernels is the state-of-the-art CNN backbone for 
    3D medical image segmentation. In rigorous benchmarking (Isensee et al.
    MICCAI 2024), it consistently outperforms nnU-Net, Transformers, and
    Mamba across BraTS, AMOS, BTCV, and LiTS. For head-and-neck PET/CT
    the analogous HNTS-MRG 2024 challenge found ResEnc+MedNeXt ensemble
    to be the winning approach.

Reference: Roy et al. "MedNeXt" MICCAI 2023. arXiv:2303.09975
"""

from dataclasses import dataclass, field 
from typing import Tuple
from src.segmentation.config.base_config import BaseConfig


@dataclass
class MedNeXtConfig(BaseConfig):


    experiment_name: str = "mednext_l"

    #   MedNeXt-L: large kernel (5×5×5), expansion ratio 4, depth 2 per stage
    in_channels: int = 2             # CT + PET (2-channel input)
    out_channels: int = 3            # background / GTVp / GTVn
    kernel_size: int = 5             # hallmark of MedNeXt-L; do NOT reduce
    enc_exp_r: Tuple = field(default_factory=lambda: (4, 4, 4, 4))
    dec_exp_r: Tuple = field(default_factory=lambda: (4, 4, 4, 4))
    enc_kernel_size: Tuple = field(default_factory=lambda: (5, 5, 5, 5))
    dec_kernel_size: Tuple = field(default_factory=lambda: (5, 5, 5, 5))
    enc_num_blocks: Tuple = field(default_factory=lambda: (2, 2, 2, 2))
    dec_num_blocks: Tuple = field(default_factory=lambda: (2, 2, 2, 2))
    stem_blocks: int = 2
    dim: int = 32                    # base feature dimension (MedNeXt-L default)
    do_res: bool = True              # residual connections in encoder/decoder
    do_res_up_d: bool = True         # residual connections in upsampler
    grn: bool = True                 # Global Response Normalization (MedNeXt v2)

    spatial_size: Tuple = (128, 128, 128)

    batch_size: int = 2
    learning_rate: float = 1e-4      # AdamW; lower than baseline SegResNet (1e-2)
    weight_decay: float = 1e-5
    num_epochs: int = 500            # MedNeXt needs longer training than UNet3D
    poly_lr_power: float = 0.9
    poly_lr_min_lr: float = 1e-7

    use_augmentation: bool = True
    aug_probability: float = 0.5
    num_samples: int = 2             # patches per image per RandCropByLabel call

    cache_rate: float = 0.25         # increase to 1.0 if RAM > 256 GB
    num_workers: int = 4

    use_deep_supervision: bool = True   # enabled by default; top teams use it
    deep_supervision_weights: Tuple = (1.0, 0.5, 0.25)

    save_checkpoint_every: int = 1
    use_tensorboard: bool = True