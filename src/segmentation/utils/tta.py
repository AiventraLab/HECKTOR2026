"""
Test-Time Augmentation (TTA) for HECKTOR 2026 Task 1 inference.

Evidence base:
- NVAUTO 2022 (winner): flips + rotations at TTA
- Salahuddin et al. 2022: 8-orientation TTA
- Isensee nnU-Net: mirror TTA as default (all 8 flip combinations)
- Expected gain: +0.005 to +0.015 DSC at inference — free improvement

Strategy: all 8 axis-flip combinations (2^3), average softmax outputs.
No rotation TTA — rotation is expensive and gains are marginal vs flips.

Reference: arXiv:2209.10809; nnU-Net paper Isensee 2021 Nature Methods.
"""

import torch
import torch.nn.functional as F
from itertools import product


def tta_predict(model, inputs: dict, device: str = "cuda") -> torch.Tensor:
    """
    8-flip TTA inference for HECKTOR segmentation.

    Args:
        model:  trained segmentation model
        inputs: dict with keys "ct" [B,1,D,H,W] and "pet" [B,1,D,H,W]
                OR single tensor [B,2,D,H,W]
        device: cuda or cpu

    Returns:
        Averaged softmax probability map [B, C, D, H, W]
    """
    model.eval()
    flip_axes_list = list(product([False, True], repeat=3))  # 8 combinations

    accum = None

    with torch.no_grad():
        for flip_x, flip_y, flip_z in flip_axes_list:
            # Build flipped input
            if isinstance(inputs, dict):
                ct  = inputs["ct"].to(device)
                pet = inputs["pet"].to(device)
                x = torch.cat([ct, pet], dim=1)
            else:
                x = inputs.to(device)

            # Apply flips
            axes = []
            if flip_x: axes.append(2)
            if flip_y: axes.append(3)
            if flip_z: axes.append(4)
            if axes:
                x = torch.flip(x, dims=axes)

            logits = model(x)
            # Handle deep supervision output (take first/full-res)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            probs = F.softmax(logits, dim=1)

            # Unflip the probability map
            if axes:
                probs = torch.flip(probs, dims=axes)

            if accum is None:
                accum = probs
            else:
                accum = accum + probs

    return accum / len(flip_axes_list)  # averaged softmax


def tta_sliding_window(model, image: torch.Tensor,
                        roi_size=(128, 128, 128),
                        sw_batch_size=2,
                        overlap=0.5,
                        device="cuda") -> torch.Tensor:
    """
    Sliding window inference with TTA.
    Combines MONAI sliding_window_inference with 8-flip TTA.

    Args:
        model:       trained model
        image:       [B, 2, D, H, W] (CT+PET concatenated)
        roi_size:    patch size (match training spatial_size)
        sw_batch_size: patches per forward pass
        overlap:     sliding window overlap (0.5 recommended)

    Returns:
        [B, C, D, H, W] averaged prediction
    """
    from monai.inferers import sliding_window_inference

    flip_axes_list = list(product([False, True], repeat=3))
    accum = None

    image = image.to(device)
    model.eval()

    with torch.no_grad():
        for flip_x, flip_y, flip_z in flip_axes_list:
            axes = []
            if flip_x: axes.append(2)
            if flip_y: axes.append(3)
            if flip_z: axes.append(4)

            x = torch.flip(image, dims=axes) if axes else image

            pred = sliding_window_inference(
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",   # gaussian weighting reduces boundary artifacts
            )

            # Handle deep supervision
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            pred = F.softmax(pred, dim=1)

            if axes:
                pred = torch.flip(pred, dims=axes)

            accum = pred if accum is None else accum + pred

    return accum / len(flip_axes_list)
