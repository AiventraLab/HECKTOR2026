"""
Post-processing for HECKTOR 2026 Task 1 segmentation outputs.

Evidence base:
- Salahuddin et al. 2022 (3rd place): outlier removal on z-axis
- Myronenko NVAUTO 2022: median filtering on resampled masks
- Sun et al. 2022 (2nd place): coarse-to-fine cascade removes FP nodes
- General finding across all years: small disconnected components are
  almost always false positives, especially for GTVn

Strategy:
  1. Keep only the largest connected component for GTVp (single primary tumor)
  2. For GTVn: remove components < min_node_voxels (paper rule: nodes
     must be >= 1cm diameter → at 1x1x1mm = ~524 voxels for sphere)
  3. Optional: morphological closing to fill small holes in GTVp

Reference: arXiv:2209.10809; Andrearczyk et al. HECKTOR 2022 overview.
"""

import numpy as np
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def remove_small_components(
    pred_array: np.ndarray,
    gtvp_label: int = 1,
    gtvn_label: int = 2,
    min_gtvp_voxels: int = 100,
    min_gtvn_voxels: int = 500,   # ~1cm sphere at 1x1x1mm = 524 voxels
    keep_largest_gtvp: bool = True,
) -> np.ndarray:
    """
    Remove anatomically implausible small components from segmentation.

    Args:
        pred_array:        [D, H, W] integer array (0=bg, 1=GTVp, 2=GTVn)
        min_gtvp_voxels:   minimum GTVp component size to keep
        min_gtvn_voxels:   minimum GTVn component size (paper: >= 1cm diameter)
        keep_largest_gtvp: if True, keep only largest GTVp component
                           (primary tumor is a single lesion per patient)

    Returns:
        Cleaned [D, H, W] array
    """
    if not HAS_SCIPY:
        print("scipy not available — skipping post-processing")
        return pred_array

    result = pred_array.copy()

    # --- GTVp: single primary tumor → keep only largest component ---
    gtvp_mask = (pred_array == gtvp_label).astype(np.uint8)
    if gtvp_mask.sum() > 0:
        labeled, n_comp = ndimage.label(gtvp_mask)
        if n_comp > 1:
            sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
            if keep_largest_gtvp:
                # Keep only the largest
                largest = np.argmax(sizes) + 1
                result[labeled != largest] = np.where(
                    result[labeled != largest] == gtvp_label, 0,
                    result[labeled != largest]
                )
            else:
                # Remove components below threshold
                for i, sz in enumerate(sizes, start=1):
                    if sz < min_gtvp_voxels:
                        result[labeled == i] = 0

    # --- GTVn: multiple nodes allowed, but each must be >= min size ---
    gtvn_mask = (pred_array == gtvn_label).astype(np.uint8)
    if gtvn_mask.sum() > 0:
        labeled, n_comp = ndimage.label(gtvn_mask)
        for i in range(1, n_comp + 1):
            comp_size = (labeled == i).sum()
            if comp_size < min_gtvn_voxels:
                # Remove this component — too small to be a real node
                result[labeled == i] = np.where(
                    result[labeled == i] == gtvn_label, 0,
                    result[labeled == i]
                )

    return result


def morphological_closing(
    pred_array: np.ndarray,
    label: int = 1,
    closing_radius: int = 2,
) -> np.ndarray:
    """
    Binary morphological closing to fill small holes in GTVp.
    Closing radius of 2 voxels at 1x1x1mm is clinically safe.
    """
    if not HAS_SCIPY:
        return pred_array

    result = pred_array.copy()
    mask = (pred_array == label).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, closing_radius)
    closed = ndimage.binary_closing(mask, structure=struct).astype(np.uint8)
    result[closed == 1] = label
    return result


def postprocess_segmentation(
    pred_array: np.ndarray,
    apply_closing: bool = True,
) -> np.ndarray:
    """
    Full post-processing pipeline for HECKTOR 2026 Task 1.

    Call this on raw argmax output before saving/evaluating:
        pred = model_output.argmax(dim=1).cpu().numpy()[0]  # [D,H,W]
        pred_clean = postprocess_segmentation(pred)
    """
    pred = remove_small_components(pred_array)
    if apply_closing:
        pred = morphological_closing(pred, label=1, closing_radius=2)
    return pred
