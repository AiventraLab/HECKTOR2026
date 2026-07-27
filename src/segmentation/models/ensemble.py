"""Ensemble the segmentation arms by averaging softmax probabilities, then
optional node-aware post-processing.

Diversity (nnU-Net ResEnc + MedNeXt + SegResNet) is the single biggest lever in
HECKTOR segmentation - different architectures make different errors, so the
average is more robust than any one. Validate that the ensemble actually beats
the best single model (it sometimes doesn't).
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage


def average_probs(prob_list, weights=None) -> np.ndarray:
    """prob_list: [P_i] each [C,H,W,D]. Returns argmax label map [H,W,D]."""
    weights = weights or [1.0] * len(prob_list)
    acc = np.zeros_like(prob_list[0], dtype=np.float32)
    for p, w in zip(prob_list, weights):
        acc += w * p
    return np.argmax(acc, axis=0).astype(np.uint8)


def postprocess(label: np.ndarray, pet_suv: np.ndarray | None = None,
                min_node_voxels=20, min_suv=2.0) -> np.ndarray:
    """Drop tiny / low-uptake spurious node components (false positives).
    Keep GTVp untouched; only filter GTVn (label 2). Conservative thresholds -
    over-pruning hurts true small nodes."""
    out = label.copy()
    cc, n = ndimage.label(out == 2)
    for i in range(1, n + 1):
        m = cc == i
        if m.sum() < min_node_voxels:
            out[m] = 0
            continue
        if pet_suv is not None and pet_suv[m].max() < min_suv:
            out[m] = 0
    return out


class SegmentationEnsemble:
    """Wrapper that averages softmax from multiple segmentation arms."""

    def __init__(self, models: dict):
        self.models = models

    def predict_prob(self, image):
        probs = []
        for name, m in self.models.items():
            if hasattr(m, "predict_prob"):
                probs.append(m.predict_prob(image))
        if not probs:
            raise RuntimeError("No model returned probabilities")
        w = 1.0 / len(probs)
        acc = np.zeros_like(probs[0], dtype=np.float32)
        for p in probs:
            acc += w * p
        return acc

    def postprocess(self, label, pet_suv=None, **kw):
        return postprocess(label, pet_suv=pet_suv, **kw)
