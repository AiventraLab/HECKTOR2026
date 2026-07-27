"""Derive tabular features from the predicted (or ground-truth) GTVp/GTVn masks
+ PET/CT. These feed the TN-staging and prognosis models.

Two families:
  - geometric / clinically-aligned: volume, MTV, SUV stats, node count/size,
    laterality — these mirror how T (tumor extent) and N (nodal burden) are
    *defined*, so they are highly predictive and data-efficient.
  - radiomics (PyRadiomics): shape + first-order + texture per modality/ROI.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

from src.shared.columns import LABEL_GTVP, LABEL_GTVN

# AJCC 8th-ed oropharynx thresholds (cm): T by primary longest dimension,
# N by largest-node longest dimension. Used as ordinal-prior features.
_T_THRESH = [(2.0, 1), (4.0, 2), (6.0, 3)]   # else 4
_N_THRESH = [(0.0, 0), (3.0, 1), (6.0, 2)]   # >6 -> 3 ; 0 -> N0


def _longest_diameter_mm(mask, spacing_mm):
    """True max-Feret diameter (mm): max distance between any two ROI voxels,
    computed on the convex-hull vertices (cheap, exact for the extremes).
    AJCC T/N staging is defined by this longest dimension, NOT sphere-equiv."""
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    pts = idx.astype(np.float64) * np.asarray(spacing_mm)   # (z,y,x) mm
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import pdist
        v = pts[ConvexHull(pts).vertices]
        return float(pdist(v).max())
    except Exception:
        # fallback: bounding-box diagonal in mm
        ext = (idx.max(0) - idx.min(0)).astype(np.float64) * np.asarray(spacing_mm)
        return float(np.sqrt((ext ** 2).sum()))


def _t_rule_index(diam_cm):
    for thr, idx in _T_THRESH:
        if diam_cm <= thr:
            return idx
    return 4


def _n_rule_index(diam_cm):
    if diam_cm <= 0:
        return 0
    for thr, idx in ((3.0, 1), (6.0, 2)):
        if diam_cm <= thr:
            return idx
    return 3


# ----------------------------------------------------------------------------
# Geometric / SUV features
# ----------------------------------------------------------------------------
def geometric_features(label: np.ndarray, pet_suv: np.ndarray,
                       spacing_mm=(1.0, 1.0, 1.0), suv_thr=2.5) -> dict:
    """label, pet_suv: numpy (z,y,x). spacing in mm. suv_thr defines MTV."""
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0   # mm^3 -> mL
    f = {}
    for name, lab in (("gtvp", LABEL_GTVP), ("gtvn", LABEL_GTVN)):
        m = label == lab
        vox = int(m.sum())
        f[f"{name}_volume_ml"] = vox * voxel_ml
        if vox:
            suv = pet_suv[m]
            f[f"{name}_suv_max"] = float(suv.max())
            f[f"{name}_suv_mean"] = float(suv.mean())
            f[f"{name}_suv_peak"] = float(np.percentile(suv, 95))
            mtv = m & (pet_suv >= suv_thr)
            f[f"{name}_mtv_ml"] = int(mtv.sum()) * voxel_ml
            f[f"{name}_tlg"] = float(pet_suv[mtv].sum()) * voxel_ml  # total lesion glycolysis
        else:
            for k in ("suv_max", "suv_mean", "suv_peak", "mtv_ml", "tlg"):
                f[f"{name}_{k}"] = 0.0
    # nodal burden (drives N-stage)
    cc, nnodes = ndimage.label(label == LABEL_GTVN)
    f["n_nodes"] = int(nnodes)
    if nnodes:
        sizes = ndimage.sum(np.ones_like(cc), cc, index=range(1, nnodes + 1))
        f["largest_node_ml"] = float(sizes.max()) * voxel_ml
        f["gtvn_total_volume_ml"] = float(sum(sizes)) * voxel_ml
        f["gtvn_mean_node_ml"] = float(np.mean(sizes)) * voxel_ml
        # laterality: nodes left vs right of the volume mid-plane (x axis)
        cx = label.shape[2] / 2.0
        coms = ndimage.center_of_mass(label == LABEL_GTVN, cc, index=range(1, nnodes + 1))
        xs = [c[2] for c in coms]
        f["nodes_left"] = int(sum(x < cx for x in xs))
        f["nodes_right"] = int(sum(x >= cx for x in xs))
        f["bilateral"] = int(f["nodes_left"] > 0 and f["nodes_right"] > 0)
        # largest node longest dimension (AJCC N metric)
        largest_lab = int(np.argmax(sizes)) + 1
        f["gtvn_largest_diameter_cm"] = _longest_diameter_mm(cc == largest_lab, spacing_mm) / 10.0
    else:
        f.update(largest_node_ml=0.0, gtvn_total_volume_ml=0.0,
                 gtvn_mean_node_ml=0.0, nodes_left=0, nodes_right=0, bilateral=0,
                 gtvn_largest_diameter_cm=0.0)
    f["n_rule_index"] = _n_rule_index(f["gtvn_largest_diameter_cm"])

    # GTVp-to-GTVn centroid distance (mm) — strong N-stage / prognosis predictor
    if (label == LABEL_GTVP).any() and (label == LABEL_GTVN).any():
        com_p = ndimage.center_of_mass(label == LABEL_GTVP)
        com_n = ndimage.center_of_mass(label == LABEL_GTVN)
        dz = (com_p[0] - com_n[0]) * spacing_mm[0]
        dy = (com_p[1] - com_n[1]) * spacing_mm[1]
        dx = (com_p[2] - com_n[2]) * spacing_mm[2]
        f["gtvp_gtvn_distance_mm"] = float(np.sqrt(dz**2 + dy**2 + dx**2))
    else:
        f["gtvp_gtvn_distance_mm"] = 0.0

    # ---- derived shape / combined features ----
    gtvp_vol = f["gtvp_volume_ml"]
    gtvn_vol = f["gtvn_total_volume_ml"]
    f["total_tumor_volume_ml"] = gtvp_vol + gtvn_vol

    # estimated equivalent sphere diameter (cm) — kept for backward compat
    if gtvp_vol > 0:
        f["gtvp_diameter_cm"] = 2.0 * ((3.0 * gtvp_vol / (4.0 * np.pi)) ** (1.0 / 3.0))
    else:
        f["gtvp_diameter_cm"] = 0.0

    # TRUE longest dimension of GTVp (cm) — the actual AJCC T-stage metric —
    # plus the rule-based ordinal T index derived from it.
    f["gtvp_longest_diameter_cm"] = _longest_diameter_mm(label == LABEL_GTVP, spacing_mm) / 10.0
    f["t_rule_index"] = _t_rule_index(f["gtvp_longest_diameter_cm"])

    # metabolic density (TLG per mL) — higher = more aggressive
    if gtvp_vol > 0:
        f["gtvp_tlg_density"] = f["gtvp_tlg"] / gtvp_vol
        f["gtvp_mtv_fraction"] = f["gtvp_mtv_ml"] / gtvp_vol
    else:
        f["gtvp_tlg_density"] = 0.0
        f["gtvp_mtv_fraction"] = 0.0

    if gtvn_vol > 0:
        f["gtvn_tlg_density"] = f["gtvn_tlg"] / gtvn_vol
    else:
        f["gtvn_tlg_density"] = 0.0

    # SUV heterogeneity (max/mean ratio)
    if f["gtvp_suv_mean"] > 0:
        f["gtvp_suv_heterogeneity"] = f["gtvp_suv_max"] / f["gtvp_suv_mean"]
    else:
        f["gtvp_suv_heterogeneity"] = 1.0

    # surface area proxy via voxel boundary count (for sphericity)
    # guard + fallbacks MUST match inference.py::_geometric_features exactly
    m_p = label == LABEL_GTVP
    if m_p.any() and label.size <= 200_000_000:
        eroded = ndimage.binary_erosion(m_p)
        boundary_vox = int((m_p & ~eroded).sum())
        sa_proxy_mm2 = boundary_vox * float((spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) ** (2.0/3.0))
        vol_mm3 = gtvp_vol * 1000.0
        if sa_proxy_mm2 > 0:
            # sphericity: ratio of sphere surface for same volume to actual surface
            f["gtvp_sphericity"] = (np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0)) / sa_proxy_mm2
        else:
            f["gtvp_sphericity"] = 1.0
    elif m_p.any():
        f["gtvp_sphericity"] = 0.8
    else:
        f["gtvp_sphericity"] = 0.0

    return f


# ----------------------------------------------------------------------------
# PyRadiomics
# ----------------------------------------------------------------------------
def _extractor(modality: str):
    from radiomics import featureextractor
    settings = {
        "binWidth": 0.5 if modality == "PET" else 25,
        "resampledPixelSpacing": None,        # already resampled upstream
        "normalize": modality == "PET",
        "geometryTolerance": 1e-4,
        "label": 1,
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    return ext


def radiomics_features(image_sitk, mask_sitk, modality: str, roi_label=1,
                       prefix="") -> dict:
    """Extract PyRadiomics features for one (image, binary-mask) pair.
    ``mask_sitk`` must be a binary mask (1 = ROI) sharing image geometry."""
    import SimpleITK as sitk
    ext = _extractor(modality)
    res = ext.execute(image_sitk, mask_sitk, label=roi_label)
    return {f"{prefix}{k}": float(v) for k, v in res.items()
            if not k.startswith("diagnostics_")}
