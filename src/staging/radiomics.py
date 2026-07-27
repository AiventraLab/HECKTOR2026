"""
Hand-crafted radiomics features using numpy/scipy/skimage.
Extracts first-order statistics + GLCM texture on PET and CT within ROI.
Proven predictors in HNC literature for TN staging and RFS prognosis.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops


# ── First-order statistics ────────────────────────────────────────────────────
def first_order(values: np.ndarray) -> dict:
    """First-order stats on 1-D intensity array within ROI."""
    if len(values) == 0:
        return {k: 0.0 for k in ("fo_mean","fo_std","fo_skewness","fo_kurtosis",
                                  "fo_entropy","fo_energy","fo_p10","fo_p90","fo_iqr")}
    from scipy.stats import skew, kurtosis
    v = values.astype(np.float64)
    hist, _ = np.histogram(v, bins=64, density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
    return {
        "fo_mean":     float(v.mean()),
        "fo_std":      float(v.std()),
        "fo_skewness": float(skew(v)),
        "fo_kurtosis": float(kurtosis(v)),
        "fo_entropy":  entropy,
        "fo_energy":   float((v**2).sum() / len(v)),
        "fo_p10":      float(np.percentile(v, 10)),
        "fo_p90":      float(np.percentile(v, 90)),
        "fo_iqr":      float(np.percentile(v, 75) - np.percentile(v, 25)),
    }


# ── GLCM texture ─────────────────────────────────────────────────────────────
def glcm_features(img_roi: np.ndarray, n_levels: int = 32) -> dict:
    """GLCM texture features on a 3-D ROI patch (uses max 2-D slice for speed)."""
    zero = {k: 0.0 for k in ("glcm_contrast","glcm_dissimilarity",
                               "glcm_homogeneity","glcm_energy","glcm_correlation")}
    if img_roi.size == 0:
        return zero
    # Use central slice along z for speed (acceptable approximation)
    mid = img_roi.shape[0] // 2
    sl = img_roi[mid].astype(np.float64)
    if sl.std() < 1e-6:
        return zero
    # Quantise to n_levels
    mn, mx = sl.min(), sl.max()
    if mx == mn:
        return zero
    sl_q = ((sl - mn) / (mx - mn) * (n_levels - 1)).astype(np.uint8)
    glcm = graycomatrix(sl_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=n_levels, symmetric=True, normed=True)
    props = {}
    for prop in ("contrast","dissimilarity","homogeneity","energy","correlation"):
        props[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    return props


# ── Shape features (3D) ───────────────────────────────────────────────────────
def shape_features(mask: np.ndarray, spacing_mm=(1.0, 1.0, 1.0)) -> dict:
    """3-D shape features from binary mask."""
    vox_vol = float(np.prod(spacing_mm))
    vol_mm3 = float(mask.sum()) * vox_vol
    if vol_mm3 == 0:
        return {"shape_elongation": 0.0, "shape_flatness": 0.0,
                "shape_surface_vol_ratio": 0.0}
    # Inertia tensor → eigenvalues → elongation/flatness
    coords = np.argwhere(mask).astype(float)
    coords = coords * np.array(spacing_mm)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]  # descending
    elongation = float(np.sqrt(eigvals[1] / (eigvals[0] + 1e-9)))
    flatness   = float(np.sqrt(eigvals[2] / (eigvals[0] + 1e-9)))
    # Surface/volume ratio
    if mask.size <= 50_000_000:
        eroded = ndimage.binary_erosion(mask)
        surface_vox = int((mask & ~eroded).sum())
        # geometric-mean face area — correct for non-isotropic voxels
        sa_mm2 = surface_vox * float((spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) ** (2.0 / 3.0))
    else:
        sa_mm2 = 4 * np.pi * ((3 * vol_mm3 / (4 * np.pi)) ** (2/3))
    svr = float(sa_mm2 / (vol_mm3 + 1e-9))
    return {"shape_elongation": elongation, "shape_flatness": flatness,
            "shape_surface_vol_ratio": svr}


# ── ROI crop helper ───────────────────────────────────────────────────────────
def _crop_roi(img: np.ndarray, mask: np.ndarray, pad: int = 2):
    """Crop img to bounding box of mask with padding."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return img[:1,:1,:1], mask[:1,:1,:1]
    mn = np.maximum(idx.min(axis=0) - pad, 0)
    mx = np.minimum(idx.max(axis=0) + pad + 1, np.array(img.shape))
    sl = tuple(slice(mn[i], mx[i]) for i in range(3))
    return img[sl], mask[sl]


# ── Main entry point ──────────────────────────────────────────────────────────
def extract_radiomics(seg: np.ndarray, pet: np.ndarray, ct: np.ndarray,
                      spacing_mm=(1.0, 1.0, 1.0),
                      label_gtvp: int = 1, label_gtvn: int = 2) -> dict:
    """
    Extract radiomics features for GTVp and GTVn from PET+CT.
    Returns flat dict with prefix 'rad_p_' (GTVp) and 'rad_n_' (GTVn).
    """
    features = {}
    for label, prefix in [(label_gtvp, "rad_p_"), (label_gtvn, "rad_n_")]:
        mask = (seg == label)
        if not mask.any():
            # Zero-fill all features
            dummy = {**first_order(np.array([])),
                     **glcm_features(np.zeros((1,1,1))),
                     **shape_features(mask, spacing_mm)}
            features.update({prefix + k: v for k, v in dummy.items()})
            continue

        pet_crop, mask_crop = _crop_roi(pet, mask)
        ct_crop,  _         = _crop_roi(ct,  mask)

        pet_vals = pet[mask]
        ct_vals  = ct[mask]

        fo_pet = first_order(pet_vals)
        fo_ct  = first_order(ct_vals)
        glcm_p = glcm_features(pet_crop * mask_crop)
        glcm_c = glcm_features(ct_crop  * mask_crop)
        shp    = shape_features(mask, spacing_mm)

        combined = {}
        combined.update({f"pet_{k}": v for k, v in fo_pet.items()})
        combined.update({f"ct_{k}":  v for k, v in fo_ct.items()})
        combined.update({f"pet_{k}": v for k, v in glcm_p.items()})
        combined.update({f"ct_{k}":  v for k, v in glcm_c.items()})
        combined.update(shp)

        features.update({prefix + k: v for k, v in combined.items()})

    return features
