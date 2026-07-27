#!/usr/bin/env python3
"""
HECKTOR 2026 — Full Pipeline
=============================
Combined pipeline:
  1. Segmentation  : nnU-Net ResEnc + SegResNetDS ensemble (Rabin)
  2. TN Staging    : FeatureGroupMamba ensemble (Prabesh)
  3. Prognosis     : RSF + Cox ensemble (Rabin)

All three subtasks are chained end-to-end. Safe defaults on failure.
"""

from __future__ import annotations
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Memory hardening for limited-VRAM environments
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_GTVP = 1
LABEL_GTVN = 2
SUV_THR = 2.5
RANDOM_SEED = 42

# Geometric feature group names (must match staging model training order)
_G = {
    "geometric": [
        "gtvp_volume_ml","gtvp_suv_max","gtvp_suv_mean","gtvp_suv_peak",
        "gtvp_mtv_ml","gtvp_tlg","gtvn_volume_ml","gtvn_suv_max","gtvn_suv_mean",
        "gtvn_suv_peak","gtvn_mtv_ml","gtvn_tlg","n_nodes","largest_node_ml",
        "gtvn_total_volume_ml","gtvn_mean_node_ml","nodes_left","nodes_right",
        "bilateral","gtvn_largest_diameter_cm","n_rule_index","gtvp_gtvn_distance_mm",
        "total_tumor_volume_ml","gtvp_diameter_cm","gtvp_longest_diameter_cm",
        "t_rule_index","gtvp_tlg_density","gtvp_mtv_fraction","gtvn_tlg_density",
        "gtvp_suv_heterogeneity","gtvp_sphericity",
    ],
    "rad_p": [
        "rad_p_pet_fo_mean","rad_p_pet_fo_std","rad_p_pet_fo_skewness",
        "rad_p_pet_fo_kurtosis","rad_p_pet_fo_entropy","rad_p_pet_fo_energy",
        "rad_p_pet_fo_p10","rad_p_pet_fo_p90","rad_p_pet_fo_iqr",
        "rad_p_ct_fo_mean","rad_p_ct_fo_std","rad_p_ct_fo_skewness",
        "rad_p_ct_fo_kurtosis","rad_p_ct_fo_entropy","rad_p_ct_fo_energy",
        "rad_p_ct_fo_p10","rad_p_ct_fo_p90","rad_p_ct_fo_iqr",
        "rad_p_pet_glcm_contrast","rad_p_pet_glcm_dissimilarity",
        "rad_p_pet_glcm_homogeneity","rad_p_pet_glcm_energy",
        "rad_p_pet_glcm_correlation",
        "rad_p_ct_glcm_contrast","rad_p_ct_glcm_dissimilarity",
        "rad_p_ct_glcm_homogeneity","rad_p_ct_glcm_energy",
        "rad_p_ct_glcm_correlation",
        "rad_p_shape_elongation","rad_p_shape_flatness","rad_p_shape_surface_vol_ratio",
    ],
    "rad_n": [
        "rad_n_pet_fo_mean","rad_n_pet_fo_std","rad_n_pet_fo_skewness",
        "rad_n_pet_fo_kurtosis","rad_n_pet_fo_entropy","rad_n_pet_fo_energy",
        "rad_n_pet_fo_p10","rad_n_pet_fo_p90","rad_n_pet_fo_iqr",
        "rad_n_ct_fo_mean","rad_n_ct_fo_std","rad_n_ct_fo_skewness",
        "rad_n_ct_fo_kurtosis","rad_n_ct_fo_entropy","rad_n_ct_fo_energy",
        "rad_n_ct_fo_p10","rad_n_ct_fo_p90","rad_n_ct_fo_iqr",
        "rad_n_pet_glcm_contrast","rad_n_pet_glcm_dissimilarity",
        "rad_n_pet_glcm_homogeneity","rad_n_pet_glcm_energy",
        "rad_n_pet_glcm_correlation",
        "rad_n_ct_glcm_contrast","rad_n_ct_glcm_dissimilarity",
        "rad_n_ct_glcm_homogeneity","rad_n_ct_glcm_energy",
        "rad_n_ct_glcm_correlation",
        "rad_n_shape_elongation","rad_n_shape_flatness","rad_n_shape_surface_vol_ratio",
    ],
}
_C = [
    "Age","Gender_missing","Gender=0.0","Gender=1.0",
    "Tobacco Consumption_missing","Tobacco Consumption=0.0",
    "Tobacco Consumption=1.0","Alcohol Consumption_missing",
    "Alcohol Consumption=0.0","Alcohol Consumption=1.0",
    "Performance Status_missing","Performance Status=0.0",
    "Performance Status=1.0","Performance Status=2.0",
    "Performance Status=3.0","Performance Status=4.0",
    "HPV Status_missing","HPV Status=0.0","HPV Status=1.0",
    "Treatment_missing","Treatment=0.0","Treatment=1.0",
]
_R2 = [
    "t_rule_index","n_rule_index","gtvp_diameter_cm",
    "gtvp_longest_diameter_cm","gtvn_largest_diameter_cm",
    "n_nodes","bilateral",
]


# ===========================================================================
# Subtask 1 — Segmentation (Rabin: nnU-Net ResEnc + SegResNetDS ensemble)
# ===========================================================================
def _hn_body_centroid_xy(ct_arr, z0, z1, hu_thr=-500.0):
    slab = ct_arr[z0:z1]
    body = slab > hu_thr
    if not body.any():
        return ct_arr.shape[2] // 2, ct_arr.shape[1] // 2
    ys = np.where(body.any(axis=(0, 2)))[0]
    xs = np.where(body.any(axis=(0, 1)))[0]
    return int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2)


def _hn_pet_tumor_center(pet_arr, nz, sz):
    petr = np.clip(pet_arr, 0, None).copy()
    te = int(round(50.0 / sz))
    petr[nz - te:, :, :] = 0
    bot = max(0, nz - int(round(360.0 / sz)))
    petr[:bot, :, :] = 0
    if petr.max() <= 0:
        return None
    thr = np.percentile(petr[petr > 0], 99)
    zz, yy, xx = np.where(petr >= thr)
    return int(zz.mean()), int(yy.mean()), int(xx.mean())


def _hn_bbox_lps(ct_lps, label_lps_arr=None, pet_lps_arr=None):
    a = sitk.GetArrayFromImage(ct_lps)
    nz, ny, nx = a.shape
    sx, sy, sz = ct_lps.GetSpacing()
    bz = int(round(330.0 / sz))
    by = int(round(210.0 / sy))
    bx = int(round(210.0 / sx))
    if label_lps_arr is not None and label_lps_arr.max() > 0:
        zz, yy, xx = np.where(label_lps_arr > 0)
        cz = int((zz.min() + zz.max()) // 2)
        cy = int((yy.min() + yy.max()) // 2)
        cx = int((xx.min() + xx.max()) // 2)
        mz = int(round(20.0 / sz))
        my = int(round(20.0 / sy))
        mx = int(round(20.0 / sx))
        bz = max(bz, (zz.max() - zz.min()) + 2 * mz)
        by = max(by, (yy.max() - yy.min()) + 2 * my)
        bx = max(bx, (xx.max() - xx.min()) + 2 * mx)
    else:
        center = _hn_pet_tumor_center(pet_lps_arr, nz, sz) if pet_lps_arr is not None else None
        if center is not None:
            cz, cy, cx = center
        else:
            cz = (nz - 1 - int(round(20.0 / sz))) - bz // 2
            cx, cy = _hn_body_centroid_xy(a, max(0, nz - bz), nz)
    z0 = max(0, cz - bz // 2); z1 = min(nz, z0 + bz); z0 = max(0, z1 - bz)
    y0 = max(0, cy - by // 2); y1 = min(ny, y0 + by); y0 = max(0, y1 - by)
    x0 = max(0, cx - bx // 2); x1 = min(nx, x0 + bx); x0 = max(0, x1 - bx)
    return (z0, z1, y0, y1, x0, x1)


def _hn_crop_resample(img_lps, bbox, is_label):
    z0, z1, y0, y1, x0, x1 = bbox
    cropped = img_lps[x0:x1, y0:y1, z0:z1]
    osz = cropped.GetSize()
    osp = cropped.GetSpacing()
    nsz = [int(round(osz[i] * osp[i] / (1.0, 1.0, 3.0)[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((1.0, 1.0, 3.0))
    rs.SetSize(nsz)
    rs.SetOutputOrigin(cropped.GetOrigin())
    rs.SetOutputDirection(cropped.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    rs.SetDefaultPixelValue(0 if is_label else -1000)
    return rs.Execute(cropped)


def _hn_crop_case(ct, pet_on_ct, label=None):
    ct_l = sitk.DICOMOrient(ct, "LPS")
    pet_l = sitk.DICOMOrient(pet_on_ct, "LPS")
    lab_l = sitk.DICOMOrient(label, "LPS") if label is not None else None
    lab_arr = sitk.GetArrayFromImage(lab_l) if lab_l is not None else None
    pet_arr = sitk.GetArrayFromImage(pet_l) if label is None else None
    bbox = _hn_bbox_lps(ct_l, lab_arr, pet_arr)
    ct_roi = _hn_crop_resample(ct_l, bbox, is_label=False)
    pet_roi = _hn_crop_resample(pet_l, bbox, is_label=False)
    lab_roi = _hn_crop_resample(lab_l, bbox, is_label=True) if lab_l is not None else None
    return ct_roi, pet_roi, lab_roi


def _hn_map_roi_to_native(seg_roi_arr, ct_roi_ref, ct_native):
    seg_img = sitk.GetImageFromArray(seg_roi_arr.astype(np.uint8))
    seg_img.CopyInformation(ct_roi_ref)
    return sitk.Resample(seg_img, ct_native, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def _register_pet_to_ct(pet, ct):
    reg = sitk.Resample(pet, ct, sitk.Transform(),
                        sitk.sitkLinear, 0.0, pet.GetPixelID())
    return sitk.Clamp(reg, lowerBound=0.0)


def _run_nnunet(ct_roi, pet_roi, model_dir: Path, device):
    """nnU-Net ResEnc inference on the H&N ROI. Returns softmax (3,Z,Y,X)."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import torch

    nn_model = model_dir / "nnunet" / "Dataset021_HECKTOR2026" / \
               "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
    if not nn_model.exists():
        raise FileNotFoundError(f"nnU-Net model not found at {nn_model}")

    tmp = Path(tempfile.mkdtemp())
    try:
        os.environ["nnUNet_raw"] = str(tmp)
        os.environ["nnUNet_preprocessed"] = str(tmp)
        os.environ["nnUNet_results"] = str(tmp)

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device), verbose=False, allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(nn_model), use_folds=(0,), checkpoint_name="checkpoint_best.pth"
        )

        tmp_ct = str(tmp / "case_0000.nii.gz")
        tmp_pet = str(tmp / "case_0001.nii.gz")
        sitk.WriteImage(ct_roi, tmp_ct, useCompression=False)
        sitk.WriteImage(pet_roi, tmp_pet, useCompression=False)

        io = SimpleITKIO()
        img, props = io.read_images([tmp_ct, tmp_pet])
        result = predictor.predict_single_npy_array(
            input_image=img, image_properties=props,
            segmentation_previous_stage=None, output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        if isinstance(result, (tuple, list)) and len(result) == 2:
            probs = np.asarray(result[1]).astype(np.float32)
        else:
            seg = np.asarray(result[0] if isinstance(result, (tuple, list)) else result)
            probs = np.zeros((3,) + seg.shape, dtype=np.float32)
            for c in range(3):
                probs[c] = (seg == c).astype(np.float32)
        return probs
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def _run_segresnet(ct_roi, pet_roi, model_dir: Path, device):
    """SegResNetDS inference on the H&N ROI. Returns softmax (3,Z,Y,X) or None."""
    ckpt = model_dir / "segresnet_best.pt"
    if not ckpt.exists():
        return None
    try:
        from monai.networks.nets import SegResNetDS
        from monai.inferers import sliding_window_inference
        import torch

        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = SegResNetDS(
            spatial_dims=3, init_filters=32, in_channels=2,
            out_channels=3, blocks_down=(1, 2, 2, 4),
            norm="instance", dsdepth=4
        ).to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
        model.eval()

        ct_a = np.clip(sitk.GetArrayFromImage(ct_roi).astype(np.float32), -250.0, 250.0)
        pet_a = sitk.GetArrayFromImage(pet_roi).astype(np.float32)
        nz = pet_a > 0
        if nz.any():
            pet_a[nz] = (pet_a[nz] - pet_a[nz].mean()) / (pet_a[nz].std() + 1e-8)

        img_t = torch.from_numpy(np.stack([ct_a, pet_a])[None]).to(device)
        ROI = (192, 192, 192)
        acc = None
        for flip_dims in [(), (2,), (3,), (4,)]:
            x = torch.flip(img_t, dims=flip_dims) if flip_dims else img_t
            with torch.amp.autocast(device.type):
                raw = sliding_window_inference(x, ROI, 2, model, overlap=0.5, mode="gaussian")
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            p = torch.softmax(raw.float(), dim=1)
            if flip_dims:
                p = torch.flip(p, dims=flip_dims)
            acc = p if acc is None else acc + p
        prob = (acc / 4)[0].cpu().numpy()
        del model, img_t, acc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return prob.astype(np.float32)
    except Exception as e:
        print(f"[seg] SegResNet failed ({e}); using nnU-Net only")
        return None


def _postprocess_seg(seg, pet_nat_arr, voxel_ml=1.0, min_gtvn_ml=0.5, suv_thr=2.0):
    """Drop tiny / low-SUV spurious GTVn components."""
    result = seg.copy()
    cc, n = ndimage.label(seg == LABEL_GTVN)
    if n == 0:
        return result
    for i in range(1, n + 1):
        mask = cc == i
        vox = int(mask.sum())
        keep = True
        if vox * voxel_ml < min_gtvn_ml:
            keep = False
        elif pet_nat_arr is not None:
            peak_suv = float(pet_nat_arr[mask].max()) if mask.any() else 0.0
            if peak_suv < suv_thr:
                keep = False
        if not keep:
            result[mask] = 0
    return result


def run_segmentation(ct_sitk, pet_reg_sitk, pet_nat_arr, model_dir: Path, device="cuda"):
    """Full segmentation: H&N crop → nnU-Net + SegResNet ensemble → postproc → native."""
    ct_roi, pet_roi, _ = _hn_crop_case(ct_sitk, pet_reg_sitk, label=None)
    prob_nn = _run_nnunet(ct_roi, pet_roi, model_dir, device)
    prob_sr = _run_segresnet(ct_roi, pet_roi, model_dir, device)

    if prob_sr is not None:
        w_nn, w_sr = 0.7, 0.3
        wpath = model_dir / "seg_weights.json"
        if wpath.exists():
            try:
                w = json.loads(wpath.read_text())
                w_nn, w_sr = float(w.get("nnunet", 0.7)), float(w.get("segresnet", 0.3))
            except Exception:
                pass
        fg_nn = int((np.argmax(prob_nn, axis=0) > 0).sum())
        fg_sr = int((np.argmax(prob_sr, axis=0) > 0).sum())
        if fg_nn > 50 and fg_sr < 10:
            prob = prob_nn
        elif fg_sr > 50 and fg_nn < 10:
            prob = prob_sr
        else:
            prob = w_nn * prob_nn + w_sr * prob_sr
    else:
        prob = prob_nn

    seg_roi = np.argmax(prob, axis=0).astype(np.uint8)
    seg_native = _hn_map_roi_to_native(seg_roi, ct_roi, ct_sitk)
    seg = sitk.GetArrayFromImage(seg_native).astype(np.uint8)

    sp = ct_sitk.GetSpacing()
    voxel_ml = float(sp[0] * sp[1] * sp[2]) / 1000.0
    seg = _postprocess_seg(seg, pet_nat_arr, voxel_ml=voxel_ml)
    return seg


# ===========================================================================
# Subtask 2 — TN Staging (Prabesh: FeatureGroupMamba ensemble)
# ===========================================================================
def _make_feature_groups(flat: dict) -> dict:
    g = {}
    for name, cols in _G.items():
        g[name] = torch.tensor([[float(flat.get(col, 0.0)) for col in cols]],
                               dtype=torch.float32)
    g["clinical"] = torch.tensor([[float(flat.get(c, 0.0)) for c in _C]],
                                 dtype=torch.float32)
    g["rules"] = torch.tensor([[float(flat.get(c, 0.0)) for c in _R2]],
                              dtype=torch.float32)
    return g


def _candidate_onehot_keys(col, val):
    forms = {str(val)}
    try:
        fv = float(val)
        forms.add(str(fv))
        if fv.is_integer():
            forms.add(str(int(fv)))
    except (TypeError, ValueError):
        pass
    return {f"{col}={s}" for s in forms}


def _encode_ehr(ehr: dict) -> dict:
    clin = {}
    clin["Age"] = float(ehr.get("Age", np.nan)) if ehr.get("Age") is not None else np.nan
    cat_cols = ["Gender", "Tobacco Consumption", "Alcohol Consumption",
                "Performance Status", "HPV Status", "M-stage", "Treatment"]
    for col in cat_cols:
        val = ehr.get(col)
        missing = val is None or (isinstance(val, float) and np.isnan(val))
        clin[f"{col}_missing"] = 1 if missing else 0
        if not missing:
            for key in _candidate_onehot_keys(col, val):
                clin[key] = 1
    return clin


def run_tn_staging(geo: dict, ehr: dict, model_dir: Path) -> tuple[str, str]:
    """Run FeatureGroupMamba ensemble for T-stage and N-stage."""
    try:
        sys_path = str(Path(__file__).parent / "src" / "staging")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from mamba_tn_staging import FeatureGroupMamba
    except ImportError as e:
        print(f"[staging] Mamba import failed ({e}); defaulting T2/N0")
        return "T2", "N0"

    bundle_path = model_dir / "mamba_tn_ensemble.pth"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Mamba ensemble not found at {bundle_path}")

    X = {**geo, **_encode_ehr(ehr)}
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    gdim = bundle["gdim"]
    seeds = bundle["seeds"]
    T_CLS = bundle["T_CLS"]
    N_CLS = bundle["N_CLS"]

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    groups = _make_feature_groups(X)
    groups = {k: v.to(dev) for k, v in groups.items()}

    t_logits_avg = None
    n_logits_avg = None
    for sd in bundle["models"]:
        m = FeatureGroupMamba(gdim, d_model=64, dropout=0.3).to(dev)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            tl, nl = m(groups)
        t_logits_avg = tl if t_logits_avg is None else t_logits_avg + tl
        n_logits_avg = nl if n_logits_avg is None else n_logits_avg + nl

    t_stage = str(T_CLS[t_logits_avg.argmax(1).item()])
    n_stage = str(N_CLS[n_logits_avg.argmax(1).item()])
    return t_stage, n_stage


# ===========================================================================
# Subtask 3 — Prognosis (Rabin: RSF + Cox ensemble)
# ===========================================================================
def _build_prognosis_row(geo: dict, ehr: dict, feature_cols: list, medians: dict) -> pd.DataFrame:
    X = dict(geo)
    for col in ["Age", "Gender", "Tobacco Consumption", "Alcohol Consumption",
                "Performance Status", "Treatment", "HPV Status"]:
        X[col] = _to_numeric(ehr.get(col))
    row = pd.DataFrame([{c: X.get(c, medians.get(c, np.nan)) for c in feature_cols}])
    for c in feature_cols:
        row[c] = pd.to_numeric(row[c], errors="coerce")
        row[c] = row[c].fillna(medians.get(c, 0))
    return row


def _risk_to_rfs(risk: float) -> float:
    """Map risk score to RFS-like value (higher risk → lower output)."""
    if not np.isfinite(risk):
        return 1000.0
    return float(1000.0 - 200.0 * risk)


def run_prognosis(geo: dict, ehr: dict, model_dir: Path) -> float:
    """Run RSF + Cox ensemble prognosis. Returns RFS risk score."""
    ens_path = model_dir / "prognosis_ensemble.pkl"
    if not ens_path.exists():
        raise FileNotFoundError(f"Prognosis ensemble not found at {ens_path}")

    with open(ens_path, "rb") as f:
        E = pickle.load(f)

    weights = E.get("weights", {})

    def _z(score, stats):
        return (score - stats["mean"]) / stats["std"] if stats else score

    # RSF arm
    rsf = E["rsf"]["model"]
    rfc = E["rsf"]["feature_cols"]
    imp = E["rsf"].get("imputer")
    rsf_row = _build_prognosis_row(geo, ehr, rfc, {})
    imputed = imp.transform(rsf_row) if imp is not None else rsf_row.values
    rsf_score = float(rsf.predict(imputed)[0])

    total = weights.get("rsf", 1.0) * _z(rsf_score, E.get("rsf_stats"))
    wsum = weights.get("rsf", 1.0)

    # Cox arm
    cox_bundle = E.get("cox")
    if cox_bundle and weights.get("cox", 0) > 0:
        cfc = cox_bundle["feature_cols"]
        cox_row = _build_prognosis_row(geo, ehr, cfc, cox_bundle.get("medians", {}))
        scaler = cox_bundle.get("scaler")
        arr_in = scaler.transform(cox_row) if scaler is not None else cox_row.values
        cox_in = pd.DataFrame(np.asarray(arr_in), columns=cfc)
        cox_score = float(cox_bundle["model"].predict_log_partial_hazard(cox_in).values[0])
        total += weights["cox"] * _z(cox_score, E.get("cox_stats"))
        wsum += weights["cox"]

    risk = total / wsum if wsum > 0 else rsf_score
    return _risk_to_rfs(risk)


# ===========================================================================
# Feature extraction (shared by TN staging and prognosis)
# ===========================================================================
def _longest_diameter_mm(mask, spacing_mm):
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    pts = idx.astype(np.float64) * np.asarray(spacing_mm)
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import pdist
        v = pts[ConvexHull(pts).vertices]
        return float(pdist(v).max())
    except Exception:
        ext = (idx.max(0) - idx.min(0)).astype(np.float64) * np.asarray(spacing_mm)
        return float(np.sqrt((ext ** 2).sum()))


def _t_rule_index(diam_cm):
    for thr, idx in ((2.0, 1), (4.0, 2), (6.0, 3)):
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


def geometric_features(seg: np.ndarray, pet_suv: np.ndarray,
                       spacing_mm=(1.0, 1.0, 1.0)) -> dict:
    """Geometric + SUV features from predicted mask + PET volume."""
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    f = {}
    for name, lab in (("gtvp", LABEL_GTVP), ("gtvn", LABEL_GTVN)):
        m = seg == lab
        vox = int(m.sum())
        f[f"{name}_volume_ml"] = vox * voxel_ml
        if vox:
            suv = pet_suv[m]
            f[f"{name}_suv_max"] = float(suv.max())
            f[f"{name}_suv_mean"] = float(suv.mean())
            f[f"{name}_suv_peak"] = float(np.percentile(suv, 95))
            mtv = m & (pet_suv >= SUV_THR)
            f[f"{name}_mtv_ml"] = int(mtv.sum()) * voxel_ml
            f[f"{name}_tlg"] = float(suv[mtv].sum()) * voxel_ml
        else:
            for k in ("suv_max", "suv_mean", "suv_peak", "mtv_ml", "tlg"):
                f[f"{name}_{k}"] = 0.0

    cc, nnodes = ndimage.label(seg == LABEL_GTVN)
    f["n_nodes"] = int(nnodes)
    if nnodes:
        sizes = ndimage.sum(np.ones_like(cc), cc, index=range(1, nnodes + 1))
        f["largest_node_ml"] = float(sizes.max()) * voxel_ml
        f["gtvn_total_volume_ml"] = float(sum(sizes)) * voxel_ml
        f["gtvn_mean_node_ml"] = float(np.mean(sizes)) * voxel_ml
        cx = seg.shape[2] / 2.0
        coms = ndimage.center_of_mass(seg == LABEL_GTVN, cc, index=range(1, nnodes + 1))
        xs = [c[2] for c in coms]
        f["nodes_left"] = int(sum(x < cx for x in xs))
        f["nodes_right"] = int(sum(x >= cx for x in xs))
        f["bilateral"] = int(f["nodes_left"] > 0 and f["nodes_right"] > 0)
        largest_lab = int(np.argmax(sizes)) + 1
        f["gtvn_largest_diameter_cm"] = _longest_diameter_mm(cc == largest_lab, spacing_mm) / 10.0
    else:
        f.update(largest_node_ml=0.0, gtvn_total_volume_ml=0.0,
                 gtvn_mean_node_ml=0.0, nodes_left=0, nodes_right=0, bilateral=0,
                 gtvn_largest_diameter_cm=0.0)
    f["n_rule_index"] = _n_rule_index(f["gtvn_largest_diameter_cm"])

    if (seg == LABEL_GTVP).any() and (seg == LABEL_GTVN).any():
        com_p = ndimage.center_of_mass(seg == LABEL_GTVP)
        com_n = ndimage.center_of_mass(seg == LABEL_GTVN)
        dz = (com_p[0] - com_n[0]) * spacing_mm[0]
        dy = (com_p[1] - com_n[1]) * spacing_mm[1]
        dx = (com_p[2] - com_n[2]) * spacing_mm[2]
        f["gtvp_gtvn_distance_mm"] = float(np.sqrt(dz**2 + dy**2 + dx**2))
    else:
        f["gtvp_gtvn_distance_mm"] = 0.0

    gtvp_vol = f["gtvp_volume_ml"]
    gtvn_vol = f["gtvn_total_volume_ml"]
    f["total_tumor_volume_ml"] = gtvp_vol + gtvn_vol

    if gtvp_vol > 0:
        f["gtvp_diameter_cm"] = 2.0 * ((3.0 * gtvp_vol / (4.0 * np.pi)) ** (1.0 / 3.0))
    else:
        f["gtvp_diameter_cm"] = 0.0

    f["gtvp_longest_diameter_cm"] = _longest_diameter_mm(seg == LABEL_GTVP, spacing_mm) / 10.0
    f["t_rule_index"] = _t_rule_index(f["gtvp_longest_diameter_cm"])

    if gtvp_vol > 0:
        f["gtvp_tlg_density"] = f["gtvp_tlg"] / gtvp_vol
        f["gtvp_mtv_fraction"] = f["gtvp_mtv_ml"] / gtvp_vol
    else:
        f["gtvp_tlg_density"] = 0.0
        f["gtvp_mtv_fraction"] = 0.0

    f["gtvn_tlg_density"] = (f["gtvn_tlg"] / gtvn_vol) if gtvn_vol > 0 else 0.0

    if f["gtvp_suv_mean"] > 0:
        f["gtvp_suv_heterogeneity"] = f["gtvp_suv_max"] / f["gtvp_suv_mean"]
    else:
        f["gtvp_suv_heterogeneity"] = 1.0

    m_p = seg == LABEL_GTVP
    if m_p.any() and seg.size <= 200_000_000:
        eroded = ndimage.binary_erosion(m_p)
        boundary_vox = int((m_p & ~eroded).sum())
        sa_mm2 = boundary_vox * float((spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) ** (2.0 / 3.0))
        vol_mm3 = gtvp_vol * 1000.0
        if sa_mm2 > 0:
            f["gtvp_sphericity"] = (np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0)) / sa_mm2
        else:
            f["gtvp_sphericity"] = 1.0
    elif m_p.any():
        f["gtvp_sphericity"] = 0.8
    else:
        f["gtvp_sphericity"] = 0.0

    return f


def _first_order(values: np.ndarray) -> dict:
    keys = ("fo_mean","fo_std","fo_skewness","fo_kurtosis",
            "fo_entropy","fo_energy","fo_p10","fo_p90","fo_iqr")
    if len(values) == 0:
        return {k: 0.0 for k in keys}
    from scipy.stats import skew, kurtosis
    v = values.astype(np.float64)
    hist, _ = np.histogram(v, bins=64, density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
    return {
        "fo_mean": float(v.mean()), "fo_std": float(v.std()),
        "fo_skewness": float(skew(v)), "fo_kurtosis": float(kurtosis(v)),
        "fo_entropy": entropy, "fo_energy": float((v**2).sum() / len(v)),
        "fo_p10": float(np.percentile(v, 10)), "fo_p90": float(np.percentile(v, 90)),
        "fo_iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
    }


def _glcm_features(img_roi, n_levels=32):
    zero = {k: 0.0 for k in ("glcm_contrast","glcm_dissimilarity",
                             "glcm_homogeneity","glcm_energy","glcm_correlation")}
    if img_roi.size == 0:
        return zero
    mid = img_roi.shape[0] // 2
    sl = img_roi[mid].astype(np.float64)
    if sl.std() < 1e-6:
        return zero
    mn, mx = sl.min(), sl.max()
    if mx == mn:
        return zero
    sl_q = ((sl - mn) / (mx - mn) * (n_levels - 1)).astype(np.uint8)
    try:
        from skimage.feature import graycomatrix, graycoprops
        glcm = graycomatrix(sl_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=n_levels, symmetric=True, normed=True)
        props = {}
        for prop in ("contrast","dissimilarity","homogeneity","energy","correlation"):
            props[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
        return props
    except ImportError:
        return zero


def _shape_features(mask, spacing_mm=(1.0, 1.0, 1.0)):
    vox_vol = float(np.prod(spacing_mm))
    vol_mm3 = float(mask.sum()) * vox_vol
    if vol_mm3 == 0:
        return {"shape_elongation": 0.0, "shape_flatness": 0.0,
                "shape_surface_vol_ratio": 0.0}
    coords = np.argwhere(mask).astype(float) * np.array(spacing_mm)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
    elongation = float(np.sqrt(eigvals[1] / (eigvals[0] + 1e-9)))
    flatness = float(np.sqrt(eigvals[2] / (eigvals[0] + 1e-9)))
    if mask.size <= 50_000_000:
        eroded = ndimage.binary_erosion(mask)
        surface_vox = int((mask & ~eroded).sum())
        sa_mm2 = surface_vox * float((spacing_mm[0]*spacing_mm[1]*spacing_mm[2]) ** (2.0/3.0))
    else:
        sa_mm2 = 4 * np.pi * ((3 * vol_mm3 / (4 * np.pi)) ** (2/3))
    svr = float(sa_mm2 / (vol_mm3 + 1e-9))
    return {"shape_elongation": elongation, "shape_flatness": flatness,
            "shape_surface_vol_ratio": svr}


def _crop_roi(img, mask, pad=2):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return img[:1, :1, :1], mask[:1, :1, :1]
    mn = np.maximum(idx.min(axis=0) - pad, 0)
    mx = np.minimum(idx.max(axis=0) + pad + 1, np.array(img.shape))
    sl = tuple(slice(mn[i], mx[i]) for i in range(3))
    return img[sl], mask[sl]


def extract_radiomics(seg, pet, ct, spacing_mm=(1.0, 1.0, 1.0)):
    """First-order + GLCM + shape radiomics for GTVp and GTVn."""
    features = {}
    for label, prefix in [(LABEL_GTVP, "rad_p_"), (LABEL_GTVN, "rad_n_")]:
        mask = seg == label
        if not mask.any():
            dummy = {**_first_order(np.array([])),
                     **_glcm_features(np.zeros((1, 1, 1))),
                     **_shape_features(mask, spacing_mm)}
            features.update({prefix + k: v for k, v in dummy.items()})
            continue
        pet_crop, mask_crop = _crop_roi(pet, mask)
        ct_crop, _ = _crop_roi(ct, mask)
        fo_pet = _first_order(pet[mask])
        fo_ct = _first_order(ct[mask])
        glcm_p = _glcm_features(pet_crop * mask_crop)
        glcm_c = _glcm_features(ct_crop * mask_crop)
        shp = _shape_features(mask, spacing_mm)
        combined = {}
        combined.update({f"pet_{k}": v for k, v in fo_pet.items()})
        combined.update({f"ct_{k}": v for k, v in fo_ct.items()})
        combined.update({f"pet_{k}": v for k, v in glcm_p.items()})
        combined.update({f"ct_{k}": v for k, v in glcm_c.items()})
        combined.update(shp)
        features.update({prefix + k: v for k, v in combined.items()})
    return features


# ===========================================================================
# Helpers
# ===========================================================================
def _to_numeric(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ===========================================================================
# Main end-to-end pipeline
# ===========================================================================
def run_pipeline(
    patient_id: str,
    ct_path: str,
    pet_path: str,
    ehr: dict,
    model_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
) -> dict:
    """Run the full HECKTOR 2026 pipeline for one patient."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_sitk = sitk.ReadImage(str(ct_path))
    pet_sitk = sitk.ReadImage(str(pet_path))
    pet_reg = _register_pet_to_ct(pet_sitk, ct_sitk)
    pet_nat = np.clip(sitk.GetArrayFromImage(pet_reg).astype(np.float32), 0, None)

    # 1. Segmentation
    print(f"[{patient_id}] Running segmentation...")
    try:
        seg = run_segmentation(ct_sitk, pet_reg, pet_nat, model_dir, device)
    except Exception as e:
        print(f"[{patient_id}] Segmentation FAILED ({e}); writing empty mask")
        seg = np.zeros(sitk.GetArrayFromImage(ct_sitk).shape, dtype=np.uint8)

    seg_img = sitk.GetImageFromArray(seg.astype(np.uint8))
    seg_img.CopyInformation(ct_sitk)
    sitk.WriteImage(seg_img, str(output_dir / f"{patient_id}_seg.mha"), useCompression=True)
    print(f"[{patient_id}] Seg saved: {output_dir / f'{patient_id}_seg.mha'}")

    # 2. Feature extraction
    sp = ct_sitk.GetSpacing()
    spacing_zyx = (sp[2], sp[1], sp[0])
    ct_arr = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    try:
        geo = geometric_features(seg, pet_nat, spacing_mm=spacing_zyx)
        rad = extract_radiomics(seg, pet_nat, ct_arr, spacing_mm=spacing_zyx)
        features = {**geo, **rad}
    except Exception as e:
        print(f"[{patient_id}] Feature extraction FAILED ({e}); using zeros")
        features = {}

    # 3. TN staging
    print(f"[{patient_id}] Running TN staging...")
    try:
        t_stage, n_stage = run_tn_staging(features, ehr, model_dir)
    except Exception as e:
        print(f"[{patient_id}] TN staging FAILED ({e}); defaulting T2/N0")
        t_stage, n_stage = "T2", "N0"

    # 4. Prognosis
    print(f"[{patient_id}] Running prognosis...")
    try:
        rfs = float(run_prognosis(features, ehr, model_dir))
        if not np.isfinite(rfs):
            rfs = 1000.0
    except Exception as e:
        print(f"[{patient_id}] Prognosis FAILED ({e}); defaulting 1000.0")
        rfs = 1000.0

    result = {
        "patient_id": patient_id,
        "segmentation": str(output_dir / f"{patient_id}_seg.mha"),
        "t_stage": t_stage,
        "n_stage": n_stage,
        "rfs": rfs,
        **features,
    }

    with open(output_dir / f"{patient_id}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{patient_id}] T={t_stage}  N={n_stage}  RFS={rfs:.4f}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--pet", required=True)
    parser.add_argument("--ehr", default="{}")
    parser.add_argument("--model_dir", default="work/models")
    parser.add_argument("--output_dir", default="work/output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_pipeline(
        patient_id=args.patient_id,
        ct_path=args.ct,
        pet_path=args.pet,
        ehr=json.loads(args.ehr),
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
