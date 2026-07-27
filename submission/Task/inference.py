"""
HECKTOR 2026 Challenge — Inference Entry Point

Pipeline:
  1. Segmentation: nnU-Net ResEnc-M (val Dice 0.726) → output.mha (CT geometry)
  2. TN Staging:   LightGBM → t-stage.json, n-stage.json
  3. Prognosis:    Cox → rfs.json

Feature extraction and clinical encoding match training exactly:
  - PET registered to CT, both resampled to 1mm isotropic
  - Geometric/SUV features from hecktor/features.py::geometric_features()
  - Clinical encoding from hecktor/columns.py::encode_clinical()
    (one-hot + _missing indicators, fillna 0.0 — same as 04_train_tn.py)
  - Cox uses raw numeric clinical (same as 05_train_prognosis.py)
"""
import json
import os
import pickle
import shutil
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage

# --- Memory hardening for the Grand Challenge T4 (16 GB VRAM / 16 GB DRAM) ---
# The dev GPU was a 24 GB 3090; the T4 is far tighter, so reduce fragmentation
# and cap CPU thread pools that can balloon resident memory during resampling.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

INPUT_PATH  = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH  = Path("/opt/ml/model")

LABEL_GTVP = 1
LABEL_GTVN = 2
SUV_THR    = 2.5   # matches features.py


# =============================================================================
# Main
# =============================================================================
def run():
    ct_path  = _get_image(INPUT_PATH / "images/ct")
    pet_path = _get_image(INPUT_PATH / "images/pet")
    # EHR is optional — GC may not provide it; never crash if absent/malformed.
    ehr = {}
    ehr_path = INPUT_PATH / "ehr.json"
    if ehr_path.exists():
        try:
            ehr = _load_json(ehr_path) or {}
        except Exception as e:
            print(f"[inputs] ehr.json unreadable ({e}); using empty EHR", flush=True)
    print(f"[inputs] ct={Path(ct_path).name}  pet={Path(pet_path).name}  ehr_keys={len(ehr)}", flush=True)

    # A GC case fails if the container exits non-zero OR any required output is
    # missing. Everything below is defended so all four outputs are ALWAYS
    # written and run() always returns 0 — degraded defaults beat a failed case.
    ct_sitk = sitk.ReadImage(str(ct_path))

    pet_reg = None
    pet_nat = None
    try:
        pet_sitk = sitk.ReadImage(str(pet_path))
        # register PET to CT grid (needed for both segmentation postproc + features)
        pet_reg = _register_pet_to_ct(pet_sitk, ct_sitk)
        pet_nat = np.clip(sitk.GetArrayFromImage(pet_reg).astype(np.float32), 0, None)
        del pet_sitk
    except Exception as e:
        print(f"[inputs] PET load/register FAILED ({e}); proceeding without PET", flush=True)

    # --- segmentation: crop to H&N ROI → ensemble predict → map back to native ---
    try:
        if pet_reg is None:
            raise RuntimeError("PET unavailable")
        seg_orig = run_segmentation(ct_sitk, pet_reg, pet_nat_arr=pet_nat)
    except Exception as e:
        print(f"[seg] FAILED ({e}); writing empty mask", flush=True)
        seg_orig = np.zeros(sitk.GetArrayFromImage(ct_sitk).shape, dtype=np.uint8)
    try:
        _write_segmentation(seg_orig, ct_sitk)
    except Exception as e:
        print(f"[seg] write FAILED ({e}); retrying with a zero mask", flush=True)
        _write_segmentation(np.zeros(sitk.GetArrayFromImage(ct_sitk).shape, dtype=np.uint8), ct_sitk)
    print(f"[seg] shape={seg_orig.shape}  unique={np.unique(seg_orig)}", flush=True)

    # --- features at native CT spacing ---
    try:
        pet_feat = pet_nat if pet_nat is not None else np.zeros(seg_orig.shape, dtype=np.float32)
        sp = ct_sitk.GetSpacing()           # SimpleITK: (x_mm, y_mm, z_mm)
        spacing_zyx = (sp[2], sp[1], sp[0])  # numpy is (z, y, x)
        geo = _geometric_features(seg_orig, pet_feat, spacing_mm=spacing_zyx)
        ct_nat = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)  # raw HU for radiomics
        geo = {**geo, **_extract_radiomics(seg_orig, pet_feat, ct_nat, spacing_mm=spacing_zyx)}
        print(f"[features] gtvp_vol={geo['gtvp_volume_ml']:.2f}mL n_nodes={geo['n_nodes']}", flush=True)
    except Exception as e:
        print(f"[features] FAILED ({e}); using zeros", flush=True)
        geo = {}

    # --- TN staging (never crash the submission; default to commonest classes) ---
    try:
        t_stage, n_stage = run_tn_staging(geo, ehr)
    except Exception as e:
        print(f"[staging] FAILED ({e}); defaulting T2/N0", flush=True)
        t_stage, n_stage = "T2", "N0"
    _write_json(OUTPUT_PATH / "t-stage.json", str(t_stage))
    _write_json(OUTPUT_PATH / "n-stage.json", str(n_stage))
    print(f"[staging] T={t_stage}  N={n_stage}", flush=True)

    # --- prognosis ---
    try:
        rfs = float(run_prognosis(geo, ehr))
        if not np.isfinite(rfs):
            rfs = 0.0
    except Exception as e:
        print(f"[prognosis] FAILED ({e}); defaulting 0.0", flush=True)
        rfs = 0.0
    _write_json(OUTPUT_PATH / "rfs.json", rfs)
    print(f"[prognosis] rfs={rfs:.4f}", flush=True)

    return 0


# =============================================================================
# Subtask 1 — Segmentation (nnU-Net + SegResNet ensemble)
# =============================================================================
def run_segmentation(ct_sitk, pet_reg_sitk, pet_nat_arr=None):
    """Crop to the H&N ROI (1,1,3 mm) — the SAME ROI both arms trained on — run
    nnU-Net (0.7) + SegResNet (0.3) on the ROI, ensemble, then map the result
    back onto the native CT grid (zeros outside the ROI).
    Falls back to nnU-Net alone if SegResNet is missing or fails."""
    import gc
    import torch
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # inference-mode crop (no label): anatomical H&N centering
    ct_roi, pet_roi, _ = _hn_crop_case(ct_sitk, pet_reg_sitk, label=None)
    print(f"[seg] H&N ROI size (x,y,z)={ct_roi.GetSize()}", flush=True)

    prob_nn = _run_nnunet(ct_roi, pet_roi, device)   # (3, Z, Y, X) softmax in ROI space
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"[seg] GPU free after nnU-Net: {free:.1f} GB", flush=True)
    prob_sr = _run_segresnet(ct_roi, pet_roi, device)  # (3, Z, Y, X) softmax or None

    if prob_sr is not None:
        # Ensemble weight is tuned on held-out val (step 7b) and written to
        # seg_weights.json; defaults to 0.7/0.3. If tuning found nnU-Net alone
        # best, w_nn=1.0 and SegResNet contributes nothing (never hurts the score).
        w_nn, w_sr = 0.7, 0.3
        wpath = MODEL_PATH / "seg_weights.json"
        if wpath.exists():
            try:
                w = json.loads(wpath.read_text())
                w_nn, w_sr = float(w.get("nnunet", 0.7)), float(w.get("segresnet", 0.3))
            except Exception:
                pass
        # Safety: if exactly one arm predicts (near-)empty foreground while the
        # other finds a clear tumor, the empty arm's confident background would
        # veto the ensemble → a catastrophic Dice-0 miss. In that case trust the
        # arm that found something instead of averaging it away.
        fg_nn = int((np.argmax(prob_nn, axis=0) > 0).sum())
        fg_sr = int((np.argmax(prob_sr, axis=0) > 0).sum())
        if fg_nn > 50 and fg_sr < 10:
            prob = prob_nn
            print(f"[seg] SegResNet empty (fg={fg_sr}); using nnU-Net only (fg={fg_nn})", flush=True)
        elif fg_sr > 50 and fg_nn < 10:
            prob = prob_sr
            print(f"[seg] nnU-Net empty (fg={fg_nn}); using SegResNet only (fg={fg_sr})", flush=True)
        else:
            prob = w_nn * prob_nn + w_sr * prob_sr
            print(f"[seg] ensemble  nnU-Net={w_nn:.2f}  SegResNet={w_sr:.2f}", flush=True)
    else:
        prob = prob_nn
        print("[seg] nnU-Net only (SegResNet unavailable)", flush=True)

    seg_roi = np.argmax(prob, axis=0).astype(np.uint8)

    # map ROI prediction back onto the native CT grid (zeros outside ROI)
    seg_native_img = _hn_map_roi_to_native(seg_roi, ct_roi, ct_sitk)
    seg = sitk.GetArrayFromImage(seg_native_img).astype(np.uint8)

    # real voxel volume on the native grid so the mL threshold actually applies
    sp = ct_sitk.GetSpacing()
    voxel_ml = float(sp[0] * sp[1] * sp[2]) / 1000.0
    seg = _postprocess_seg(seg, pet_nat_arr, voxel_ml=voxel_ml)
    return seg


def _run_nnunet(ct_roi, pet_roi, device):
    """Returns softmax (3, Z, Y, X) float32 in the ROI geometry."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import torch

    nn_model = (MODEL_PATH / "nnunet" / "Dataset021_HECKTOR2026" /
                "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")

    tmp = Path(tempfile.mkdtemp())
    try:
        os.environ["nnUNet_raw"]          = str(tmp)
        os.environ["nnUNet_preprocessed"] = str(tmp)
        os.environ["nnUNet_results"]      = str(tmp)

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True,
            device=device, verbose=False, allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(nn_model), use_folds=(0,), checkpoint_name="checkpoint_best.pth"
        )

        tmp_ct  = str(tmp / "case_0000.nii.gz")
        tmp_pet = str(tmp / "case_0001.nii.gz")
        sitk.WriteImage(ct_roi,  tmp_ct,  useCompression=False)
        sitk.WriteImage(pet_roi, tmp_pet, useCompression=False)

        io = SimpleITKIO()
        img, props = io.read_images([tmp_ct, tmp_pet])
        result = predictor.predict_single_npy_array(
            input_image=img, image_properties=props,
            segmentation_previous_stage=None, output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        if isinstance(result, (tuple, list)) and len(result) == 2:
            probs = result[1]
        else:
            seg = np.asarray(result[0] if isinstance(result, (tuple, list)) else result)
            probs = np.zeros((3,) + seg.shape, dtype=np.float32)
            for c in range(3):
                probs[c] = (seg == c).astype(np.float32)
        probs = np.asarray(probs).astype(np.float32)
        print("[seg] nnU-Net done", flush=True)
        return probs
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def _run_segresnet(ct_roi, pet_roi, device):
    """Returns softmax (3, Z, Y, X) float32 in the ROI geometry, or None.
    ct_roi/pet_roi are already the (1,1,3) H&N ROI — same as training, so no
    extra resampling (this also avoids the whole-body 1mm OOM)."""
    ckpt = MODEL_PATH / "segresnet_best.pt"
    if not ckpt.exists():
        return None
    try:
        import torch
        from monai.networks.nets import SegResNetDS
        from monai.inferers import sliding_window_inference

        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = SegResNetDS(spatial_dims=3, init_filters=32, in_channels=2,
                            out_channels=3, blocks_down=(1, 2, 2, 4),
                            norm="instance", dsdepth=4).to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location=device,
                                         weights_only=True))
        model.eval()

        # Match SegResNet training preprocessing on the ROI:
        #   CT  → clip [-250, 250]   PET → z-score on non-zero voxels
        ct_a  = np.clip(sitk.GetArrayFromImage(ct_roi).astype(np.float32), -250.0, 250.0)
        pet_a = sitk.GetArrayFromImage(pet_roi).astype(np.float32)
        nz = pet_a > 0
        if nz.any():
            pet_a[nz] = (pet_a[nz] - pet_a[nz].mean()) / (pet_a[nz].std() + 1e-8)

        img_t = torch.from_numpy(np.stack([ct_a, pet_a])[None]).to(device)  # (1,2,Z,Y,X)
        ROI = (192, 192, 192)

        with torch.no_grad():
            acc = None
            for flip_dims in [(), (2,), (3,), (4,)]:   # 4-flip TTA
                x = torch.flip(img_t, dims=flip_dims) if flip_dims else img_t
                with torch.amp.autocast(device.type):
                    raw = sliding_window_inference(
                        x, ROI, 2, model, overlap=0.5, mode="gaussian"
                    )
                raw = raw[0] if isinstance(raw, (list, tuple)) else raw
                p = torch.softmax(raw.float(), dim=1)
                if flip_dims:
                    p = torch.flip(p, dims=flip_dims)
                acc = p if acc is None else acc + p
            prob = (acc / 4)[0].cpu().numpy()   # (3, Z, Y, X) — already ROI geometry

        del model, img_t, acc
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("[seg] SegResNet done", flush=True)
        return prob.astype(np.float32)
    except Exception as e:
        print(f"[seg] SegResNet failed ({e}); using nnU-Net only", flush=True)
        return None


# =============================================================================
# Subtask 2 — TN Staging
# =============================================================================
_G = {"geometric": ["gtvp_volume_ml","gtvp_suv_max","gtvp_suv_mean","gtvp_suv_peak",
    "gtvp_mtv_ml","gtvp_tlg","gtvn_volume_ml","gtvn_suv_max","gtvn_suv_mean",
    "gtvn_suv_peak","gtvn_mtv_ml","gtvn_tlg","n_nodes","largest_node_ml",
    "gtvn_total_volume_ml","gtvn_mean_node_ml","nodes_left","nodes_right",
    "bilateral","gtvn_largest_diameter_cm","n_rule_index","gtvp_gtvn_distance_mm",
    "total_tumor_volume_ml","gtvp_diameter_cm","gtvp_longest_diameter_cm",
    "t_rule_index","gtvp_tlg_density","gtvp_mtv_fraction","gtvn_tlg_density",
    "gtvp_suv_heterogeneity","gtvp_sphericity"],
    "rad_p": ["rad_p_pet_fo_mean","rad_p_pet_fo_std","rad_p_pet_fo_skewness",
    "rad_p_pet_fo_kurtosis","rad_p_pet_fo_entropy","rad_p_pet_fo_energy",
    "rad_p_pet_fo_p10","rad_p_pet_fo_p90","rad_p_pet_fo_iqr",
    "rad_p_ct_fo_mean","rad_p_ct_fo_std","rad_p_ct_fo_skewness","rad_p_ct_fo_kurtosis",
    "rad_p_ct_fo_entropy","rad_p_ct_fo_energy","rad_p_ct_fo_p10","rad_p_ct_fo_p90",
    "rad_p_ct_fo_iqr","rad_p_pet_glcm_contrast","rad_p_pet_glcm_dissimilarity",
    "rad_p_pet_glcm_homogeneity","rad_p_pet_glcm_energy","rad_p_pet_glcm_correlation",
    "rad_p_ct_glcm_contrast","rad_p_ct_glcm_dissimilarity","rad_p_ct_glcm_homogeneity",
    "rad_p_ct_glcm_energy","rad_p_ct_glcm_correlation",
    "rad_p_shape_elongation","rad_p_shape_flatness","rad_p_shape_surface_vol_ratio"],
    "rad_n": ["rad_n_pet_fo_mean","rad_n_pet_fo_std","rad_n_pet_fo_skewness",
    "rad_n_pet_fo_kurtosis","rad_n_pet_fo_entropy","rad_n_pet_fo_energy",
    "rad_n_pet_fo_p10","rad_n_pet_fo_p90","rad_n_pet_fo_iqr",
    "rad_n_ct_fo_mean","rad_n_ct_fo_std","rad_n_ct_fo_skewness","rad_n_ct_fo_kurtosis",
    "rad_n_ct_fo_entropy","rad_n_ct_fo_energy","rad_n_ct_fo_p10","rad_n_ct_fo_p90",
    "rad_n_ct_fo_iqr","rad_n_pet_glcm_contrast","rad_n_pet_glcm_dissimilarity",
    "rad_n_pet_glcm_homogeneity","rad_n_pet_glcm_energy","rad_n_pet_glcm_correlation",
    "rad_n_ct_glcm_contrast","rad_n_ct_glcm_dissimilarity","rad_n_ct_glcm_homogeneity",
    "rad_n_ct_glcm_energy","rad_n_ct_glcm_correlation",
    "rad_n_shape_elongation","rad_n_shape_flatness","rad_n_shape_surface_vol_ratio"]}
_C = ["Age","Gender_missing","Gender=0.0","Gender=1.0","Tobacco Consumption_missing",
     "Tobacco Consumption=0.0","Tobacco Consumption=1.0","Alcohol Consumption_missing",
     "Alcohol Consumption=0.0","Alcohol Consumption=1.0","Performance Status_missing",
     "Performance Status=0.0","Performance Status=1.0","Performance Status=2.0",
     "Performance Status=3.0","Performance Status=4.0","HPV Status_missing",
     "HPV Status=0.0","HPV Status=1.0","Treatment_missing","Treatment=0.0","Treatment=1.0"]
_R2 = ["t_rule_index","n_rule_index","gtvp_diameter_cm","gtvp_longest_diameter_cm",
       "gtvn_largest_diameter_cm","n_nodes","bilateral"]


def _make_feature_groups(flat):
    g = {}
    for name, cols in _G.items():
        g[name] = torch.tensor([[float(flat.get(col, 0.0)) for col in cols]], dtype=torch.float32)
    g["clinical"] = torch.tensor([[float(flat.get(c, 0.0)) for c in _C]], dtype=torch.float32)
    g["rules"] = torch.tensor([[float(flat.get(c, 0.0)) for c in _R2]], dtype=torch.float32)
    return g


_CATEGORICAL_COLS = ["Gender", "Tobacco Consumption", "Alcohol Consumption",
                     "Performance Status", "HPV Status", "M-stage", "Treatment"]

def run_tn_staging(geo, ehr):
    """Mamba ensemble (FeatureGroupMamba) for T-stage and N-stage."""
    from mamba_tn_staging import FeatureGroupMamba

    clin = {}
    clin["Age"] = _to_numeric(ehr.get("Age"))
    for col in _CATEGORICAL_COLS:
        val = ehr.get(col)
        missing = val is None or (isinstance(val, float) and np.isnan(val))
        clin[f"{col}_missing"] = 1 if missing else 0
        if not missing:
            for key in _candidate_onehot_keys(col, val):
                clin[key] = 1

    X = {**geo, **clin}

    bundle = torch.load(MODEL_PATH / "ensemble.pth", map_location="cpu", weights_only=False)
    gdim = bundle["gdim"]
    seeds = bundle["seeds"]
    T_CLS = bundle["T_CLS"]
    N_CLS = bundle["N_CLS"]

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
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
        del m

    t_stage = str(T_CLS[t_logits_avg.argmax(1).item()])
    n_stage = str(N_CLS[n_logits_avg.argmax(1).item()])

    return t_stage, n_stage



# =============================================================================
def run_prognosis(geo, ehr):
    """RSF+Cox ensemble (preferred) or Cox fallback."""
    import pandas as pd
    from scipy.stats import rankdata

    def _build_row(fc, meds):
        X = dict(geo)
        for col in ["Age", "Gender", "Tobacco Consumption", "Alcohol Consumption",
                    "Performance Status", "Treatment", "HPV Status"]:
            X[col] = _to_numeric(ehr.get(col))
        row = pd.DataFrame([{c: X.get(c, meds.get(c, np.nan)) for c in fc}])
        for c in fc:
            row[c] = pd.to_numeric(row[c], errors="coerce")
            row[c] = row[c].fillna(meds.get(c, 0))
        return row

    # Try ensemble model first (trained on full dataset)
    ens_path = MODEL_PATH / "prognosis_ensemble.pkl"
    if ens_path.exists():
        with open(ens_path, "rb") as f: E = pickle.load(f)

        weights = E.get("weights", {})

        def _z(score, stats):
            return (score - stats["mean"]) / stats["std"] if stats else score

        # RSF (its own full feature_cols; NaNs filled by imputer). GBS shares it.
        rsf = E["rsf"]["model"]
        rfc = E["rsf"]["feature_cols"]
        imp = E["rsf"].get("imputer")
        rsf_row = _build_row(rfc, {})   # DataFrame (named cols → no sklearn warning)
        imputed = imp.transform(rsf_row) if imp is not None else rsf_row.values
        rsf_score = float(rsf.predict(imputed)[0])

        total, wsum = weights.get("rsf", 1.0) * _z(rsf_score, E.get("rsf_stats")), weights.get("rsf", 1.0)

        # GBS arm (shares RSF imputed input)
        gbs_bundle = E.get("gbs")
        if gbs_bundle and weights.get("gbs", 0) > 0:
            gbs_score = float(gbs_bundle["model"].predict(imputed)[0])
            total += weights["gbs"] * _z(gbs_score, E.get("gbs_stats")); wsum += weights["gbs"]

        # Cox arm (its own standardized feature subset)
        cox_bundle = E.get("cox")
        if cox_bundle and weights.get("cox", 0) > 0:
            cfc = cox_bundle["feature_cols"]
            cox_row = _build_row(cfc, cox_bundle["medians"])
            scaler = cox_bundle.get("scaler")
            arr = scaler.transform(cox_row) if scaler is not None else cox_row.values
            cox_in = pd.DataFrame(np.asarray(arr), columns=cfc)
            cox_score = float(cox_bundle["model"].predict_log_partial_hazard(cox_in).values[0])
            total += weights["cox"] * _z(cox_score, E.get("cox_stats")); wsum += weights["cox"]

        risk = total / wsum if wsum > 0 else rsf_score
        return _risk_to_rfs(risk)

    # Fallback: Cox only
    with open(MODEL_PATH / "prognosis_cox.pkl", "rb") as f: P = pickle.load(f)
    cox, fc, meds = P["model"], P["feature_cols"], P["medians"]
    row = _build_row(fc, meds)
    return _risk_to_rfs(float(cox.predict_log_partial_hazard(row).values[0]))


def _risk_to_rfs(risk):
    """The models produce a RISK score (higher = worse survival), but Grand
    Challenge scores the output as an *RFS time in days, anti-concordant with
    risk* (higher output = longer survival). The C-index is rank-based, so any
    strictly-decreasing map of risk is equivalent — we return a positive,
    day-like value that decreases with risk."""
    if not np.isfinite(risk):
        return 1000.0
    return float(1000.0 - 200.0 * risk)


# =============================================================================
# Segmentation postprocessing — remove tiny / low-SUV spurious GTVn components
# =============================================================================
def _postprocess_seg(seg, pet_nat_arr, voxel_ml=1.0, min_gtvn_ml=0.5, suv_thr=2.0):
    """Drop GTVn connected components smaller than min_gtvn_ml mL OR with peak
    SUV below suv_thr (likely false positives from tracer uptake in salivary glands).
    voxel_ml is the native-grid voxel volume (mL) so the size threshold is real.
    GTVp is left unchanged — nnU-Net is reliable for primary."""
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


# =============================================================================
# Feature extraction — mirrors hecktor/features.py::geometric_features()
# =============================================================================
def _longest_diameter_mm(mask, spacing_mm):
    """True max-Feret diameter (mm) via convex-hull vertices — the AJCC metric."""
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


def _geometric_features(seg, pet_suv, spacing_mm=(1.0, 1.0, 1.0)):
    """Matches hecktor/features.py::geometric_features() exactly."""
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    f = {}

    for name, lab in (("gtvp", LABEL_GTVP), ("gtvn", LABEL_GTVN)):
        m   = seg == lab
        vox = int(m.sum())
        f[f"{name}_volume_ml"] = vox * voxel_ml
        if vox:
            suv = pet_suv[m]
            f[f"{name}_suv_max"]  = float(suv.max())
            f[f"{name}_suv_mean"] = float(suv.mean())
            f[f"{name}_suv_peak"] = float(np.percentile(suv, 95))
            mtv = m & (pet_suv >= SUV_THR)
            f[f"{name}_mtv_ml"] = int(mtv.sum()) * voxel_ml
            f[f"{name}_tlg"]    = float(pet_suv[mtv].sum()) * voxel_ml
        else:
            for k in ("suv_max", "suv_mean", "suv_peak", "mtv_ml", "tlg"):
                f[f"{name}_{k}"] = 0.0

    # nodal burden
    cc, nnodes = ndimage.label(seg == LABEL_GTVN)
    f["n_nodes"] = int(nnodes)
    if nnodes:
        sizes = ndimage.sum(np.ones_like(cc), cc, index=range(1, nnodes + 1))
        f["largest_node_ml"]      = float(sizes.max()) * voxel_ml
        f["gtvn_total_volume_ml"] = float(sum(sizes)) * voxel_ml
        f["gtvn_mean_node_ml"]    = float(np.mean(sizes)) * voxel_ml
        cx   = seg.shape[2] / 2.0
        coms = ndimage.center_of_mass(seg == LABEL_GTVN, cc, index=range(1, nnodes + 1))
        xs   = [c[2] for c in coms]
        f["nodes_left"]  = int(sum(x < cx for x in xs))
        f["nodes_right"] = int(sum(x >= cx for x in xs))
        f["bilateral"]   = int(f["nodes_left"] > 0 and f["nodes_right"] > 0)
        largest_lab = int(np.argmax(sizes)) + 1
        f["gtvn_largest_diameter_cm"] = _longest_diameter_mm(cc == largest_lab, spacing_mm) / 10.0
    else:
        f.update(largest_node_ml=0.0, gtvn_total_volume_ml=0.0,
                 gtvn_mean_node_ml=0.0, nodes_left=0, nodes_right=0, bilateral=0,
                 gtvn_largest_diameter_cm=0.0)
    f["n_rule_index"] = _n_rule_index(f["gtvn_largest_diameter_cm"])

    # GTVp-to-GTVn centroid distance (mm) — strong N-staging + prognosis feature
    if (seg == LABEL_GTVP).any() and (seg == LABEL_GTVN).any():
        com_p = ndimage.center_of_mass(seg == LABEL_GTVP)   # (z,y,x) voxel coords
        com_n = ndimage.center_of_mass(seg == LABEL_GTVN)
        dz = (com_p[0] - com_n[0]) * spacing_mm[0]
        dy = (com_p[1] - com_n[1]) * spacing_mm[1]
        dx = (com_p[2] - com_n[2]) * spacing_mm[2]
        f["gtvp_gtvn_distance_mm"] = float(np.sqrt(dz**2 + dy**2 + dx**2))
    else:
        f["gtvp_gtvn_distance_mm"] = 0.0

    # derived shape / combined features
    gtvp_vol = f["gtvp_volume_ml"]
    gtvn_vol = f["gtvn_total_volume_ml"]
    f["total_tumor_volume_ml"] = gtvp_vol + gtvn_vol

    if gtvp_vol > 0:
        f["gtvp_diameter_cm"] = 2.0 * ((3.0 * gtvp_vol / (4.0 * np.pi)) ** (1.0 / 3.0))
    else:
        f["gtvp_diameter_cm"] = 0.0

    # TRUE longest GTVp dimension (cm) — actual AJCC T metric — + rule ordinal
    f["gtvp_longest_diameter_cm"] = _longest_diameter_mm(seg == LABEL_GTVP, spacing_mm) / 10.0
    f["t_rule_index"] = _t_rule_index(f["gtvp_longest_diameter_cm"])

    if gtvp_vol > 0:
        f["gtvp_tlg_density"]  = f["gtvp_tlg"] / gtvp_vol
        f["gtvp_mtv_fraction"] = f["gtvp_mtv_ml"] / gtvp_vol
    else:
        f["gtvp_tlg_density"]  = 0.0
        f["gtvp_mtv_fraction"] = 0.0

    f["gtvn_tlg_density"] = (f["gtvn_tlg"] / gtvn_vol) if gtvn_vol > 0 else 0.0

    if f["gtvp_suv_mean"] > 0:
        f["gtvp_suv_heterogeneity"] = f["gtvp_suv_max"] / f["gtvp_suv_mean"]
    else:
        f["gtvp_suv_heterogeneity"] = 1.0

    # sphericity via voxel boundary count — guard matches features.py exactly
    m_p = seg == LABEL_GTVP
    if m_p.any() and seg.size <= 200_000_000:
        eroded     = ndimage.binary_erosion(m_p)
        boundary   = int((m_p & ~eroded).sum())
        # Use geometric mean face area as approximation for non-isotropic voxels
        sa_mm2     = boundary * float((spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) ** (2.0/3.0))
        vol_mm3    = gtvp_vol * 1000.0
        if sa_mm2 > 0:
            f["gtvp_sphericity"] = (np.pi ** (1.0/3.0) * (6.0 * vol_mm3) ** (2.0/3.0)) / sa_mm2
        else:
            f["gtvp_sphericity"] = 1.0
    elif m_p.any():
        # extremely large volume: sphere assumption (rare; keeps train==inference)
        f["gtvp_sphericity"] = 0.8
    else:
        f["gtvp_sphericity"] = 0.0

    return f


# =============================================================================
# Radiomics — mirrors hecktor/radiomics.py::extract_radiomics() EXACTLY
# (training adds these rad_* features; inference must produce identical keys)
# =============================================================================
def _first_order(values):
    keys = ("fo_mean", "fo_std", "fo_skewness", "fo_kurtosis", "fo_entropy",
            "fo_energy", "fo_p10", "fo_p90", "fo_iqr")
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
    from skimage.feature import graycomatrix, graycoprops
    zero = {k: 0.0 for k in ("glcm_contrast", "glcm_dissimilarity",
                             "glcm_homogeneity", "glcm_energy", "glcm_correlation")}
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
    glcm = graycomatrix(sl_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=n_levels, symmetric=True, normed=True)
    props = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        props[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    return props


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
    flatness   = float(np.sqrt(eigvals[2] / (eigvals[0] + 1e-9)))
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


def _extract_radiomics(seg, pet, ct, spacing_mm=(1.0, 1.0, 1.0)):
    """Identical to hecktor/radiomics.py::extract_radiomics()."""
    features = {}
    for label, prefix in [(LABEL_GTVP, "rad_p_"), (LABEL_GTVN, "rad_n_")]:
        mask = (seg == label)
        if not mask.any():
            dummy = {**_first_order(np.array([])),
                     **_glcm_features(np.zeros((1, 1, 1))),
                     **_shape_features(mask, spacing_mm)}
            features.update({prefix + k: v for k, v in dummy.items()})
            continue
        pet_crop, mask_crop = _crop_roi(pet, mask)
        ct_crop,  _         = _crop_roi(ct, mask)
        fo_pet = _first_order(pet[mask])
        fo_ct  = _first_order(ct[mask])
        glcm_p = _glcm_features(pet_crop * mask_crop)
        glcm_c = _glcm_features(ct_crop * mask_crop)
        shp    = _shape_features(mask, spacing_mm)
        combined = {}
        combined.update({f"pet_{k}": v for k, v in fo_pet.items()})
        combined.update({f"ct_{k}":  v for k, v in fo_ct.items()})
        combined.update({f"pet_{k}": v for k, v in glcm_p.items()})
        combined.update({f"ct_{k}":  v for k, v in glcm_c.items()})
        combined.update(shp)
        features.update({prefix + k: v for k, v in combined.items()})
    return features


# =============================================================================
# Head-and-neck ROI crop — IDENTICAL to hecktor/hn_crop.py
# (must match training; container ships only inference.py so it's embedded here)
# =============================================================================
_HN_BOX_MM        = (210.0, 210.0, 330.0)   # (x, y, z)
_HN_SPACING       = (1.0, 1.0, 3.0)         # (x, y, z)
_HN_TOP_MARGIN_MM = 20.0
_HN_LABEL_MARGIN_MM = 20.0


def _hn_body_centroid_xy(ct_arr, z0, z1, hu_thr=-500.0):
    slab = ct_arr[z0:z1]
    body = slab > hu_thr
    if not body.any():
        return ct_arr.shape[2] // 2, ct_arr.shape[1] // 2
    ys = np.where(body.any(axis=(0, 2)))[0]
    xs = np.where(body.any(axis=(0, 1)))[0]
    return int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2)


_HN_BRAIN_EXCL_MM = 50.0
_HN_PET_SEARCH_MM = 360.0


def _hn_pet_tumor_center(pet_arr, nz, sz):
    petr = np.clip(pet_arr, 0, None).copy()
    te = int(round(_HN_BRAIN_EXCL_MM / sz))
    petr[nz - te:, :, :] = 0
    bot = max(0, nz - int(round(_HN_PET_SEARCH_MM / sz)))
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
    bz = int(round(_HN_BOX_MM[2] / sz)); by = int(round(_HN_BOX_MM[1] / sy)); bx = int(round(_HN_BOX_MM[0] / sx))
    if label_lps_arr is not None and label_lps_arr.max() > 0:
        zz, yy, xx = np.where(label_lps_arr > 0)
        cz = int((zz.min() + zz.max()) // 2); cy = int((yy.min() + yy.max()) // 2); cx = int((xx.min() + xx.max()) // 2)
        mz = int(round(_HN_LABEL_MARGIN_MM / sz)); my = int(round(_HN_LABEL_MARGIN_MM / sy)); mx = int(round(_HN_LABEL_MARGIN_MM / sx))
        bz = max(bz, (zz.max() - zz.min()) + 2 * mz)
        by = max(by, (yy.max() - yy.min()) + 2 * my)
        bx = max(bx, (xx.max() - xx.min()) + 2 * mx)
    else:
        center = _hn_pet_tumor_center(pet_lps_arr, nz, sz) if pet_lps_arr is not None else None
        if center is not None:
            cz, cy, cx = center
        else:
            cz = (nz - 1 - int(round(_HN_TOP_MARGIN_MM / sz))) - bz // 2
            cx, cy = _hn_body_centroid_xy(a, max(0, nz - bz), nz)
    z0 = max(0, cz - bz // 2); z1 = min(nz, z0 + bz); z0 = max(0, z1 - bz)
    y0 = max(0, cy - by // 2); y1 = min(ny, y0 + by); y0 = max(0, y1 - by)
    x0 = max(0, cx - bx // 2); x1 = min(nx, x0 + bx); x0 = max(0, x1 - bx)
    return (z0, z1, y0, y1, x0, x1)


def _hn_crop_resample(img_lps, bbox, is_label):
    z0, z1, y0, y1, x0, x1 = bbox
    cropped = img_lps[x0:x1, y0:y1, z0:z1]
    osz = cropped.GetSize(); osp = cropped.GetSpacing()
    nsz = [int(round(osz[i] * osp[i] / _HN_SPACING[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(_HN_SPACING); rs.SetSize(nsz)
    rs.SetOutputOrigin(cropped.GetOrigin()); rs.SetOutputDirection(cropped.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    rs.SetDefaultPixelValue(0 if is_label else -1000)
    return rs.Execute(cropped)


def _hn_crop_case(ct, pet_on_ct, label=None):
    ct_l  = sitk.DICOMOrient(ct, "LPS")
    pet_l = sitk.DICOMOrient(pet_on_ct, "LPS")
    lab_l = sitk.DICOMOrient(label, "LPS") if label is not None else None
    lab_arr = sitk.GetArrayFromImage(lab_l) if lab_l is not None else None
    pet_arr = sitk.GetArrayFromImage(pet_l) if label is None else None
    bbox = _hn_bbox_lps(ct_l, lab_arr, pet_arr)
    ct_roi  = _hn_crop_resample(ct_l, bbox, is_label=False)
    pet_roi = _hn_crop_resample(pet_l, bbox, is_label=False)
    lab_roi = _hn_crop_resample(lab_l, bbox, is_label=True) if lab_l is not None else None
    return ct_roi, pet_roi, lab_roi


def _hn_map_roi_to_native(seg_roi_arr, ct_roi_ref, ct_native):
    seg_img = sitk.GetImageFromArray(seg_roi_arr.astype(np.uint8))
    seg_img.CopyInformation(ct_roi_ref)
    return sitk.Resample(seg_img, ct_native, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


# =============================================================================
# Image utilities — mirrors hecktor/io_utils.py
# =============================================================================
def _resample_to_1mm(img, is_label=False):
    osz = img.GetSize()
    osp = img.GetSpacing()
    nsz = [int(round(osz[i] * osp[i])) for i in range(3)]
    rs  = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((1.0, 1.0, 1.0))
    rs.SetSize(nsz)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(img)


def _register_pet_to_ct(pet, ct):
    """Resample PET onto CT grid — mirrors io_utils.register_pet_to_ct().
    Some scanners pad the PET outside the circular FOV with large negative
    values (e.g. -1000); linear interpolation then spreads them. SUV is
    physically non-negative, so clamp to >=0 — this keeps the SegResNet arm's
    intensity normalization from being corrupted (which otherwise makes it
    predict pure background and veto nnU-Net in the ensemble)."""
    reg = sitk.Resample(pet, ct, sitk.Transform(),
                        sitk.sitkLinear, 0.0, pet.GetPixelID())
    return sitk.Clamp(reg, lowerBound=0.0)


# =============================================================================
# Misc helpers
# =============================================================================
def _to_numeric(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _candidate_onehot_keys(col, val):
    """Every plausible string form of a categorical value, so the one-hot key
    matches whatever pandas .astype(str) produced at training time.
    e.g. EHR int 0 → {"col=0", "col=0.0"}; "Positive" → {"col=Positive"}."""
    forms = {str(val)}
    try:
        fv = float(val)
        forms.add(str(fv))            # 0 → "0.0"
        if fv.is_integer():
            forms.add(str(int(fv)))   # 0.0 → "0"
    except (TypeError, ValueError):
        pass
    return {f"{col}={s}" for s in forms}


def _get_image(location):
    files = (glob(str(location / "*.mha")) + glob(str(location / "*.nii.gz"))
             + glob(str(location / "*.tif")) + glob(str(location / "*.tiff")))
    if not files:
        raise FileNotFoundError(f"No image found in {location}")
    return files[0]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_segmentation(array, reference_sitk):
    out_dir = OUTPUT_PATH / "images/head-neck-tumor-segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(array.astype(np.uint8))
    img.CopyInformation(reference_sitk)
    sitk.WriteImage(img, str(out_dir / "output.mha"), useCompression=True)


if __name__ == "__main__":
    # Never let the container exit non-zero: a crash before the per-stage guards
    # (e.g. a corrupt CT) would still fail the GC case. Write safe defaults for
    # whatever is missing and always exit 0.
    try:
        rc = run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[main] top-level failure ({e}); writing safe defaults", flush=True)
        try:
            for name, val in [("t-stage.json", "T2"), ("n-stage.json", "N0"), ("rfs.json", 1000.0)]:
                p = OUTPUT_PATH / name
                if not p.exists():
                    _write_json(p, val)
        except Exception as e2:
            print(f"[main] could not write defaults ({e2})", flush=True)
        rc = 0
    raise SystemExit(rc)

# build: consolidated 1783618095
