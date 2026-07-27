"""Head-and-neck ROI cropping for HECKTOR 2026.

Whole-body PET/CT (USZ, CHUP, MDA) is up to 568 slices / 471 MB per case —
storing 704 of them as nnU-Net raw is ~158 GB, and nnU-Net preprocessed at
native spacing would be ~80 GB. Neither fits a storage-limited machine.

The task is head-and-neck: all GTVp + GTVn disease sits within ~0-34 cm below
the top of the head (measured across all 8 centers). Cropping to that ROI and
working at (1,1,3) mm:
  * shrinks raw to ~10 GB and preprocessed to ~25 GB (fits easily)
  * keeps fine in-plane resolution (1 mm) for small nodes
  * removes chest/abdomen/legs that only cause false positives

Orientation is normalised to LPS first (head at the max-z end), so the crop is
deterministic regardless of how a center stored its scan.

Train vs inference:
  * train  — center on the label, and expand the box so the FULL label is always
             inside (never train on a cut mask).
  * infer  — no label: center axially on the body, and place the box top a fixed
             distance below the top-of-head so the oropharynx + neck + supra-
             clavicular region is reliably contained.
"""
from __future__ import annotations
import numpy as np
import SimpleITK as sitk

# Physical ROI size (mm) and working spacing (mm). 21x21x33 cm comfortably holds
# bilateral nodal disease (official HECKTOR box was only 14.4 cm) and reaches
# 33 cm inferiorly (max observed disease depth was 31.7 cm).
BOX_MM      = (210.0, 210.0, 330.0)   # (x, y, z)
SPACING     = (1.0, 1.0, 3.0)         # (x, y, z) — fine in-plane, native-ish z
TOP_MARGIN_MM = 20.0                  # gap between top-of-head and box top
LABEL_MARGIN_MM = 20.0                # padding around label when expanding (train)


def _to_lps(img, is_label=False):
    return sitk.DICOMOrient(img, "LPS")


def _body_centroid_xy(ct_arr, z0, z1, hu_thr=-500.0):
    """Axial (x,y) centroid of body voxels in the superior slab [z0:z1]."""
    slab = ct_arr[z0:z1]
    body = slab > hu_thr
    if not body.any():
        return ct_arr.shape[2] // 2, ct_arr.shape[1] // 2
    ys = np.where(body.any(axis=(0, 2)))[0]
    xs = np.where(body.any(axis=(0, 1)))[0]
    return int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2)


BRAIN_EXCL_MM = 50.0    # drop top 5 cm (brain) when PET-locating the tumor
PET_SEARCH_MM = 360.0   # only search the superior 36 cm for the tumor hot-spot


def _pet_tumor_center(pet_arr, nz, sz):
    """(cz,cy,cx) of the PET hot-spot in the H&N band (brain excluded).
    The primary tumour is the dominant FDG-avid focus below the brain, so this
    reliably localizes it for the inference crop. Returns None if no uptake."""
    petr = np.clip(pet_arr, 0, None).copy()
    te = int(round(BRAIN_EXCL_MM / sz))
    petr[nz - te:, :, :] = 0                          # head at max-z → drop brain
    bot = max(0, nz - int(round(PET_SEARCH_MM / sz)))
    petr[:bot, :, :] = 0
    if petr.max() <= 0:
        return None
    thr = np.percentile(petr[petr > 0], 99)
    zz, yy, xx = np.where(petr >= thr)
    return int(zz.mean()), int(yy.mean()), int(xx.mean())


def compute_bbox_lps(ct_lps, label_lps_arr=None, pet_lps_arr=None):
    """Return an (z0,z1,y0,y1,x0,x1) array-index box in the LPS-oriented CT.

    Train (label given): center on the label and enlarge so the whole mask fits.
    Inference (no label): center on the PET tumor hot-spot — this guarantees the
    primary is inside the crop AND matches the label-centered training FOV
    (anatomical top-of-head centering clipped the tumour in ~18% of cases).
    """
    a = sitk.GetArrayFromImage(ct_lps)          # (z, y, x)
    nz, ny, nx = a.shape
    sx, sy, sz = ct_lps.GetSpacing()            # mm (x, y, z)

    bz = int(round(BOX_MM[2] / sz))
    by = int(round(BOX_MM[1] / sy))
    bx = int(round(BOX_MM[0] / sx))

    if label_lps_arr is not None and label_lps_arr.max() > 0:
        zz, yy, xx = np.where(label_lps_arr > 0)
        cz = int((zz.min() + zz.max()) // 2)
        cy = int((yy.min() + yy.max()) // 2)
        cx = int((xx.min() + xx.max()) // 2)
        mz = int(round(LABEL_MARGIN_MM / sz)); my = int(round(LABEL_MARGIN_MM / sy)); mx = int(round(LABEL_MARGIN_MM / sx))
        bz = max(bz, (zz.max() - zz.min()) + 2 * mz)
        by = max(by, (yy.max() - yy.min()) + 2 * my)
        bx = max(bx, (xx.max() - xx.min()) + 2 * mx)
    else:
        center = _pet_tumor_center(pet_lps_arr, nz, sz) if pet_lps_arr is not None else None
        if center is not None:
            cz, cy, cx = center
        else:   # no PET uptake → fall back to anatomical top-of-head + body centroid
            cz = (nz - 1 - int(round(TOP_MARGIN_MM / sz))) - bz // 2
            cx, cy = _body_centroid_xy(a, max(0, nz - bz), nz)

    z0 = max(0, cz - bz // 2); z1 = min(nz, z0 + bz); z0 = max(0, z1 - bz)
    y0 = max(0, cy - by // 2); y1 = min(ny, y0 + by); y0 = max(0, y1 - by)
    x0 = max(0, cx - bx // 2); x1 = min(nx, x0 + bx); x0 = max(0, x1 - bx)
    return (z0, z1, y0, y1, x0, x1)


def _crop_resample(img_lps, bbox, is_label):
    """Crop the LPS image to bbox (array indices) then resample to SPACING."""
    z0, z1, y0, y1, x0, x1 = bbox
    # SimpleITK indexing is (x, y, z)
    cropped = img_lps[x0:x1, y0:y1, z0:z1]
    osz = cropped.GetSize(); osp = cropped.GetSpacing()
    nsz = [int(round(osz[i] * osp[i] / SPACING[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(SPACING)
    rs.SetSize(nsz)
    rs.SetOutputOrigin(cropped.GetOrigin())
    rs.SetOutputDirection(cropped.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1000 if not is_label else 0)
    return rs.Execute(cropped)


def crop_case(ct, pet_on_ct, label=None):
    """Crop CT, PET (already on CT grid), and optional label to the H&N ROI at
    SPACING. Returns (ct_roi, pet_roi, label_roi_or_None) as SimpleITK images
    sharing identical geometry. ``label`` presence selects train-mode centering.
    """
    ct_l = _to_lps(ct)
    pet_l = _to_lps(pet_on_ct)
    lab_l = _to_lps(label, is_label=True) if label is not None else None
    lab_arr = sitk.GetArrayFromImage(lab_l) if lab_l is not None else None
    pet_arr = sitk.GetArrayFromImage(pet_l) if label is None else None

    bbox = compute_bbox_lps(ct_l, lab_arr, pet_arr)
    ct_roi = _crop_resample(ct_l, bbox, is_label=False)
    pet_roi = _crop_resample(pet_l, bbox, is_label=False)
    lab_roi = _crop_resample(lab_l, bbox, is_label=True) if lab_l is not None else None
    return ct_roi, pet_roi, lab_roi


def map_roi_seg_to_native(seg_roi_arr, ct_roi_ref, ct_native):
    """Map a predicted ROI segmentation (numpy in ct_roi geometry) back onto the
    original native CT grid (zeros outside the ROI). Used at inference so output
    matches the input CT exactly."""
    seg_img = sitk.GetImageFromArray(seg_roi_arr.astype(np.uint8))
    seg_img.CopyInformation(ct_roi_ref)
    return sitk.Resample(seg_img, ct_native, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
