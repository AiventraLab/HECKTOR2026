"""I/O and geometry helpers built on SimpleITK (matches the official
preprocessing in BioMedIA-MBZUAI/HECKTOR2026)."""
from __future__ import annotations
import numpy as np
import SimpleITK as sitk


def read(path: str) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write(img: sitk.Image, path: str, compress: bool = True) -> None:
    sitk.WriteImage(img, str(path), useCompression=compress)


def arr(img: sitk.Image) -> np.ndarray:
    """SimpleITK image -> numpy array (z, y, x)."""
    return sitk.GetArrayFromImage(img)


def like(reference: sitk.Image, array: np.ndarray, dtype=np.uint8) -> sitk.Image:
    """Wrap a numpy array as an image sharing ``reference`` geometry."""
    out = sitk.GetImageFromArray(array.astype(dtype))
    out.CopyInformation(reference)
    return out


def resample_to(img: sitk.Image, spacing=(1.0, 1.0, 1.0), is_label=False) -> sitk.Image:
    """Resample to isotropic ``spacing`` (mm). Labels use nearest neighbour."""
    osz = img.GetSize(); osp = img.GetSpacing()
    nsz = [int(round(osz[i] * osp[i] / spacing[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(spacing)
    rs.SetSize(nsz)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(img)


def resample_to_reference(img: sitk.Image, reference: sitk.Image, is_label=True) -> sitk.Image:
    """Resample ``img`` onto ``reference``'s grid (used to put a predicted mask
    back at the original CT geometry before writing output.mha)."""
    return sitk.Resample(
        img, reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline,
        0,
        img.GetPixelID(),
    )


def register_pet_to_ct(pet: sitk.Image, ct: sitk.Image) -> sitk.Image:
    """Resample PET onto the CT grid so the two share identical geometry
    (required before stacking as channels). Assumes prior rigid alignment, as
    in the HECKTOR data."""
    return sitk.Resample(pet, ct, sitk.Transform(), sitk.sitkLinear, 0.0, pet.GetPixelID())


def clip_ct(ct_arr: np.ndarray, lo=-250.0, hi=250.0) -> np.ndarray:
    """CT HU window used by the official preprocess.py."""
    return np.clip(ct_arr, lo, hi)


def zscore(a: np.ndarray, nonzero=True) -> np.ndarray:
    """Per-image z-score (PET). ``nonzero`` ignores background voxels."""
    mask = a != 0 if nonzero else np.ones_like(a, dtype=bool)
    if mask.sum() < 2:
        return a.astype(np.float32)
    m, s = a[mask].mean(), a[mask].std()
    return ((a - m) / (s + 1e-8)).astype(np.float32)
