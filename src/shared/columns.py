"""Clinical CSV schema for HECKTOR 2026 and helpers to build the tabular
feature matrix.

Confirmed from the official repo (BioMedIA-MBZUAI/HECKTOR2026):
  - training CSV: HECKTOR_2026_Training.csv
  - imaging:      <CASE>/<CASE>__CT.nii.gz, <CASE>__PT.nii.gz (PET = "PT")
  - label:        <CASE>/<CASE>.nii.gz   (0 background, 1 GTVp, 2 GTVn)
  - survival:     time column "RFS" (days), event column "Relapse" (1=event)
  - TN targets:   "T_stage" (T1-T4), "N_stage" (N0-N3)  [training-only]

Multi-word column capitalisation should be re-verified against the real CSV
once data access is granted; they are centralised here so a single edit
propagates everywhere.
"""
from __future__ import annotations
import pandas as pd

ID_COL = "PatientID"
CENTER_COL = "Center"

# Clinical fields available at BOTH train and test time (model inputs).
CLINICAL_NUMERIC = ["Age"]
CLINICAL_CATEGORICAL = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "HPV Status",      # may be missing; also the optional HPV-diagnosis target
    "M-stage",
    "Treatment",
]
CLINICAL_INPUTS = CLINICAL_NUMERIC + CLINICAL_CATEGORICAL

# Training-only labels (NOT provided at inference).
T_TARGET = "T-stage"
N_TARGET = "N-stage"
SURV_TIME = "RFS"
SURV_EVENT = "Relapse"

# Segmentation label values.
LABEL_BG, LABEL_GTVP, LABEL_GTVN = 0, 1, 2


def load_clinical(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.set_index(ID_COL) if ID_COL in df.columns else df
    return df


def encode_clinical(df: pd.DataFrame, fit_on: pd.DataFrame | None = None) -> pd.DataFrame:
    """One-hot encode categoricals + keep numerics. ``fit_on`` lets a test
    frame reuse the training column space (pass the training frame).  Missing
    values are preserved as an explicit ``<col>_missing`` indicator so that
    'unknown HPV' carries signal instead of being silently imputed."""
    src = fit_on if fit_on is not None else df
    out = {}
    for col in CLINICAL_NUMERIC:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    base = pd.DataFrame(out, index=df.index)
    for col in CLINICAL_CATEGORICAL:
        if col not in df.columns:
            continue
        base[f"{col}_missing"] = df[col].isna().astype(int)
        cats = sorted(src[col].dropna().astype(str).unique())
        for c in cats:
            base[f"{col}={c}"] = (df[col].astype(str) == c).astype(int)
    return base
