"""Patient-level cross-validation splits.

Stratify on the event indicator (so censoring rate is stable across folds) and,
when available, keep centers balanced. One split file is reused by every
subtask so segmentation / TN / prognosis all train on identical folds.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_folds(df: pd.DataFrame, n_splits=5, seed=42,
               strat_col="Relapse", id_index=True) -> dict:
    """Return {case_id: fold_index}. ``df`` indexed by PatientID (id_index)."""
    ids = df.index.to_list() if id_index else df["PatientID"].to_list()
    if strat_col in df.columns:
        y = df[strat_col].fillna(0).astype(int).to_numpy()
    else:
        y = np.zeros(len(ids), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = {}
    for k, (_, val_idx) in enumerate(skf.split(ids, y)):
        for i in val_idx:
            fold[ids[i]] = k
    return fold


def save_folds(fold: dict, path: str):
    with open(path, "w") as f:
        json.dump(fold, f, indent=2)


def load_folds(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def from_manifest(manifest_path: str) -> tuple:
    """Load split_manifest.json produced by extract_subsample.py.

    Returns (case_dirs, fold_map) where:
      case_dirs  — {patient_id: absolute_path_to_patient_folder}
      fold_map   — {patient_id: fold_index}  train→0, val→1  (test excluded)
    """
    from pathlib import Path
    root = Path(manifest_path).parent
    with open(manifest_path) as f:
        manifest = json.load(f)
    case_dirs: dict = {}
    fold_map: dict = {}
    for pid, info in manifest["patients"].items():
        split = info["split"]
        case_dirs[pid] = str(root / split / pid)
        if split == "train":
            fold_map[pid] = 0
        elif split == "val":
            fold_map[pid] = 1
        # test cases are recorded in case_dirs but excluded from fold_map
    return case_dirs, fold_map
