#!/usr/bin/env python3
"""
Stream feature extraction from zip — no full extraction needed.

Reads each case directly from dataset.zip, computes:
  - geometric + SUV features (hecktor/features.py)
  - radiomics features (hecktor/radiomics.py)

Skips test cases (no label leakage).

Usage:
  python scripts/03b_stream_features.py \
      --zip /home/hecktor/data/dataset.zip \
      --split work/split_full.json \
      --csv /home/hecktor/data/extracted/HECKTOR_2026_training_data.csv \
      --out work/features_full.csv \
      --resume
"""
import argparse, json, os, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hecktor.features import geometric_features
from hecktor.radiomics import extract_radiomics
from hecktor.io_utils import register_pet_to_ct


def process_case(case_id, case_dir: Path):
    ct_path  = case_dir / f"{case_id}__CT.nii.gz"
    pet_path = case_dir / f"{case_id}__PT.nii.gz"
    seg_path = case_dir / f"{case_id}.nii.gz"

    if not ct_path.exists() or not pet_path.exists() or not seg_path.exists():
        return None

    ct_sitk  = sitk.ReadImage(str(ct_path))
    pet_sitk = sitk.ReadImage(str(pet_path))
    seg_sitk = sitk.ReadImage(str(seg_path))

    # Register PET to CT
    pet_reg = register_pet_to_ct(pet_sitk, ct_sitk)

    # Resample seg to CT space if needed
    if seg_sitk.GetSize() != ct_sitk.GetSize():
        seg_sitk = sitk.Resample(seg_sitk, ct_sitk, sitk.Transform(),
                                 sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    sp = ct_sitk.GetSpacing()  # (x, y, z)
    # numpy arrays are (z, y, x); spacing must match that axis order
    spacing_zyx = (sp[2], sp[1], sp[0])

    seg_arr = sitk.GetArrayFromImage(seg_sitk).astype(np.uint8)
    pet_arr = np.clip(sitk.GetArrayFromImage(pet_reg).astype(np.float32), 0, None)
    ct_arr  = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)

    # Geometric + SUV features at native spacing
    geo = geometric_features(seg_arr, pet_arr, spacing_mm=spacing_zyx)

    # Radiomics features at native spacing
    rad = extract_radiomics(seg_arr, pet_arr, ct_arr, spacing_mm=spacing_zyx)

    return {**{"PatientID": case_id}, **geo, **rad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip",    default="/home/hecktor/data/dataset.zip")
    ap.add_argument("--split",  default="work/split_full.json")
    ap.add_argument("--csv",    default="/home/hecktor/data/extracted/HECKTOR_2026_training_data.csv")
    ap.add_argument("--out",    default="work/features_full.csv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.split) as f:
        split_map = json.load(f)

    # Only extract for train + val (NOT test — no leakage)
    target_cases = set(split_map["train"] + split_map["val"])
    print(f"Extracting features for {len(target_cases)} cases (train+val only)")

    # Resume: skip already-done cases
    done = set()
    if args.resume and Path(args.out).exists():
        done = set(pd.read_csv(args.out)["PatientID"].tolist())
        print(f"Resuming — {len(done)} already done, {len(target_cases)-len(done)} remaining")

    todo = target_cases - done
    rows = []

    # --- Process already-extracted cases first ---
    extracted_root = Path("/home/hecktor/data/extracted")
    extracted_done = set()
    for split_name in ("train", "val"):
        split_dir = extracted_root / split_name
        if not split_dir.exists():
            continue
        for case_dir in split_dir.iterdir():
            cid = case_dir.name
            if cid not in todo:
                continue
            print(f"  [extracted] {cid}", end=" ", flush=True)
            row = process_case(cid, case_dir)
            if row:
                rows.append(row)
                extracted_done.add(cid)
                print("✓", flush=True)
            else:
                print("✗", flush=True)

    todo -= extracted_done
    print(f"\n{len(todo)} cases remaining from zip...")

    # --- Stream from zip ---
    with zipfile.ZipFile(args.zip) as zf:
        from collections import defaultdict
        case_members = defaultdict(list)
        for m in zf.namelist():
            parts = m.split("/")
            if len(parts) >= 3 and parts[0] == "HECKTOR 2026 Training Data":
                cid = parts[1]
                if cid and cid in todo:
                    case_members[cid].append(m)

        total = len(case_members)
        for i, (cid, members) in enumerate(sorted(case_members.items()), 1):
            print(f"[{i}/{total}] {cid}", end=" ", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_case = Path(tmp) / cid
                tmp_case.mkdir()
                for m in members:
                    fname = Path(m).name
                    if fname and "._" not in fname:
                        with zf.open(m) as src:
                            (tmp_case / fname).write_bytes(src.read())
                row = process_case(cid, tmp_case)
                if row:
                    rows.append(row)
                    print("✓", flush=True)
                else:
                    print("✗", flush=True)

            # Append to CSV incrementally every 50 cases
            if len(rows) % 50 == 0 and rows:
                _save(rows, args.out, done)

    _save(rows, args.out, done)
    print(f"\nDone. Features saved → {args.out}")


def _save(rows, out_path, done_set):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if done_set and Path(out_path).exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="PatientID", keep="last")
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
