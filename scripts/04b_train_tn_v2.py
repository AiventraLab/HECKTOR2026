#!/usr/bin/env python3
"""
Improved TN staging: for T-stage, compare LightGBM vs CORN-ordinal vs their
probability blend (5-fold OOF balanced accuracy) and keep the winner; for
N-stage keep tuned LightGBM. Evaluates on the held-out VAL split (GT-mask
features) and saves bundles in a format inference understands:

  T bundle: {"kind":"ensemble", "lgb":model, "corn":corn_bundle|None,
             "blend_w":w, "classes":[...], "feature_cols":[...]}
  N bundle: {"kind":"lgb", "lgb":model, "classes":[...], "feature_cols":[...]}

Usage:
  python scripts/04b_train_tn_v2.py --features work/features_full.csv \
      --csv .../HECKTOR_2026_training_data.csv --split work/split_full.json --out work
"""
import argparse, json, pickle
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score

from hecktor import columns, staging


def _eval_val(bundle, Xval, yval_str, classes):
    yv = np.array([classes.index(v) for v in yval_str])
    fc = bundle["feature_cols"]
    Xv = Xval.reindex(columns=fc, fill_value=0.0)
    if bundle["kind"] == "ensemble" and bundle.get("corn") is not None and bundle["blend_w"] < 1.0:
        p = (bundle["blend_w"] * bundle["lgb"].predict_proba(Xv)
             + (1 - bundle["blend_w"]) * staging.corn_class_proba(bundle["corn"], Xv.values))
        pred = p.argmax(1)
    else:
        pred = bundle["lgb"].predict(Xv)
    return balanced_accuracy_score(yv, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="work")
    ap.add_argument("--n_trials", type=int, default=120)
    ap.add_argument("--broader", action="store_true",
                    help="Use expanded hyperparameter search space")
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text())
    train_ids, val_ids = set(split["train"]), set(split["val"])
    feats = pd.read_csv(args.features).set_index("PatientID")
    clin = columns.load_clinical(args.csv)

    # encode clinical on the FULL frame so train/val share columns
    idx = feats.index.intersection(clin.index)
    clin_enc = columns.encode_clinical(clin.loc[idx])
    X_all = feats.loc[idx].join(clin_enc).fillna(0.0)
    Xtr_all = X_all[X_all.index.isin(train_ids)]
    Xval_all = X_all[X_all.index.isin(val_ids)]

    for target, classes, tag in ((columns.T_TARGET, staging.T_CLASSES, "T"),
                                 (columns.N_TARGET, staging.N_CLASSES, "N")):
        y = clin.loc[Xtr_all.index, target].astype(str)
        keep = y.isin(classes)
        Xk, yk = Xtr_all[keep.values], y[keep]
        print(f"\n===== {tag}-stage =====  train n={len(Xk)}")

        best = staging.tune_lightgbm(Xk, yk, classes, n_trials=args.n_trials,
                                       broader=args.broader)
        lgb_cv = staging.cv_lightgbm(Xk, yk, classes, params=best)
        print(f"[{tag}] LightGBM CV balanced_acc = {lgb_cv['balanced_accuracy']:.3f}")

        if tag == "T":
            ens = staging.cv_ensemble_t(Xk, yk, classes, lgb_params=best)
            print(f"[T] CV  lgb={ens['lgb']:.3f}  corn={ens['corn']:.3f}  "
                  f"blend={ens['blend']:.3f} (w={ens['blend_w']:.2f})")
            use_blend = ens["blend"] >= max(ens["lgb"], ens["corn"]) and ens["blend_w"] < 1.0
            lgb_final = staging.fit_full(Xk, yk, classes, params=best)
            corn_final = staging.fit_corn_full(Xk, yk, classes) if (use_blend or ens["corn"] > ens["lgb"]) else None
            blend_w = ens["blend_w"] if use_blend else (0.0 if (corn_final is not None and ens["corn"] > ens["lgb"]) else 1.0)
            bundle = {"kind": "ensemble", "lgb": lgb_final, "corn": corn_final,
                      "blend_w": blend_w, "classes": classes, "feature_cols": list(X_all.columns)}
        else:
            lgb_final = staging.fit_full(Xk, yk, classes, params=best)
            bundle = {"kind": "lgb", "lgb": lgb_final, "classes": classes,
                      "feature_cols": list(X_all.columns)}

        # held-out VAL eval
        yval = clin.loc[Xval_all.index, target].astype(str)
        vkeep = yval.isin(classes)
        if vkeep.sum() > 0:
            va_ba = _eval_val(bundle, Xval_all[vkeep.values], yval[vkeep], list(classes))
            print(f"[{tag}] HELD-OUT VAL balanced_acc = {va_ba:.3f}  (n={int(vkeep.sum())})")

        with open(Path(args.out) / f"tn_{tag}.pkl", "wb") as f:
            pickle.dump(bundle, f)
        print(f"[{tag}] saved tn_{tag}.pkl  (kind={bundle['kind']}"
              + (f", blend_w={bundle['blend_w']:.2f}" if bundle['kind'] == 'ensemble' else "") + ")")


if __name__ == "__main__":
    main()
