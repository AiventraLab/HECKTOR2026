#!/usr/bin/env python3
"""
Train RSF + Cox ensemble prognosis model on full dataset.

Strategy (research-backed):
  1. Random Survival Forest (RSF) — handles nonlinearity, interactions, missing
  2. Lifelines CoxPH — strong linear baseline, generalises well
  3. Rank-average ensemble (RSF 0.6 + Cox 0.4) — scale-free, robust

Features: geometric (27) + radiomics (from hecktor/radiomics.py)
Clinical:  Age, Gender, Tobacco, Alcohol, Performance Status, Treatment, HPV

Saved: work/prognosis_rsf.pkl, work/prognosis_cox_full.pkl, work/prognosis_ensemble.pkl

Usage:
  python scripts/05b_train_prognosis_full.py \
      --features work/features_full.csv \
      --csv /home/hecktor/data/extracted/HECKTOR_2026_training_data.csv \
      --split work/split_full.json \
      --out work
"""
import argparse, pickle
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import rankdata

from hecktor import columns
from hecktor.metrics import c_index
from hecktor.prognosis import make_y

COX_CLINICAL = ["Age", "Gender", "Tobacco Consumption", "Alcohol Consumption",
                "Performance Status", "Treatment", "HPV Status"]


def _build_X(feats: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    X = feats.copy().astype(float)
    for col in COX_CLINICAL:
        X[col] = pd.to_numeric(clin.get(col, np.nan), errors="coerce")
    return X


def _tune_rsf(X_imp, y, seed=42, broader=True):
    """Grid search using OOB C-index to avoid overfitting on train set."""
    from sksurv.ensemble import RandomSurvivalForest
    best_c, best_params = 0, {}
    if broader:
        grid = (
            [dict(n_estimators=n, min_samples_leaf=m, max_features=f)
             for n in (100, 200, 300, 500, 750, 1000)
             for m in (5, 10, 15, 20, 30, 50)
             for f in ("sqrt", 0.3, 0.5, 0.8)] +
            # deeper leaves for fine-grained patterns
            [dict(n_estimators=n, min_samples_leaf=m, max_features=f)
             for n in (300, 500)
             for m in (1, 2, 3)
             for f in ("sqrt", 0.5)]
        )
    else:
        grid = [
            dict(n_estimators=300, min_samples_leaf=10,  max_features="sqrt"),
            dict(n_estimators=300, min_samples_leaf=15,  max_features="sqrt"),
            dict(n_estimators=300, min_samples_leaf=20,  max_features=0.5),
            dict(n_estimators=500, min_samples_leaf=15,  max_features="sqrt"),
        ]
    for p in grid:
        # oob_score=True enables out-of-bag C-index — unbiased generalisation estimate
        m = RandomSurvivalForest(**p, oob_score=True, n_jobs=-1, random_state=seed)
        m.fit(X_imp, y)
        c = m.oob_score_   # Harrell C-index on OOB samples
        print(f"  RSF {p}  C-index(OOB)={c:.3f}")
        if c > best_c:
            best_c, best_params = c, p
    return best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="work/features_full.csv")
    ap.add_argument("--csv",      default="/home/hecktor/data/extracted/HECKTOR_2026_training_data.csv")
    ap.add_argument("--split",    default="work/split_full.json")
    ap.add_argument("--out",      default="work")
    ap.add_argument("--broader", action="store_true",
                    help="Expanded RSF grid search (72 combos) + GBS tuning")
    args = ap.parse_args()

    import json
    with open(args.split) as f:
        split_map = json.load(f)
    train_ids = set(split_map["train"])

    feats = pd.read_csv(args.features).set_index("PatientID")
    clin  = columns.load_clinical(args.csv)

    # Intersect train IDs
    idx = feats.index.intersection(clin.index).intersection(list(train_ids))
    feats = feats.loc[idx]; clin = clin.loc[idx]

    # Survival labels
    df = feats.join(clin[[columns.SURV_TIME, columns.SURV_EVENT]]).dropna(
        subset=[columns.SURV_TIME, columns.SURV_EVENT])
    feats = feats.loc[df.index]; clin = clin.loc[df.index]

    X = _build_X(feats, clin)
    y = make_y(df)

    print(f"Training on {len(X)} cases with survival data")
    print(f"Features: {X.shape[1]}")
    print(f"Events: {df[columns.SURV_EVENT].sum():.0f}/{len(df)}")

    # Impute + scale for Cox
    imp  = SimpleImputer(strategy="median")
    scl  = StandardScaler()
    X_imp_raw = imp.fit_transform(X)
    X_scaled  = scl.fit_transform(X_imp_raw)
    X_imp = pd.DataFrame(X_imp_raw, columns=X.columns, index=X.index)

    # ── RSF ──────────────────────────────────────────────────────────────────
    print("\n=== Random Survival Forest ===")
    from sksurv.ensemble import RandomSurvivalForest
    best_rsf_params = _tune_rsf(X_imp_raw, y, broader=args.broader)
    print(f"Best RSF params: {best_rsf_params}")
    rsf = RandomSurvivalForest(**best_rsf_params, n_jobs=-1, random_state=42)
    rsf.fit(X_imp_raw, y)
    rsf_risk = rsf.predict(X_imp_raw)
    rsf_c = c_index(df[columns.SURV_TIME], df[columns.SURV_EVENT], rsf_risk)
    print(f"RSF C-index (train): {rsf_c:.3f}")

    # ── Gradient-Boosted Survival (sksurv) — strong nonlinear third arm ────────
    print("\n=== Gradient-Boosted Survival ===")
    gbs, gbs_risk, gbs_c = None, None, float("nan")
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sklearn.model_selection import KFold
        best_gbs_c, best_gbs = 0, None
        gbs_grid = [
            dict(n_estimators=n, learning_rate=lr, max_depth=d, subsample=s,
                 min_samples_leaf=ms, random_state=42)
            for n in (100, 200, 300, 500)
            for lr in (0.01, 0.05, 0.1)
            for d in (2, 3, 4)
            for s in (0.7, 0.8, 1.0)
            for ms in (5, 10, 20)
        ]
        # limit to a diverse subset to avoid excessive runtime
        import random; random.seed(42)
        gbs_grid = random.sample(gbs_grid, min(len(gbs_grid), 40))
        for gbs_p in gbs_grid:
            gbs_cv = GradientBoostingSurvivalAnalysis(**gbs_p)
            cv_c = []
            for tr, va in KFold(3, shuffle=True, random_state=42).split(X_imp_raw):
                gbs_cv.fit(X_imp_raw[tr], y[tr])
                cv_c.append(c_index(
                    df.iloc[va][columns.SURV_TIME], df.iloc[va][columns.SURV_EVENT],
                    gbs_cv.predict(X_imp_raw[va])))
            cv_c = float(np.mean(cv_c))
            print(f"  GBS {gbs_p}  C-index(CV)={cv_c:.3f}")
            if cv_c > best_gbs_c:
                best_gbs_c, best_gbs = cv_c, gbs_p
        print(f"Best GBS params: {best_gbs}  (CV C-index={best_gbs_c:.3f})")
        gbs = GradientBoostingSurvivalAnalysis(**best_gbs)
        gbs.fit(X_imp_raw, y)
        gbs_risk = gbs.predict(X_imp_raw)
        gbs_c = c_index(df[columns.SURV_TIME], df[columns.SURV_EVENT], gbs_risk)
        print(f"GBS C-index (train): {gbs_c:.3f}")
    except Exception as e:
        print(f"GBS failed ({type(e).__name__}: {str(e)[:80]}) — skipping")
        gbs = None

    # ── Cox (lifelines) — robust: drop zero-variance, standardize, penalize ────
    print("\n=== Cox PH (lifelines) ===")
    from lifelines import CoxPHFitter

    # Zero-variance / near-constant columns make the Cox information matrix
    # singular → NaN delta → ConvergenceError. Drop them, then standardize
    # (lifelines converges far better on z-scored features, esp. radiomics).
    cox_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    dropped = [c for c in X.columns if c not in cox_cols]
    print(f"Cox uses {len(cox_cols)}/{X.shape[1]} features (dropped {len(dropped)} constant)")
    cox_meds = X[cox_cols].median()
    cox_scaler = StandardScaler()
    Xc = pd.DataFrame(
        cox_scaler.fit_transform(X[cox_cols].fillna(cox_meds)),
        columns=cox_cols, index=X.index)

    def _fit_cox(pen):
        tr = Xc.copy()
        tr[columns.SURV_TIME]  = df[columns.SURV_TIME].values
        tr[columns.SURV_EVENT] = df[columns.SURV_EVENT].values
        m = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
        m.fit(tr, duration_col=columns.SURV_TIME, event_col=columns.SURV_EVENT,
              fit_options={"step_size": 0.3})
        return m

    cox_final, best_pen, best_c_cox = None, None, -1.0
    for pen in (0.5, 1.0, 2.0, 5.0, 10.0):
        try:
            m = _fit_cox(pen)
            risk = m.predict_log_partial_hazard(Xc).values
            c = c_index(df[columns.SURV_TIME], df[columns.SURV_EVENT], risk)
            print(f"  penalizer={pen}  C-index(train)={c:.3f}")
            if c > best_c_cox:
                best_c_cox, best_pen, cox_final = c, pen, m
        except Exception as e:
            print(f"  penalizer={pen} failed: {type(e).__name__}: {str(e)[:80]}")

    cox_ok = cox_final is not None
    if cox_ok:
        cox_risk = cox_final.predict_log_partial_hazard(Xc).values
        cox_c = c_index(df[columns.SURV_TIME], df[columns.SURV_EVENT], cox_risk)
        print(f"Best Cox penalizer={best_pen}  C-index (train): {cox_c:.3f}")
    else:
        cox_risk, cox_c = None, float("nan")
        print("Cox failed for all penalizers → ensemble will use RSF only.")

    # ── Ensemble (skill-weighted rank-average of RSF + Cox + GBS) ─────────────
    # Weight each arm by its skill above random (C-index - 0.5) so weak arms are
    # down-weighted automatically; weights are normalised over available models.
    print("\n=== Ensemble (RSF + Cox + GBS) ===")
    arms = {"rsf": (rsf_risk, rsf_c)}
    if cox_ok:            arms["cox"] = (cox_risk, cox_c)
    if gbs is not None:   arms["gbs"] = (gbs_risk, gbs_c)
    raw_w = {k: max(c - 0.5, 0.02) for k, (_, c) in arms.items()}
    tot = sum(raw_w.values())
    weights = {k: raw_w[k] / tot for k in raw_w}
    # default zero weight for any missing arm
    for k in ("rsf", "cox", "gbs"):
        weights.setdefault(k, 0.0)
    ens_rank = sum(weights[k] * (rankdata(arms[k][0]) / len(arms[k][0])) for k in arms)
    ens_c = c_index(df[columns.SURV_TIME], df[columns.SURV_EVENT], ens_rank)
    print(f"  weights: " + ", ".join(f"{k}={weights[k]:.2f}" for k in ('rsf','cox','gbs')))
    print(f"  Ensemble C-index (train): {ens_c:.3f}")

    # per-arm standardisation stats for single-case blending at inference
    rsf_stats = {"mean": float(np.mean(rsf_risk)), "std": float(np.std(rsf_risk) + 1e-8)}
    cox_stats = ({"mean": float(np.mean(cox_risk)), "std": float(np.std(cox_risk) + 1e-8)}
                 if cox_ok else {"mean": 0.0, "std": 1.0})
    gbs_stats = ({"mean": float(np.mean(gbs_risk)), "std": float(np.std(gbs_risk) + 1e-8)}
                 if gbs is not None else {"mean": 0.0, "std": 1.0})
    w_rsf, w_cox = weights["rsf"], weights["cox"]   # kept for summary print

    # ── Save all models ───────────────────────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rsf_bundle = {"model": rsf, "imputer": imp, "feature_cols": list(X.columns)}
    with open(out / "prognosis_rsf.pkl", "wb") as f:
        pickle.dump(rsf_bundle, f)
    print(f"Saved prognosis_rsf.pkl")

    cox_bundle = None
    if cox_ok:
        # scaler + cox_cols let inference reproduce the standardized Cox input
        cox_bundle = {"model": cox_final, "feature_cols": cox_cols,
                      "medians": cox_meds.to_dict(), "scaler": cox_scaler}
        with open(out / "prognosis_cox_full.pkl", "wb") as f:
            pickle.dump(cox_bundle, f)
        print(f"Saved prognosis_cox_full.pkl")

    # GBS bundle (shares RSF's imputer + full feature_cols at inference)
    gbs_bundle = {"model": gbs} if gbs is not None else None

    # Ensemble bundle (used by inference.py)
    ens_bundle = {
        "rsf": rsf_bundle,
        "cox": cox_bundle,                 # None if Cox failed
        "gbs": gbs_bundle,                 # None if GBS failed
        "weights": weights,                # {rsf, cox, gbs} normalised
        "feature_cols": list(X.columns),
        "rsf_stats": rsf_stats,            # per-arm z-score stats for blending
        "cox_stats": cox_stats,
        "gbs_stats": gbs_stats,
    }
    with open(out / "prognosis_ensemble.pkl", "wb") as f:
        pickle.dump(ens_bundle, f)
    print(f"Saved prognosis_ensemble.pkl  (← use this in inference.py)")

    print(f"\nSummary:")
    print(f"  RSF C-index:      {rsf_c:.3f}")
    print(f"  Cox C-index:      {cox_c if cox_ok else float('nan'):.3f}")
    print(f"  GBS C-index:      {gbs_c if gbs is not None else float('nan'):.3f}")
    print(f"  Ensemble C-index: {ens_c:.3f}  ← submission model  weights={weights}")


if __name__ == "__main__":
    main()
