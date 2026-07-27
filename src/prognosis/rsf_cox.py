"""RFS prognosis — continuous risk score (higher = higher recurrence risk),
evaluated by C-index (+ Brier). Tabular features: mask-derived + radiomics +
clinical.

Strategy proven across HECKTOR editions ("simplicity wins"):
  1. clinical-only Cox  -> the baseline to beat (~0.70 historically)
  2. ICARE (BaggedIcareSurvival) on all features  -> best ranking (won 2022)
  3. Deep-MTLR (pycox)  -> best calibration
  -> rank-average ICARE + MTLR for the final risk score.

Orientation contract: every ``predict_*`` returns risk where higher = higher
risk (concordant with sksurv's concordance_index_censored).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.shared.columns import SURV_TIME, SURV_EVENT


def make_y(df: pd.DataFrame):
    from sksurv.util import Surv
    return Surv.from_arrays(event=df[SURV_EVENT].astype(bool).to_numpy(),
                            time=df[SURV_TIME].astype(float).to_numpy())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _patch_icare():
    """sklearn>=1.6 renamed force_all_finite -> ensure_all_finite; patch icare."""
    try:
        import icare.base as _ib
        from sklearn.utils import check_array as _ca
        import functools
        orig = _ib.format_x
        def _fmt(X):
            import pandas as pd
            X = X.copy()
            _ca(X, ensure_all_finite=False)
            if not isinstance(X, pd.DataFrame):
                import numpy as np
                X = pd.DataFrame(data=X, columns=np.arange(X.shape[1]).astype("str"))
            return X
        _ib.format_x = _fmt
    except Exception:
        pass


def fit_icare(X: pd.DataFrame, y):
    """ICARE handles NaN natively -> no imputation needed."""
    _patch_icare()
    from icare.survival import BaggedIcareSurvival
    m = BaggedIcareSurvival(n_estimators=50, aggregation_method="mean",
                            n_jobs=1, random_state=42)
    m.fit(X, y)
    return m


def fit_cox(X: pd.DataFrame, y):
    """Ridge-penalised Cox (sksurv) — needs imputed, scaled, finite X."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    m = CoxPHSurvivalAnalysis(alpha=1.0)
    m.fit(X.values, y)
    return m


def fit_rsf(X: pd.DataFrame, y, n_estimators=300, min_samples_leaf=15,
            max_features="sqrt", n_jobs=-1, seed=42):
    """Random Survival Forest — handles nonlinearity + missing values.
    Consistently top-performing in HECKTOR prognosis tasks.
    """
    from sksurv.ensemble import RandomSurvivalForest
    m = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=seed,
    )
    m.fit(X.values, y)
    return m


def ensemble_risk(models_weights: list, X: pd.DataFrame) -> np.ndarray:
    """Rank-average ensemble of (model, weight) pairs → final risk scores."""
    scores = []
    ws = []
    for m, w in models_weights:
        try:
            pred = m.predict(X.values)
        except Exception:
            pred = m.predict(X)
        scores.append(rankdata(pred) / len(pred))
        ws.append(w)
    ws = np.array(ws, dtype=float) / sum(ws)
    return np.average(np.vstack(scores), axis=0, weights=ws)


def fit_mtlr(X_tr, y_tr_df, X_val, y_val_df, num_durations=20, epochs=200):
    """Deep-MTLR via pycox. Returns (model, labtrans) and a risk function."""
    import torchtuples as tt
    from pycox.models import MTLR
    xt = X_tr.values.astype("float32"); xv = X_val.values.astype("float32")
    get = lambda d: (d[SURV_TIME].astype("float32").to_numpy(),
                     d[SURV_EVENT].astype("float32").to_numpy())
    lab = MTLR.label_transform(num_durations)
    yt = lab.fit_transform(*get(y_tr_df)); yv = lab.transform(*get(y_val_df))
    net = tt.practical.MLPVanilla(xt.shape[1], [32, 32], lab.out_features,
                                  batch_norm=True, dropout=0.2)
    model = MTLR(net, tt.optim.Adam, duration_index=lab.cuts)
    model.optimizer.set_lr(0.01)
    model.fit(xt, yt, batch_size=64, epochs=epochs,
              callbacks=[tt.callbacks.EarlyStopping()], val_data=(xv, yv), verbose=False)
    return model


def mtlr_risk(model, X) -> np.ndarray:
    """Single risk score from the survival curve = negative expected survival."""
    surv = model.predict_surv_df(X.values.astype("float32"))
    dt = np.diff(surv.index.values, prepend=0)
    expected = (surv.values * dt[:, None]).sum(axis=0)
    return -expected


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def rank_average(*scores, weights=None) -> np.ndarray:
    """Scale-free combine (ideal for C-index, which only needs ranking)."""
    n = len(scores[0]); weights = weights or [1.0] * len(scores)
    ranks = [rankdata(s) / n for s in scores]
    return np.average(np.vstack(ranks), axis=0, weights=weights)
