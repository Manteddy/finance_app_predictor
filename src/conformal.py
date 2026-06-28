"""Adaptive prediction intervals via normalized split conformal.

Design notes (learned empirically on this ~46-week series):

* The original fan was wide mostly due to a CALIBRATION BUG: expanding-CV early
  folds trained on ~8 points produced garbage residuals that inflated the margin
  (and it still under-covered). We instead calibrate on **prequential** (online)
  residuals — one-step errors from models trained on >=50% of the data — which
  are realistic and use every point. This roughly halves the band and fixes
  coverage.
* The band is then scaled by a BOUNDED local-volatility multiplier
  ``rel(x) = clip(vol(x)/median(vol), 0.6, 1.6)`` so it is tighter in calm weeks
  and wider in volatile ones, without the instability of an unbounded
  feature/quantile sigma on tiny data.

    interval(x) = yhat(x)  ±  margin_c * rel(x)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

_Z80 = 1.2816
SIGMA_COL = "nf_roll_std_4"   # weekly rolling-volatility feature
LO_B, HI_B = 0.6, 1.6         # bounds on the volatility multiplier
MIN_TRAIN_FRAC = 0.5


def _rel(spec, X):
    col = spec["col"]
    if col and col in X.columns:
        v = np.abs(X[col].fillna(0.0).values) / max(spec["ref"], 1.0)
        return np.clip(v, spec["lo_b"], spec["hi_b"])
    return np.ones(len(X))


def fit_normalized(point_unfit, Xtr, ytr, levels, sigma_col=SIGMA_COL):
    """Calibrate per-level margins on prequential residuals. Returns a spec for
    ``interval`` plus the prequential residuals (for a fair marginal 'before')."""
    n = len(Xtr)
    start = max(8, int(n * MIN_TRAIN_FRAC))
    col = sigma_col if sigma_col in Xtr.columns else None
    ref = float(np.median(np.abs(Xtr[col]))) if col else 1.0
    spec0 = {"col": col, "ref": ref, "lo_b": LO_B, "hi_b": HI_B}

    resid, rels = [], []
    for i in range(start, n):
        m = clone(point_unfit).fit(Xtr.iloc[:i], ytr.iloc[:i])
        resid.append(float(ytr.iloc[i] - m.predict(Xtr.iloc[[i]])[0]))
        rels.append(float(_rel(spec0, Xtr.iloc[[i]])[0]))
    resid = np.asarray(resid)
    scores = np.abs(resid) / np.asarray(rels)
    nn = len(scores)
    margins = {c: float(np.quantile(scores, min(1.0, np.ceil((nn + 1) * c) / nn)))
               for c in levels}
    return {**spec0, "margins": margins, "cal_resid": resid}


def interval(spec, point_model, c, X):
    """Adaptive (lo, hi) for central level ``c`` using an external point model."""
    yhat = point_model.predict(X)
    half = spec["margins"][c] * _rel(spec, X)
    return yhat - half, yhat + half


def project_intervals_mc(point_model, spec, df, freq, deterministic, horizon,
                         decomposed, last_balance, n_paths=300, levels=None,
                         seed=42):
    """Monte-Carlo multi-step projection fan: simulate residual paths drawn from
    N(0, s) with s = margin_0.8 * rel(x) / z80 (adaptive), accumulate balances,
    take per-step quantiles. Returns (dates, p50, {level: (lo, hi)})."""
    from .features import compute_feature_matrix

    levels = levels or sorted(spec["margins"].keys())
    rng = np.random.default_rng(seed)
    grp_cols = [c for c in df.columns if c.startswith("grp_")]
    recent = df.tail(8 if freq == "W" else 30)
    grp_means = {c: recent[c].mean() for c in grp_cols}
    txn_mean = recent["txn_count"].mean()
    step = pd.Timedelta(weeks=1) if freq == "W" else pd.Timedelta(days=1)
    m80 = spec["margins"].get(0.8, 1.0)

    paths = np.zeros((n_paths, horizon))
    dates = None
    for p in range(n_paths):
        work = df.copy()
        balance = last_balance
        d_list, b_list = [], []
        for _ in range(horizon):
            feat = compute_feature_matrix(work, freq, deterministic)
            x = feat.iloc[[-1]].fillna(0.0)
            point = float(point_model.predict(x)[0])
            s = float(m80 * _rel(spec, x)[0] / _Z80)
            resid = rng.normal(0.0, max(s, 1.0))
            nxt = work.index[-1] + step
            det_next = float(deterministic.get(nxt, 0.0)) if deterministic is not None else 0.0
            flow = (det_next + point if decomposed else point) + resid
            balance += flow
            row = {c: 0.0 for c in work.columns}
            row["net_flow"], row["end_balance"], row["txn_count"] = flow, balance, txn_mean
            for c in grp_cols:
                row[c] = grp_means[c]
            work.loc[nxt] = row
            d_list.append(nxt)
            b_list.append(balance)
        paths[p] = b_list
        dates = d_list

    p50 = np.median(paths, axis=0)
    bands = {c: (np.percentile(paths, (1 - c) / 2 * 100, axis=0),
                 np.percentile(paths, (1 + c) / 2 * 100, axis=0)) for c in levels}
    return dates, p50, bands
