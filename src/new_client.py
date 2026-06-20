"""New-client forecasting benchmark on the real Berka panel.

Question: can a model forecast a NEW client's finances accumulation from that
client's own historical records? We hold out whole accounts as "new clients" and
compare four model families + a naive baseline under MASE (the scale-free metric
that is comparable across heterogeneous accounts).

  * linear / random-forest / boosted-trees : GLOBAL models, trained only on the
    train accounts, applied zero-shot to each new client's own features.
  * SARIMA / baseline : per-series, fit on the new client's own history.

Each model's capacity is tuned for the best performance-to-time ratio. The best
model is then shown forecasting a randomly chosen held-out client (line + fan).

Run:  python -m src.new_client
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import berka, conformal, evaluate, models, plots
from .features import aggregate, build_features, compute_feature_matrix

OUT_DIR = "outputs"
N_ACCOUNTS = 360
N_TRAIN = 300
TEST_WEEKS = 12
SIGMA_COL = "nf_roll_std_4"
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def account_frame(rows):
    """Weekly (X, y, frame, df) for one account in DIRECT total-flow mode."""
    df = aggregate(berka.account_daily(rows), "W")
    if len(df) < TEST_WEEKS + 45:
        return None
    X, y, frame = build_features(df, "W", "total", None)
    if len(X) < TEST_WEEKS + 25:
        return None
    return X, y, frame, df


def pooled(by_acct, ids):
    Xs, ys, gs = [], [], []
    for aid in ids:
        r = account_frame(by_acct[aid])
        if r is None:
            continue
        X, y, _, _ = r
        Xs.append(X); ys.append(y); gs.append(np.full(len(X), int(aid)))
    X = pd.concat(Xs).fillna(0.0).reset_index(drop=True)
    return X, pd.concat(ys).values, np.concatenate(gs)


def _ridge():
    return Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0))])


# --------------------------------------------------------------------------- #
# capacity tuning (performance-to-time)
# --------------------------------------------------------------------------- #
def tune(name, factory, grid, X, y, g, scale):
    """GroupKFold(3) sweep; return (best_estimator_factory, rows). Rank by val
    MASE; pick the knee = cheapest config within 3% of the best MASE."""
    gkf = GroupKFold(n_splits=3)
    rows = []
    for cfg in grid:
        maes, t0 = [], time.time()
        for tr, va in gkf.split(X, y, g):
            m = factory(cfg).fit(X.iloc[tr], y[tr])
            maes.append(mean_absolute_error(y[va], m.predict(X.iloc[va])))
        rows.append({"cfg": cfg, "mase": float(np.mean(maes)) / scale,
                     "fit_s": (time.time() - t0) / 3})
    best = min(r["mase"] for r in rows)
    knee = min((r for r in rows if r["mase"] <= best * 1.03), key=lambda r: r["fit_s"])
    print(f"\n[tune {name}] (val MASE, fit s):")
    for r in sorted(rows, key=lambda r: r["mase"]):
        mark = " <- knee" if r["cfg"] == knee["cfg"] else ""
        print(f"    {str(r['cfg']):52s} MASE {r['mase']:.3f}  {r['fit_s']:.2f}s{mark}")
    return (lambda: factory(knee["cfg"])), knee


# --------------------------------------------------------------------------- #
# global conformal (calibrated cross-account) -> spec for src.conformal
# --------------------------------------------------------------------------- #
def global_conformal(model, Xcal, ycal, levels):
    col = SIGMA_COL if SIGMA_COL in Xcal.columns else None
    ref = float(np.median(np.abs(Xcal[col]))) if col else 1.0
    spec = {"col": col, "ref": ref, "lo_b": 0.6, "hi_b": 1.6}
    rel = conformal._rel(spec, Xcal)
    resid = np.abs(ycal - model.predict(Xcal))
    if col is None:
        spec["floor"] = 1.0
    scores = resid / np.maximum(rel, 1e-9)
    n = len(scores)
    spec["margins"] = {c: float(np.quantile(scores, min(1.0, np.ceil((n + 1) * c) / n)))
                       for c in levels}
    return spec


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by = berka.load_transactions()
    accounts = berka.select_accounts(by, n=N_ACCOUNTS)
    train_ids, test_ids = accounts[:N_TRAIN], accounts[N_TRAIN:]
    assert not (set(train_ids) & set(test_ids)), "account leakage!"
    print(f"Berka new-client benchmark: {len(train_ids)} train accts, "
          f"{len(test_ids)} held-out new clients")

    Xtr, ytr, gtr = pooled(by, train_ids)
    scale = float(np.mean(np.abs(np.diff(ytr))))      # global MASE scale for tuning
    print(f"pooled train: {len(Xtr)} weekly rows, {len(np.unique(gtr))} accounts")

    # ---- capacity tuning per global family ---------------------------------
    rf_grid = [{"n_estimators": n, "max_depth": d, "min_samples_leaf": 20}
               for n in (100, 300) for d in (8, None)]
    gb_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05}
               for n in (200, 400) for d in (2, 3)]
    li_grid = [{"alpha": a} for a in (1.0, 10.0, 100.0)]

    li_fac, li_knee = tune("linear", lambda c: Pipeline(
        [("s", StandardScaler()), ("m", Ridge(**c))]), li_grid, Xtr, ytr, gtr, scale)
    rf_fac, rf_knee = tune("random_forest", lambda c: RandomForestRegressor(
        random_state=42, n_jobs=-1, **c), rf_grid, Xtr, ytr, gtr, scale)
    gb_fac, gb_knee = tune("boosted_trees", lambda c: GradientBoostingRegressor(
        subsample=0.8, min_samples_leaf=20, random_state=42, **c),
        gb_grid, Xtr, ytr, gtr, scale)

    # ---- fit global models on all train accounts; split off a calib slice ---
    cal_mask = np.isin(gtr, [int(a) for a in train_ids[-50:]])
    fitX, fity = Xtr[~cal_mask], ytr[~cal_mask]
    calX, caly = Xtr[cal_mask], ytr[cal_mask]
    globals_ = {"linear": li_fac().fit(fitX, fity),
                "random_forest": rf_fac().fit(fitX, fity),
                "boosted_trees": gb_fac().fit(fitX, fity)}

    # ---- evaluate every model on each held-out new client ------------------
    per = {m: {"mase": [], "r2": []} for m in
           ["baseline", "sarima", "linear", "random_forest", "boosted_trees"]}
    usable = []
    for aid in test_ids:
        r = account_frame(by[aid])
        if r is None:
            continue
        X, y, frame, df = r
        (Xa, ya, fa), (Xte, yte, fte) = evaluate.temporal_split(X, y, frame, test_days=TEST_WEEKS)
        ytrue = fte["total_next"].values
        sc = float(np.mean(np.abs(np.diff(fa["net_flow"].values)))) or 1.0
        usable.append(aid)

        preds = {
            "baseline": fte["net_flow"].values,                         # naive RW
            "linear": globals_["linear"].predict(Xte),
            "random_forest": globals_["random_forest"].predict(Xte),
            "boosted_trees": globals_["boosted_trees"].predict(Xte),
        }
        # SARIMA: per-series on this client's own net flow.
        series = df["net_flow"]
        tgt = df.index[df.index.get_indexer(fte.index) + 1]
        sar, _, _ = models.sarima_one_step(series[series.index < tgt[0]], series, tgt, "W")
        preds["sarima"] = sar

        for m, p in preds.items():
            per[m]["mase"].append(evaluate.mase(ytrue, p, fa["net_flow"].values))
            per[m]["r2"].append(evaluate.flow_metrics(ytrue, p)["r2"])

    # ---- aggregate + select best -------------------------------------------
    print(f"\n{'='*64}\nNEW-CLIENT RESULTS  (n={len(usable)} held-out clients)\n{'='*64}")
    print(f"{'model':16s} {'median MASE':>12s} {'%MASE<1':>9s} {'median R2':>10s}")
    summary = {}
    for m, d in per.items():
        med = float(np.nanmedian(d["mase"]))
        beat = float(np.mean(np.array(d["mase"]) < 1.0) * 100)
        r2 = float(np.nanmedian(d["r2"]))
        summary[m] = {"median_mase": round(med, 3), "pct_beats_naive": round(beat, 1),
                      "median_r2": round(r2, 3)}
        print(f"{m:16s} {med:12.3f} {beat:8.0f}% {r2:10.3f}")
    strong = {m: v for m, v in summary.items() if m != "baseline"}
    best = min(strong, key=lambda m: strong[m]["median_mase"])
    print(f"\nBEST model (lowest median MASE): {best}  "
          f"(MASE {summary[best]['median_mase']}, beats naive on "
          f"{summary[best]['pct_beats_naive']:.0f}% of clients)")

    # ---- probabilistic scoring for the best model --------------------------
    levels = plots.FAN_LEVELS
    if best in globals_:
        spec = global_conformal(globals_[best], calX, caly, levels)
        cov, pin = [], []
        for aid in usable:
            r = account_frame(by[aid]);
            if r is None: continue
            X, y, frame, df = r
            _, (Xte, yte, fte) = evaluate.temporal_split(X, y, frame, test_days=TEST_WEEKS)
            lo, hi = conformal.interval(spec, globals_[best], 0.8, Xte)
            cov.append(evaluate.interval_coverage(fte["total_next"].values, lo, hi))
            p50 = globals_[best].predict(Xte)
            pin.append(np.mean([evaluate.pinball_loss(fte["total_next"].values,
                       conformal.interval(spec, globals_[best], c, Xte)[j], q)
                       for c, q, j in [(0.8, 0.1, 0), (0.8, 0.9, 1)]]))
        print(f"  P10-P90 coverage median {np.median(cov):.2f} (target ~0.80); "
              f"pinball median {np.median(pin):.1f}")
        summary["_best_coverage"] = round(float(np.median(cov)), 3)

    # ---- worked example on a random held-out client ------------------------
    example_id = str(RNG.choice(usable))
    _example_fan(by[example_id], globals_.get(best), best,
                 spec if best in globals_ else None, example_id)

    out = {"n_clients": len(usable), "best_model": best,
           "tuned": {"linear": li_knee["cfg"], "random_forest": rf_knee["cfg"],
                     "boosted_trees": gb_knee["cfg"]},
           "metrics": summary, "example_account": example_id}
    with open(os.path.join(OUT_DIR, "new_client_metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    _benchmark_bar(summary, os.path.join(OUT_DIR, "new_client_benchmark.png"))
    print(f"\nSaved new_client_metrics.json, new_client_benchmark.png, "
          f"new_client_example.png (account {example_id})")
    return out


def _example_fan(rows, model, best_name, spec, aid):
    X, y, frame, df = account_frame(rows)
    _, (Xte, yte, fte) = evaluate.temporal_split(X, y, frame, test_days=TEST_WEEKS)
    anchors = fte["end_balance"].values
    actual_total = fte["total_next"].values
    target_dates = fte.index + pd.Timedelta(weeks=1)
    p50 = model.predict(Xte)

    holdout_bands = []
    for c in sorted(plots.FAN_LEVELS, reverse=True):
        lo, hi = conformal.interval(spec, model, c, Xte)
        holdout_bands.append((anchors + lo, anchors + hi))

    last_balance = df["end_balance"].iloc[-1]
    pdates, pp50, pbands = conformal.project_intervals_mc(
        model, spec, df, "W", None, 8, False, last_balance, n_paths=200)
    proj = pd.DataFrame({"p50": pp50}, index=pdates)
    for c in plots.FAN_LEVELS:
        proj[f"lo_{c}"], proj[f"hi_{c}"] = pbands[c]
    anchor = pd.DataFrame({col: [last_balance] for col in proj.columns},
                          index=[df.index[-1]])
    projection_df = pd.concat([anchor, proj])

    hist = df.iloc[-(TEST_WEEKS + 26):][["end_balance"]]
    plots.plot_fan_chart(
        target_dates, anchors + actual_total, anchors + p50, holdout_bands,
        projection_df, os.path.join(OUT_DIR, "new_client_example.png"),
        history_tail=hist,
        title=f"New client (Berka acct {aid}) — {best_name} forecast + adaptive fan")


def _benchmark_bar(summary, path):
    order = ["baseline", "sarima", "linear", "random_forest", "boosted_trees"]
    mases = [summary[m]["median_mase"] for m in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["0.6"] + ["tab:blue"] * 4
    ax.bar(order, mases, color=colors)
    ax.axhline(1.0, color="tab:red", ls="--", lw=1, label="naive (MASE=1)")
    for i, v in enumerate(mases):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("median MASE across held-out new clients (lower = better)")
    ax.set_title("New-client forecasting — model families vs naive (Berka)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
