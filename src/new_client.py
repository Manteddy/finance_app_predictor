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

import copy
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
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
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

    # ---- cold-start vs warm-start (fine-tuning) cost/benefit ---------------
    warm = warmstart_comparison(by, usable, globals_["random_forest"])

    # ---- worked example on a random held-out client ------------------------
    example_id = str(RNG.choice(usable))
    _example_fan(by[example_id], globals_.get(best), best,
                 spec if best in globals_ else None, example_id)

    out = {"n_clients": len(usable), "best_model": best,
           "tuned": {"linear": li_knee["cfg"], "random_forest": rf_knee["cfg"],
                     "boosted_trees": gb_knee["cfg"]},
           "metrics": summary, "warmstart": warm, "example_account": example_id}
    with open(os.path.join(OUT_DIR, "new_client_metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    _benchmark_bar(summary, os.path.join(OUT_DIR, "new_client_benchmark.png"))
    print(f"\nSaved new_client_metrics.json, new_client_benchmark.png, "
          f"new_client_example.png (account {example_id})")
    return out


# --------------------------------------------------------------------------- #
# cold-start vs warm-start (fine-tuning the global model on the client)
# --------------------------------------------------------------------------- #
WARM_EXTRA_TREES = 50          # trees appended on the client's own data
WARM_RIDGE_ALPHA = 10.0        # per-client linear model (regularised for tiny n)


def _blend_weight(g_tr, Xa, ya):
    """Pick w for  w*global + (1-w)*client_ridge  by 3-fold TimeSeriesSplit on
    the client's own training data (out-of-fold, so the weight does not overfit).
    Returns (w, ridge_fit_on_all_Xa)."""
    n = len(Xa)
    oof_client = np.full(n, np.nan)
    tscv = TimeSeriesSplit(n_splits=min(3, max(2, n // 12)))
    for tr, va in tscv.split(Xa):
        rg = Pipeline([("s", StandardScaler()),
                       ("m", Ridge(alpha=WARM_RIDGE_ALPHA))]).fit(Xa.iloc[tr], ya[tr])
        oof_client[va] = rg.predict(Xa.iloc[va])
    mask = ~np.isnan(oof_client)
    best_w, best_mae = 1.0, np.inf
    if mask.sum() >= 3:
        for w in np.linspace(0.0, 1.0, 11):
            blend = w * g_tr[mask] + (1 - w) * oof_client[mask]
            mae = mean_absolute_error(ya[mask], blend)
            if mae < best_mae:
                best_mae, best_w = mae, float(w)
    ridge_full = Pipeline([("s", StandardScaler()),
                           ("m", Ridge(alpha=WARM_RIDGE_ALPHA))]).fit(Xa, ya)
    return best_w, ridge_full


def warmstart_comparison(by, usable, global_rf):
    """Compare cold-start (global RF zero-shot) against three warm-start
    (per-client fine-tuning) strategies, on the SAME held-out clients and the
    SAME holdout weeks. Reports the performance lift vs the per-client cost.

      * cold        - global RF, predict only (no per-client fit).
      * client_only - a fresh Ridge trained on the client's own history (no
                      transfer): tests whether transfer helps at all.
      * warm_rf     - the global RF's trees are kept and WARM_EXTRA_TREES more
                      are grown on the client's data (sklearn warm_start) - the
                      tree-ensemble analogue of fine-tuning.
      * blended     - global RF frozen, blended with a small per-client Ridge at
                      a weight learned out-of-fold (the cheapest adaptation).
    """
    strategies = ["cold", "client_only", "warm_rf", "blended"]
    res = {s: {"mase": [], "r2": [], "ms": []} for s in strategies}
    blend_ws = []

    for aid in usable:
        r = account_frame(by[aid])
        if r is None:
            continue
        X, y, frame, df = r
        (Xa, ya, fa), (Xte, yte, fte) = evaluate.temporal_split(
            X, y, frame, test_days=TEST_WEEKS)
        ya = np.asarray(ya, dtype=float)
        ytrue = fte["total_next"].values
        scale_ref = fa["net_flow"].values

        def _record(name, pred, ms):
            res[name]["mase"].append(evaluate.mase(ytrue, pred, scale_ref))
            res[name]["r2"].append(evaluate.flow_metrics(ytrue, pred)["r2"])
            res[name]["ms"].append(ms * 1e3)

        # 1. COLD: global model, predict only (no per-client fitting).
        t0 = time.perf_counter()
        p_cold = global_rf.predict(Xte)
        _record("cold", p_cold, time.perf_counter() - t0)

        # 2. CLIENT-ONLY: fresh Ridge on the client's own data (no transfer).
        t0 = time.perf_counter()
        rg = Pipeline([("s", StandardScaler()),
                       ("m", Ridge(alpha=WARM_RIDGE_ALPHA))]).fit(Xa, ya)
        p_local = rg.predict(Xte)
        _record("client_only", p_local, time.perf_counter() - t0)

        # 3. WARM RF: keep the global trees, grow extra trees on client data.
        #    deepcopy preserves the fitted ensemble; clone() would reset it.
        t0 = time.perf_counter()
        warm = copy.deepcopy(global_rf)
        warm.warm_start = True
        warm.n_estimators = warm.n_estimators + WARM_EXTRA_TREES
        warm.fit(Xa, ya)
        p_warm = warm.predict(Xte)
        _record("warm_rf", p_warm, time.perf_counter() - t0)

        # 4. BLENDED: frozen global RF + small per-client Ridge, weight learned
        #    out-of-fold on the client's own training data.
        t0 = time.perf_counter()
        g_tr = global_rf.predict(Xa)
        w, ridge_full = _blend_weight(g_tr, Xa, ya)
        p_blend = w * p_cold + (1 - w) * ridge_full.predict(Xte)
        _record("blended", p_blend, time.perf_counter() - t0)
        blend_ws.append(w)

    # ---- aggregate ---------------------------------------------------------
    print(f"\n{'='*70}\nCOLD-START vs WARM-START  (n={len(res['cold']['mase'])} clients)\n{'='*70}")
    print(f"{'strategy':14s} {'median MASE':>12s} {'%beats naive':>13s} "
          f"{'median R2':>10s} {'ms/client':>10s} {'p90 ms':>9s}")
    cold_mase = float(np.nanmedian(res["cold"]["mase"]))
    cold_ms = float(np.median(res["cold"]["ms"]))
    summary = {}
    for s in strategies:
        mase = np.array(res[s]["mase"], dtype=float)
        med = float(np.nanmedian(mase))
        beat = float(np.mean(mase < 1.0) * 100)
        r2 = float(np.nanmedian(res[s]["r2"]))
        ms = float(np.median(res[s]["ms"]))
        p90 = float(np.percentile(res[s]["ms"], 90))
        d_mase = (med - cold_mase) / cold_mase * 100 if s != "cold" else 0.0
        d_ms = ms / cold_ms if cold_ms > 0 else float("inf")
        summary[s] = {"median_mase": round(med, 3), "pct_beats_naive": round(beat, 1),
                      "median_r2": round(r2, 3), "median_ms": round(ms, 3),
                      "p90_ms": round(p90, 3),
                      "mase_vs_cold_pct": round(d_mase, 1),
                      "cost_x_vs_cold": round(d_ms, 1)}
        print(f"{s:14s} {med:12.3f} {beat:12.0f}% {r2:10.3f} {ms:10.2f} {p90:9.2f}")

    print(f"\nReference (cold-start): median MASE {cold_mase:.3f}, "
          f"{cold_ms*1e3:.0f} µs/client (predict only).")
    print("Warm-start lift vs cost:")
    for s in strategies[1:]:
        sm = summary[s]
        verdict = "better" if sm["mase_vs_cold_pct"] < 0 else "worse"
        print(f"  {s:12s}: MASE {sm['median_mase']:.3f} "
              f"({sm['mase_vs_cold_pct']:+.1f}% vs cold = {verdict}) "
              f"at {sm['cost_x_vs_cold']:.1f}x cost "
              f"({sm['median_ms']-cold_ms:+.1f} ms/client)")
    summary["_cold_ref_ms"] = round(cold_ms, 4)
    summary["_median_blend_weight"] = round(float(np.median(blend_ws)), 2) if blend_ws else None

    _warmstart_plot(summary, strategies, os.path.join(OUT_DIR, "warmstart_comparison.png"))
    print("Saved warmstart_comparison.png")
    return summary


def _warmstart_plot(summary, strategies, path):
    labels = {"cold": "cold\n(zero-shot)", "client_only": "client-only\n(Ridge)",
              "warm_rf": "warm RF\n(+50 trees)", "blended": "blended\n(global+Ridge)"}
    names = [labels[s] for s in strategies]
    mases = [summary[s]["median_mase"] for s in strategies]
    costs = [max(summary[s]["median_ms"], 1e-3) for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cold_mase = summary["cold"]["median_mase"]
    colors = ["0.6"] + ["tab:blue"] * (len(strategies) - 1)
    ax1.bar(names, mases, color=colors)
    ax1.axhline(cold_mase, color="tab:green", ls="--", lw=1, label="cold-start MASE")
    ax1.axhline(1.0, color="tab:red", ls=":", lw=1, label="naive (MASE=1)")
    for i, v in enumerate(mases):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylabel("median MASE  (lower = better)")
    ax1.set_title("Performance")
    ax1.legend(fontsize=8)

    ax2.bar(names, costs, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("median compute per client (ms, log scale)")
    ax2.set_title("Cost")
    for i, v in enumerate(costs):
        ax2.text(i, v * 1.1, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("Cold-start vs warm-start (fine-tuning) — Berka new clients",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


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
