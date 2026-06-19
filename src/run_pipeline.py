"""End-to-end pipeline.

Runs a 2x2 grid of tracks — {daily, weekly} x {direct, decomposed} — across four
models (moving-average baseline, linear w/ Lasso->Ridge workflow, trees,
SARIMA), compares them on the reconstructed *total* next-period flow and balance,
and produces the headline **balance fan chart** for the weekly decomposed track.

Run with:  python -m src.run_pipeline
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import etl, evaluate, explain, models, plots, recurring
from .features import aggregate, build_features, load_daily

OUT_DIR = "outputs"
COEF_TRIGGER = 150.0            # |std coef| (EUR) that flags an inflated OLS fit
TEST_PERIODS = {"D": 60, "W": 12}
HORIZON = {"D": 30, "W": 8}
Z80 = 1.2816


def _select_linear(Xtr, ytr):
    """Run the OLS -> Lasso -> Ridge regularization workflow; return chosen fit."""
    lin_models = models.build_linear_models()
    max_coef = models.ols_max_abs_coef(lin_models["linear_ols"], Xtr, ytr)
    cv = {n: models._cv_rmse(m, Xtr, ytr) for n, m in lin_models.items()}
    best_reg = min(cv["linear_lasso"], cv["linear_ridge"])
    regularize = (max_coef > COEF_TRIGGER) or (cv["linear_ols"] > 1.15 * best_reg)
    if regularize:
        chosen = min(["linear_lasso", "linear_ridge"], key=cv.get)
    else:
        chosen = min(cv, key=cv.get)
    return chosen, lin_models[chosen].fit(Xtr, ytr), max_coef, regularize, cv


def run_track(daily, tx, freq, mode):
    """Train + evaluate one (granularity, mode) track. Returns (rows, artifacts)."""
    df = aggregate(daily, freq)
    n_test = TEST_PERIODS[freq]
    cutoff = df.index[-n_test]                     # first test period (train < cutoff)

    decomposed = mode == "decomposed"
    deterministic = None
    if decomposed:
        rules = recurring.detect_recurring(tx, cutoff)
        habitual = recurring.habitual_rates(daily, cutoff, freq)
        # Deterministic backbone over history + future horizon.
        future_idx = pd.date_range(df.index[-1], periods=HORIZON[freq] + 1,
                                   freq=freq)[1:]
        ext_idx = df.index.append(future_idx)
        deterministic = recurring.build_deterministic(ext_idx, rules, habitual, freq)

    target_mode = "residual" if decomposed else "total"
    X, y, frame = build_features(df, freq, target_mode, deterministic)
    (Xtr, ytr, ftr), (Xte, yte, fte) = evaluate.temporal_split(X, y, frame,
                                                               test_days=n_test)

    # Predicted *periods* and the actual total flow / balance anchors there.
    pos = df.index.get_indexer(fte.index)
    target_periods = df.index[pos + 1]
    anchors = fte["end_balance"].values
    actual_total = fte["total_next"].values
    det_next = fte["det_next"].values
    ytr_total = ftr["total_next"]                   # baseline always targets total

    def to_total(pred):                             # residual -> total flow
        return det_next + pred if decomposed else pred

    rows, pred_totals = [], {}

    # ---- baseline (naive total-flow moving average, identical in both modes) --
    base = models.MovingAverageBaseline().fit(Xtr, ytr_total)
    pred_totals["baseline"] = base.predict(Xte)

    # ---- linear (regularization workflow) ------------------------------------
    lin_name, lin, max_coef, regularize, lin_cv = _select_linear(Xtr, ytr)
    pred_totals["linear"] = to_total(lin.predict(Xte))

    # ---- trees ---------------------------------------------------------------
    tree_name, tree_model, tree_cv = models.select_best_tree(Xtr, ytr)
    tree = tree_model.fit(Xtr, ytr)
    pred_totals["tree"] = to_total(tree.predict(Xte))

    # ---- SARIMA (on total or residual series, one-step-ahead) ----------------
    series = df["net_flow"].copy()
    if decomposed:
        series = series - deterministic.reindex(df.index)
    sar_mean, sar_lo, sar_hi = models.sarima_one_step(
        series[series.index < target_periods[0]], series, target_periods, freq)
    pred_totals["sarima"] = (det_next + sar_mean) if decomposed else sar_mean

    # ---- metrics for every model (on reconstructed TOTAL flow + balance) -----
    for name, pt in pred_totals.items():
        fm = evaluate.flow_metrics(actual_total, pt)
        bm, _, _ = evaluate.balance_one_step(anchors, actual_total, pt)
        rows.append({"track": f"{freq}-{mode}", "freq": freq, "mode": mode,
                     "model": name, **fm, **bm})

    # ---- residual fan via split-conformal calibration ------------------------
    # Conditional quantile regression is unreliable on ~56 weekly points, so we
    # build the P10/P90 band from out-of-fold residuals across the training set
    # (split conformal => ~80% marginal coverage). The best strong model gives
    # the P50 point; conformal offsets widen it to the calibrated band.
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    r2_by = {r["model"]: r["r2"] for r in rows}
    best_strong = max(("linear", "tree"), key=lambda m: r2_by[m])
    est_unfit = {"linear": lin, "tree": tree_model}[best_strong]

    oof_err = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(Xtr):
        m = clone(est_unfit).fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
        oof_err.extend(ytr.iloc[va_idx].values - m.predict(Xtr.iloc[va_idx]))
    off10, off90 = np.percentile(oof_err, 10), np.percentile(oof_err, 90)

    flow_p50 = pred_totals[best_strong]
    flow_p10, flow_p90 = flow_p50 + off10, flow_p50 + off90
    pinball = float(np.mean([
        evaluate.pinball_loss(actual_total, flow_p10, 0.1),
        evaluate.pinball_loss(actual_total, flow_p50, 0.5),
        evaluate.pinball_loss(actual_total, flow_p90, 0.9)]))
    coverage = evaluate.interval_coverage(actual_total, flow_p10, flow_p90)
    for r in rows:
        if r["model"] == best_strong:
            r["pinball"], r["coverage"] = round(pinball, 2), round(coverage, 3)

    artifacts = {
        "df": df, "freq": freq, "deterministic": deterministic,
        "target_periods": target_periods, "anchors": anchors,
        "actual_total": actual_total,
        "pred_totals": pred_totals, "best_strong": best_strong,
        "flow_p10": flow_p10, "flow_p50": flow_p50, "flow_p90": flow_p90,
        "Xtr": Xtr, "Xte": Xte, "tree": tree, "tree_name": tree_name,
        "lin": lin, "lin_name": lin_name, "feature_names": list(X.columns),
        "max_coef": max_coef, "regularize": regularize,
        "n_train": len(Xtr), "n_test": len(Xte),
    }
    return rows, artifacts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = etl.build_database()
    print("=" * 72)
    print(f"ETL: {summary['n_transactions']} txns, {summary['n_days']} days, "
          f"{summary['date_range'][0]}..{summary['date_range'][1]}, "
          f"final balance EUR {summary['final_balance']:.2f}")

    daily = load_daily()
    tx = recurring.load_transactions()

    all_rows, artifacts = [], {}
    for freq in ("D", "W"):
        for mode in ("direct", "decomposed"):
            rows, art = run_track(daily, tx, freq, mode)
            all_rows.extend(rows)
            artifacts[(freq, mode)] = art

    comp = pd.DataFrame(all_rows)
    print("\n" + "=" * 72)
    print("MODEL COMPARISON — next-period TOTAL flow (holdout). Higher R² better.")
    print("=" * 72)
    show = comp[["track", "model", "rmse", "mae", "r2", "balance_rmse"]].copy()
    print(show.round(2).to_string(index=False))

    # Headline track: weekly decomposed.
    head = artifacts[("W", "decomposed")]
    freq = head["freq"]
    det = head["deterministic"]
    future_idx = det.index[det.index > head["df"].index[-1]]
    det_future = det.loc[future_idx]
    sigma_resid = float(np.std(head["actual_total"] - head["flow_p50"]))
    last_balance = head["df"]["end_balance"].iloc[-1]
    # Historical mean residual (= typical net of irregular income the
    # deterministic backbone omits) anchors the projection's central line.
    resid_hist = head["df"]["net_flow"] - det.reindex(head["df"].index)
    drift = float(resid_hist.dropna().mean())
    projection = plots.project_fan(det_future, last_balance, sigma_resid,
                                   HORIZON[freq], drift=drift)

    hist = head["df"].iloc[-(head["n_test"] + 26):][["end_balance"]]
    plots.plot_fan_chart(
        head["target_periods"], head["anchors"], head["actual_total"],
        head["flow_p10"], head["flow_p50"], head["flow_p90"], projection,
        os.path.join(OUT_DIR, "fan_chart.png"), history_tail=hist,
        title="Weekly balance forecast — deterministic line + P10–P90 fan")

    # Multi-model holdout comparison (weekly decomposed).
    plots.plot_balance_forecast(
        head["target_periods"], head["anchors"], head["actual_total"],
        head["pred_totals"], os.path.join(OUT_DIR, "balance_forecast.png"),
        history_tail=hist, title="Weekly balance: actual vs models (holdout)")

    # Daily-vs-weekly R² comparison.
    plots.plot_track_comparison(comp, os.path.join(OUT_DIR, "daily_vs_weekly.png"))

    # Explainability: SHAP for the tree, coefficients for the linear model.
    shap_rank = explain.shap_tree(head["tree"], head["Xtr"], head["Xte"],
                                  os.path.join(OUT_DIR, "shap_summary.png"))
    explain.plot_importance_bar(shap_rank, "mean_abs_shap",
                                f"SHAP — weekly {head['tree_name']} (residual)",
                                os.path.join(OUT_DIR, "shap_bar.png"))
    lin_rank = explain.linear_contributions(head["lin"], head["feature_names"])
    explain.plot_importance_bar(lin_rank, "abs_coef",
                                f"Standardized coefficients — {head['lin_name']}",
                                os.path.join(OUT_DIR, "linear_coefficients.png"))

    # ---- narrative summary -------------------------------------------------
    print("\n" + "-" * 72)
    best_daily = comp[comp["freq"] == "D"].set_index(["mode", "model"])["r2"]
    best_weekly = comp[comp["freq"] == "W"].set_index(["mode", "model"])["r2"]
    print(f"Weekly lifts R²: daily best {best_daily.max():+.3f} -> "
          f"weekly best {best_weekly.max():+.3f}")
    wd = comp[(comp["freq"] == "W") & (comp["mode"] == "decomposed")].set_index("model")
    hb = head["best_strong"]
    print(f"Weekly-decomposed: baseline R² {wd.loc['baseline', 'r2']:+.3f} vs "
          f"best strong ({hb}) {wd.loc[hb, 'r2']:+.3f}  |  P10–P90 coverage "
          f"{wd.loc[hb, 'coverage']:.2f} (target ~0.80), "
          f"pinball {wd.loc[hb, 'pinball']:.1f}")

    out = {
        "data": {k: summary[k] for k in
                 ("n_transactions", "n_days", "date_range", "final_balance")},
        "tracks": comp.round(3).to_dict(orient="records"),
        "headline": {
            "track": "W-decomposed",
            "best_strong_model": hb,
            "best_strong_r2": round(float(wd.loc[hb, "r2"]), 3),
            "regularization_triggered": bool(head["regularize"]),
            "ols_max_abs_coef": round(head["max_coef"], 2),
            "chosen_linear": head["lin_name"], "chosen_tree": head["tree_name"],
            "coverage_p10_p90": round(float(wd.loc[hb, "coverage"]), 3),
            "pinball": round(float(wd.loc[hb, "pinball"]), 2),
            "n_train": head["n_train"], "n_test": head["n_test"],
        },
        "top_shap_features": shap_rank.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    comp.round(3).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print(f"\nSaved figures + metrics to {OUT_DIR}/")
    return out


if __name__ == "__main__":
    main()
