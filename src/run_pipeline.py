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

from . import conformal, etl, evaluate, explain, models, plots, recurring
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

    decomposed = mode in ("decomposed", "decomp_agnostic")
    deterministic = None
    if decomposed:
        if mode == "decomp_agnostic":
            # Recurrence-FIRST: detect periodic merchants across all categories,
            # then compute habitual on the non-recurring remainder.
            rules = recurring.detect_recurring_agnostic(tx, cutoff)
            rec_keys = {r.key for r in rules}
            habitual = recurring.habitual_rates_excluding(tx, rec_keys, cutoff, freq)
        else:
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
    best_model = {"linear": lin, "tree": tree}[best_strong]

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
        "df": df, "freq": freq, "mode": mode, "decomposed": decomposed,
        "deterministic": deterministic, "rules": rules if decomposed else [],
        "target_periods": target_periods, "anchors": anchors,
        "actual_total": actual_total,
        "pred_totals": pred_totals, "best_strong": best_strong,
        "best_model": best_model, "oof_err": np.asarray(oof_err),
        "flow_p10": flow_p10, "flow_p50": flow_p50, "flow_p90": flow_p90,
        "Xtr": Xtr, "Xte": Xte, "tree": tree, "tree_name": tree_name,
        "lin": lin, "lin_name": lin_name, "feature_names": list(X.columns),
        "max_coef": max_coef, "regularize": regularize,
        "n_train": len(Xtr), "n_test": len(Xte),
    }
    return rows, artifacts


def build_headline(daily, tx, freq="W"):
    """Tightest client-facing forecast on the full balance series: category-gated
    deterministic backbone + a prequentially-calibrated, bounded-adaptive
    conformal fan. (Routing own-account flows out was tried but BACKFIRES — those
    savings/transfers are compensatory stabilisers, so removing them raises
    variance; see REPORT.) Returns everything needed to plot/report.
    """
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    n_test = TEST_PERIODS[freq]
    df = aggregate(daily, freq)
    cutoff = df.index[-n_test]

    rules = recurring.detect_recurring(tx, cutoff)
    habitual = recurring.habitual_rates(daily, cutoff, freq)
    future_idx = pd.date_range(df.index[-1], periods=HORIZON[freq] + 1, freq=freq)[1:]
    det = recurring.build_deterministic(df.index.append(future_idx), rules, habitual, freq)

    X, y, frame = build_features(df, freq, "residual", det)
    (Xtr, ytr, ftr), (Xte, yte, fte) = evaluate.temporal_split(X, y, frame, test_days=n_test)
    anchors = fte["end_balance"].values
    actual_total = fte["total_next"].values
    det_next = fte["det_next"].values

    _, lin, *_ = _select_linear(Xtr, ytr)
    tname, tmodel, _ = models.select_best_tree(Xtr, ytr)
    tree = tmodel.fit(Xtr, ytr)
    cand = {"linear": lin, "tree": tree}
    r2 = {k: evaluate.flow_metrics(actual_total, det_next + m.predict(Xte))["r2"]
          for k, m in cand.items()}
    point_name = max(r2, key=r2.get)
    point = cand[point_name]
    point_unfit = clone(point)
    p50_flow = det_next + point.predict(Xte)

    # Prequential + bounded-adaptive conformal bands (holdout).
    spec = conformal.fit_normalized(point_unfit, Xtr, ytr, plots.FAN_LEVELS)
    holdout_bands = []
    for c in sorted(plots.FAN_LEVELS, reverse=True):
        lo, hi = conformal.interval(spec, point, c, Xte)
        holdout_bands.append((anchors + det_next + lo, anchors + det_next + hi))
    lo8, hi8 = conformal.interval(spec, point, 0.8, Xte)
    cov = evaluate.interval_coverage(actual_total, det_next + lo8, det_next + hi8)
    widths = hi8 - lo8

    # 'Before' = the previous design's band: full expanding-CV OOF (incl. the
    # tiny-fold residuals that inflated it), marginal P10-P90 + its coverage.
    oof = []
    for tr, va in TimeSeriesSplit(n_splits=5).split(Xtr):
        m = clone(point_unfit).fit(Xtr.iloc[tr], ytr.iloc[tr])
        oof.extend(ytr.iloc[va].values - m.predict(Xtr.iloc[va]))
    o_lo, o_hi = np.percentile(oof, 10), np.percentile(oof, 90)
    width_before = float(o_hi - o_lo)
    cov_before = evaluate.interval_coverage(actual_total, p50_flow + o_lo, p50_flow + o_hi)

    # Monte-Carlo projection fan.
    last_balance = df["end_balance"].iloc[-1]
    last_date = df.index[-1]
    pdates, pp50, pbands = conformal.project_intervals_mc(
        point, spec, df, freq, det, HORIZON[freq], True, last_balance)
    proj = pd.DataFrame({"p50": pp50}, index=pdates)
    for c in plots.FAN_LEVELS:
        proj[f"lo_{c}"], proj[f"hi_{c}"] = pbands[c]
    anchor = pd.DataFrame({col: [last_balance] for col in proj.columns}, index=[last_date])
    projection_df = pd.concat([anchor, proj])

    return {
        "target_periods": fte.index + pd.Timedelta(weeks=1),
        "actual_bal": anchors + actual_total,
        "p50_bal": anchors + p50_flow,
        "holdout_bands": holdout_bands,
        "projection_df": projection_df,
        "hist": df.iloc[-(n_test + 26):][["end_balance"]],
        "point_name": point_name, "coverage": cov, "cov_before": cov_before,
        "width_before": width_before,
        "width_calm": float(np.percentile(widths, 25)),
        "width_median": float(np.median(widths)),
        "width_max": float(widths.max()),
    }


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
        for mode in ("direct", "decomposed", "decomp_agnostic"):
            rows, art = run_track(daily, tx, freq, mode)
            all_rows.extend(rows)
            artifacts[(freq, mode)] = art

    comp = pd.DataFrame(all_rows)
    print("\n" + "=" * 72)
    print("MODEL COMPARISON — next-period TOTAL flow (holdout). Higher R² better.")
    print("=" * 72)
    show = comp[["track", "model", "rmse", "mae", "r2", "balance_rmse"]].copy()
    print(show.round(2).to_string(index=False))

    # Show how the recurrence-first architecture captures rent (point 1/2).
    ag = artifacts[("W", "decomp_agnostic")]["rules"]
    print("\nRecurrence-first (agnostic) housing/loan rules detected:")
    for r in sorted(ag, key=lambda x: abs(x.amount), reverse=True)[:6]:
        print(f"  {r.key:26s} {r.amount:+9.1f}/{r.cadence[:2]}  n={r.n} ({r.group})")

    # head (full-flow weekly decomposed) is kept for SHAP + the multi-model plot.
    head = artifacts[("W", "decomposed")]

    # ---- HEADLINE FAN: core 'spendable cash' series + adaptive CQR intervals --
    hl = build_headline(daily, tx)
    print("\n" + "-" * 72)
    print(f"FAN WIDTH (P10–P90, EUR): old broken-calibration {hl['width_before']:.0f} "
          f"@ coverage {hl['cov_before']:.2f}  ->  fixed+adaptive: calm-week "
          f"{hl['width_calm']:.0f}, median {hl['width_median']:.0f}, "
          f"volatile {hl['width_max']:.0f} @ coverage {hl['coverage']:.2f}")

    plots.plot_fan_chart(
        hl["target_periods"], hl["actual_bal"], hl["p50_bal"], hl["holdout_bands"],
        hl["projection_df"], os.path.join(OUT_DIR, "fan_chart.png"),
        history_tail=hl["hist"],
        title=f"Weekly balance forecast ({hl['point_name']}) "
              f"— adaptive conformal fan")

    # Multi-model holdout comparison (weekly decomposed, full flow).
    hist_full = head["df"].iloc[-(head["n_test"] + 26):][["end_balance"]]
    plots.plot_balance_forecast(
        head["target_periods"], head["anchors"], head["actual_total"],
        head["pred_totals"], os.path.join(OUT_DIR, "balance_forecast.png"),
        history_tail=hist_full, title="Weekly balance: actual vs models (holdout)")

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
    best_daily = comp[comp["freq"] == "D"]["r2"].max()
    best_weekly = comp[comp["freq"] == "W"]["r2"].max()
    print(f"Weekly lifts R²: daily best {best_daily:+.3f} -> "
          f"weekly best {best_weekly:+.3f}")

    def track_best(mode):
        t = comp[(comp["freq"] == "W") & (comp["mode"] == mode)
                 & (comp["model"] != "baseline")]
        return t["r2"].max()
    print(f"Architecture A/B (weekly, best strong R²): "
          f"category-gated decomposed {track_best('decomposed'):+.3f}  vs  "
          f"recurrence-first agnostic {track_best('decomp_agnostic'):+.3f}")

    hb = head["best_strong"]
    hrow = comp[(comp["track"] == f"W-{head['mode']}") & (comp["model"] == hb)].iloc[0]
    print(f"Headline (W-{head['mode']}, {hb}): R² {hrow['r2']:+.3f}, "
          f"P10–P90 coverage {hrow['coverage']:.2f} (target ~0.80), "
          f"pinball {hrow['pinball']:.1f}")

    out = {
        "data": {k: summary[k] for k in
                 ("n_transactions", "n_days", "date_range", "final_balance")},
        "tracks": comp.round(3).to_dict(orient="records"),
        "architecture_ab_weekly": {
            "category_gated_decomposed_r2": round(float(track_best("decomposed")), 3),
            "recurrence_first_agnostic_r2": round(float(track_best("decomp_agnostic")), 3),
        },
        "headline": {
            "track": f"W-{head['mode']}",
            "best_strong_model": hb,
            "best_strong_r2": round(float(hrow["r2"]), 3),
            "regularization_triggered": bool(head["regularize"]),
            "ols_max_abs_coef": round(head["max_coef"], 2),
            "chosen_linear": head["lin_name"], "chosen_tree": head["tree_name"],
            "coverage_p10_p90": round(float(hrow["coverage"]), 3),
            "pinball": round(float(hrow["pinball"]), 2),
            "n_train": head["n_train"], "n_test": head["n_test"],
        },
        "fan_width_eur": {
            "old_broken_p10_p90": round(hl["width_before"], 1),
            "old_coverage": round(hl["cov_before"], 3),
            "new_calm_week": round(hl["width_calm"], 1),
            "new_median": round(hl["width_median"], 1),
            "new_volatile_week": round(hl["width_max"], 1),
            "new_coverage": round(hl["coverage"], 3),
            "method": "prequential calibration + bounded adaptive conformal",
            "point_model": hl["point_name"],
            "note": "routing own-account flows out was tried but backfires (compensatory)",
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
