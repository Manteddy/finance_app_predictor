"""End-to-end pipeline: ETL -> features -> train 3 models -> evaluate ->
regularization workflow -> explainability -> plots + metrics.

Run with:  python -m src.run_pipeline
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import etl, evaluate, explain, models, plots
from .features import build_features, load_daily

OUT_DIR = "outputs"
# On standardized features, a coefficient larger than this (in target units of
# EUR/day) flags an inflated/unstable OLS fit and triggers regularization.
COEF_TRIGGER = 150.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------- ETL ---
    summary = etl.build_database()
    print("=" * 70)
    print("ETL — mock database built")
    print(f"  {summary['n_transactions']} transactions, {summary['n_days']} days, "
          f"{summary['date_range'][0]} .. {summary['date_range'][1]}")
    print(f"  final reconstructed balance: EUR {summary['final_balance']:.2f} "
          f"(statement closing = 206.37)")

    # ----------------------------------------------------------- features ---
    daily = load_daily()
    X, y, frame = build_features(daily)
    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    (Xtr, ytr, ftr), (Xte, yte, fte) = evaluate.temporal_split(X, y, frame,
                                                               test_days=60)
    print(f"Train: {Xtr.index.min().date()} .. {Xtr.index.max().date()} "
          f"({len(Xtr)})   Test: {Xte.index.min().date()} .. "
          f"{Xte.index.max().date()} ({len(Xte)})")

    # Anchors = actual balance on each holdout day; one-step-ahead predicted
    # balance = anchor + predicted next-day flow. Target days are the days the
    # flows land on (the day after each feature row).
    anchors = fte["end_balance"].values
    target_dates = fte.index + pd.Timedelta(days=1)
    actual_test_flows = yte.values

    results = {}        # name -> metrics
    test_flows = {}     # short-name -> predicted flows on holdout

    # ---------------------------------------------------------- baseline ---
    base = models.MovingAverageBaseline().fit(Xtr, ytr)
    base_pred = base.predict(Xte)
    bm = evaluate.flow_metrics(yte, base_pred)
    bbal, _, _ = evaluate.balance_one_step(anchors, actual_test_flows, base_pred)
    results["baseline_movavg"] = {**bm, **bbal, "note": f"window={base.window}"}
    test_flows["baseline"] = base_pred

    # ----------------------------------- linear family + regularization ---
    lin_models = models.build_linear_models()
    max_coef = models.ols_max_abs_coef(lin_models["linear_ols"], Xtr, ytr)
    linear_cv = {name: models._cv_rmse(m, Xtr, ytr) for name, m in lin_models.items()}
    best_reg_cv = min(linear_cv["linear_lasso"], linear_cv["linear_ridge"])

    # Regularize if OLS weights are inflated OR OLS overfits (its CV error is
    # materially worse than the regularized variants) — both signal that the
    # strong linear model is leaning on unstable weights.
    coef_high = max_coef > COEF_TRIGGER
    overfit = linear_cv["linear_ols"] > 1.15 * best_reg_cv
    regularize = coef_high or overfit
    print(f"\nOLS max |standardized coef| = {max_coef:.1f} EUR/day "
          f"(threshold {COEF_TRIGGER:.0f}); OLS CV RMSE {linear_cv['linear_ols']:.0f} "
          f"vs best regularized {best_reg_cv:.0f}")
    print(f"  -> regularization {'TRIGGERED' if regularize else 'not needed'} "
          f"(high_weights={coef_high}, overfit={overfit}); applying Lasso then Ridge")
    print("  linear CV RMSE:", {k: round(v, 2) for k, v in linear_cv.items()})

    # Choose the linear representative: if OLS weights are inflated, prefer the
    # best regularized variant (Lasso first, then Ridge); else best overall.
    if regularize:
        order = ["linear_lasso", "linear_ridge", "linear_ols"]
        chosen_linear = min(["linear_lasso", "linear_ridge"], key=linear_cv.get)
    else:
        chosen_linear = min(linear_cv, key=linear_cv.get)
    print(f"  chosen linear model: {chosen_linear}")

    lin = lin_models[chosen_linear].fit(Xtr, ytr)
    lin_pred = lin.predict(Xte)
    lm = evaluate.flow_metrics(yte, lin_pred)
    lbal, _, _ = evaluate.balance_one_step(anchors, actual_test_flows, lin_pred)
    results[chosen_linear] = {**lm, **lbal, "cv_rmse": round(linear_cv[chosen_linear], 2)}
    test_flows["linear"] = lin_pred

    # ------------------------------------------------------------- trees ---
    best_tree_name, tree_model, tree_cv = models.select_best_tree(Xtr, ytr)
    print("\nTree CV RMSE:", {k: round(v, 2) for k, v in tree_cv.items()},
          "-> chosen:", best_tree_name)
    tree = tree_model.fit(Xtr, ytr)
    tree_pred = tree.predict(Xte)
    tm = evaluate.flow_metrics(yte, tree_pred)
    tbal, _, _ = evaluate.balance_one_step(anchors, actual_test_flows, tree_pred)
    results[best_tree_name] = {**tm, **tbal, "cv_rmse": round(tree_cv[best_tree_name], 2)}
    test_flows["tree"] = tree_pred

    # ------------------------------------------------ comparison table ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON  (holdout, lower RMSE/MAE is better)")
    print("=" * 70)
    comp = pd.DataFrame(results).T[["rmse", "mae", "r2", "balance_rmse", "balance_mae"]]
    comp = comp.astype(float).round(2)
    print(comp.to_string())

    strong = comp.drop(index="baseline_movavg")
    winner = strong["balance_rmse"].idxmin()
    base_bal_rmse = comp.loc["baseline_movavg", "balance_rmse"]
    win_bal_rmse = comp.loc[winner, "balance_rmse"]
    improvement = 100 * (base_bal_rmse - win_bal_rmse) / base_bal_rmse
    print(f"\nBest model by balance RMSE: {winner} "
          f"(EUR {win_bal_rmse:.2f} vs baseline EUR {base_bal_rmse:.2f}, "
          f"{improvement:+.1f}% vs baseline)")

    # ---------------------------------------------------- explainability ---
    print("\nExplainability:")
    shap_rank = explain.shap_tree(tree, Xtr, Xte,
                                  os.path.join(OUT_DIR, "shap_summary.png"))
    explain.plot_importance_bar(shap_rank, "mean_abs_shap",
                                f"SHAP contribution — {best_tree_name}",
                                os.path.join(OUT_DIR, "shap_bar.png"))
    print("  top tree-model contributors (mean |SHAP|):")
    print(shap_rank.head(8).to_string(index=False))

    lin_rank = explain.linear_contributions(lin, X.columns)
    explain.plot_importance_bar(lin_rank, "abs_coef",
                                f"Standardized coefficients — {chosen_linear}",
                                os.path.join(OUT_DIR, "linear_coefficients.png"))

    # ---------------------------------------------------- balance figure ---
    final_model = tree if winner == best_tree_name else lin
    projections = {
        "tree" if winner == best_tree_name else "linear":
            plots.project_future(final_model, daily, horizon=30),
        "baseline": _baseline_projection(daily, base.window, horizon=30),
    }
    plots.plot_balance_forecast(
        target_dates, anchors, actual_test_flows, test_flows, projections,
        os.path.join(OUT_DIR, "balance_forecast.png"),
        history_tail=daily.iloc[-(len(Xte) + 90):][["end_balance"]],
    )

    # ------------------------------------------------------ persist ---
    out = {
        "data": {
            "n_transactions": summary["n_transactions"],
            "n_days": summary["n_days"],
            "date_range": summary["date_range"],
            "final_balance": summary["final_balance"],
        },
        "split": {"train": len(Xtr), "test": len(Xte)},
        "regularization_triggered": bool(regularize),
        "ols_max_abs_coef": round(max_coef, 2),
        "chosen_linear": chosen_linear,
        "chosen_tree": best_tree_name,
        "winner": winner,
        "improvement_vs_baseline_pct": round(improvement, 1),
        "metrics": {k: {kk: round(float(vv), 3) for kk, vv in v.items()
                        if isinstance(vv, (int, float))} for k, v in results.items()},
        "top_shap_features": shap_rank.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    comp.to_csv(os.path.join(OUT_DIR, "metrics.csv"))
    print(f"\nSaved figures + metrics to {OUT_DIR}/")
    return out


def _baseline_projection(daily, window, horizon=30):
    """Flat-drift projection: future daily flow = mean of last `window` days."""
    flow = daily["net_flow"].tail(window).mean()
    balance = daily["end_balance"].iloc[-1]
    dates, bals = [], []
    for _ in range(horizon):
        balance += flow
        d = (dates[-1] if dates else daily.index[-1]) + pd.Timedelta(days=1)
        dates.append(d)
        bals.append(balance)
    return pd.DataFrame({"net_flow": flow, "end_balance": bals}, index=dates)


if __name__ == "__main__":
    main()
