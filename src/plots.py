"""Plotting: the headline balance-vs-time forecast and a recursive projection."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import _SPEND_GROUPS, compute_feature_matrix


def project_future(model, daily: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Recursively forecast net flow and balance ``horizon`` days ahead.

    Future spending-group features are held at their trailing-30-day average
    (their levels are unknown), while net-flow lags/rollings evolve with the
    model's own predictions. This is an approximation, flagged as such.
    """
    work = daily.copy()
    grp_cols = [f"grp_{g}" for g in _SPEND_GROUPS if f"grp_{g}" in work.columns]
    recent = work.tail(30)
    grp_means = {c: recent[c].mean() for c in grp_cols}
    txn_mean = recent["txn_count"].mean()

    out_dates, out_flows, out_bal = [], [], []
    balance = work["end_balance"].iloc[-1]

    for _ in range(horizon):
        feat = compute_feature_matrix(work)
        x_last = feat.iloc[[-1]].fillna(0.0)
        flow = float(model.predict(x_last)[0])
        next_date = work.index[-1] + pd.Timedelta(days=1)
        balance = balance + flow

        new_row = {c: 0.0 for c in work.columns}
        new_row["net_flow"] = flow
        new_row["end_balance"] = balance
        new_row["txn_count"] = txn_mean
        for c in grp_cols:
            new_row[c] = grp_means[c]
        work.loc[next_date] = new_row

        out_dates.append(next_date)
        out_flows.append(flow)
        out_bal.append(balance)

    return pd.DataFrame(
        {"net_flow": out_flows, "end_balance": out_bal}, index=out_dates
    )


def plot_balance_forecast(
    target_dates,
    anchors,
    actual_flows,
    model_pred_flows: dict,
    projections: dict,
    out_png: str,
    history_tail=None,
):
    """Headline figure: actual vs predicted balance over the holdout, plus a
    forward projection for the chosen model and the baseline.

    Holdout lines use one-step-ahead re-anchored balances (predicted next-day
    balance = actual balance today + predicted next-day flow).
    ``model_pred_flows`` maps model name -> predicted next-day flows (test).
    ``projections`` maps model name -> projected balance DataFrame.
    """
    dates = pd.DatetimeIndex(target_dates)
    anchors = np.asarray(anchors, dtype=float)
    actual_bal = anchors + np.asarray(actual_flows)

    plt.figure(figsize=(13, 6))

    if history_tail is not None:
        plt.plot(history_tail.index, history_tail["end_balance"],
                 color="0.6", lw=1, label="actual (history)")

    plt.plot(dates, actual_bal, color="black", lw=2.2, label="actual (holdout)")

    colors = {"baseline": "tab:orange", "linear": "tab:green",
              "tree": "tab:blue"}
    for name, flows in model_pred_flows.items():
        pred_bal = anchors + np.asarray(flows)
        plt.plot(dates, pred_bal, "--", lw=1.6,
                 color=colors.get(name, None), label=f"{name} (1-step pred)")

    if dates.size:
        plt.axvline(dates[0], color="0.8", ls=":", lw=1)
    for name, proj in projections.items():
        plt.plot(proj.index, proj["end_balance"], ":", lw=2,
                 color=colors.get(name, None), label=f"{name} (projection)")

    plt.title("Account balance: actual vs predicted (holdout) and forward projection")
    plt.xlabel("Date")
    plt.ylabel("Balance (EUR)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
