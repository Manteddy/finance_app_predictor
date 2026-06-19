"""Plotting: balance fan chart (deterministic line + uncertainty band),
multi-model holdout comparison, and daily-vs-weekly comparison."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# z for an 80% central interval (P10-P90).
_Z80 = 1.2816


def project_fan(det_future: pd.Series, last_balance: float,
                sigma_resid: float, horizon: int, drift: float = 0.0):
    """Forward balance projection with a widening uncertainty band.

    Central line = deterministic backbone + the historical mean residual
    ``drift`` (which captures typical irregular income the backbone omits); the
    band widens as residual variance accumulates over the horizon (independent
    residuals => std grows like sqrt(k)).
    """
    det = det_future.iloc[:horizon]
    central = last_balance + np.cumsum(det.values + drift)
    k = np.arange(1, len(det) + 1)
    band = _Z80 * sigma_resid * np.sqrt(k)
    return pd.DataFrame(
        {"p50": central, "p10": central - band, "p90": central + band},
        index=det.index[:len(central)],
    )


def plot_fan_chart(target_dates, anchors, actual_total_flows,
                   flow_p10, flow_p50, flow_p90, projection, out_png,
                   history_tail=None, title="Balance forecast — fan chart"):
    """Headline figure: deterministic P50 balance line + P10-P90 band vs actual,
    over the holdout, then a forward projection fan.

    Holdout balances are one-step re-anchored: balance = actual balance today +
    predicted next-period flow (per quantile).
    """
    dates = pd.DatetimeIndex(target_dates)
    anchors = np.asarray(anchors, dtype=float)
    actual_bal = anchors + np.asarray(actual_total_flows)
    p10, p50, p90 = (anchors + np.asarray(f) for f in (flow_p10, flow_p50, flow_p90))

    plt.figure(figsize=(13, 6))
    if history_tail is not None:
        plt.plot(history_tail.index, history_tail["end_balance"],
                 color="0.6", lw=1, label="actual (history)")

    plt.plot(dates, actual_bal, color="black", lw=2.2, label="actual")
    plt.plot(dates, p50, color="tab:blue", lw=1.8, label="forecast P50")
    plt.fill_between(dates, p10, p90, color="tab:blue", alpha=0.20,
                     label="P10–P90 (holdout)")

    if projection is not None and len(projection):
        plt.axvline(projection.index[0], color="0.8", ls=":", lw=1)
        plt.plot(projection.index, projection["p50"], color="tab:purple",
                 lw=1.8, ls="--", label="projection P50")
        plt.fill_between(projection.index, projection["p10"], projection["p90"],
                         color="tab:purple", alpha=0.15, label="P10–P90 (projection)")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Balance (EUR)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def plot_balance_forecast(target_dates, anchors, actual_flows,
                          model_pred_flows: dict, out_png, history_tail=None,
                          title="Account balance: actual vs predicted (holdout)"):
    """Multi-model one-step holdout comparison (one line per model)."""
    dates = pd.DatetimeIndex(target_dates)
    anchors = np.asarray(anchors, dtype=float)
    actual_bal = anchors + np.asarray(actual_flows)

    plt.figure(figsize=(13, 6))
    if history_tail is not None:
        plt.plot(history_tail.index, history_tail["end_balance"],
                 color="0.6", lw=1, label="actual (history)")
    plt.plot(dates, actual_bal, color="black", lw=2.2, label="actual")
    for name, flows in model_pred_flows.items():
        plt.plot(dates, anchors + np.asarray(flows), "--", lw=1.5, label=name)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Balance (EUR)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def plot_track_comparison(summary: pd.DataFrame, out_png):
    """Bar chart comparing R² (total flow) across granularity x mode x model."""
    piv = summary.pivot_table(index="model", columns="track", values="r2")
    ax = piv.plot(kind="bar", figsize=(11, 6))
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("R²  (next-period total flow, holdout)")
    ax.set_title("Daily vs Weekly  ×  direct vs decomposed — R² by model")
    ax.legend(title="track", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
