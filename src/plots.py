"""Plotting: balance fan chart (deterministic line + gradient uncertainty band),
multi-model holdout comparison, and daily-vs-weekly comparison."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import compute_feature_matrix

# z-scores for central-interval levels (normal approx; avoids a scipy dep).
_Z = {0.5: 0.674, 0.7: 1.036, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960}
FAN_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]


def project_recursive(model, df, freq, deterministic, horizon, decomposed,
                      last_balance):
    """Recursive multi-step projection that carries the model's own dynamics.

    Each future period: build features from the evolving series, predict the
    (residual or total) flow, add the deterministic backbone, update the balance
    and append the row. Future spending-group columns are held at trailing means.
    This yields week-to-week texture (from lags/Fourier) plus the monthly salary,
    rather than a flat deterministic sawtooth. Returns a balance Series.
    """
    work = df.copy()
    grp_cols = [c for c in work.columns if c.startswith("grp_")]
    recent = work.tail(8 if freq == "W" else 30)
    grp_means = {c: recent[c].mean() for c in grp_cols}
    txn_mean = recent["txn_count"].mean()
    step = pd.Timedelta(weeks=1) if freq == "W" else pd.Timedelta(days=1)

    dates, balances = [], []
    balance = last_balance
    for _ in range(horizon):
        feat = compute_feature_matrix(work, freq, deterministic)
        x = feat.iloc[[-1]].fillna(0.0)
        pred = float(model.predict(x)[0])
        nxt = work.index[-1] + step
        det_next = float(deterministic.get(nxt, 0.0)) if deterministic is not None else 0.0
        flow = det_next + pred if decomposed else pred
        balance += flow

        row = {c: 0.0 for c in work.columns}
        row["net_flow"], row["end_balance"], row["txn_count"] = flow, balance, txn_mean
        for c in grp_cols:
            row[c] = grp_means[c]
        work.loc[nxt] = row
        dates.append(nxt)
        balances.append(balance)
    return pd.Series(balances, index=dates)


def build_projection_bands(proj_p50: pd.Series, sigma: float,
                           last_date, last_balance, levels=FAN_LEVELS):
    """Wrap a P50 projection with widening central intervals + a connecting
    anchor at (last_date, last_balance) so the fan starts where actual ends.
    Returns a DataFrame indexed by date with p50 and lo/hi per level.
    """
    k = np.arange(1, len(proj_p50) + 1)
    data = {"p50": proj_p50.values}
    for lv in levels:
        half = _Z[lv] * sigma * np.sqrt(k)
        data[f"lo_{lv}"] = proj_p50.values - half
        data[f"hi_{lv}"] = proj_p50.values + half
    df = pd.DataFrame(data, index=proj_p50.index)
    anchor = pd.DataFrame(
        {c: [last_balance] for c in df.columns}, index=[pd.Timestamp(last_date)])
    return pd.concat([anchor, df])


def _gradient_band(ax, x, p50, bands, color):
    """Draw nested central intervals widest->narrowest with rising alpha
    (darker near the centre line). ``bands`` = list of (lo, hi) widest first."""
    n = len(bands)
    for i, (lo, hi) in enumerate(bands):
        ax.fill_between(x, lo, hi, color=color, alpha=0.07 + 0.05 * i, lw=0)


def plot_fan_chart(dates, actual_bal, p50_bal, holdout_bands,
                   projection_df, out_png, history_tail=None,
                   title="Balance forecast — fan chart"):
    """Headline figure: P50 balance + gradient P-interval fan over the holdout,
    then a connected forward projection fan.

    ``holdout_bands`` = list of (lo_bal, hi_bal) widest->narrowest.
    ``projection_df`` from ``build_projection_bands`` (anchor row included).
    """
    dates = pd.DatetimeIndex(dates)
    fig, ax = plt.subplots(figsize=(13, 6))
    if history_tail is not None:
        ax.plot(history_tail.index, history_tail["end_balance"],
                color="0.6", lw=1, label="actual (history)")

    _gradient_band(ax, dates, p50_bal, holdout_bands, "tab:blue")
    ax.plot(dates, actual_bal, color="black", lw=2.2, label="actual")
    ax.plot(dates, p50_bal, color="tab:blue", lw=1.8, label="forecast P50")

    if projection_df is not None and len(projection_df) > 1:
        px = pd.DatetimeIndex(projection_df.index)
        ax.axvline(px[0], color="0.8", ls=":", lw=1)
        proj_bands = [(projection_df[f"lo_{lv}"].values, projection_df[f"hi_{lv}"].values)
                      for lv in sorted(FAN_LEVELS, reverse=True)]
        _gradient_band(ax, px, projection_df["p50"].values, proj_bands, "tab:purple")
        ax.plot(px, projection_df["p50"].values, color="tab:purple", lw=1.8,
                ls="--", label="projection P50")

    # Legend proxy for the band gradient.
    ax.fill_between([], [], [], color="tab:blue", alpha=0.2,
                    label="P50→P95 fan (darker = more likely)")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance (EUR)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    ax = piv.plot(kind="bar", figsize=(12, 6))
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylim(-0.7, 0.6)  # clip SARIMA daily-decomposed outliers (~-3.3) for readability
    ax.set_ylabel("R²  (next-period total flow, holdout)")
    ax.set_title("Daily vs Weekly  ×  direct / decomposed / agnostic — R² by model "
                 "(y clipped at −0.7)")
    ax.legend(title="track", fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
