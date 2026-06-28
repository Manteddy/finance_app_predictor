"""Time-series-aware feature engineering for daily AND weekly forecasting.

Target = next-period net cash-flow (the change in balance). Forecasting the flow
rather than the balance level keeps the problem stationary; the balance line is
reconstructed by cumulative sum (balance_t = balance_{t-1} + flow_t).

Two target modes:
  * "total"    - predict the full next-period net flow.
  * "residual" - predict only the stochastic residual (next-period flow minus
                 the deterministic backbone from ``recurring.py``); the total is
                 then reconstructed as deterministic + residual.

All features describe information available *up to and including period t* and
predict period t+1, so there is no look-ahead leakage into the target.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

# Spending groups whose recent rate is informative for upcoming flow.
_SPEND_GROUPS = ["groceries", "dining", "transport", "shopping",
                 "subscriptions", "utilities", "savings", "income"]

# Lag / rolling windows per granularity (weekly uses fewer, wider windows).
LAGS = {"D": [1, 2, 3, 7, 14], "W": [1, 2, 3, 4, 8]}
ROLL_WINDOWS = {"D": [3, 7, 14, 30], "W": [4, 8, 12]}
GROUP_ROLLS = {"D": [7, 30], "W": [4, 12]}
# Monthly-cycle period in each granularity's units, for Fourier seasonality.
MONTH_PERIOD = {"D": 30.44, "W": 4.345}
FOURIER_K = {"D": 3, "W": 2}


def load_daily(db_path: str = "data/finance.db") -> pd.DataFrame:
    """Load the daily aggregation from the mock database, date-indexed."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM daily_balance ORDER BY date", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def to_core(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a 'spendable cash' view that routes own-account flows out.

    NOTE: this was an explored width-reduction lever that **backfires** and is NOT
    used by the headline. Empirically the ``savings`` group (Rahakoguja round-ups,
    Easy Saver, transfers between own accounts) is *compensatory* — the client
    tops up from savings when spending spikes — so it has negative covariance with
    net flow and removing it *raises* variance (target std 438 -> 763). Kept for
    reference / experimentation; see REPORT.md.

    Subtracts savings from net flow, recomputes the balance as the cumulative core
    flow from the opening balance, and zeroes grp_savings to avoid double counting.
    """
    df = daily.copy()
    savings = df["grp_savings"] if "grp_savings" in df.columns else 0.0
    opening = float(df["end_balance"].iloc[0] - df["net_flow"].iloc[0])
    df["net_flow"] = df["net_flow"] - savings
    df["end_balance"] = opening + df["net_flow"].cumsum()
    if "grp_savings" in df.columns:
        df["grp_savings"] = 0.0
    return df


def aggregate(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Return the series at the requested granularity ('D' or 'W')."""
    if freq == "D":
        return daily.copy()
    agg = {c: "sum" for c in daily.columns if c != "end_balance"}
    agg["end_balance"] = "last"
    out = daily.resample("W").agg(agg)
    return out.dropna(subset=["end_balance"])


def _fourier(index: pd.DatetimeIndex, period: float, k: int) -> pd.DataFrame:
    """Sin/cos harmonics of a seasonal cycle (smooth periodic features)."""
    t = np.arange(len(index))
    out = {}
    for h in range(1, k + 1):
        out[f"fourier_sin_{h}"] = np.sin(2 * np.pi * h * t / period)
        out[f"fourier_cos_{h}"] = np.cos(2 * np.pi * h * t / period)
    return pd.DataFrame(out, index=index)


def compute_feature_matrix(df: pd.DataFrame, freq: str,
                           deterministic: pd.Series | None = None) -> pd.DataFrame:
    """Build the full feature matrix (one row per period, target-agnostic).

    Rows may contain NaNs in the warm-up window; the final row is retained so it
    can seed the first future-period forecast.
    """
    nf = df["net_flow"]
    feat = pd.DataFrame(index=df.index)

    # --- calendar (raw bits only where they carry sub-monthly signal) ---
    if freq == "D":
        feat["dow"] = df.index.dayofweek
        feat["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    feat["month"] = df.index.month
    # Smooth monthly seasonality via Fourier harmonics (replaces raw day/week).
    feat = feat.join(_fourier(df.index, MONTH_PERIOD[freq], FOURIER_K[freq]))

    # --- net-flow lags ---
    for lag in LAGS[freq]:
        feat[f"nf_lag_{lag}"] = nf.shift(lag)

    # --- net-flow rolling stats (closed on the past, includes period t) ---
    for w in ROLL_WINDOWS[freq]:
        feat[f"nf_roll_mean_{w}"] = nf.rolling(w).mean()
        feat[f"nf_roll_std_{w}"] = nf.rolling(w).std()
        feat[f"nf_roll_sum_{w}"] = nf.rolling(w).sum()

    # --- transaction intensity ---
    feat["txn_roll_mean"] = df["txn_count"].rolling(ROLL_WINDOWS[freq][0]).mean()

    # --- spending-group rates (trailing sums) ---
    for g in _SPEND_GROUPS:
        col = f"grp_{g}"
        if col in df.columns:
            for w in GROUP_ROLLS[freq]:
                feat[f"{g}_roll{w}"] = df[col].rolling(w).sum()

    # --- periods since last income inflow ---
    has_income = (df["grp_income"] > 0).astype(int).values
    counter, since = 0, []
    for v in has_income:
        counter = 0 if v else counter + 1
        since.append(counter)
    feat["periods_since_income"] = since

    # --- balance level (helps tree model anchor) ---
    feat["balance_lag_1"] = df["end_balance"].shift(1)

    # --- deterministic-backbone signal (known in advance, exogenous) ---
    if deterministic is not None:
        det = deterministic.reindex(df.index)
        feat["expected_flow"] = det                 # this period's expectation
        feat["expected_flow_next"] = det.shift(-1)  # next period's expectation
    return feat


def build_features(df: pd.DataFrame, freq: str, target_mode: str = "total",
                   deterministic: pd.Series | None = None):
    """Return (X, y, frame).

    ``y`` is the next-period target: full flow ("total") or residual flow
    ("residual" = flow - deterministic). ``frame`` carries everything needed to
    reconstruct the total flow and balance:
        net_flow, end_balance, total_next (actual next flow), det_next.
    """
    feat = compute_feature_matrix(df, freq, deterministic)

    total_next = df["net_flow"].shift(-1)
    if deterministic is not None:
        det = deterministic.reindex(df.index)
    else:
        det = pd.Series(0.0, index=df.index)
    det_next = det.shift(-1)

    if target_mode == "residual":
        y = total_next - det_next
    else:
        y = total_next

    frame = pd.DataFrame({
        "net_flow": df["net_flow"],
        "end_balance": df["end_balance"],
        "total_next": total_next,
        "det_next": det_next,
    })

    valid = feat.dropna().index.intersection(y.dropna().index)
    valid = valid.intersection(frame.dropna().index)
    return feat.loc[valid], y.loc[valid], frame.loc[valid]


def reconstruct_balance(start_balance: float, flows: np.ndarray) -> np.ndarray:
    """Reconstruct a balance trajectory from a starting balance and flows."""
    return start_balance + np.cumsum(flows)
