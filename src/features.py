"""Time-series-aware feature engineering for daily balance forecasting.

Target = **next-day net cash-flow** (the change in balance). Forecasting the
flow rather than the balance level keeps the problem stationary and prevents a
naive persistence model from trivially "winning"; the balance trajectory is
then reconstructed by cumulative sum (balance_t = balance_{t-1} + flow_t).

All features describe information available *up to and including day t* and are
used to predict day t+1, so there is no look-ahead leakage into the target.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

# Spending groups whose recent rate is informative for upcoming flow.
_SPEND_GROUPS = ["groceries", "dining", "transport", "shopping",
                 "subscriptions", "utilities", "savings", "income"]

LAGS = [1, 2, 3, 7, 14]
ROLL_WINDOWS = [3, 7, 14, 30]


def load_daily(db_path: str = "data/finance.db") -> pd.DataFrame:
    """Load the daily aggregation from the mock database, date-indexed."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM daily_balance ORDER BY date", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def compute_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix (one row per day, target-agnostic).

    Rows may contain NaNs in the warm-up period; the final row is retained so
    it can be used to forecast the first future day.
    """
    df = daily.copy()
    nf = df["net_flow"]

    feat = pd.DataFrame(index=df.index)

    # --- calendar ---
    feat["dow"] = df.index.dayofweek
    feat["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    feat["day_of_month"] = df.index.day
    feat["is_month_start"] = df.index.is_month_start.astype(int)
    feat["is_month_end"] = df.index.is_month_end.astype(int)
    feat["week_of_year"] = df.index.isocalendar().week.astype(int).values

    # --- net-flow lags ---
    for lag in LAGS:
        feat[f"nf_lag_{lag}"] = nf.shift(lag)

    # --- net-flow rolling stats (closed on the past, includes day t) ---
    for w in ROLL_WINDOWS:
        feat[f"nf_roll_mean_{w}"] = nf.rolling(w).mean()
        feat[f"nf_roll_std_{w}"] = nf.rolling(w).std()
        feat[f"nf_roll_sum_{w}"] = nf.rolling(w).sum()

    # --- transaction intensity ---
    feat["txn_roll_mean_7"] = df["txn_count"].rolling(7).mean()

    # --- spending-group rates (trailing 7 and 30 day sums) ---
    for g in _SPEND_GROUPS:
        col = f"grp_{g}"
        if col in df.columns:
            feat[f"{g}_roll7"] = df[col].rolling(7).sum()
            feat[f"{g}_roll30"] = df[col].rolling(30).sum()

    # --- days since last income inflow ---
    has_income = (df["grp_income"] > 0).astype(int)
    days_since = []
    counter = 0
    for v in has_income.values:
        counter = 0 if v else counter + 1
        days_since.append(counter)
    feat["days_since_income"] = days_since

    # --- balance level (helps tree model anchor) ---
    feat["balance_lag_1"] = df["end_balance"].shift(1)
    return feat


def build_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, frame) where y is next-day net_flow.

    ``frame`` keeps net_flow / end_balance alongside X for reconstruction.
    All features describe day t and predict the flow on day t+1 (no leakage).
    """
    feat = compute_feature_matrix(daily)
    y = daily["net_flow"].shift(-1)  # next-day net flow
    frame = daily[["net_flow", "end_balance"]].copy()

    # Drop rows with NaNs from the longest lag/rolling window or the final
    # (target-less) row.
    valid = feat.dropna().index.intersection(y.dropna().index)
    return feat.loc[valid], y.loc[valid], frame.loc[valid]


def reconstruct_balance(start_balance: float, flows: np.ndarray) -> np.ndarray:
    """Reconstruct a balance trajectory from a starting balance and flows."""
    return start_balance + np.cumsum(flows)
