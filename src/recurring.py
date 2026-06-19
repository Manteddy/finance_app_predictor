"""Deterministic backbone: recurring-event detection + habitual spend rates.

The forecasting architecture splits net flow into:

    net_flow  =  deterministic  +  residual

where ``deterministic`` is the *predictable* part — fixed recurring debits
(subscriptions, utilities, rent) on a stable monthly cadence, plus stable
habitual spend rates for everyday categories (e.g. ~EUR X/week on groceries) —
and ``residual`` is the genuinely stochastic discretionary part that the ML
models forecast and around which we draw an uncertainty fan.

To avoid look-ahead leakage, recurring rules and habitual rates are estimated on
the **training portion only** (``cutoff`` date) and then applied everywhere,
including the future projection.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Everyday categories with a roughly stable per-period spend rate.
HABITUAL_GROUPS = ["groceries", "dining", "transport"]
# Groups with fixed, scheduled, recurring flows worth detecting. "income"
# captures the monthly salary (Wise, last working day of the month) — the
# largest and most regular flow, which dominates the deterministic backbone.
RECURRING_GROUPS = ["income", "subscriptions", "utilities"]


@dataclass
class RecurringRule:
    key: str            # normalised merchant key
    group: str
    amount: float       # median signed amount per occurrence
    day_of_month: int   # typical calendar day it lands on
    n: int              # occurrences seen in training


def load_transactions(db_path: str = "data/finance.db") -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT txn_date, counterparty, details, amount, "
                     "spending_group FROM transactions", con,
                     parse_dates=["txn_date"])
    con.close()
    return df


def _merchant_key(counterparty: str, details: str) -> str:
    """Stable key for grouping the same recurring merchant together."""
    base = (counterparty or "").strip() or (details or "")
    base = re.sub(r"\d", "", base).upper()
    return re.sub(r"\s+", " ", base).strip()[:24]


def detect_recurring(tx: pd.DataFrame, cutoff: pd.Timestamp) -> list[RecurringRule]:
    """Detect monthly fixed-amount recurring transactions from training data."""
    train = tx[tx["txn_date"] < cutoff].copy()
    train = train[train["spending_group"].isin(RECURRING_GROUPS)]
    train["key"] = [_merchant_key(c, d)
                    for c, d in zip(train["counterparty"], train["details"])]

    rules: list[RecurringRule] = []
    for key, g in train.groupby("key"):
        if len(g) < 3:
            continue
        dates = g["txn_date"].sort_values()
        gaps = dates.diff().dropna().dt.days
        # Monthly cadence with a stable amount.
        if not (20 <= gaps.median() <= 40):
            continue
        amt = g["amount"].median()
        if abs(amt) < 1:
            continue
        rules.append(RecurringRule(
            key=key, group=g["spending_group"].iloc[0],
            amount=float(amt), day_of_month=int(dates.dt.day.median()),
            n=len(g),
        ))
    return rules


def _monthly_occurrences(rule: RecurringRule, start, end) -> list[pd.Timestamp]:
    """All calendar dates a monthly rule fires between start and end."""
    out = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        dom = min(rule.day_of_month, period.days_in_month)
        d = pd.Timestamp(year=period.year, month=period.month, day=dom)
        if start <= d <= end:
            out.append(d)
    return out


def habitual_rates(daily: pd.DataFrame, cutoff: pd.Timestamp, freq: str) -> dict:
    """Median per-period spend for each habitual group, learned on training."""
    train = daily[daily.index < cutoff]
    if freq == "W":
        train = train.resample("W").sum(numeric_only=True)
    rates = {}
    for grp in HABITUAL_GROUPS:
        col = f"grp_{grp}"
        if col in train.columns:
            rates[grp] = float(train[col].median())
    return rates


def build_deterministic(period_index: pd.DatetimeIndex,
                        rules: list[RecurringRule],
                        habitual: dict,
                        freq: str) -> pd.Series:
    """Expected (deterministic) net flow for each period in ``period_index``.

    = scheduled recurring debits landing in the period + constant habitual
    spend levels. Defined over the full history *and* any future periods passed
    in ``period_index``, so it powers both training residuals and projection.
    """
    idx = pd.DatetimeIndex(period_index)
    det = pd.Series(0.0, index=idx)

    # Habitual baseline: applied every period.
    habitual_total = sum(habitual.values())
    det += habitual_total

    # Recurring events: bucket each occurrence into its period.
    start, end = idx.min(), idx.max()
    for rule in rules:
        for d in _monthly_occurrences(rule, start, end):
            if freq == "W":
                # Assign to the week-ending label (resample('W') uses Sunday).
                bucket = d + pd.offsets.Week(weekday=6)
            else:
                bucket = d
            pos = idx.searchsorted(bucket)
            if 0 <= pos < len(idx):
                det.iloc[pos] += rule.amount
    return det
