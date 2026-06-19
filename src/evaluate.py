"""Time-respecting evaluation: temporal split, metrics and balance reconstruction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def temporal_split(X, y, frame, test_days: int = 60):
    """Split chronologically: the last ``test_days`` rows are the holdout.

    No shuffling — the test period is strictly after the training period.
    """
    n_test = min(test_days, len(X) // 4)
    Xtr, Xte = X.iloc[:-n_test], X.iloc[-n_test:]
    ytr, yte = y.iloc[:-n_test], y.iloc[-n_test:]
    ftr, fte = frame.iloc[:-n_test], frame.iloc[-n_test:]
    assert Xtr.index.max() < Xte.index.min(), "test must follow train in time"
    return (Xtr, ytr, ftr), (Xte, yte, fte)


def flow_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def pinball_loss(y_true, q_pred, alpha) -> float:
    """Average pinball (quantile) loss — the proper score for a quantile."""
    y_true = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    diff = y_true - q_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def interval_coverage(y_true, lo, hi) -> float:
    """Fraction of actuals inside [lo, hi] (target ~0.8 for a P10-P90 band)."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= np.asarray(lo)) & (y_true <= np.asarray(hi))))


def balance_one_step(anchors, true_flows, pred_flows):
    """One-step-ahead balance backtest, re-anchored on the actual balance.

    For each holdout day, predicted next-day balance = actual balance today +
    predicted next-day flow. This is the standard, fair way to score a balance
    forecast: it does not let small daily biases compound over the whole window
    (a free-running multi-step projection, used only for the future line, does).
    """
    anchors = np.asarray(anchors, dtype=float)
    true_bal = anchors + np.asarray(true_flows)
    pred_bal = anchors + np.asarray(pred_flows)
    return {
        "balance_rmse": float(np.sqrt(mean_squared_error(true_bal, pred_bal))),
        "balance_mae": float(mean_absolute_error(true_bal, pred_bal)),
    }, true_bal, pred_bal
