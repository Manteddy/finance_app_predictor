"""The three models: a baseline plus two strong learners, with the
Lasso-then-Ridge regularization workflow.

  * Baseline      - moving-average of recent net flow (no learned weights).
  * Linear family - OLS, then Lasso, then Ridge (regularization workflow).
  * Trees         - RandomForest vs GradientBoosting, better one chosen by CV.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
class MovingAverageBaseline:
    """Predict next-day net flow as the mean of the last ``window`` days.

    The window is selected by time-series CV on the training set. This model
    has no learned weights, so the regularization workflow does not apply to it
    (documented in the report).
    """

    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, X, y):
        # The trailing-mean features are precomputed; pick the window whose
        # corresponding feature best predicts y on expanding CV folds. Windows
        # are discovered from the feature columns so this works at any freq.
        candidates = sorted(
            int(c.rsplit("_", 1)[1]) for c in X.columns
            if c.startswith("nf_roll_mean_")
        )
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 10)))
        best_w, best_rmse = candidates[0], np.inf
        for w in candidates:
            col = f"nf_roll_mean_{w}"
            rmses = []
            for _, te in tscv.split(X):
                pred = X.iloc[te][col].values
                rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
            if np.mean(rmses) < best_rmse:
                best_rmse, best_w = np.mean(rmses), w
        self.window = best_w
        self._col = f"nf_roll_mean_{best_w}"
        return self

    def predict(self, X):
        return X[self._col].values


# --------------------------------------------------------------------------- #
# Linear family with regularization workflow
# --------------------------------------------------------------------------- #
def _cv_rmse(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
    return float(np.mean(rmses))


def build_linear_models():
    """Return OLS, Lasso(CV) and Ridge(CV) pipelines (all standardized)."""
    tscv = TimeSeriesSplit(n_splits=5)
    return {
        "linear_ols": Pipeline([
            ("scale", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "linear_lasso": Pipeline([
            ("scale", StandardScaler()),
            ("reg", LassoCV(cv=tscv, random_state=RANDOM_STATE, max_iter=20000)),
        ]),
        "linear_ridge": Pipeline([
            ("scale", StandardScaler()),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]),
    }


def ols_max_abs_coef(pipeline, X, y) -> float:
    """Fit an OLS pipeline and return the largest absolute (standardized) coef.

    Used to decide whether the regularization workflow is needed: on
    standardized features, very large coefficients signal multicollinearity /
    overfitting.
    """
    pipeline.fit(X, y)
    return float(np.max(np.abs(pipeline.named_steps["reg"].coef_)))


# --------------------------------------------------------------------------- #
# Tree family
# --------------------------------------------------------------------------- #
def build_tree_models():
    """Return RandomForest and GradientBoosting regressors (regularized).

    Complexity controls (depth, leaf size, estimators, subsample) are the
    tree-model analogue of regularization on this small single-account series.
    """
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    }


def select_best_tree(X, y):
    """Pick the better tree model by time-series CV RMSE."""
    results = {name: _cv_rmse(m, X, y) for name, m in build_tree_models().items()}
    best = min(results, key=results.get)
    return best, build_tree_models()[best], results


# --------------------------------------------------------------------------- #
# Classical seasonal model (4th contender)
# --------------------------------------------------------------------------- #
def sarima_one_step(train_series, full_series, test_index, freq):
    """One-step-ahead SARIMA forecast over the holdout.

    Params are estimated on the training series, then applied (filtered) to the
    full series so ``get_prediction(dynamic=False)`` yields genuine
    one-step-ahead predictions on the test period (each uses the *actual* past).
    A data-efficient structural model well-suited to short seasonal series.
    Returns (point, lower, upper) aligned to ``test_index``.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    m = 7 if freq == "D" else 4  # weekly seasonality (daily) / monthly (weekly)
    order, seasonal = (1, 1, 1), (1, 0, 0, m)
    try:
        fitted = SARIMAX(train_series, order=order, seasonal_order=seasonal,
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
        applied = fitted.apply(full_series, refit=False)
        pred = applied.get_prediction(start=test_index[0], dynamic=False)
        mean = pred.predicted_mean.reindex(test_index)
        ci = pred.conf_int(alpha=0.2).reindex(test_index)  # 80% interval
        return mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values
    except Exception as exc:  # pragma: no cover - robustness on tiny series
        print(f"  [sarima] fell back to flat forecast: {exc}")
        last = float(train_series.iloc[-1])
        n = len(test_index)
        return np.full(n, last), np.full(n, last), np.full(n, last)
