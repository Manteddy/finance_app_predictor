# finance_app_predictor

Predicts a client's **future account balance trajectory** ("finances
accumulation") from past transaction activity. Built on two real Swedbank
statements that represent the kind of transaction-level data ingested into a
data lake.

The pipeline ingests the statements into a **mock SQLite database** (with every
transaction labelled by spending group), engineers **time-series-aware
features**, trains **three models** (two strong + one baseline), evaluates them
with a **time-respecting split**, applies a **Lasso→Ridge regularization
workflow**, and explains the predictions with **SHAP**.

---

## How to run

```bash
pip install -r requirements.txt
python -m src.etl            # build data/finance.db only (stdlib-only, optional)
python -m src.run_pipeline   # full pipeline: DB + features + 3 models + plots
```

Outputs land in `outputs/` (figures + `metrics.json` / `metrics.csv`).
The mock DB is written to `data/finance.db`.

---

## Data

| | |
|---|---|
| Source | 2 Swedbank statements, **same account / same person**, contiguous |
| Span | **2024-12-01 → 2026-06-18 (~18.5 months)** |
| Transactions | **3,552** |
| Daily points after aggregation | **565** |
| Final balance | €206.37 (matches the statement closing balance exactly — reconciliation check) |

The two statements stitch seamlessly: statement #2 closes at €491.55, which is
exactly the opening balance of statement #1.

### Mock database (`data/finance.db`)

- **`transactions`** — one row per real transaction: date, counterparty,
  details, signed amount, running balance, and a **`spending_group`** label.
- **`spending_groups`** — reference table describing each group.
- **`daily_balance`** — daily aggregation (net flow, end balance, transaction
  count, debit/credit sums, and per-group sums) used for feature engineering.

#### Spending-group labelling (data, not a pipeline)

Every transaction is assigned one of: `groceries`, `dining`, `transport`,
`travel`, `utilities`, `subscriptions`, `leisure`, `shopping`, `health`,
`cash`, `savings` (round-up "Rahakoguja" + Easy Saver moves), `income`,
`transfer`, `other`. This is a **one-time deterministic labelling** curated from
the actual merchant strings in these statements (`src/categorize.py`) — *not* a
trained classifier or categorisation service. ~95% of transactions land in a
specific group; the remaining ~5% (`other`) is a long tail of one-off merchants.

---

## Modelling approach

The models forecast **next-day net cash-flow** (the change in balance); the
balance line is then `balance_t = balance_{t-1} + flow_t`. Forecasting the flow
rather than the highly auto-correlated balance level keeps the target
stationary and stops a naive persistence model from trivially "winning".

- **Features** (`src/features.py`, 42 total, all strictly backward-looking to
  avoid leakage): calendar (day-of-week, day-of-month, month start/end, week),
  net-flow lags (1/2/3/7/14), rolling mean/std/sum over 3/7/14/30 days,
  per-spending-group trailing sums (7/30), transaction intensity, days since
  last income, and the lagged balance level.

- **Three models** (`src/models.py`):
  1. **Baseline** — moving average of recent net flow (window chosen by CV). A
     deliberately simple reference; it has no learned weights.
  2. **Linear regression** (strong, explainable) — standardised OLS, with the
     **Lasso→Ridge** regularization workflow below.
  3. **Random Forest / Gradient Boosting** (strong, pattern-capturing) — the
     better of the two is chosen by time-series CV (Random Forest here).

- **Time-respecting split** (`src/evaluate.py`): the last **60 days** are a
  pure holdout (test strictly after train); **expanding-window
  `TimeSeriesSplit`** is used for CV and hyper-parameter / window selection.
  Balance accuracy is scored **one-step-ahead, re-anchored** on the actual
  previous balance (the fair way to score a balance forecast — a free-running
  multi-step sum is shown only as the future projection line).

- **Regularization workflow** (per the brief): fit OLS, then trigger
  regularization if its weights are inflated **or** it overfits (its CV error
  is materially worse than the regularized variants). Here OLS overfits badly
  (CV RMSE **517** vs **284**), so **Lasso then Ridge** are fit and **Ridge** is
  selected. Tree complexity controls (depth, leaf size, subsample) are the
  tree-model analogue; the baseline has no weights to regularize.

- **Explainability** (`src/explain.py`): **SHAP** (the method that attributes a
  prediction additively across features — "how much each parameter
  contributed") for the tree model, plus standardised coefficients for the
  linear model.

---

## Results (60-day holdout)

| model | RMSE (flow, €/day) | MAE | R² | balance RMSE | balance MAE |
|---|---|---|---|---|---|
| baseline (moving avg) | **252.1** | **131.4** | −0.04 | **252.1** | **131.4** |
| linear (Ridge) | 278.4 | 209.9 | −0.27 | 278.4 | 209.9 |
| random forest | 276.7 | 189.4 | −0.25 | 276.7 | 189.4 |

**Honest read of these numbers:**

- **Daily individual cash-flow is largely noise-dominated.** R² is negative for
  *all* models, including the strong ones — at daily granularity for a single
  account there is little learnable point-by-point signal beyond "recent
  average", and the simple moving-average baseline is therefore hard to beat
  (it edges the strong models by ~10% on RMSE here). This is exactly why a
  baseline was included, and the result is reported rather than hidden.
- The strong models still add value the baseline cannot: they **expose the
  structure** driving the balance (see SHAP below) and give explainable,
  per-feature attributions. Among the strong models the **Random Forest** is the
  recommended choice (best balance error of the two, plus SHAP explainability).
- The recurring **monthly salary spike** (balance jumps to ~€2,200 then drains)
  is the dominant pattern; it shows up as the importance of `day_of_month` and
  the 30-day flow trend.

### What drives the prediction (SHAP)

Top contributors for the Random Forest: `nf_roll_sum_30` / `nf_roll_mean_30`
(the 30-day net-flow trend), `day_of_month` (salary timing), and
`savings_roll30` (recent round-up savings behaviour). See
`outputs/shap_summary.png` and `outputs/balance_forecast.png`.

### ⚠️ Data-size caveat

This is **one account over ~18.5 months**. ~565 daily points is enough to *demo
the methodology* end-to-end, but: (a) the model learns *this client's* habits
and will **not** generalise to other clients, and (b) daily net flow is
intrinsically noisy. For production-grade accumulation forecasting you would
want **many accounts** (panel data) and likely a **coarser horizon** (weekly /
monthly) where the signal-to-noise ratio is far better. Treat the numbers here
as a demonstrator, not a production benchmark.

---

## Repository layout

```
src/
  categorize.py     # spending-group labels (data, not a pipeline)
  etl.py            # parse 2 xlsx -> categorize -> SQLite mock DB (stdlib only)
  features.py       # daily aggregation + time-series features
  models.py         # baseline, linear (OLS->Lasso->Ridge), trees
  evaluate.py       # temporal split, metrics, one-step balance backtest
  explain.py        # SHAP + standardized coefficients
  plots.py          # balance-vs-time figure + forward projection
  run_pipeline.py   # end-to-end orchestration
data/raw/           # the two source statements (committed for reproducibility)
data/finance.db     # generated mock database
outputs/            # generated figures + metrics
```
