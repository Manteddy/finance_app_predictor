# finance_app_predictor

Predicts a client's **future account balance trajectory** ("finances
accumulation") from past transaction activity. Built on two real Swedbank
statements that represent the kind of transaction-level data ingested into a
data lake.

The pipeline ingests the statements into a **mock SQLite database** (every
transaction labelled by spending group), engineers **time-series-aware
features**, and trains four models across a **2×2 grid** of granularity
(daily / weekly) × architecture (direct / decomposed). It applies a
**Lasso→Ridge regularization workflow**, explains predictions with **SHAP**, and
produces a **balance fan chart** — a deterministic forecast line with a
calibrated P10–P90 uncertainty band.

---

## How to run

```bash
pip install -r requirements.txt
python -m src.etl            # build data/finance.db only (stdlib-only, optional)
python -m src.run_pipeline   # full grid: DB + features + 4 models x 4 tracks + plots
```

Outputs land in `outputs/` (figures + `metrics.json` / `metrics.csv`).

---

## Data

| | |
|---|---|
| Source | 2 Swedbank statements, **same account / same person**, contiguous |
| Span | **2024-12-01 → 2026-06-18 (~18.5 months)** |
| Transactions | **3,552** |
| Daily points | **565**  ·  Weekly points: **~81** |
| Final balance | €206.37 (matches the statement closing balance exactly) |

### Mock database (`data/finance.db`)
- **`transactions`** — date, counterparty, details, signed amount, balance, and a
  **`spending_group`** label.
- **`spending_groups`** — reference table describing each group.
- **`daily_balance`** — daily aggregation (net flow, balance, counts, per-group sums).

Spending groups are assigned by a **one-time deterministic labelling** curated
from the real merchant strings (`src/categorize.py`) — *not* a trained
classifier. ~95% of transactions land in a specific group.

---

## Modelling approach

Models forecast **next-period net cash-flow**; the balance line is reconstructed
as `balanceₜ = balanceₜ₋₁ + flowₜ`. Two architectures are compared:

- **Direct** — predict the full next-period net flow from features.
- **Decomposed** — split `flow = deterministic + residual`:
  - **Deterministic backbone** (`src/recurring.py`): detected monthly recurring
    debits (subscriptions, utilities, dormitory/rent) on their schedule **plus**
    stable habitual category rates (e.g. median weekly groceries). Fit on the
    **training portion only** (no leakage), extendable into the future.
  - **Residual** — the genuinely stochastic discretionary part, which the ML
    models forecast and around which we draw the **fan chart**.

### Features (`src/features.py`), all strictly backward-looking
Calendar + **Fourier harmonics** of the monthly cycle (smooth seasonality),
net-flow lags & rolling mean/std/sum, per-spending-group trailing sums,
transaction intensity, periods-since-income, lagged balance, and the
**deterministic-backbone signal** (`expected_flow`, `expected_flow_next`).

### Four models (`src/models.py`)
1. **Baseline** — moving average of recent net flow (window via CV). No weights.
2. **Linear regression** — standardised OLS with the **Lasso→Ridge**
   regularization workflow (triggered on inflated/overfit OLS weights).
3. **Trees** — RandomForest vs GradientBoosting, better chosen by CV.
4. **SARIMA** (`statsmodels`) — a data-efficient classical seasonal model
   (one-step-ahead), strong on short seasonal series.

### Evaluation (`src/evaluate.py`)
Time-respecting holdout (60 days / 12 weeks), strictly after train, with
expanding-window `TimeSeriesSplit` CV. Point metrics (RMSE/MAE/R²) are always
computed on the reconstructed **total** flow so all tracks are comparable.
Probabilistic quality is scored with **pinball loss** and **P10–P90 interval
coverage**.

### Fan chart (`src/plots.py`)
The P10–P90 band uses **split-conformal calibration** (out-of-fold residuals
across the training set) rather than fragile conditional quantile regression —
appropriate for small data. The forward projection's central line follows the
deterministic backbone plus the historical mean residual (typical income), with
the band widening as residual variance accumulates over the horizon.

---

## Results — R² on next-period total flow (holdout)

| track | baseline | linear | tree | sarima |
|---|---|---|---|---|
| **Daily**, direct | −0.04 | −0.15 | −0.25 | **+0.10** |
| **Daily**, decomposed | −0.04 | −0.21 | −0.34 | +0.02 |
| **Weekly**, direct | −0.09 | +0.07 | **+0.38** | −0.02 |
| **Weekly**, decomposed | −0.09 | **+0.30** | +0.35 | −0.07 |

**What this shows (the whole point of the exercise):**

1. **Daily is noise-dominated.** Every learner is ≤0 except SARIMA (+0.10); the
   naive baseline is essentially unbeatable. Individual daily cash-flow is mostly
   irreducible timing noise.
2. **Weekly aggregation rescues the signal.** Best R² jumps **+0.10 → +0.38**.
   Summing 7 days cancels day-to-day timing noise (incoherently, ~×7) while the
   weekly *budget* structure adds coherently — a ~7× signal-to-noise gain. At
   weekly cadence the **strong models clearly beat the baseline** (which stays
   negative).
3. **Decomposition helps the linear model a lot** (weekly **+0.07 → +0.30**):
   removing the deterministic salary/rent/subscription structure lets the linear
   model focus on the learnable residual. (For trees the gain is neutral — they
   could already approximate that structure.)
4. **SHAP** on the weekly model ranks `fourier_sin_1` (monthly cycle) and
   `expected_flow_next` (the deterministic backbone) as top contributors —
   confirming the new architectural features carry the signal.

Headline track (**weekly, decomposed**): regularization **triggered** (OLS max
|coef| €450 → **Lasso** chosen); best strong model R² **+0.35** vs baseline
**−0.09**; P10–P90 coverage **0.58**, pinball **131**.

### On "is R² ≈ 0.3–0.4 good?" — the irreducible-variance ceiling
Discretionary human spending contains genuine randomness, so there is a Bayes
floor: no architecture drives single-account R² near 0.9. That is *why* the
deliverable is a **fan chart**, not a single line — the predictable backbone is
drawn as a line and the irreducible part as an honest uncertainty band. (The
band slightly under-covers here — 0.58 vs 0.80 — because the final 12 weeks were
unusually volatile, including a travel period; it covers ~0.80 of the
out-of-fold calibration residuals by construction.)

### Figures (`outputs/`)
- `fan_chart.png` — **headline**: deterministic P50 line + P10–P90 band + actuals + projection.
- `daily_vs_weekly.png` — R² by model across all four tracks.
- `balance_forecast.png` — weekly multi-model holdout comparison.
- `shap_summary.png`, `shap_bar.png`, `linear_coefficients.png`, `metrics.{json,csv}`.

### ⚠️ Data-size caveat
This is **one account**. ~565 daily / ~81 weekly points demonstrate the
methodology end-to-end, but the model learns *this client's* habits and will not
generalise cross-client. Aggregation fixes the **noise** problem; only
**panel data (many accounts)** fixes the **single-subject** problem.

---

## Repository layout
```
src/
  categorize.py     # spending-group labels (data, not a pipeline)
  etl.py            # parse 2 xlsx -> categorize -> SQLite mock DB (stdlib only)
  recurring.py      # deterministic backbone: recurring-event + habitual rates
  features.py       # daily & weekly features (+ Fourier, backbone signal)
  models.py         # baseline, linear (OLS->Lasso->Ridge), trees, SARIMA, quantiles
  evaluate.py       # temporal split, point + probabilistic metrics
  explain.py        # SHAP + standardized coefficients
  plots.py          # fan chart, track comparison, holdout comparison
  run_pipeline.py   # orchestrate the daily/weekly x direct/decomposed grid
data/raw/           # the two source statements (committed for reproducibility)
data/finance.db     # generated mock database
outputs/            # generated figures + metrics
```
