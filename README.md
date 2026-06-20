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
    flows — the **Wise salary** (income, month-end), subscriptions, utilities,
    dormitory/rent — on their schedule **plus** stable habitual category rates
    (e.g. median weekly groceries). Fit on the **training portion only** (no
    leakage), extendable into the future. Two detection strategies are compared:
    **category-gated** (`detect_recurring`, scans income/subscriptions/utilities)
    and **recurrence-first / agnostic** (`detect_recurring_agnostic`, groups *all*
    merchants then finds periodicity — see the A/B below).
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

### Fan chart (`src/plots.py`, `src/conformal.py`)
The uncertainty band is an **adaptive normalized conformal** interval
(`src/conformal.py`): calibrated on **prequential** (online) residuals — which
fixed a bug where expanding-CV's tiny early folds inflated the margin *and* let it
under-cover (0.67) — and scaled by a **bounded local-volatility multiplier** so it
is **tight in calm weeks and wide only in volatile ones**. It is rendered as a
**gradient** of nested intervals (50→95%, darker near P50). The forward projection
is a **Monte-Carlo** simulation (recursive, model-driven), so it carries
week-level texture plus the monthly salary; it **starts exactly where the actual
line ends** and widens with horizon.

**Two width-reduction levers were tried (see [REPORT.md](REPORT.md)):**
- **Adaptive intervals** ✅ — calm-week 80% band ≈ €1.1k vs volatile ≈ €2.0k, and
  honest coverage (~0.9 vs the old 0.67).
- **Routing own-account flows out** ❌ — it *backfires*: savings/Easy-Saver
  transfers are **compensatory** (the client tops up from savings when spending
  spikes), so removing them *raises* variance. Reverted; documented.

The remaining width is largely **irreducible** for a single account over a
volatile period — the decisive lever is **data** (more clients, all accounts per
client, longer history), detailed in **[REPORT.md](REPORT.md)**.

---

## Real-data validation across 4,500 accounts (Berka panel)

The model's biggest drawback was being trained on **one** account — so we
validated it on the **real PKDD'99 "Berka" Czech-bank panel** (~4,500 accounts,
~1.06M transactions, with a running balance per transaction). `src/berka.py`
maps it into our schema; `src/cross_account.py` runs two regimes on 250 real
accounts (`python -m src.cross_account`). (Of the three real datasets researched,
only Berka is both a balance-bearing *panel* and retrievable; MoneyData is tiny
single-account and MBD has no balance — see [REPORT.md](REPORT.md).)

| regime | R² (weekly net-flow) |
|---|---|
| Estonian single account (this repo's headline) | **+0.49** |
| Berka **per-account** (median of 250), ridge | **+0.36** (IQR −0.04…+0.56; **72% of accounts > 0**) |
| Berka **global, leave-accounts-out**, ridge | **+0.30** on *unseen* accounts |
| Berka **global, leave-accounts-out**, GradientBoosting | **+0.37** on *unseen* accounts |

**The performance change / key result:** the approach **generalises**. A single
global model trained on some accounts forecasts **completely unseen accounts at
R² +0.37** (GroupKFold by account — no account leakage), and the per-account
method beats the naive baseline (median **−0.15**, 0% positive) on **~70% of real
accounts**. The adaptive-conformal band stays calibrated out-of-the-box across
accounts (**median P10–P90 coverage 0.83**). The Estonian account (+0.49) sits on
the *higher* end of the real distribution, so **+0.36 median is the more honest
expectation** for a typical account. See `outputs/berka_generalization.png`.

Caveats: Berka is 1990s Czech data (dnoeth mirror; dates shifted +20 yr, amounts
intact), a different category taxonomy, research-use licence — so the takeaway is
the **generalisation result**, not a like-for-like accuracy number.

## The dominant pattern: a monthly salary

The single strongest, most regular flow is a **salary paid via Wise on the last
working day of every month** (~€2,086; 19 occurrences across the data). It is
detected as a recurring `income` event and placed at month-end in the
deterministic backbone — so the **forward projection now shows the expected
monthly sawtooth** (a spike at each month-end, then a draw-down through the month
as rent/subscriptions/spending land). Capturing it was the largest single
accuracy win (see below).

## Rent: is it captured correctly?

Originally **no**. The landlord `MARIKA LAOS` was **split across two categories**
by a keyword quirk (details containing "utilities" → `utilities`, rent-only →
`transfer`), so the category-gated detector saw only 3 of 9 payments and used
**−€310/mo**. The true picture: rent **€700/mo for two roommates** (the user's
share ≈ €350) **+ €85–310 utilities**, paid on shifting mid-month dates, with
roommate **Bohdan reimbursing ≈ +€250/mo**. The recurrence-first detector
(below) groups the landlord correctly and recovers **−€700/mo**. But note: the
amount genuinely **varies €85–1070** month to month, so even the corrected rule
is only an approximation — the variable part stays in the stochastic residual.

## Architecture A/B: category-first vs recurrence-first

The recurrence-first (agnostic) detector is **more universal** — grouping all
merchants first surfaces the full landlord payment, the **gym** (€19/mo ×17), the
**dormitory**, **SportsDirect**, etc., regardless of category. **But it forecasts
slightly worse**: weekly best-strong R² **+0.43 (agnostic) vs +0.49
(category-gated)**. Why: variable-amount / variable-date items like rent (€85–
1070, paid the 12th–20th) are poorly represented by a single fixed deterministic
value, so they inject phase/amplitude noise into the residual. The disciplined
category-gated backbone (only clean, fixed recurrences — salary, subscriptions,
dormitory) yields cleaner residuals. **Lesson: more complete detection ≠ better
forecasting; only *stably* deterministic patterns belong in the backbone.** The
agnostic scan is still valuable as a discovery tool (it correctly found the rent).

## Results — R² on next-period total flow (holdout)

| track | baseline | linear | tree | sarima |
|---|---|---|---|---|
| **Daily**, direct | −0.04 | −0.16 | −0.03 | **+0.10** |
| **Daily**, decomposed | −0.04 | −0.24 | −0.01 | −3.03 |
| **Daily**, agnostic | −0.04 | −0.23 | 0.00 | −3.30 |
| **Weekly**, direct | −0.09 | +0.49 | **+0.52** | −0.02 |
| **Weekly**, decomposed | −0.09 | **+0.49** | +0.32 | −0.08 |
| **Weekly**, agnostic | −0.09 | +0.43 | +0.20 | −0.38 |

**What this shows (the whole point of the exercise):**

1. **Daily is noise-dominated.** Every learner is ≈0 or negative except SARIMA
   (+0.10); the naive baseline is essentially unbeatable. Individual daily
   cash-flow is mostly irreducible timing noise. (SARIMA on the *decomposed*
   daily residual diverges, −3.03 — that combination is simply unsuitable.)
2. **Weekly aggregation rescues the signal.** Best R² jumps **+0.10 → +0.52**.
   Summing 7 days cancels day-to-day timing noise (incoherently, ~×7) while the
   weekly *budget* structure adds coherently — a ~7× signal-to-noise gain. At
   weekly cadence the **strong models clearly beat the baseline** (which stays
   negative).
3. **Modelling the recurring salary + decomposition** lifts the linear model
   from weekly **+0.07** (before the salary was captured) to **+0.49**. Removing
   the deterministic salary/rent/subscription structure lets the linear model
   focus on the learnable residual. (For trees the decomposition is roughly
   neutral — they can approximate that structure directly, so the weekly *direct*
   tree is marginally best at +0.52.)
4. **SHAP** on the weekly model ranks `fourier_sin_1` (monthly cycle) and
   `expected_flow_next` (the deterministic backbone, incl. the salary) as top
   contributors — confirming the new architectural features carry the signal.

Headline track (**weekly, decomposed**): best strong model **linear** (Lasso;
regularization **triggered**, OLS max |coef| ≈ €450) at R² **+0.49** vs baseline
**−0.09**; P10–P90 coverage **0.67**, pinball **116**.

### On "is R² ≈ 0.5 good?" — the irreducible-variance ceiling
Discretionary human spending contains genuine randomness, so there is a Bayes
floor: no architecture drives single-account R² near 0.9. That is *why* the
deliverable is a **fan chart**, not a single line — the predictable backbone
(salary, rent, subscriptions, habits) is drawn as a line and the irreducible
part as an honest uncertainty band. (The band under-covers slightly — 0.67 vs
0.80 — because the final 12 weeks were unusually volatile, including a travel
period; it covers ~0.80 of the out-of-fold calibration residuals by
construction.)

### Figures (`outputs/`)
- `fan_chart.png` — **headline**: P50 line + **gradient** uncertainty fan + actuals,
  with a connected, recursive forward projection.
- `daily_vs_weekly.png` — R² by model across all six tracks (y clipped at −0.7).
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
  models.py         # baseline, linear (OLS->Lasso->Ridge), trees, SARIMA
  conformal.py      # adaptive normalized-conformal prediction intervals
  evaluate.py       # temporal split, point + probabilistic metrics
  explain.py        # SHAP + standardized coefficients
  plots.py          # fan chart, track comparison, holdout comparison
  berka.py          # loader for the real Berka multi-account panel
  cross_account.py  # cross-account generalisation eval (per-account + leave-accounts-out)
  run_pipeline.py   # orchestrate the daily/weekly x direct/decomposed grid
REPORT.md           # data strategy: narrowing the fan + cross-client generalisation
data/raw/           # the two source statements (committed for reproducibility)
data/finance.db     # generated mock database
outputs/            # generated figures + metrics
```
