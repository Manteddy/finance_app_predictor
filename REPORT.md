# Data strategy: narrowing the fan & generalising across clients

This report answers two linked questions: **how to make the uncertainty fan
tighter and more useful**, and **which learning/testing data to prioritise so the
model performs well universally across clients.** They are linked because, on the
current data, most of the fan's width is *irreducible* — it can only be reduced
with better data, not a cleverer estimator.

## What we learned from the single-account model (the drawbacks)

1. **Weekly cash-flow is genuinely volatile and heteroscedastic.** Consumption is
   stable (weekly std ≈ €285), but income lumpiness and a 44× swing in weekly
   volatility (calm €30 → travel €1300 rolling std) mean an honest 80% band is
   ~±€750 in volatile weeks. We made this *adaptive* (tight in calm weeks, wide in
   volatile ones) and *correctly calibrated* (the previous band silently
   under-covered at 0.67), but the core width is real.
2. **Own-account flows are compensatory, not noise.** Routing savings/Easy-Saver
   transfers out *increased* variance — when spending spikes, the client tops up
   from savings, which stabilises the current-account net flow. With only the
   current account visible, these stabilising moves look like noise.
3. **Single account ⇒ no generalisation.** The model learns *this* person's
   salary date, rent, and habits. It cannot transfer to another client, and we
   cannot even *measure* cross-client performance.
4. **18 months ⇒ no annual seasonality**, and fuzzy merchant-string matching is
   brittle (e.g. the landlord was split across two categories).

## Data to prioritise (ranked)

### 1. Panel data — many clients (the decisive lever) — NOW VALIDATED
Hundreds–thousands of accounts with the same schema. This is the single most
important acquisition. It enables:
- a **global model with per-client effects** (hierarchical / mixed-effects, or a
  shared model with client embeddings) that borrows strength across clients and
  shrinks the irreducible-looking variance of any one account;
- **cold-start** for new clients from population priors;
- the only valid measure of universality — **leave-clients-out** evaluation.

**We tested this empirically** on the real **Berka panel** (~4,500 accounts; see
`src/cross_account.py`, README "Real-data validation"). Result: a single global
model forecasts **unseen accounts at R² +0.37** (leave-accounts-out), and the
per-account method beats the baseline on **~70% of 250 real accounts** (median
R² +0.36) with conformal coverage **0.83**. So the panel-data lever is confirmed:
the method generalises across clients — the remaining need is *scale + breadth*
(more clients, modern data, all-accounts-per-client) rather than a new method.

The public datasets we found (and why most don't suffice):
- **Berka / PKDD'99** — real, ~4,500-account panel **with running balances**;
  the only public real panel that fits. Research-use licence; 1990s Czech data.
- **MoneyData/MoneyVis** — real, has balance, but ~single account (too small).
- **MBD (Sber)** — real, 1.5M clients, but **no balance**, scaled, corporate.
- **Synthetic** (IBM TabFormer, Sparkov, PaySim) — either no balance, or
  artificial balance dynamics; useful only to *augment*, not to ground truth.
- **Yodlee/Envestnet, Smart Data Foundry** — real consumer panels but
  **commercial / safe-haven only** (partnership required, not downloadable).

### 2. All accounts per client (current + savings + credit card)
Directly fixes the width problem in finding #2: internal transfers and savings
top-ups **net out** when you see both sides, instead of inflating the forecast
band. Model the **household liquidity position**, not one account in isolation.

### 3. Merchant / MCC enrichment
Standardised merchant IDs and **MCC category codes** instead of fuzzy strings →
robust, country-agnostic categorisation and recurring-payment detection (no more
landlord-split-across-categories). Also enables transfer/own-account flagging.

### 4. Longer history per client (≥ 2–3 years)
To learn **annual seasonality** — holidays, summer travel, tuition, tax,
insurance renewals — that 18 months cannot reveal. Improves both the point
forecast and the calibration of multi-month projections.

### 5. Known-future / scheduled data
Standing orders, direct debits, confirmed **salary date & amount**, loan
amortisation schedules, and (where available) calendar/travel signals. Every item
moved from "stochastic" to "scheduled/deterministic" **directly narrows the fan**.

### 6. Client attributes & balanced segment sampling
Income band, employment type, household size, tenure, region — features that
*generalise* across clients. Sample **representatively across segments** (income,
behaviour, country) so the model is fair and accurate for everyone, with enough
volume per segment to detect failures.

## Evaluation data & protocol (equally important)

- **Leave-clients-out CV** (hold out whole clients) to measure universality —
  *plus* a time-based split within clients to measure forecasting. Report both.
- **Stratify metrics by segment** (income, age, country, tenure) to surface where
  the model underperforms, not just a global average.
- **Probabilistic scoring**: keep pinball loss + interval coverage **per
  horizon**, and check calibration *per client* (not just marginally).
- Hold out enough **future** per client to test multi-week/▒month projections.

## Bottom line

Better estimators (adaptive conformal, decomposition, recurrence detection) have
been pushed about as far as one account allows. The remaining fan width is
dominated by **irreducible single-account volatility**. The step-change comes from
**data**: many clients (for generalisation + variance shrinkage), all of each
client's accounts (so compensatory flows net out), MCC enrichment, longer history,
and scheduled-payment feeds — evaluated with a **leave-clients-out** protocol.
