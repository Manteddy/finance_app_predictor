"""Validate the forecaster on the REAL Berka panel and report the performance
change vs the single-account (Estonian) baseline.

Two regimes on the weekly next-period net-flow target:

  (A) Per-account  - our existing decomposed approach applied to each of N real
      accounts -> the DISTRIBUTION of R²/coverage (does the method hold up?).
  (B) Global, leave-accounts-out - one pooled model, GroupKFold by account, so
      test accounts are unseen in training -> R² on UNSEEN accounts. This is the
      cross-client generalisation we could never measure with one account.

Run:  python -m src.cross_account
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from . import berka, conformal, evaluate, models, recurring
from .features import aggregate, build_features

OUT_DIR = "outputs"
N_ACCOUNTS = 250
TEST_WEEKS = 12
ESTONIAN_BASELINE_R2 = 0.49   # weekly net-flow R², from src.run_pipeline headline


def _per_account(rows):
    """Decomposed weekly forecast for one account. Returns metrics or None."""
    daily = berka.account_daily(rows)
    df = aggregate(daily, "W")
    if len(df) < TEST_WEEKS + 40:
        return None
    tx = berka.account_tx(rows)
    cutoff = df.index[-TEST_WEEKS]
    rules = recurring.detect_recurring(tx, cutoff)
    hab = recurring.habitual_rates(daily, cutoff, "W")
    det = recurring.build_deterministic(df.index, rules, hab, "W")

    X, y, frame = build_features(df, "W", "residual", det)
    if len(X) < TEST_WEEKS + 25:
        return None
    (Xtr, ytr, ftr), (Xte, yte, fte) = evaluate.temporal_split(X, y, frame, test_days=TEST_WEEKS)
    det_next = fte["det_next"].values
    actual = fte["total_next"].values
    if np.var(actual) < 1e-6:
        return None

    out = {}
    base = models.MovingAverageBaseline().fit(Xtr, ftr["total_next"])
    out["r2_baseline"] = r2_score(actual, base.predict(Xte))
    ridge = clone(models.build_linear_models()["linear_ridge"]).fit(Xtr, ytr)
    out["r2_ridge"] = r2_score(actual, det_next + ridge.predict(Xte))
    tree = clone(models.build_tree_models()["random_forest"]).fit(Xtr, ytr)
    out["r2_tree"] = r2_score(actual, det_next + tree.predict(Xte))

    spec = conformal.fit_normalized(clone(models.build_linear_models()["linear_ridge"]),
                                    Xtr, ytr, [0.8])
    lo, hi = conformal.interval(spec, ridge, 0.8, Xte)
    out["coverage"] = evaluate.interval_coverage(actual, det_next + lo, det_next + hi)
    return out


def _pooled_features(by_acct, accounts):
    """Build pooled (X, y, group) across accounts using the DIRECT total-flow
    target (clean for a shared cross-account model)."""
    Xs, ys, gs = [], [], []
    for aid in accounts:
        try:
            df = aggregate(berka.account_daily(by_acct[aid]), "W")
            if len(df) < TEST_WEEKS + 40:
                continue
            X, y, _ = build_features(df, "W", "total", None)
            if len(X) < 25:
                continue
            Xs.append(X); ys.append(y); gs.append(np.full(len(X), int(aid)))
        except Exception:
            continue
    X = pd.concat(Xs).fillna(0.0)
    y = pd.concat(ys).values
    g = np.concatenate(gs)
    return X.reset_index(drop=True), y, g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_acct = berka.load_transactions()
    accounts = berka.select_accounts(by_acct, n=N_ACCOUNTS)
    print(f"Berka: {len(by_acct)} accounts; evaluating {len(accounts)}")

    # ---- (A) per-account ----------------------------------------------------
    per = []
    fails = 0
    for aid in accounts:
        try:
            m = _per_account(by_acct[aid])
        except Exception:
            m = None
        if m is None:
            fails += 1
        else:
            per.append(m)
    P = pd.DataFrame(per)
    print(f"\n(A) PER-ACCOUNT  (n={len(P)} usable, {fails} skipped)")
    for col, lbl in [("r2_baseline", "baseline"), ("r2_ridge", "ridge"),
                     ("r2_tree", "random_forest")]:
        q = P[col].quantile([0.25, 0.5, 0.75]).values
        print(f"  R² {lbl:13s} median {q[1]:+.3f}  IQR [{q[0]:+.3f}, {q[2]:+.3f}]  "
              f"%>0: {100*(P[col]>0).mean():.0f}%")
    print(f"  P10–P90 coverage median {P['coverage'].median():.2f} (target ~0.80)")

    # ---- (B) global, leave-accounts-out ------------------------------------
    X, y, g = _pooled_features(by_acct, accounts)
    print(f"\n(B) GLOBAL leave-accounts-out  (pooled {len(X)} weekly rows, "
          f"{len(np.unique(g))} accounts, GroupKFold=5)")
    gkf = GroupKFold(n_splits=5)
    glob = {}
    for name, mk in [("ridge", lambda: clone(models.build_linear_models()["linear_ridge"])),
                     ("grad_boost", lambda: GradientBoostingRegressor(
                         n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, min_samples_leaf=20, random_state=42))]:
        yt, yp = [], []
        for tr, va in gkf.split(X, y, g):
            m = mk().fit(X.iloc[tr], y[tr])
            yt.append(y[va]); yp.append(m.predict(X.iloc[va]))
        glob[name] = float(r2_score(np.concatenate(yt), np.concatenate(yp)))
        print(f"  R² {name:11s} on UNSEEN accounts: {glob[name]:+.3f}")

    # ---- report + figure ----------------------------------------------------
    out = {
        "n_accounts_evaluated": len(P),
        "estonian_single_account_r2": ESTONIAN_BASELINE_R2,
        "per_account": {
            k: {"median": round(float(P[k].median()), 3),
                "iqr": [round(float(P[k].quantile(.25)), 3), round(float(P[k].quantile(.75)), 3)],
                "pct_positive": round(float((P[k] > 0).mean()) * 100, 1)}
            for k in ["r2_baseline", "r2_ridge", "r2_tree"]},
        "per_account_coverage_median": round(float(P["coverage"].median()), 3),
        "global_leave_accounts_out_r2": {k: round(v, 3) for k, v in glob.items()},
    }
    with open(os.path.join(OUT_DIR, "berka_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(P["r2_ridge"].clip(-1, 1), bins=30, color="tab:blue", alpha=0.8)
    ax1.axvline(P["r2_ridge"].median(), color="k", ls="--",
                label=f"median {P['r2_ridge'].median():+.2f}")
    ax1.axvline(0, color="0.5", lw=0.8)
    ax1.set_title(f"(A) Per-account weekly R² — ridge (n={len(P)} real accounts)")
    ax1.set_xlabel("R² (clipped to [-1,1])"); ax1.legend()

    labels = ["Estonian\n(1 account)", "Berka per-acct\n(median)",
              "Berka global LOO\n(ridge)", "Berka global LOO\n(GBoost)"]
    vals = [ESTONIAN_BASELINE_R2, P["r2_ridge"].median(), glob["ridge"], glob["grad_boost"]]
    ax2.bar(labels, vals, color=["tab:gray", "tab:blue", "tab:green", "tab:olive"])
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_ylabel("R² (weekly net-flow)")
    ax2.set_title("(B) Generalisation: single-account vs real panel")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.01, f"{v:+.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "berka_generalization.png"), dpi=120)
    plt.close(fig)

    print(f"\nBaseline (Estonian single account): R² {ESTONIAN_BASELINE_R2:+.2f}")
    print(f"Saved outputs/berka_metrics.json + berka_generalization.png")
    return out


if __name__ == "__main__":
    main()
