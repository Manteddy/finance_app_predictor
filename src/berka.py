"""Loader for the real Berka / PKDD'99 Czech bank panel (~4,500 accounts).

Maps Berka's transaction table into the same per-account daily schema the rest of
the pipeline uses, so we can measure how the model generalises across many REAL
accounts (it was previously trained on a single account).

Source: dnoeth/1999_Czech_financial_dataset_Teradata (GitHub), a Teradata-ready
copy of the PKDD'99 dataset (dates shifted +20yr, descriptions abbreviated;
amounts/balances intact). Research use. fin_trans.tsv columns (no header):
    0 trans_id, 1 account_id, 2 date, 3 amount, 4 balance(running),
    5 type(C/D/P), 6 operation, 7 k_symbol, 8 bank, 9 partner_account
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

ZIP_PATH = "data/raw/berka.zip"
ZIP_URL = ("https://raw.githubusercontent.com/dnoeth/"
           "1999_Czech_financial_dataset_Teradata/master/financial_db_Teradata.zip")

# Berka k_symbol (abbreviated) -> our spending groups.
_KSYM = {"IC": "income", "PE": "income", "HH": "utilities", "IN": "utilities",
         "ST": "utilities", "LO": "transfer", "IO": "other"}
_GROUPS = ["income", "utilities", "transfer", "cash", "savings", "other"]


def ensure_download(path: str = ZIP_PATH) -> str:
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(ZIP_URL, path)
    return path


def _group(typ: str, op: str, ksym: str) -> str:
    k = ksym.strip()
    if k in _KSYM:
        return _KSYM[k]
    if typ == "C":
        return "income"          # credit, no symbol (cash-in / collection)
    if op in ("WIC", "CCW"):
        return "cash"
    if op == "ROB":
        return "transfer"
    return "other"


def load_transactions(path: str = ZIP_PATH) -> dict:
    """Return {account_id: list of dict rows} for all accounts."""
    ensure_download(path)
    by_acct: dict[str, list] = defaultdict(list)
    with zipfile.ZipFile(path) as z, z.open("fin_trans.tsv") as f:
        for r in csv.reader(io.TextIOWrapper(f, "utf-8", "replace"), delimiter="\t"):
            typ = r[5]
            signed = float(r[3])          # the amount column is already signed
            by_acct[r[1]].append({
                "trans_id": int(r[0]), "txn_date": r[2], "amount": signed,
                "balance": float(r[4]), "is_credit": signed > 0,
                "spending_group": _group(typ, r[6], r[7]),
                "counterparty": (r[9].strip() or r[7].strip() or r[6].strip()),
                "details": f"{r[6]}|{r[7].strip()}",
            })
    return by_acct


def account_tx(rows: list) -> pd.DataFrame:
    """Per-account transaction frame for recurring detection."""
    df = pd.DataFrame(rows)
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    return df.sort_values(["txn_date", "trans_id"]).reset_index(drop=True)


def account_daily(rows: list) -> pd.DataFrame:
    """Per-account daily frame shaped like the Estonian `daily_balance`."""
    df = account_tx(rows)
    df["date"] = df["txn_date"].dt.normalize()

    grp_cols = {g: df["amount"].where(df["spending_group"] == g, 0.0) for g in _GROUPS}
    agg = pd.DataFrame({
        "txn_count": 1,
        "debit_sum": (-df["amount"]).clip(lower=0),
        "credit_sum": df["amount"].clip(lower=0),
        "grp_income": df["amount"].where(df["is_credit"], 0.0),  # credits -> income feat
        **{f"grp_{g}": grp_cols[g] for g in _GROUPS if g != "income"},
    })
    agg["date"] = df["date"].values
    daily = agg.groupby("date").sum()
    # End-of-day running balance = last balance that day (rows are time-ordered);
    # this column is authoritative, so net_flow is its day-to-day difference.
    daily["end_balance"] = df.groupby("date")["balance"].last()

    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(idx)
    daily["end_balance"] = daily["end_balance"].ffill()
    fill0 = [c for c in daily.columns if c != "end_balance"]
    daily[fill0] = daily[fill0].fillna(0.0)

    opening = float(rows_first_balance(rows))
    daily["net_flow"] = daily["end_balance"].diff()
    daily.iloc[0, daily.columns.get_loc("net_flow")] = daily["end_balance"].iloc[0] - opening
    daily.index.name = "date"
    return daily


def rows_first_balance(rows: list) -> float:
    """Opening balance = balance before the first transaction."""
    first = sorted(rows, key=lambda r: (r["txn_date"], r["trans_id"]))[0]
    return first["balance"] - first["amount"]


def select_accounts(by_acct: dict, min_weeks: int = 80, min_txns: int = 120,
                    n: int | None = 300, seed: int = 0) -> list[str]:
    """Accounts with enough history for the weekly temporal split."""
    ok = []
    for aid, rows in by_acct.items():
        if len(rows) < min_txns:
            continue
        ds = [r["txn_date"] for r in rows]
        span = (pd.to_datetime(max(ds)) - pd.to_datetime(min(ds))).days / 7
        if span >= min_weeks:
            ok.append(aid)
    ok.sort(key=int)
    if n and len(ok) > n:
        rng = np.random.default_rng(seed)
        ok = sorted(rng.choice(ok, n, replace=False), key=int)
    return ok


if __name__ == "__main__":
    by = load_transactions()
    n_tx = sum(len(v) for v in by.values())
    print(f"Berka: {len(by)} accounts, {n_tx} transactions")
    sel = select_accounts(by, n=None)
    print(f"accounts with >=80wk & >=120txn: {len(sel)}")
    # Reconciliation check on a few accounts.
    import random
    random.seed(1)
    for aid in random.sample(sel, 3):
        d = account_daily(by[aid])
        opening = float(by[aid][0]["balance"] - by[aid][0]["amount"])
        recon = opening + d["net_flow"].cumsum()
        err = float((recon - d["end_balance"]).abs().max())
        print(f"  acct {aid}: {len(d)} days, final bal {d['end_balance'].iloc[-1]:.0f}, "
              f"recon max-err {err:.2f}")
