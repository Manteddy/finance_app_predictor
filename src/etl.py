"""ETL: parse the two Swedbank statements into a mock SQLite database.

Uses only the Python standard library (``zipfile`` + ``sqlite3``) so the mock
database can be built without the heavier data-science dependencies. An .xlsx
file is just a zip of XML, so we read the worksheet XML directly.

Produces ``data/finance.db`` with:
  * ``transactions``   - one row per real transaction, with a spending_group
  * ``spending_groups``- reference table describing each group
  * ``daily_balance``  - daily aggregation used for feature engineering
"""

from __future__ import annotations

import os
import sqlite3
import zipfile
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

from .categorize import SPENDING_GROUPS, categorize

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_EXCEL_EPOCH = datetime(1899, 12, 30)  # Excel's day 0 (handles the 1900 bug)

# Statements in chronological order (oldest first).
STATEMENTS = [
    "data/raw/9f3f5bf7-statement_2.xlsx",  # 2024-12 .. 2025-12
    "data/raw/bdca300a-statement.xlsx",    # 2026-01 .. 2026-06
]
DB_PATH = "data/finance.db"


def _read_sheet(path: str) -> list[dict]:
    """Read a statement worksheet into a list of {column-letter: value} dicts."""
    with zipfile.ZipFile(path) as z:
        shared = ET.fromstring(z.read("xl/sharedStrings.xml"))
        strings = [
            "".join(t.text or "" for t in si.iter(_NS + "t")) for si in shared
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    def cell_value(c):
        v = c.find(_NS + "v")
        if v is None:
            return None
        return strings[int(v.text)] if c.get("t") == "s" else v.text

    rows = []
    for r in sheet.iter(_NS + "row"):
        cells = {}
        for c in r.findall(_NS + "c"):
            col = "".join(ch for ch in c.get("r") if ch.isalpha())
            cells[col] = cell_value(c)
        rows.append(cells)
    return rows


def _serial_to_date(serial: str) -> str:
    """Convert an Excel serial day number to an ISO date string."""
    return (_EXCEL_EPOCH + timedelta(days=int(float(serial)))).strftime("%Y-%m-%d")


def parse_statements(paths: list[str] = STATEMENTS) -> list[dict]:
    """Parse and concatenate the statements into clean transaction records.

    Returns rows sorted by date with a continuity check between the two
    statements (closing balance of one must equal the opening of the next).
    """
    all_txns: list[dict] = []
    prev_closing = None

    for path in paths:
        rows = _read_sheet(path)
        opening = closing = None

        for row in rows:
            a = (row.get("A") or "").strip()

            if a.startswith("EUR opening balance"):
                opening = float(row.get("E"))
                if prev_closing is not None:
                    assert abs(prev_closing - opening) < 0.01, (
                        f"Discontinuity: prev closing {prev_closing} != "
                        f"opening {opening} in {path}"
                    )
            if a.startswith("EUR closing balance"):
                closing = float(row.get("E"))

            # Transaction rows have a numeric index in column A and a turnover.
            try:
                float(a)
            except (TypeError, ValueError):
                continue
            if row.get("E") is None or row.get("B") is None:
                continue

            all_txns.append(
                {
                    "txn_date": _serial_to_date(row.get("B")),
                    "counterparty": (row.get("C") or "").strip(),
                    "details": (row.get("D") or "").strip(),
                    "amount": round(float(row.get("E")), 2),
                    "balance": float(row.get("F")) if row.get("F") else None,
                }
            )
        prev_closing = closing

    # Stable sort by date keeps the original within-day ordering (and thus the
    # running balance) intact.
    all_txns.sort(key=lambda t: t["txn_date"])
    for i, t in enumerate(all_txns):
        t["id"] = i + 1
        t["spending_group"] = categorize(t["counterparty"], t["details"], t["amount"])
    return all_txns


def _build_daily(txns: list[dict]) -> list[dict]:
    """Aggregate transactions to one row per calendar day (gap-filled)."""
    from collections import defaultdict

    groups = sorted(SPENDING_GROUPS.keys())
    by_day: dict[str, dict] = defaultdict(
        lambda: {
            "net_flow": 0.0,
            "txn_count": 0,
            "debit_sum": 0.0,
            "credit_sum": 0.0,
            "end_balance": None,
            **{f"grp_{g}": 0.0 for g in groups},
        }
    )
    for t in txns:
        d = by_day[t["txn_date"]]
        amt = t["amount"]
        d["net_flow"] += amt
        d["txn_count"] += 1
        if amt < 0:
            d["debit_sum"] += -amt
        else:
            d["credit_sum"] += amt
        d[f"grp_{t['spending_group']}"] += amt
        if t["balance"] is not None:
            d["end_balance"] = t["balance"]  # last balance seen that day

    # Fill missing calendar days so the series is regularly spaced. On a day
    # with no transactions net_flow=0 and the balance carries forward.
    start = datetime.strptime(min(by_day), "%Y-%m-%d")
    end = datetime.strptime(max(by_day), "%Y-%m-%d")
    out = []
    last_balance = None
    day = start
    while day <= end:
        key = day.strftime("%Y-%m-%d")
        if key in by_day:
            rec = by_day[key]
            if rec["end_balance"] is None:
                rec["end_balance"] = last_balance
            last_balance = rec["end_balance"]
        else:
            rec = {
                "net_flow": 0.0,
                "txn_count": 0,
                "debit_sum": 0.0,
                "credit_sum": 0.0,
                "end_balance": last_balance,
                **{f"grp_{g}": 0.0 for g in groups},
            }
        rec["date"] = key
        out.append(rec)
        day += timedelta(days=1)
    return out


def build_database(db_path: str = DB_PATH) -> dict:
    """Run the full ETL and write the mock SQLite database. Returns a summary."""
    txns = parse_statements()
    daily = _build_daily(txns)
    groups = sorted(SPENDING_GROUPS.keys())

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """CREATE TABLE spending_groups (
               name TEXT PRIMARY KEY, description TEXT)"""
    )
    cur.executemany(
        "INSERT INTO spending_groups VALUES (?, ?)",
        list(SPENDING_GROUPS.items()),
    )

    cur.execute(
        """CREATE TABLE transactions (
               id INTEGER PRIMARY KEY,
               txn_date TEXT NOT NULL,
               counterparty TEXT,
               details TEXT,
               amount REAL NOT NULL,
               balance REAL,
               spending_group TEXT NOT NULL,
               FOREIGN KEY (spending_group) REFERENCES spending_groups(name))"""
    )
    cur.executemany(
        """INSERT INTO transactions
               (id, txn_date, counterparty, details, amount, balance, spending_group)
               VALUES (:id, :txn_date, :counterparty, :details, :amount,
                       :balance, :spending_group)""",
        txns,
    )

    grp_cols = ", ".join(f"grp_{g} REAL" for g in groups)
    cur.execute(
        f"""CREATE TABLE daily_balance (
                date TEXT PRIMARY KEY,
                net_flow REAL, end_balance REAL, txn_count INTEGER,
                debit_sum REAL, credit_sum REAL, {grp_cols})"""
    )
    cols = ["date", "net_flow", "end_balance", "txn_count", "debit_sum",
            "credit_sum"] + [f"grp_{g}" for g in groups]
    placeholders = ", ".join("?" for _ in cols)
    cur.executemany(
        f"INSERT INTO daily_balance ({', '.join(cols)}) VALUES ({placeholders})",
        [tuple(rec[c] for c in cols) for rec in daily],
    )

    con.commit()

    # Summary for logging / verification.
    cur.execute("SELECT spending_group, COUNT(*), ROUND(SUM(amount),2) "
                "FROM transactions GROUP BY spending_group ORDER BY COUNT(*) DESC")
    group_breakdown = cur.fetchall()
    con.close()

    return {
        "n_transactions": len(txns),
        "n_days": len(daily),
        "date_range": (daily[0]["date"], daily[-1]["date"]),
        "final_balance": daily[-1]["end_balance"],
        "group_breakdown": group_breakdown,
        "n_other": sum(1 for t in txns if t["spending_group"] == "other"),
    }


if __name__ == "__main__":
    summary = build_database()
    print(f"Built {DB_PATH}")
    print(f"  transactions : {summary['n_transactions']}")
    print(f"  daily rows   : {summary['n_days']}")
    print(f"  date range   : {summary['date_range'][0]} .. {summary['date_range'][1]}")
    print(f"  final balance: EUR {summary['final_balance']}")
    print(f"  uncategorised ('other'): {summary['n_other']}")
    print("  spending groups (count, sum EUR):")
    for name, n, total in summary["group_breakdown"]:
        print(f"    {name:14s} {n:5d}  {total:>12.2f}")
