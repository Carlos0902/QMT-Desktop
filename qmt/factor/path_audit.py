"""Reconstruct saved cash/units/receivables independently of strategy scoring.

This audit deliberately supports the no-dividend annual paths in the current
reviewed universe. Unsupported wealth events fail instead of disappearing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path

from qmt.factor.cost_benchmark import load_execution_data
from qmt.factor.historical_bundle import HistoricalBundle


def audit_path(payload, data, history, initial_cash=100000.):
    metrics, events = payload["metrics"], payload["events"]
    cash, pending, cost = initial_cash, 0., 0.
    units = defaultdict(float)
    confirms = {e["lot_id"]: date.fromisoformat(e["date"]) for e in events
                if e["event_type"] == "buy_confirm"}
    lots = {}
    fills = 0
    for e in events:
        kind, code, d = e["event_type"], e["fund_code"], date.fromisoformat(e["date"])
        if kind.startswith("dividend"):
            raise ValueError("dividend audit requires entitlement ledger support")
        cost += e["cost_money"]
        if kind == "buy_fill":
            fills += 1
            placed = date.fromisoformat(e["order_date"] or e["decision_date"])
            quote = history.fees(code, placed)
            assert quote is not None and history.statuses(code, placed, "buy").allowed
            assert abs(e["units"] * e["nav"] - e["gross_money"]) < 1e-6
            assert abs(e["gross_money"] * quote.purchase_fee_rate - e["cost_money"]) < 1e-6
            assert abs(-e["cash_money"] - e["gross_money"] - e["cost_money"]) < 1e-6
            assert -e["cash_money"] < quote.supported_purchase_amount_max_exclusive
            cash += e["cash_money"]
            units[code] += e["units"]
            lots[e["lot_id"]] = {"fund_code": code, "units": e["units"]}
        elif kind == "sell_fill":
            fills += 1
            placed = date.fromisoformat(e["decision_date"])
            quote = history.fees(code, placed)
            assert quote is not None and history.statuses(code, placed, "sell").allowed
            assert abs(e["units"] * e["nav"] - e["gross_money"]) < 1e-6
            assert abs(e["cost_money"] - e["gross_money"] * quote.redemption_rate(e["hold_days"])) < 1e-6
            confirmation = confirms[e["lot_id"]]
            assert confirmation <= placed
            remaining = [i for i, lot in lots.items() if lot["fund_code"] == code and lot["units"] > 1e-8]
            assert e["lot_id"] == remaining[0], "FIFO mismatch"
            lots[e["lot_id"]]["units"] -= e["units"]
            units[code] -= e["units"]
            assert units[code] >= -1e-7
            pending += e["gross_money"] - e["cost_money"]
        elif kind == "redemption_confirm":
            pending -= e["cost_money"]
            assert e["hold_days"] == (d - confirms[e["lot_id"]]).days
        elif kind == "cash_settle":
            cash += e["cash_money"]
            pending -= e["cash_money"]
        assert cash >= -1e-6 and pending >= -1e-6
    end = date.fromisoformat(metrics["last_valuation_date"])
    wealth = cash + pending + sum(n * float(data.unit_nav.loc[end, c]) for c, n in units.items())
    wealth_error, cost_error = abs(wealth - metrics["final_equity"]), abs(cost - metrics["cost_money"])
    assert wealth_error < 1e-6 and cost_error < 1e-6
    return {"fills": fills, "wealth_error": wealth_error, "cost_error": cost_error,
            "cash": cash, "unpaid_redemptions": pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    history = HistoricalBundle.load(args.history)
    data = load_execution_data(args.cache, codes=history.codes, domestic_only=True,
                               conservative_nav_publication=True)
    methods = json.loads((args.study / "study_protocol.json").read_text())["methods"]
    rows = [{"file": str(p), **audit_path(json.loads(p.read_text()), data, history)}
            for method in methods for p in sorted((args.study / method).rglob("*_execution.json"))]
    assert rows, "no execution paths to audit"
    result = {"passed": True, "n_paths": len(rows), "n_fills": sum(r["fills"] for r in rows),
              "max_terminal_wealth_error": max(r["wealth_error"] for r in rows),
              "max_cost_error": max(r["cost_error"] for r in rows), "paths": rows,
              "scope": "fee arithmetic, known fees/status, FIFO, confirmation holding age, nonnegative cash/receivables, terminal wealth and cumulative cost",
              "source_fingerprints": history.fingerprints,
              "sealed_cache_values_read": False}
    (args.study / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "paths"}))


if __name__ == "__main__":
    main()
