"""Load reviewed execution histories and audit their supported date ranges.

Source archives/parsed snapshots are distinct from certified continuous coverage.
Neither a PDF download nor an empty dividend list grants trading permissions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from qmt.factor.event_history import CoverageSpan, DividendEvent, DividendHistory, StatusEvent, StatusHistory
from qmt.factor.fee_history import FeeHistory


def _read(path):
    return json.loads(path.read_text())


def _objects(cls, records):
    supported = {f.name for f in fields(cls)}
    date_names = {"start_date", "end_date", "record_date", "ex_date", "cash_pay_date",
                  "reinvest_price_date", "confirmation_date", "redeem_available_date",
                  "effective_start", "effective_end"}
    objects = []
    for record in records:
        extra = set(record) - supported
        if extra:
            raise ValueError(f"unsupported {cls.__name__} fields: {sorted(extra)}")
        values = dict(record)
        for name in date_names & set(values):
            if values[name] is not None:
                values[name] = pd.Timestamp(values[name]).date()
        if values.get("available_at") is not None:
            stamp = pd.Timestamp(values["available_at"])
            # A date-only notice is not assumed to precede a 14:30 decision.
            if len(str(values["available_at"])) <= 10:
                stamp += pd.Timedelta(hours=23, minutes=59, seconds=59)
            values["available_at"] = stamp.to_pydatetime()
        objects.append(cls(**values))
    return objects


@dataclass
class HistoricalBundle:
    directory: Path
    fees: FeeHistory
    statuses: StatusHistory
    dividends: DividendHistory
    manifest: dict
    fingerprints: dict

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        names = ["standard_fees.json", "status_history.json", "dividend_history.json", "coverage_manifest.json"]
        missing = [n for n in names if not (directory / n).exists()]
        if missing:
            raise ValueError("missing historical inputs: " + ", ".join(missing))
        fee, status, dividend, manifest = [_read(directory / n) for n in names]
        if fee.get("channel") != "manager_direct_counter_standard" or manifest.get("channel") != fee["channel"]:
            raise ValueError("this research channel is ordinary manager-direct counter standard rates")
        return cls(directory, FeeHistory(fee),
                   StatusHistory(_objects(StatusEvent, status.get("events", [])),
                                 _objects(CoverageSpan, status.get("coverage", []))),
                   DividendHistory(_objects(DividendEvent, dividend.get("events", [])),
                                   _objects(CoverageSpan, dividend.get("coverage", []))),
                   manifest, {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in names})

    @property
    def codes(self):
        return tuple(sorted(set(self.manifest.get("selected_codes", []))))

    def validate(self, request):
        missing = []
        domain = self.manifest.get("supported_execution_domain")
        if domain:
            # A certificate for bounded orders cannot authorize an arbitrary
            # execution policy. Require the same enforceable engine guards.
            checks = {
                "initial_cash": math.isfinite(request.initial_cash) and 0 < request.initial_cash <= domain["max_initial_cash"],
                "daily_subscription_scope": (getattr(request, "covered_daily_buy_money_exclusive", None) is not None
                    and 0 < request.covered_daily_buy_money_exclusive <= domain["daily_subscription_money_exclusive"]),
                "remaining_units_policy": (getattr(request, "research_minimum_remaining_units", 0.0)
                    >= domain["minimum_remaining_units"]),
                "redemption_units_policy": (getattr(request, "research_minimum_redemption_units", 0.0)
                    >= domain["minimum_partial_redemption_units"]),
                "channel": domain["channel"] == self.fees.channel,
            }
            missing.extend({"component": "execution_domain", "constraint": name,
                            "reason": "requested execution policy exceeds the reviewed source domain"}
                           for name, passed in checks.items() if not passed)
        for code in request.fund_codes:
            for kind in ("fees", "status", "dividend"):
                eligible = [r for r in self.manifest.get("coverage", [])
                            if r.get("fund_code") == code and r.get("kind") == kind
                            and r.get("certified") is True]
                cursor = request.start_date
                for row in sorted(eligible, key=lambda r: r["start_date"]):
                    lo, hi = pd.Timestamp(row["start_date"]).date(), pd.Timestamp(row["end_date"]).date()
                    if hi < cursor:
                        continue
                    if lo > cursor:
                        break
                    if hi >= request.end_date:
                        cursor = None
                        break
                    cursor = (pd.Timestamp(hi) + pd.Timedelta(days=1)).date()
                if cursor is not None:
                    evidence = [r for r in self.manifest.get("coverage", [])
                                if r.get("fund_code") == code and r.get("kind") == kind]
                    missing.append({"fund_code": code, "component": kind,
                                    "first_unverified": str(cursor), "required_end": str(request.end_date),
                                    "reasons": [reason for r in evidence for reason in r.get("reasons", [])]
                                               or ["no reviewed continuous coverage"]})
        return {"complete": not missing, "missing": missing, "channel": self.fees.channel,
                "manifest_fingerprints": self.fingerprints,
                "supported_execution_domain": domain,
                "universe_selection_basis": self.manifest.get("universe_selection_basis"),
                "historical_fee_status_authentication": "reviewed input manifest; not inferred from current state"}
