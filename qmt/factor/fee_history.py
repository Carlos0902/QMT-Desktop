"""Date/channel fee history supplied from fund documents or investor records.

This loader validates data and publication timing, not the authenticity of the
documents. Missing fund/date coverage returns None and blocks execution rather
than silently applying zero/default fees.
"""
from __future__ import annotations

from datetime import date
from dataclasses import dataclass
import json
import math
from pathlib import Path

import pandas as pd

from qmt.types import FeeSchedule


@dataclass(frozen=True)
class ContractFeeQuote:
    """A verified fee band with its documented counter dealing constraints.

    Upper band boundaries represent missing rate coverage, not a purchase cap.
    Execution must stop if an actual order leaves the supported amount band.
    """
    schedule: FeeSchedule
    supported_purchase_amount_max_exclusive: float | None = None
    minimum_initial_purchase_money: float = 0.0
    minimum_additional_purchase_money: float = 0.0
    minimum_redemption_units: float = 0.0

    @property
    def fund_code(self):
        return self.schedule.fund_code

    @property
    def purchase_fee_rate(self):
        return self.schedule.purchase_fee_rate

    def redemption_rate(self, hold_days):
        return self.schedule.redemption_rate(hold_days)


def _local_timestamp(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("Asia/Shanghai").tz_localize(None) if stamp.tzinfo else stamp


class FeeHistory:
    def __init__(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("fee history must be a JSON object")
        self.channel = str(payload.get("channel", ""))
        if not self.channel.strip():
            raise ValueError("fee history requires an explicit sales channel")
        self.records = []
        for record in payload.get("records", []):
            source = str(record.get("source", "")).strip()
            if not source:
                raise ValueError("every fee record requires document/statement source")
            start = pd.Timestamp(record["effective_from"]).date()
            end = pd.Timestamp(record["effective_to"]).date() if record.get("effective_to") else date.max
            if end < start:
                raise ValueError("fee effective_to precedes effective_from")
            known = _local_timestamp(record["available_at"])
            if pd.isna(known):
                raise ValueError("fee available_at must be present")
            if len(str(record["available_at"])) <= 10:
                known += pd.Timedelta(hours=23, minutes=59, seconds=59)
            raw_schedule = record["schedule"]
            if "purchase_fee_rate" not in raw_schedule or not raw_schedule.get("redemption_tiers"):
                raise ValueError("fee records must explicitly declare purchase and full redemption rates")
            schedule = FeeSchedule.model_validate(raw_schedule)
            if not math.isfinite(schedule.purchase_fee_rate):
                raise ValueError("purchase fee must be finite")
            next_start = 0
            for tier in sorted(schedule.redemption_tiers, key=lambda t: t.min_hold_days):
                if (tier.min_hold_days != next_start or not math.isfinite(tier.rate)
                        or not 0 <= tier.rate <= 1 or (tier.max_hold_days is not None
                        and tier.max_hold_days <= tier.min_hold_days)):
                    raise ValueError("redemption tiers must be finite, contiguous and nonoverlapping from day zero")
                next_start = tier.max_hold_days
            if next_start is not None:
                raise ValueError("redemption tiers must include an open-ended final tier")
            if record.get("contract_rules") or record.get("supported_purchase_amount_max_exclusive") is not None:
                rules = record.get("contract_rules", {})
                valid_names = {"minimum_initial_purchase_money", "minimum_additional_purchase_money",
                               "minimum_redemption_units"}
                if set(rules) - valid_names:
                    raise ValueError("unsupported contract rule; cannot silently ignore execution constraints")
                bound = record.get("supported_purchase_amount_max_exclusive")
                if bound is not None and (not math.isfinite(bound) or bound <= 0):
                    raise ValueError("supported purchase amount bound must be positive and finite")
                if any(not math.isfinite(v) or v < 0 for v in rules.values()):
                    raise ValueError("minimum dealing rules must be finite and nonnegative")
                schedule = ContractFeeQuote(schedule, bound, **rules)
            self.records.append((start, end, known, source, schedule))
        if not self.records:
            raise ValueError("fee history requires at least one record")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text()))

    def __call__(self, code, decision_date):
        d = pd.Timestamp(decision_date).date()
        cutoff = pd.Timestamp(d) + pd.Timedelta(hours=14, minutes=30)
        eligible = [r for r in self.records if r[4].fund_code == code
                    and r[0] <= d <= r[1] and r[2] <= cutoff]
        if not eligible:
            return None
        return max(eligible, key=lambda r: (r[0], r[2]))[4]

    def provenance(self):
        return {"channel": self.channel, "n_records": len(self.records),
                "sources": sorted({r[3] for r in self.records}),
                "certification": "caller supplied; source authenticity not independently verified",
                "missing_coverage": "blocks execution, no default fee substitution"}
