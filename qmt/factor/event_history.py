"""Point-in-time dividend and dealing-status histories with explicit coverage.

An empty event list proves nothing. Coverage spans must come from a caller's
audited archive/master and carry provenance. Status OPEN additionally requires
an explicit credible opening/contract record; absence of suspension notices is
never interpreted as permission to trade. No network or cache reads occur here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Sequence

import math
import pandas as pd

__all__ = ["CoverageSpan", "DividendEvent", "DividendHistory", "StatusEvent",
           "TradingStatus", "StatusHistory", "publication_time", "decision_cutoff"]


def publication_time(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")


def decision_cutoff(value: date | datetime) -> pd.Timestamp:
    if isinstance(value, datetime):
        return publication_time(value)
    return publication_time(datetime.combine(pd.Timestamp(value).date(), time(14, 30)))


@dataclass(frozen=True)
class CoverageSpan:
    fund_code: str
    start_date: date
    end_date: date
    source: str
    kind: str = "dividend"
    available_at: datetime | None = None

    def __post_init__(self):
        if self.start_date > self.end_date or not self.source:
            raise ValueError("coverage requires ordered dates and source provenance")


def _assert_coverage(coverage, codes, start, end, kind):
    for code in codes:
        spans = sorted((s for s in coverage if s.fund_code == code and s.kind == kind),
                       key=lambda s: s.start_date)
        cursor = pd.Timestamp(start).date()
        for span in spans:
            if span.end_date < cursor:
                continue
            if span.start_date > cursor:
                break
            if span.end_date >= end:
                cursor = None
                break
            cursor = (pd.Timestamp(span.end_date) + pd.Timedelta(days=1)).date()
        if cursor is not None:
            raise ValueError(f"incomplete {kind} coverage for {code}: {start}..{end}")


@dataclass(frozen=True)
class DividendEvent:
    event_id: str
    fund_code: str
    record_date: date
    ex_date: date
    per_unit_cash: float
    available_at: datetime | None
    source: str
    cash_pay_date: date | None = None
    reinvest_price_date: date | None = None
    confirmation_date: date | None = None
    redeem_available_date: date | None = None
    reinvest_price: float | None = None
    record_date_buy_entitled: bool = False
    record_date_sell_entitled: bool = True
    version: int = 1

    def __post_init__(self):
        if not self.event_id or not self.source:
            raise ValueError("dividend event_id and source are required")
        if not isinstance(self.record_date_buy_entitled, bool) or not isinstance(self.record_date_sell_entitled, bool):
            raise ValueError("dividend entitlement application rules must be explicit booleans")
        if self.record_date > self.ex_date:
            raise ValueError("dividend record_date must not follow ex_date")
        if not math.isfinite(self.per_unit_cash) or self.per_unit_cash < 0:
            raise ValueError("per_unit_cash must be finite and nonnegative")
        if self.cash_pay_date is not None and self.cash_pay_date < self.ex_date:
            raise ValueError("cash_pay_date must not precede ex_date")
        if self.reinvest_price_date is not None and self.reinvest_price_date < self.ex_date:
            raise ValueError("reinvest_price_date must not precede ex_date")
        if self.reinvest_price is not None and (not math.isfinite(self.reinvest_price) or self.reinvest_price <= 0):
            raise ValueError("reinvest_price must be finite and positive")
        if (self.confirmation_date is not None and self.reinvest_price_date is not None
                and self.confirmation_date < self.reinvest_price_date):
            raise ValueError("dividend confirmation must not precede reinvestment pricing")
        if (self.redeem_available_date is not None and self.confirmation_date is not None
                and self.redeem_available_date < self.confirmation_date):
            raise ValueError("redeem availability must not precede confirmation")


def _latest_known(events, cutoff, publications=None):
    latest = {}
    for event in events:
        published = publications.get(id(event)) if publications is not None else publication_time(event.available_at)
        if published is None or published > cutoff:
            continue
        key = (event.fund_code, event.event_id)
        rank = (published, event.version)
        previous = latest.get(key)
        if previous is None or rank > previous[0]:
            latest[key] = (rank, event)
    return [item[1] for _, item in sorted(latest.items())]


class DividendHistory:
    def __init__(self, events: Sequence[DividendEvent], coverage: Sequence[CoverageSpan] = ()):
        self.events = tuple(events)
        self.coverage = tuple(coverage)
        self._publications = {id(e): publication_time(e.available_at) for e in self.events}
        self._cache = {}

    def as_of(self, cutoff: date | datetime) -> list[DividendEvent]:
        at = decision_cutoff(cutoff)
        if at not in self._cache:
            self._cache[at] = tuple(_latest_known(self.events, at, self._publications))
        return list(self._cache[at])

    def assert_coverage(self, codes, start: date, end: date) -> None:
        _assert_coverage(self.coverage, codes, start, end, "dividend")


@dataclass(frozen=True)
class StatusEvent:
    event_id: str
    fund_code: str
    side: str  # buy | sell | both
    effective_start: date
    effective_end: date | None
    allowed: bool
    available_at: datetime | None
    source: str
    purchase_limit_money: float | None = None
    min_hold_days: int | None = None
    version: int = 1
    credible_open: bool = False
    clears_purchase_limit: bool = False
    clears_min_hold_days: bool = False
    intraday_start: datetime | None = None
    intraday_end: datetime | None = None

    def __post_init__(self):
        if not self.event_id or not self.source or self.side not in {"buy", "sell", "both"}:
            raise ValueError("status event requires id, provenance and buy/sell/both side")
        if not isinstance(self.allowed, bool) or not isinstance(self.credible_open, bool):
            raise ValueError("allowed and credible_open must be explicit booleans")
        if self.effective_end is not None and self.effective_end <= self.effective_start:
            raise ValueError("status effective_end is exclusive and must follow start")
        if self.purchase_limit_money is not None and (not math.isfinite(self.purchase_limit_money) or self.purchase_limit_money < 0):
            raise ValueError("purchase_limit_money must be finite and nonnegative")
        if self.min_hold_days is not None and self.min_hold_days < 0:
            raise ValueError("min_hold_days must be nonnegative")
        if self.clears_purchase_limit and self.purchase_limit_money is not None:
            raise ValueError("a status cannot both set and clear a purchase limit")
        if self.clears_min_hold_days and self.min_hold_days is not None:
            raise ValueError("a status cannot both set and clear minimum holding days")
        if (self.intraday_start is not None and self.intraday_end is not None
                and publication_time(self.intraday_end) <= publication_time(self.intraday_start)):
            raise ValueError("intraday_end must follow intraday_start")


@dataclass(frozen=True)
class TradingStatus:
    allowed: bool
    purchase_limit_money: float | None = None
    min_hold_days: int | None = None
    source: str = ""
    available_at: datetime | None = None

    def __post_init__(self):
        if not isinstance(self.allowed, bool):
            raise ValueError("TradingStatus.allowed must be a boolean")
        if self.purchase_limit_money is not None and (not math.isfinite(self.purchase_limit_money) or self.purchase_limit_money < 0):
            raise ValueError("TradingStatus purchase limit must be finite and nonnegative")
        if self.min_hold_days is not None and self.min_hold_days < 0:
            raise ValueError("TradingStatus minimum hold must be nonnegative")


class StatusHistory:
    def __init__(self, events: Sequence[StatusEvent], coverage: Sequence[CoverageSpan] = ()):
        self.events = tuple(events)
        self.coverage = tuple(coverage)
        self._publications = {id(e): publication_time(e.available_at) for e in self.events}
        self._by_code = {}
        for event in self.events:
            self._by_code.setdefault(event.fund_code, []).append(event)
        self._cache = {}

    def assert_coverage(self, codes, start: date, end: date) -> None:
        _assert_coverage(self.coverage, codes, start, end, "status")

    def status(self, code: str, at: date | datetime, side: str) -> TradingStatus | None:
        key = (code, decision_cutoff(at), side)
        if key not in self._cache:
            self._cache[key] = self._status(code, at, side)
        return self._cache[key]

    def _status(self, code: str, at: date | datetime, side: str) -> TradingStatus | None:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        cutoff = decision_cutoff(at)
        d = cutoff.date()
        if not any(s.fund_code == code and s.kind == "status" and s.start_date <= d <= s.end_date
                   for s in self.coverage):
            return None
        known = [e for e in _latest_known(self._by_code.get(code, ()), cutoff, self._publications)
                 if e.fund_code == code and e.side in {side, "both"}
                 and e.effective_start <= d and (e.effective_end is None or d < e.effective_end)
                 and (e.intraday_start is None or publication_time(e.intraday_start) <= cutoff)
                 and (e.intraday_end is None or cutoff < publication_time(e.intraday_end))]
        if not known:
            return None
        open_proven = any(e.allowed and e.credible_open for e in known)
        def priority(event):
            start = publication_time(event.intraday_start or datetime.combine(event.effective_start, time.min))
            return start, self._publications[id(event)], event.version
        best = max(priority(e) for e in known)
        peers = [e for e in known if priority(e) == best]
        # Equal-date notices do not arbitrarily lose caps depending on event IDs;
        # conflicting permissions resolve to the restrictive published state.
        chosen = next((e for e in peers if not e.allowed), peers[0])
        if chosen.allowed and not open_proven:
            return None
        cap_events = [e for e in known if e.purchase_limit_money is not None or e.clears_purchase_limit]
        hold_events = [e for e in known if e.min_hold_days is not None or e.clears_min_hold_days]
        cap = None
        hold = None
        if cap_events:
            cap_best = max(priority(e) for e in cap_events)
            limits = [e.purchase_limit_money for e in cap_events if priority(e) == cap_best and e.purchase_limit_money is not None]
            cap = min(limits) if limits else None
        if hold_events:
            hold_best = max(priority(e) for e in hold_events)
            holds = [e.min_hold_days for e in hold_events if priority(e) == hold_best and e.min_hold_days is not None]
            hold = max(holds) if holds else None
        return TradingStatus(chosen.allowed, cap, hold,
                             chosen.source, chosen.available_at)

    def __call__(self, code: str, at: date | datetime, side: str) -> TradingStatus | None:
        return self.status(code, at, side)
