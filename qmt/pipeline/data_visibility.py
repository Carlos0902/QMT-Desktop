"""data_visibility — the first point-in-time gate (§4 sec:mod-visibility, §1.8-2).

Responsibility (doc): confirm which data is observable now, which is not yet disclosed,
and which is too old. It stamps every :class:`~qmt.types.DataField` with a
{visible / hidden} state and a {fresh / stale / expired} freshness, and disables any
field that lacks an ``available_at`` (invariant 1). This is the first gate of
point-in-time correctness — the whole system's foundation.

Doc signature (04-modules.tex, lst:vis)::

    def resolve_visibility(fields: list[DataField],
                           as_of: datetime) -> VisibilitySet:
        \"\"\"对每个字段判断 available_at <= as_of 是否成立（visible / hidden），
        并计算新鲜度（fresh / stale / expired）；禁用任何无 available_at 的字段。\"\"\"

The doc leaves the freshness day-boundaries unspecified; they come from
:class:`~qmt.config.FreshnessConfig` (decision D02), keyed by update cadence because a
daily NAV and a quarterly holdings report age at wildly different rates.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union

from qmt.config import CONFIG, FreshnessConfig
from qmt.pit import TradingCalendar
from qmt.types import (
    DataField,
    Freshness,
    UpdateFreq,
    Visibility,
    VisibilitySet,
    VisibilitySnapshot,
)

__all__ = ["resolve_visibility", "classify_freshness"]


def _as_datetime(as_of: Union[datetime, date]) -> datetime:
    if isinstance(as_of, datetime):
        return as_of
    return datetime.combine(as_of, datetime.min.time())


def _cutoffs_for(update_freq: UpdateFreq, cfg: FreshnessConfig) -> tuple[int, int]:
    """Return ``(fresh_max_days, stale_max_days)`` for a field's update cadence (D02).

    Daily-cadence data (NAV, index bars) ages in a few trading days; quarterly holdings
    age over months (the 15/60/90-day disclosure lags). Everything else uses the
    documented default fallback.
    """
    if update_freq is UpdateFreq.DAILY:
        return cfg.nav_fresh_max_days, cfg.nav_stale_max_days
    if update_freq is UpdateFreq.QUARTERLY:
        return cfg.holdings_fresh_max_days, cfg.holdings_stale_max_days
    return cfg.default_fresh_max_days, cfg.default_stale_max_days


def classify_freshness(
    age_days: int,
    update_freq: UpdateFreq,
    cfg: Optional[FreshnessConfig] = None,
) -> Freshness:
    """Map an observable-data age to {fresh / stale / expired}.

    Inclusive on the lower state (per ``FreshnessConfig`` docstring):
    ``fresh`` iff ``age <= fresh_max``; ``stale`` iff ``age <= stale_max``; else
    ``expired``.
    """
    fcfg = cfg or CONFIG.freshness
    fresh_max, stale_max = _cutoffs_for(update_freq, fcfg)
    if age_days <= fresh_max:
        return Freshness.FRESH
    if age_days <= stale_max:
        return Freshness.STALE
    return Freshness.EXPIRED


def _age_days(
    available_at: datetime,
    as_of_dt: datetime,
    update_freq: UpdateFreq,
    calendar: TradingCalendar,
) -> int:
    """Age of an observable field in the unit its freshness cutoffs are defined in.

    Daily-cadence data (NAV, index bars) ages in **trading** days — its fresh<=3 /
    stale<=10 cutoffs (D02, config) count sessions, so a NAV 3 trading days old over a
    weekend/holiday is still FRESH rather than reading STALE on the calendar-day gap.
    Everything else (quarterly holdings with the 45/100-day cutoffs, and the default
    fallback) ages in **calendar** days.
    """
    if update_freq is UpdateFreq.DAILY:
        return calendar.count_trading_days(available_at.date(), as_of_dt.date())
    return max((as_of_dt - available_at).days, 0)


def resolve_visibility(
    fields: list[DataField],
    as_of: Union[datetime, date],
    cfg: Optional[FreshnessConfig] = None,
    calendar: Optional[TradingCalendar] = None,
) -> VisibilitySet:
    """Classify each :class:`DataField` visible/hidden + fresh/stale/expired at ``as_of``.

    * ``visible`` iff ``available_at <= as_of`` (invariant 1); otherwise ``hidden``.
    * A visible field's ``freshness`` follows its age via :func:`classify_freshness`;
      daily-cadence fields are aged in **trading** days (via ``calendar``, default the
      business-day :class:`~qmt.pit.TradingCalendar`), quarterly/other in calendar days.
    * A hidden field is *not yet observable*; its ``age_days`` is 0 and its ``freshness``
      fails **closed** to ``EXPIRED`` (invariant 6) so any consumer that ignores
      ``visible`` and only inspects ``freshness`` still never treats it as usable.

    ``DataField.available_at`` is mandatory at the type layer (pydantic), which is where
    the doc's "禁用任何无 available_at 的字段" clause is enforced — a field cannot reach
    this function without it.
    """
    fcfg = cfg or CONFIG.freshness
    cal = calendar or TradingCalendar()
    as_of_dt = _as_datetime(as_of)

    snapshots: list[VisibilitySnapshot] = []
    for f in fields:
        is_visible = f.available_at <= as_of_dt
        if is_visible:
            age_days = _age_days(f.available_at, as_of_dt, f.update_freq, cal)
            visibility = Visibility.VISIBLE
            freshness = classify_freshness(age_days, f.update_freq, fcfg)
        else:
            # Not yet observable → unusable → most-cautious defaults (fail-closed).
            age_days = 0
            visibility = Visibility.HIDDEN
            freshness = Freshness.EXPIRED

        snapshots.append(
            VisibilitySnapshot(
                field_name=f.name,
                visible=visibility,
                freshness=freshness,
                available_at=f.available_at,
                age_days=age_days,
            )
        )

    return VisibilitySet(as_of=as_of_dt, snapshots=snapshots)
