"""Timeline backtest engine (slice: data-priming + backtest, part 2).

A walk-forward, point-in-time-correct backtester that honours the design doc's ten-point
backtest protocol (07-testing.tex §sec:test-pit). It is deliberately a *self-built
Timeline engine* (D17), not a third-party framework, so every one of the ten rules is
visible in the code:

1. **Only then-visible data.** Every signal is computed from
   :meth:`PointInTimeStore.get_asof(decision_moment)` — never from a NAV whose
   ``available_at`` is after the decision. In particular a quarter-end holdings *date* is
   never treated as the date it was *known* (that is enforced upstream in the store).
2. **Time-ordered.** The engine steps forward through the trading grid; nothing is
   shuffled and train/eval order is chronological.
3. **Survivorship.** The universe at a past date is reconstructed from
   ``found_date`` / ``delist_date`` (a fund contributes only while it was alive), NOT from
   the current ``status`` snapshot — delisted / merged funds are retained.
4. **No peeking.** Signals read only ``available_at <= decision_moment``; the leak-detector
   test (:mod:`tests.test_backtest`) asserts the engine never surfaces a future row.
5. **Execution + confirmation delay.** An order decided on day *T* fills at the NAV of the
   next executable point from :func:`qmt.pit.build_decision_timeline` (``horizon_start`` =
   next trading day, T+1 OTC open-end confirmation); the position only earns returns from
   that fill forward.
6. **Costs.** Investor subscription/redemption charges come from FeeSchedule;
   annual management/custody/service fees already embedded in NAV are not charged
   again. Defaults are labeled research assumptions, not verified fund/channel
   schedules. This legacy weight-return engine is an approximation; the new
   factor.fund_execution engine supplies cash, units, FIFO and settlement paths.
7. **Short-holding fees.** Redemption fees are looked up by the *actual* holding period
   (:meth:`FeeSchedule.redemption_rate`), so a short round-trip pays the punitive tier —
   never assumed zero.
8. **By market regime.** Per-decision returns are labelled by a (point-in-time) regime and
   aggregated in :attr:`BacktestResult.net_return_by_regime`. The label set spans the doc's
   five regimes (rule 8: 上涨/下跌/震荡/高波动/事件冲击) — trend (up/down/range) with a
   high-vol suffix, plus a dominant ``"shock"`` (事件冲击) regime raised by a fed
   ``market_events`` SHOCK flag or a large single-day drawdown proxy (D-BT-8).
9. **By fund type.** Results also slice by ``FundProfile.fund_type``
   (:attr:`BacktestResult.net_return_by_fund_type`).
10. **Net of cost vs. baselines.** The strategy is always compared, net of cost, against the
    doc's **five** naive baselines (§sec:test-baselines: 长期持有 ``buy_and_hold`` / 固定定投
    ``fixed_sip`` / 简单均线 ``ma`` / 简单趋势 ``trend`` / 简单波动控制 ``vol_control``) via
    :attr:`BacktestResult.baselines` and per-baseline deltas in
    :attr:`BacktestResult.vs_baseline`.

Modelling choice (documented, DECISIONS D-BT-1): the engine is a **position-return**
backtester. At each trading day *T* a strategy emits a target invested fraction
``w in [0, 1]`` from PIT data; the realised fund return over each interval is earned by the
position that was *already executable* at the interval's start (a two-step lag: decide at
*T* using data through *T-1*, fill at *T+1*, first earn the *T+1 → T+2* move). This keeps
the equity curve built only from then-visible marks while measuring the actual outcome, and
avoids a fragile pending-order state machine. Fees are charged on turnover as it takes
effect. This is the transparent baseline; a full lot-level order book is a candidate hook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from qmt.config import CONFIG, get_settings
from qmt.data.sources import TABLE_FUND_BASIC, TABLE_FUND_NAV
from qmt.fees import default_fee_schedule
from qmt.pit import (
    AVAILABLE_AT,
    PointInTimeStore,
    TradingCalendar,
    build_decision_timeline,
)
from qmt.types import FeeSchedule, Horizon, MarketEvent

logger = logging.getLogger(__name__)

__all__ = [
    "BacktestResult",
    "BaselineOutcome",
    "DecisionRecord",
    "StrategyContext",
    "Strategy",
    "run_backtest",
    "STRATEGIES",
    "strategy_buy_and_hold",
    "strategy_ma_cross",
    "strategy_trend",
    "strategy_vol_control",
]

# Number of trading days a return is lagged behind the decision that drives it (see the
# module docstring / D-BT-1): decide at T (data <= T-1) -> fill at T+1 -> first earned move
# is T+1 -> T+2. So the interval ending at grid index k is driven by the target that became
# executable at index k-1.
_ANNUALIZATION_DAYS = 252.0

# A single-day return at/below this on the fund's own visible path is treated as an
# event/shock (事件冲击) regime proxy (protocol rule 8; doc-silent, DECISIONS D-BT-8). A
# −5% single session is a genuine shock for a broad A-share equity fund.
_SHOCK_SINGLE_DAY_RETURN = -0.05


# =============================================================================
# Strategy interface
# =============================================================================
@dataclass(frozen=True)
class StrategyContext:
    """Point-in-time context handed to a strategy at one decision date.

    ``prices`` is the fund's ``adj_nav`` series **visible at** the decision moment
    (``available_at <= decision_moment``), oldest→newest — the only market data a strategy
    may use. A strategy returns a target invested fraction in ``[0, 1]`` (0 = all cash,
    1 = fully invested); the engine clips out-of-range values.
    """

    fund_code: str
    as_of: date
    decision_moment: datetime
    prices: np.ndarray


Strategy = Callable[[StrategyContext], float]


def strategy_buy_and_hold(ctx: StrategyContext) -> float:
    """Naive baseline: always fully invested (enter at the first executable point, hold)."""
    return 1.0


def strategy_ma_cross(ctx: StrategyContext) -> float:
    """Baseline strategy — a simple fast/slow moving-average rule (doc's «简单均线规则»).

    Fully invested when the fast MA is above the slow MA by more than the neutral band,
    otherwise in cash. Reuses ``CONFIG.market_state`` (MA20/60, ``direction_band_pct``).
    Fail-closed: too little visible history → in cash (never a blind long).
    """
    cfg = CONFIG.market_state
    p = ctx.prices
    if p.size < cfg.ma_slow_days:
        return 0.0
    ma_fast = float(np.mean(p[-cfg.ma_fast_days:]))
    ma_slow = float(np.mean(p[-cfg.ma_slow_days:]))
    if ma_slow <= 0.0:
        return 0.0
    return 1.0 if (ma_fast - ma_slow) / ma_slow > cfg.direction_band_pct else 0.0


def strategy_trend(ctx: StrategyContext) -> float:
    """Naive baseline: simple momentum — invested iff the last ``lookback`` return is positive."""
    cfg = CONFIG.market_state
    lookback = cfg.ma_fast_days
    p = ctx.prices
    if p.size < lookback + 1:
        return 0.0
    return 1.0 if p[-1] > p[-lookback - 1] else 0.0


def strategy_vol_control(ctx: StrategyContext) -> float:
    """Naive baseline: simple volatility control — scale exposure down when recent vol is high.

    Target fraction = clip(target_vol / realised_vol, 0, 1) on the visible return series
    (annualised realised vol over ``realized_vol_window_days``). Fail-closed: thin history
    → in cash.
    """
    cfg = CONFIG.market_state
    p = ctx.prices
    win = cfg.realized_vol_window_days
    if p.size < win + 1:
        return 0.0
    rets = np.diff(p[-(win + 1):]) / p[-(win + 1):-1]
    realized = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    if realized <= 0.0:
        return 1.0
    target_daily_vol = 0.01  # ~16% annualised target (doc-silent baseline constant, D-BT-2)
    return float(np.clip(target_daily_vol / realized, 0.0, 1.0))


STRATEGIES: dict[str, Strategy] = {
    "baseline": strategy_ma_cross,
    "ma_cross": strategy_ma_cross,
    "buy_and_hold": strategy_buy_and_hold,
    "trend": strategy_trend,
    "vol_control": strategy_vol_control,
}

# The doc's five naive baselines (§sec:test-baselines). ``fixed_sip`` is money-weighted and has no
# weight function. A caller may request a SUBSET (``run_backtest(baselines=...)``) — a parameter
# search re-evaluates one strategy hundreds of times and does not need the baselines it never
# reports, and most of them are invariant to the swept parameters anyway (D-BT-16).
_WEIGHT_BASELINES: dict[str, Strategy] = {
    "buy_and_hold": strategy_buy_and_hold,
    "ma": strategy_ma_cross,
    "trend": strategy_trend,
    "vol_control": strategy_vol_control,
}
ALL_BASELINES: tuple[str, ...] = ("buy_and_hold", "fixed_sip", "ma", "trend", "vol_control")

# regime labeller: (as_of, visible prices) -> label
RegimeLabeler = Callable[[date, np.ndarray], str]


def _default_regime_labeler(as_of: date, prices: np.ndarray, win: Optional[int] = None) -> str:
    """Point-in-time regime label from the fund's own visible trailing trend + vol.

    Uses only ``prices`` (already ``available_at <= decision_moment``). Trailing
    ``ma_fast_days`` return classifies up / down / range; a high recent realised vol adds a
    ``_highvol`` suffix. A large single-day drawdown in the trailing window raises the
    dominant ``"shock"`` (事件冲击) regime — the price-only half of the doc's fifth regime
    (protocol rule 8; the fed-flag half is added by :func:`_make_default_regime_labeler`).
    Coarse by design — enough to *slice* results (rules 8/9), not a market-state model.
    """
    cfg = CONFIG.market_state
    # ``win`` may be PINNED by the caller. The regime partition must not depend on a parameter the
    # caller is sweeping, or two runs of the same window get different regime buckets and become
    # incomparable slice-by-slice (D-BT-17).
    win = cfg.ma_fast_days if win is None else int(win)
    if prices.size < win + 1:
        return "warmup"
    rets = np.diff(prices[-(win + 1):]) / prices[-(win + 1):-1]
    # event/shock proxy dominates: a large single-day drawdown in the visible window.
    if rets.size and float(np.min(rets)) <= _SHOCK_SINGLE_DAY_RETURN:
        return "shock"
    trail = prices[-1] / prices[-win - 1] - 1.0
    if trail > 0.02:
        direction = "up"
    elif trail < -0.02:
        direction = "down"
    else:
        direction = "range"
    vol = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    return f"{direction}_highvol" if vol > 0.02 else direction


def _fed_shock_active(store: PointInTimeStore, as_of: date) -> bool:
    """True iff a SHOCK flag in the ``market_events`` PIT table is observable+active at ``as_of``.

    Mirrors :func:`qmt.pipeline.market_state._assess_event` (SHOCK dominates POLICY, a flag
    stays active for ``event_active_days`` after its ``available_at``) but reads only for the
    shock case — the fed-flag half of the event/shock regime (protocol rule 8). Read
    point-in-time at the 14:30 decision moment, so a flag never colours a decision made
    before it was observable.
    """
    from qmt.pipeline.market_state import TABLE_MARKET_EVENTS  # local: avoid import cost/cycle

    moment = _decision_moment(as_of)
    df = store.get_asof(TABLE_MARKET_EVENTS, moment)
    if df is None or df.empty or "event" not in df.columns:
        return False
    cutoff = moment - timedelta(days=CONFIG.market_state.event_active_days)
    available = pd.to_datetime(df[AVAILABLE_AT])
    active = df[available >= cutoff]
    return not active.empty and MarketEvent.SHOCK.value in set(active["event"].astype(str))


def make_frozen_regime_labeler(
    store: PointInTimeStore, end: date, ma_fast_days: int
) -> RegimeLabeler:
    """A regime labeller whose trailing window is **pinned**, so it does not move with CONFIG.

    Comparing two runs slice-by-slice (protocol rule 8 / the "stably beats baseline" regime gate)
    is only meaningful if both are attributed under the **same** regime partition. The default
    labeller reads ``CONFIG.market_state.ma_fast_days`` at call time — which a parameter search
    sweeps — so a candidate and the baseline would bucket the same dates differently. Pin it
    (D-BT-17).
    """
    return _make_default_regime_labeler(store, _end_moment(end), ma_fast_days=ma_fast_days)


def _make_default_regime_labeler(
    store: PointInTimeStore,
    end_moment: Optional[datetime] = None,
    *,
    ma_fast_days: Optional[int] = None,
) -> RegimeLabeler:
    """The default store-aware regime labeller: fed SHOCK flag OR the price-only proxy.

    A fed ``market_events`` SHOCK flag active at the decision moment dominates (regime
    ``"shock"``); otherwise it falls back to :func:`_default_regime_labeler` (trend/vol +
    single-day-drawdown proxy). Keeps the injectable :data:`RegimeLabeler` signature so a
    caller can still supply a pure-price labeller. The SHOCK flags are read **once** into a
    :class:`_ShockIndex` (D-BT-15) instead of a ``get_asof`` per decision day; when no
    ``end_moment`` is given the per-day read path is used (back-compat for direct callers).
    """
    index = _ShockIndex(store, end_moment) if end_moment is not None else None

    def labeler(as_of: date, prices: np.ndarray) -> str:
        shock = (
            index.active_at(_decision_moment(as_of))
            if index is not None
            else _fed_shock_active(store, as_of)
        )
        if shock:
            return "shock"
        return _default_regime_labeler(as_of, prices, win=ma_fast_days)

    return labeler


# =============================================================================
# Result containers
# =============================================================================
@dataclass(frozen=True)
class DecisionRecord:
    """One decision the engine made (per fund, per decision date) — the per-decision log."""

    as_of: date
    fund_code: str
    fund_type: Optional[str]
    signal: float               # target invested fraction decided at as_of (pre-lag, clipped)
    executable_from: Optional[date]  # next executable point (build_decision_timeline)
    regime: str
    alive: bool


@dataclass(frozen=True)
class BaselineOutcome:
    """A naive baseline's net-of-cost outcome, comparable to the strategy."""

    name: str
    net_return: float
    max_drawdown: float
    equity_curve: pd.DataFrame  # columns: date, equity


@dataclass(frozen=True)
class BacktestResult:
    """The full backtest outcome (net of cost) with baselines, slices and a decision log."""

    codes: list[str]
    start: date
    end: date
    strategy_name: str
    equity_curve: pd.DataFrame        # strategy, net of cost — columns: date, equity
    net_return: float
    max_drawdown: float
    total_cost: float
    n_decisions: int
    baselines: dict[str, BaselineOutcome]
    vs_baseline: dict[str, float]     # strategy net_return - baseline net_return
    decision_log: list[DecisionRecord]
    net_return_by_regime: dict[str, float]
    net_return_by_fund_type: dict[str, float]
    # Name of the baseline that IS this strategy (its ``vs_baseline`` delta is identically 0 —
    # a self-comparison, not evidence). ``None`` when the strategy is distinct from all of them.
    self_baseline: Optional[str] = None

    def summary(self) -> dict[str, object]:
        """A compact, printable dict of the headline metrics and baseline comparison."""
        return {
            "codes": list(self.codes),
            "window": f"{self.start.isoformat()}..{self.end.isoformat()}",
            "strategy": self.strategy_name,
            "net_return": round(self.net_return, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "total_cost": round(self.total_cost, 6),
            "n_decisions": self.n_decisions,
            "vs_baseline": {k: round(v, 6) for k, v in self.vs_baseline.items()},
            "net_return_by_regime": {k: round(v, 6) for k, v in self.net_return_by_regime.items()},
            "net_return_by_fund_type": {
                k: round(v, 6) for k, v in self.net_return_by_fund_type.items()
            },
            # The baseline that IS this strategy — its vs_baseline delta is identically 0 and is
            # NOT evidence of anything (D-BT-16). ``None`` when the strategy is distinct.
            "self_baseline": self.self_baseline,
            "gaps": [],
        }


# =============================================================================
# fund metadata (survivorship) reconstruction
# =============================================================================
@dataclass(frozen=True)
class _FundMeta:
    fund_code: str
    fund_type: Optional[str]
    found_date: Optional[date]
    delist_date: Optional[date]
    m_fee: Optional[float] = None  # disclosed annual management rate (for default fees)
    c_fee: Optional[float] = None  # disclosed annual custodian rate (for default fees)

    def alive_on(self, d: date) -> bool:
        """Survivorship gate: alive iff established on/before ``d`` and not yet delisted.

        A ``None`` ``found_date`` is treated as always-established (fail-open only on the
        universe boundary — the fund's own NAV availability still bounds it point-in-time).
        """
        if self.found_date is not None and d < self.found_date:
            return False
        if self.delist_date is not None and d >= self.delist_date:
            return False
        return True


def _as_date(value: object) -> Optional[date]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.date()


def _reconstruct_meta(store: PointInTimeStore, code: str, as_of: date) -> _FundMeta:
    """Reconstruct a fund's survivorship metadata from the PIT ``fund_basic`` at ``as_of``.

    Reads ``fund_basic`` via ``get_asof`` (so only a profile observable by ``as_of`` is
    used), takes the most-recently-observable row, and lifts ``fund_type`` /
    ``found_date`` / ``delist_date``. Survivorship (protocol rule 3) is driven by these
    dates, never the current ``status`` snapshot.
    """
    df = store.get_asof(TABLE_FUND_BASIC, _decision_moment(as_of), {"fund_code": code})
    if df is None or df.empty:
        return _FundMeta(code, None, None, None)
    df = df.sort_values(AVAILABLE_AT)
    row = df.iloc[-1].to_dict()
    return _FundMeta(
        fund_code=code,
        fund_type=(str(row["fund_type"]) if row.get("fund_type") not in (None, "") else None),
        found_date=_as_date(row.get("found_date")),
        delist_date=_as_date(row.get("delist_date")),
        # m_fee/c_fee are not persisted in the PIT fund_basic table today (they live only on
        # the domain FundMeta), so these are usually None → documented default fees; read
        # opportunistically so a future schema that carries them refines fees automatically.
        m_fee=_opt_float(row.get("m_fee")),
        c_fee=_opt_float(row.get("c_fee")),
    )


def _opt_float(value: object) -> Optional[float]:
    """Coerce a store cell to a non-negative float, or ``None`` (missing / NaN / invalid)."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f if f >= 0.0 else None


# =============================================================================
# point-in-time data access
# =============================================================================
def _decision_moment(as_of: date) -> datetime:
    """T-day decision instant (14:30) — reads at 14:30 hide the same-day 15:00 close."""
    t = CONFIG.timing
    return datetime.combine(as_of, time(hour=t.decision_hour, minute=t.decision_minute))


def _end_moment(end: date) -> datetime:
    """Outcome-measurement cutoff: everything disclosed by end-of-day on the last window day.

    Reading the realised NAV path at ``end`` 23:59 includes every NAV whose ``available_at``
    is on/before ``end`` — and, crucially, EXCLUDES anything disclosed after ``end`` (no
    look-ahead past the backtest window).
    """
    return datetime.combine(end, time(hour=23, minute=59, second=59))


def _outcome_path(
    store: PointInTimeStore, code: str, end_moment: datetime
) -> tuple[list[date], np.ndarray]:
    """The fund's realised ``adj_nav`` path (by ``nav_date``) observable by ``end_moment``.

    Restatements collapse to the latest-disclosed value per ``nav_date``; ``adj_nav`` is
    preferred, falling back to ``unit_nav``. Non-positive / NaN prices are dropped. This is
    the *outcome* series the equity curve is marked on; it never includes a NAV disclosed
    after the window (``available_at <= end_moment``).
    """
    df = store.get_asof(TABLE_FUND_NAV, end_moment, {"fund_code": code})
    if df is None or df.empty or "nav_date" not in df.columns:
        return [], np.empty(0, dtype=float)
    df = df.sort_values(["nav_date", AVAILABLE_AT]).drop_duplicates(subset="nav_date", keep="last")
    col = None
    if "adj_nav" in df.columns and df["adj_nav"].notna().any():
        col = "adj_nav"
    elif "unit_nav" in df.columns and df["unit_nav"].notna().any():
        col = "unit_nav"
    if col is None:
        return [], np.empty(0, dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce")
    keep = vals.notna() & (vals > 0.0)
    dates = [pd.Timestamp(d).date() for d in df.loc[keep, "nav_date"]]
    return dates, vals[keep].to_numpy(dtype=float)


def _visible_prices(store: PointInTimeStore, code: str, moment: datetime) -> np.ndarray:
    """Fund ``adj_nav`` series visible at ``moment`` (``available_at <= moment``), oldest→newest.

    The sole signal-time read path: a strategy can never see a NAV disclosed after its
    decision moment (invariant 1 / protocol rules 1 & 4). Kept as the *reference*
    implementation; the engine itself goes through :class:`_PriceTape`, which is proven
    equivalent to this function by ``tests/test_backtest.py`` (D-BT-15).
    """
    df = store.get_asof(TABLE_FUND_NAV, moment, {"fund_code": code})
    if df is None or df.empty or "nav_date" not in df.columns:
        return np.empty(0, dtype=float)
    df = df.sort_values(["nav_date", AVAILABLE_AT]).drop_duplicates(subset="nav_date", keep="last")
    col = None
    if "adj_nav" in df.columns and df["adj_nav"].notna().any():
        col = "adj_nav"
    elif "unit_nav" in df.columns and df["unit_nav"].notna().any():
        col = "unit_nav"
    if col is None:
        return np.empty(0, dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[vals.notna() & (vals > 0.0)]
    return vals.to_numpy(dtype=float)


class _PriceTape:
    """One point-in-time read of a fund's whole NAV history, sliced per decision (D-BT-15).

    The engine used to call :func:`_visible_prices` — a full ``get_asof`` + sort + de-dup of the
    *entire* history — once **per decision day**, making a simulation O(days x history) and a
    48-candidate parameter search take minutes. The tape performs that read **once** at
    ``end_moment`` (the window's outcome cutoff, so it can never contain a row the window itself
    may not see) and then answers ``visible_at(moment)`` by slicing.

    Point-in-time semantics are preserved exactly: ``visible_at(m)`` returns, for every
    ``nav_date`` having at least one row with ``available_at <= m``, the value of that
    ``nav_date``'s **latest** such row — identical to ``_visible_prices(store, code, m)``. Because
    every decision moment inside the window is ``<= end_moment``, slicing a superset read at
    ``end_moment`` is equivalent to re-reading at each moment (``get_asof`` is monotone in ``as_of``).
    """

    __slots__ = ("nav_dates", "outcome_vals", "_nd", "_av", "_adj", "_unit", "_unique", "_monotone")

    def __init__(self, store: PointInTimeStore, code: str, end_moment: datetime) -> None:
        self.nav_dates: list[date] = []
        self.outcome_vals: np.ndarray = np.empty(0, dtype=float)
        self._nd = np.empty(0, dtype=np.int64)
        self._av = np.empty(0, dtype=np.int64)
        self._adj = np.empty(0, dtype=float)
        self._unit = np.empty(0, dtype=float)
        self._unique = True
        self._monotone = True

        df = store.get_asof(TABLE_FUND_NAV, end_moment, {"fund_code": code})
        if df is None or df.empty or "nav_date" not in df.columns:
            return
        if "adj_nav" not in df.columns and "unit_nav" not in df.columns:
            return
        # Keep EVERY visible row, values un-filtered: the reference path de-duplicates first and
        # only then drops invalid values, and it chooses the price column from the rows visible at
        # *that* moment. Both must be reproduced per slice, not decided once at end_moment.
        df = df.sort_values(["nav_date", AVAILABLE_AT])
        nd_ts = pd.to_datetime(df["nav_date"])
        n = len(df)
        self._nd = nd_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._av = pd.to_datetime(df[AVAILABLE_AT]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._adj = (
            pd.to_numeric(df["adj_nav"], errors="coerce").to_numpy(dtype=float)
            if "adj_nav" in df.columns else np.full(n, np.nan)
        )
        self._unit = (
            pd.to_numeric(df["unit_nav"], errors="coerce").to_numpy(dtype=float)
            if "unit_nav" in df.columns else np.full(n, np.nan)
        )

        last = np.ones(n, dtype=bool)
        last[:-1] = self._nd[1:] != self._nd[:-1]
        self._unique = bool(last.all())                    # no restatements → one row per nav_date
        self._monotone = bool(self._unique and np.all(np.diff(self._av) >= 0))

        # outcome path: latest disclosed row per nav_date at end_moment, then column choice + filter
        vals, keep = _choose_and_filter(self._adj[last], self._unit[last])
        self.nav_dates = [d.date() for d in nd_ts[last][keep]]
        self.outcome_vals = vals

    def visible_at(self, moment: datetime) -> np.ndarray:
        """Prices visible at ``moment`` (``available_at <= moment``), ordered by ``nav_date``.

        Exactly reproduces :func:`_visible_prices`: restrict to visible rows, keep the **latest
        disclosure per nav_date**, then pick ``adj_nav`` if any of *those* rows has one (else
        ``unit_nav``) and drop NaN / non-positive values.
        """
        if self._nd.size == 0:
            return np.empty(0, dtype=float)
        m = np.int64(pd.Timestamp(moment).value)
        if self._monotone:  # available_at rises with nav_date → the visible set is a prefix
            k = int(np.searchsorted(self._av, m, side="right"))
            adj, unit = self._adj[:k], self._unit[:k]
        else:
            mask = self._av <= m
            adj, unit, nd = self._adj[mask], self._unit[mask], self._nd[mask]
            if not self._unique and nd.size:
                last = np.ones(nd.size, dtype=bool)
                last[:-1] = nd[1:] != nd[:-1]
                adj, unit = adj[last], unit[last]
        if adj.size == 0:
            return np.empty(0, dtype=float)
        vals, _ = _choose_and_filter(adj, unit)
        return vals

    def outcome_path(self) -> tuple[list[date], np.ndarray]:
        """The realised path the equity curve is marked on (latest value per ``nav_date``)."""
        return self.nav_dates, self.outcome_vals


def _choose_and_filter(adj: np.ndarray, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the reference column choice + validity filter on an already-de-duplicated slice.

    ``adj_nav`` is preferred iff at least one of *these* rows carries one (``notna().any()``);
    otherwise ``unit_nav``. Then NaN / non-positive values are dropped. Returns ``(values, keep)``.
    """
    col = adj if bool(np.any(~np.isnan(adj))) else unit
    keep = ~np.isnan(col) & (col > 0.0)
    return col[keep], keep


class _ShockIndex:
    """Pre-read of the ``market_events`` SHOCK flags, so the regime labeller is O(log n) per day.

    Same semantics as the former per-day ``get_asof`` on ``market_events``: a SHOCK flag is active
    at a decision moment when its ``available_at`` lies in ``(moment - event_active_days, moment]``.
    Read once at ``end_moment`` (a superset of every in-window decision moment), so no future flag
    can colour an earlier decision (D-BT-15).
    """

    __slots__ = ("_av",)

    def __init__(self, store: PointInTimeStore, end_moment: datetime) -> None:
        from qmt.pipeline.market_state import TABLE_MARKET_EVENTS  # local: avoid import cost/cycle

        self._av = np.empty(0, dtype=np.int64)
        df = store.get_asof(TABLE_MARKET_EVENTS, end_moment)
        if df is None or df.empty or "event" not in df.columns:
            return
        shocks = df[df["event"].astype(str) == MarketEvent.SHOCK.value]
        if shocks.empty:
            return
        av = pd.to_datetime(shocks[AVAILABLE_AT]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._av = np.sort(av)

    def active_at(self, moment: datetime) -> bool:
        if self._av.size == 0:
            return False
        m = pd.Timestamp(moment)
        lo = np.int64(pd.Timestamp(m - timedelta(days=CONFIG.market_state.event_active_days)).value)
        hi = np.int64(m.value)
        i = int(np.searchsorted(self._av, lo, side="left"))
        j = int(np.searchsorted(self._av, hi, side="right"))
        return j > i


# =============================================================================
# core per-fund simulation (position-return, execution-lagged, net of cost)
# =============================================================================
@dataclass
class _FundRun:
    fund_code: str
    dates: list[date]                 # measured interval-end dates (within window)
    net_ret: np.ndarray               # per-interval net-of-cost return
    cost: np.ndarray                  # per-interval cost fraction
    regime: list[str]                 # regime of the responsible decision
    decisions: list[DecisionRecord]
    net_return: float
    max_drawdown: float
    total_cost: float
    fund_type: Optional[str]


def _executable_from(as_of: date, calendar: TradingCalendar) -> Optional[date]:
    """Next executable point for a decision on ``as_of`` (build_decision_timeline horizon_start).

    Returns ``None`` when ``as_of`` is not a trading day on ``calendar`` or has no next
    trading day (the decision cannot be executed within the available grid → dropped).
    """
    try:
        tl = build_decision_timeline(as_of, horizon=Horizon.SHORT, calendar=calendar)
        return tl.horizon_start
    except ValueError:
        return None


def _simulate_fund(
    tape: _PriceTape,
    code: str,
    meta: _FundMeta,
    start: date,
    end: date,
    weight_fn: Strategy,
    fees: FeeSchedule,
    calendar: TradingCalendar,
    regime_labeler: RegimeLabeler,
) -> Optional[_FundRun]:
    """Run one fund through the position-return engine; ``None`` if there is nothing to measure."""
    path_dates, path_vals = tape.outcome_path()
    n = len(path_dates)
    if n < 3:
        return None

    # --- decisions: one per in-window trading day the fund is alive on -------
    # target[j] = desired invested fraction that becomes effective at grid index j (its fill
    # point). Forward-filled to give the held position at every grid step.
    target = np.full(n, np.nan, dtype=float)
    target[0] = 0.0  # flat before the first executable decision
    decisions: list[DecisionRecord] = []
    date_to_index = {d: i for i, d in enumerate(path_dates)}

    for i, d in enumerate(path_dates):
        if not (start <= d <= end):
            continue
        alive = meta.alive_on(d)
        prices = tape.visible_at(_decision_moment(d))
        raw = weight_fn(
            StrategyContext(
                fund_code=code, as_of=d, decision_moment=_decision_moment(d), prices=prices
            )
        )
        w = 0.0 if not alive else float(np.clip(raw, 0.0, 1.0))
        exec_from = _executable_from(d, calendar)
        regime = regime_labeler(d, prices)
        decisions.append(
            DecisionRecord(
                as_of=d,
                fund_code=code,
                fund_type=meta.fund_type,
                signal=w,
                executable_from=exec_from,
                regime=regime,
                alive=alive,
            )
        )
        # Map the fill to a grid index; a decision that cannot be executed on-grid is
        # recorded (above) but never takes a position (protocol rule 5).
        if exec_from is None:
            continue
        fill_idx = date_to_index.get(exec_from)
        if fill_idx is None:
            # first grid date on/after exec_from
            later = [j for j, gd in enumerate(path_dates) if gd >= exec_from]
            if not later:
                continue
            fill_idx = later[0]
        target[fill_idx] = w

    # forward-fill the target across the grid (held position at each step)
    target_series = pd.Series(target).ffill().fillna(0.0)
    target_ffilled = target_series.to_numpy(dtype=float)

    # decision regime by fill index, for attributing each interval's return to a regime
    regime_by_fill: dict[int, str] = {}
    for rec in decisions:
        if rec.executable_from is None:
            continue
        j = date_to_index.get(rec.executable_from)
        if j is None:
            later = [k for k, gd in enumerate(path_dates) if gd >= rec.executable_from]
            if not later:
                continue
            j = later[0]
        regime_by_fill[j] = rec.regime
    regime_ffill = pd.Series(
        [regime_by_fill.get(j, np.nan) for j in range(n)]
    ).ffill().tolist()

    # --- realised returns, execution-lagged position, costs -----------------
    m_dates: list[date] = []
    m_net: list[float] = []
    m_cost: list[float] = []
    m_regime: list[str] = []
    entry_date: Optional[date] = None

    for k in range(1, n):
        d_k = path_dates[k]
        if not (start <= d_k <= end):
            continue
        r_k = path_vals[k] / path_vals[k - 1] - 1.0
        pos_k = float(target_ffilled[k - 1])       # position held over (k-1, k]
        pos_prev = float(target_ffilled[k - 2]) if k >= 2 else 0.0
        dpos = pos_k - pos_prev

        cost_k = 0.0
        if dpos > 1e-12:  # buying → subscription fee on the added fraction
            cost_k += dpos * fees.purchase_fee_rate / (1.0 + fees.purchase_fee_rate)
            if pos_prev <= 1e-9:
                entry_date = path_dates[k - 1]
        elif dpos < -1e-12:  # selling → redemption fee by ACTUAL holding period (short-hold)
            hold_days = (d_k - entry_date).days if entry_date is not None else 0
            cost_k += (-dpos) * fees.redemption_rate(hold_days)
            if pos_k <= 1e-9:
                entry_date = None
        # Published (adjusted) NAV is already net of accrued fund operating and
        # sales-service fees. Subtracting them again would double-charge returns.

        m_dates.append(d_k)
        m_net.append(pos_k * r_k - cost_k)
        m_cost.append(cost_k)
        reg = regime_ffill[k - 1]
        m_regime.append(reg if isinstance(reg, str) else "warmup")

    if not m_dates:
        return None

    net = np.asarray(m_net, dtype=float)
    equity = np.cumprod(1.0 + net)
    net_return = float(equity[-1] - 1.0)
    max_dd = _max_drawdown(equity)

    return _FundRun(
        fund_code=code,
        dates=m_dates,
        net_ret=net,
        cost=np.asarray(m_cost, dtype=float),
        regime=m_regime,
        decisions=decisions,
        net_return=net_return,
        max_drawdown=max_dd,
        total_cost=float(np.sum(m_cost)),
        fund_type=meta.fund_type,
    )


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (non-negative)."""
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / running_peak
    return float(np.max(drawdowns))


# =============================================================================
# fixed-SIP baseline (money-weighted, separate accounting)
# =============================================================================
@dataclass
class _SipSeries:
    dates: list[date]
    value: np.ndarray       # market value of accumulated units at each measured date
    contrib: np.ndarray     # cumulative contributions by each measured date


def _simulate_sip_fund(
    tape: _PriceTape,
    code: str,
    meta: _FundMeta,
    start: date,
    end: date,
    fees: FeeSchedule,
    calendar: TradingCalendar,
    contribution: float = 1.0,
) -> Optional[_SipSeries]:
    """Fixed periodic investment: contribute a fixed amount each in-window trading day.

    Each contribution buys units at the next executable NAV (same T+1 fill as the strategy),
    net of the subscription fee. The reported performance is the money-weighted index
    ``market_value / cumulative_contribution``; ``net_return`` = that at ``end`` − 1. Fully
    survivorship-aware (contributes only while alive).
    """
    path_dates, path_vals = tape.outcome_path()
    n = len(path_dates)
    if n < 2:
        return None
    date_to_index = {d: i for i, d in enumerate(path_dates)}

    fills: list[tuple[int, float, float]] = []  # (fill_index, units, contribution)
    for i, d in enumerate(path_dates):
        if not (start <= d <= end) or not meta.alive_on(d):
            continue
        exec_from = _executable_from(d, calendar)
        if exec_from is None:
            continue
        fill_idx = date_to_index.get(exec_from)
        if fill_idx is None:
            later = [j for j, gd in enumerate(path_dates) if gd >= exec_from]
            if not later:
                continue
            fill_idx = later[0]
        nav = path_vals[fill_idx]
        units = contribution / (1.0 + fees.purchase_fee_rate) / nav if nav > 0 else 0.0
        fills.append((fill_idx, units, contribution))

    if not fills:
        return None

    m_dates: list[date] = []
    m_value: list[float] = []
    m_contrib: list[float] = []
    for k in range(n):
        d_k = path_dates[k]
        if not (start <= d_k <= end):
            continue
        units_held = sum(u for (fi, u, _c) in fills if fi <= k)
        contrib = sum(c for (fi, _u, c) in fills if fi <= k)
        if contrib <= 0.0:
            continue
        m_dates.append(d_k)
        m_value.append(units_held * path_vals[k])
        m_contrib.append(contrib)

    if not m_dates:
        return None
    return _SipSeries(m_dates, np.asarray(m_value, dtype=float), np.asarray(m_contrib, dtype=float))


# =============================================================================
# combination across funds (equal-weight, survivorship-aware)
# =============================================================================
def _combine_equity(runs: list[_FundRun]) -> pd.DataFrame:
    """Equal-weight the per-fund net-return series into one portfolio equity curve.

    Aligns on the union of interval dates; each date's portfolio return is the mean over the
    funds that have a defined return that day (a delisted fund simply drops out — its series
    ends at delisting), i.e. an equal-weight, daily-rebalanced book. Returns a
    ``date, equity`` frame.
    """
    if not runs:
        return pd.DataFrame(columns=["date", "equity"])
    frames = [pd.Series(r.net_ret, index=pd.to_datetime(r.dates), name=r.fund_code) for r in runs]
    merged = pd.concat(frames, axis=1).sort_index()
    port_ret = merged.mean(axis=1, skipna=True)
    equity = (1.0 + port_ret).cumprod()
    return pd.DataFrame({"date": [d.date() for d in equity.index], "equity": equity.to_numpy()})


def _combine_sip(series: list[_SipSeries]) -> pd.DataFrame:
    """Money-weighted portfolio SIP index: Σ value / Σ contribution across funds over time."""
    if not series:
        return pd.DataFrame(columns=["date", "equity"])
    val_frames = [pd.Series(s.value, index=pd.to_datetime(s.dates)) for s in series]
    con_frames = [pd.Series(s.contrib, index=pd.to_datetime(s.dates)) for s in series]
    values = pd.concat(val_frames, axis=1).sort_index()
    contribs = pd.concat(con_frames, axis=1).sort_index()
    # a fund present since its first contribution carries its last value/contrib forward
    values = values.ffill().fillna(0.0)
    contribs = contribs.ffill().fillna(0.0)
    total_val = values.sum(axis=1)
    total_con = contribs.sum(axis=1)
    index = (total_val / total_con.replace(0.0, np.nan)).dropna()
    return pd.DataFrame({"date": [d.date() for d in index.index], "equity": index.to_numpy()})


def _curve_metrics(curve: pd.DataFrame) -> tuple[float, float]:
    """(net_return, max_drawdown) of a ``date, equity`` curve; (0, 0) if empty."""
    if curve.empty:
        return 0.0, 0.0
    equity = curve["equity"].to_numpy(dtype=float)
    return float(equity[-1] / equity[0] - 1.0), _max_drawdown(equity)


# =============================================================================
# public entrypoint
# =============================================================================
# A per-fund fee source: one schedule for every fund, a ``{code: schedule}`` map, or a
# ``code -> schedule`` callable. ``None`` → a realistic default per code (:mod:`qmt.fees`).
FeeResolver = Union[FeeSchedule, Mapping[str, FeeSchedule], Callable[[str], FeeSchedule]]


def _resolve_fee_schedule(
    fees: Optional[FeeResolver], code: str, meta: "_FundMeta"
) -> FeeSchedule:
    """Resolve the :class:`FeeSchedule` for one fund (protocol rules 6/7 — never zero-cost).

    * ``None`` → :func:`qmt.fees.default_fee_schedule` (realistic, meta-aware defaults);
    * a single :class:`FeeSchedule` → applied to every fund (back-compat);
    * a ``{code: FeeSchedule}`` mapping → that fund's schedule, else a per-code default;
    * a ``code -> FeeSchedule`` callable → its result.
    """
    if fees is None:
        return default_fee_schedule(code, meta)
    if isinstance(fees, FeeSchedule):
        return fees
    if isinstance(fees, Mapping):
        got = fees.get(code)
        return got if got is not None else default_fee_schedule(code, meta)
    if callable(fees):
        return fees(code)
    raise ValueError(f"unsupported fees type {type(fees)!r} (want FeeSchedule / mapping / callable).")


def _grid_calendar_from_tapes(tapes: Mapping[str, _PriceTape]) -> TradingCalendar:
    """Build the trading grid from the union of the funds' observable NAV dates.

    Using the actual NAV grid as the calendar makes ``build_decision_timeline``'s
    ``next_trading_day`` land exactly on the next NAV date, so execution/confirmation delay
    is modelled on the real grid rather than an assumed business-day calendar.
    """
    all_dates: set[date] = set()
    for tape in tapes.values():
        all_dates.update(tape.nav_dates)
    if not all_dates:
        return TradingCalendar()
    return TradingCalendar(sorted(all_dates))


def run_backtest(
    codes: Union[str, Sequence[str]],
    start: date,
    end: date,
    strategy: Union[str, Strategy] = "baseline",
    fees: Optional[FeeResolver] = None,
    store: Optional[PointInTimeStore] = None,
    *,
    calendar: Optional[TradingCalendar] = None,
    regime_labeler: Optional[RegimeLabeler] = None,
    baselines: Optional[Sequence[str]] = None,
) -> BacktestResult:
    """Run a point-in-time, net-of-cost backtest and compare it to the five naive baselines.

    Parameters
    ----------
    codes:
        A single 6-digit fund code or a list of them (equal-weight, survivorship-aware
        portfolio).
    start, end:
        Inclusive backtest window (decisions are made on in-window trading days).
    strategy:
        A named built-in (``"baseline"``/``ma_cross``/``buy_and_hold``/``trend``/
        ``vol_control``/``fixed_sip``) or a custom :data:`Strategy` callable returning a
        target invested fraction in ``[0, 1]`` from point-in-time data.
    fees:
        Fee source, resolved **per fund** (rules 6/7 — never zero-cost): a single
        :class:`~qmt.types.FeeSchedule` applied to every code, a ``{code: FeeSchedule}``
        mapping, or a ``code -> FeeSchedule`` callable. ``None`` → a realistic default per
        code via :func:`qmt.fees.default_fee_schedule` (subscription / short-hold redemption
        tiers / sales-service), so results are genuinely net of a non-zero cost.
    store:
        The point-in-time store to read (the sole read path is ``get_asof``). ``None`` opens
        the persistent cache store (which the ingest layer must have primed).
    calendar:
        Trading calendar for the execution-delay math. ``None`` → reconstructed from the
        funds' observable NAV grid.
    regime_labeler:
        ``(as_of, visible_prices) -> label`` for the by-regime slice. ``None`` → a default
        point-in-time trend/vol labeller.
    baselines:
        Which of the five naive baselines to compute. ``None`` → all of :data:`ALL_BASELINES`
        (the doc's protocol rule 10). A parameter search may request a subset (e.g.
        ``("buy_and_hold",)``) to avoid re-simulating baselines it never reports; a baseline
        identical to the strategy is reused rather than re-simulated and named in
        :attr:`BacktestResult.self_baseline` (D-BT-16).

    Returns
    -------
    BacktestResult
        Equity curve, net return, max drawdown, per-baseline deltas, per-decision log and
        by-regime / by-fund-type slices — all net of cost.

    Raises
    ------
    ValueError
        If nothing is measurable for any code in the window (fail-closed: a backtest with no
        observable data is a configuration error, never a silent empty result).
    """
    code_list = [codes] if isinstance(codes, str) else list(codes)
    if not code_list:
        raise ValueError("run_backtest requires at least one fund code.")
    if end < start:
        raise ValueError(f"backtest end {end} precedes start {start}.")

    requested = tuple(baselines) if baselines is not None else ALL_BASELINES
    unknown = [b for b in requested if b not in ALL_BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; known: {list(ALL_BASELINES)}.")

    st = store if store is not None else PointInTimeStore(get_settings().data_cache_dir)
    end_moment = _end_moment(end)
    # ONE point-in-time read of each fund's history; every decision-day query slices it (D-BT-15).
    tapes = {code: _PriceTape(st, code, end_moment) for code in code_list}
    cal = calendar if calendar is not None else _grid_calendar_from_tapes(tapes)
    labeler = (
        regime_labeler if regime_labeler is not None else _make_default_regime_labeler(st, end_moment)
    )

    if isinstance(strategy, str):
        strat_name = strategy
        weight_fn: Optional[Strategy] = STRATEGIES.get(strategy)
        is_sip_strategy = strategy == "fixed_sip"
        if weight_fn is None and not is_sip_strategy:
            raise ValueError(
                f"unknown strategy {strategy!r}; known: {sorted(STRATEGIES) + ['fixed_sip']}."
            )
    else:
        strat_name = getattr(strategy, "__name__", "custom")
        weight_fn = strategy
        is_sip_strategy = False

    metas = {code: _reconstruct_meta(st, code, end) for code in code_list}
    # Resolve a fee schedule PER FUND (rules 6/7 — never zero-cost; item 4 per-code fees).
    fee_map = {code: _resolve_fee_schedule(fees, code, metas[code]) for code in code_list}

    def _weight_runs(fn: Strategy) -> list[_FundRun]:
        return [
            r
            for code in code_list
            if (r := _simulate_fund(
                tapes[code], code, metas[code], start, end, fn, fee_map[code], cal, labeler
            )) is not None
        ]

    def _weight_baseline(name: str, fn: Strategy) -> tuple[BaselineOutcome, float]:
        curve = _combine_equity(_weight_runs(fn))
        ret, dd = _curve_metrics(curve)
        return BaselineOutcome(name, ret, dd, curve), ret

    def _sip() -> tuple[pd.DataFrame, float, float]:
        series = [
            s
            for code in code_list
            if (s := _simulate_sip_fund(
                tapes[code], code, metas[code], start, end, fee_map[code], cal
            )) is not None
        ]
        curve = _combine_sip(series)
        ret, dd = _curve_metrics(curve)
        return curve, ret, dd

    # --- strategy FIRST, so a requested baseline identical to it can reuse the result -----
    sip_cache: Optional[tuple[pd.DataFrame, float, float]] = None
    if is_sip_strategy:
        sip_cache = _sip()
        strat_curve, strat_ret, strat_dd = sip_cache
        total_cost = 0.0
        decision_log: list[DecisionRecord] = []
        by_regime: dict[str, float] = {}
        by_type: dict[str, float] = {}
    else:
        assert weight_fn is not None
        runs = _weight_runs(weight_fn)
        if not runs:
            raise ValueError(
                f"backtest measured nothing for {code_list} in {start}..{end} "
                f"(no observable NAV in the window — prime the store first)."
            )
        strat_curve = _combine_equity(runs)
        strat_ret, strat_dd = _curve_metrics(strat_curve)
        total_cost = float(sum(r.total_cost for r in runs))
        decision_log = [d for r in runs for d in r.decisions]
        by_regime = _slice_by_regime(runs)
        by_type = _slice_by_fund_type(runs)

    if strat_curve.empty:
        raise ValueError(
            f"backtest produced no equity curve for {code_list} in {start}..{end} "
            f"(no observable data — prime the store first)."
        )

    # --- the requested naive baselines (§sec:test-baselines), all net of cost -------------
    # 长期持有 / 固定定投 / 简单均线 / 简单趋势 / 简单波动控制. A baseline whose weight function IS the
    # strategy's is a *self-comparison* (its delta is identically 0): reuse the strategy's own run
    # instead of simulating it again, and name it in ``self_baseline`` so the degenerate 0 is
    # explicit rather than mysterious (D-BT-16).
    baselines_out: dict[str, BaselineOutcome] = {}
    baseline_ret: dict[str, float] = {}
    self_baseline: Optional[str] = None
    for name in requested:
        if name == "fixed_sip":
            if sip_cache is None:
                sip_cache = _sip()
            curve, ret, dd = sip_cache
            if is_sip_strategy:
                self_baseline = name
            baselines_out[name] = BaselineOutcome(name, ret, dd, curve)
            baseline_ret[name] = ret
            continue
        fn = _WEIGHT_BASELINES[name]
        if (not is_sip_strategy) and fn is weight_fn:  # self-comparison → reuse, no extra simulation
            self_baseline = name
            baselines_out[name] = BaselineOutcome(name, strat_ret, strat_dd, strat_curve)
            baseline_ret[name] = strat_ret
            continue
        outcome, ret = _weight_baseline(name, fn)
        baselines_out[name] = outcome
        baseline_ret[name] = ret

    return BacktestResult(
        codes=code_list,
        start=start,
        end=end,
        strategy_name=strat_name,
        equity_curve=strat_curve,
        net_return=strat_ret,
        max_drawdown=strat_dd,
        total_cost=total_cost,
        n_decisions=len(decision_log),
        baselines=baselines_out,
        vs_baseline={name: strat_ret - r for name, r in baseline_ret.items()},
        decision_log=decision_log,
        net_return_by_regime=by_regime,
        net_return_by_fund_type=by_type,
        self_baseline=self_baseline,
    )


def _slice_by_regime(runs: list[_FundRun]) -> dict[str, float]:
    """Additive net-return attribution by (point-in-time) regime (protocol rule 8).

    Sums each interval's net return into its responsible decision's regime, equal-weighted
    across funds. The sum over regimes equals the portfolio's arithmetic sum of interval
    returns (a transparent additive decomposition, not a compounded per-regime return).
    """
    n = len(runs) or 1
    acc: dict[str, float] = {}
    for r in runs:
        for ret, reg in zip(r.net_ret, r.regime):
            acc[reg] = acc.get(reg, 0.0) + float(ret) / n
    return acc


def _slice_by_fund_type(runs: list[_FundRun]) -> dict[str, float]:
    """Per-fund-type net return (protocol rule 9): equal-weighted mean of standalone returns."""
    groups: dict[str, list[float]] = {}
    for r in runs:
        key = r.fund_type or "unknown"
        groups.setdefault(key, []).append(r.net_return)
    return {k: float(np.mean(v)) for k, v in groups.items()}
