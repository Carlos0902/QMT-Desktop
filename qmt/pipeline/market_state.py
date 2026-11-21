"""market_state — the multi-dimensional market-state gate (§4 sec:mod-market, §1.5/§2.7, §1.8-4).

Responsibility (doc §3 sec:ds-marketstate, §4 sec:mod-market): judge the market as
**four INDEPENDENT dimensions** and *never* collapse them into a single label
(invariant 5). The doc is explicit that the market can be, at the same time, "up but
very volatile and policy-driven" or "down with a risk event" — so the output is a
:class:`~qmt.types.MarketStateVector` carrying all four:

======================  ================================================================
dimension               baseline rule (05-algorithms.tex, tbl 均线/波动分位/成交阈值)
======================  ================================================================
``direction``           fast/slow moving-average cross (MA20 vs MA60) on a broad index.
``volatility``          rolling-20d realized vol vs its own historical 33rd/67th pct.
``stress``              turnover ("volume") shrink vs its historical low percentile.
``event``               NONE unless a policy / shock flag is fed (point-in-time, store).
======================  ================================================================

Doc signature (04-modules.tex, lst:market)::

    def assess_market_state(as_of: date) -> MarketStateVector: ...

The concrete ``store`` / ``benchmark_index`` parameters are additive (they carry the
point-in-time data source and the broad-index choice); the doc's one-argument contract
is preserved as the leading positional parameter.

Point-in-time correctness (invariant 1): every read goes through
:meth:`PointInTimeStore.get_asof` at the T-day **decision moment** (14:30, from
``CONFIG.timing``), so the current day's EOD close (available_at = trade_date @ 15:00)
is never visible when forming the signal — the "no same-day close in a pre-close signal"
rule. The historical distribution used for the percentile comparisons is likewise only
the observable history ``available_at <= decision_moment``; nothing after ``as_of`` can
leak in.

Baseline-first: this is the doc's simple transparent rule set. The candidate direction
(regime-switching / change-point detection, E54) is left as a documented hook in
:func:`_assess_direction` / :func:`_assess_volatility` — it is **not** built here.

Fail-closed (invariant 6): if the observable history is too short to support the rules,
the function refuses (raises :class:`InsufficientMarketHistory`) rather than fabricate a
benign judgment; every emitted vector is fully data-backed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from qmt.config import CONFIG, MarketStateConfig
from qmt.data.sources import TABLE_INDEX_DAILY
from qmt.pit import AVAILABLE_AT, PointInTimeStore
from qmt.types import Direction, MarketEvent, MarketStateVector, Stress, Volatility

__all__ = [
    "assess_market_state",
    "record_market_event",
    "InsufficientMarketHistory",
    "TABLE_MARKET_EVENTS",
]

# Point-in-time table of fed policy/shock flags (Part-B addition, B07). A row carries
# ``available_at`` (when the flag became known) and ``event`` (a MarketEvent value).
TABLE_MARKET_EVENTS = "market_events"

# Turnover ("volume") column preference on an index_daily bar: 成交额 first, then 成交量.
_TURNOVER_COLUMNS = ("amount", "vol")


class InsufficientMarketHistory(ValueError):
    """Raised when the observable index history is too short to assess market state.

    Fail-closed (invariant 6): the module refuses to emit a fabricated benign vector on
    missing data. The caller (pipeline orchestrator) treats this as "cannot assess".
    """


# ---------------------------------------------------------------------------
# point-in-time data access
# ---------------------------------------------------------------------------
def _decision_moment(as_of: date) -> datetime:
    """The T-day decision instant (14:30 by ``CONFIG.timing``) used as the read as_of.

    Reading at 14:30 (< the 15:00 EOD close) guarantees the current day's own close is
    still hidden, so the market-state signal is formed only from closes through T-1.
    """
    tcfg = CONFIG.timing
    return datetime.combine(as_of, time(hour=tcfg.decision_hour, minute=tcfg.decision_minute))


def _resolve_store(store: Optional[PointInTimeStore]) -> PointInTimeStore:
    """A caller-supplied store always wins; ``None`` falls back to the default cache.

    The default is the persistent Parquet store under the configured ``data_cache`` — it
    must have been populated by the data layer (``qmt.data.sources.index_daily``). This
    module is *read-only* over the point-in-time store; it never fetches (keeps it pure
    and offline-testable).
    """
    if store is not None:
        return store
    from qmt.config import get_settings

    return PointInTimeStore(get_settings().data_cache_dir)


def _load_index_bars(
    store: PointInTimeStore, benchmark_index: str, read_asof: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(closes, turnover)`` for the benchmark index, observable at ``read_asof``.

    Both arrays are ordered oldest→newest and cleaned of unusable rows (NaN close). Only
    rows with ``available_at <= read_asof`` are ever returned (the sole PIT read path).
    """
    df = store.get_asof(TABLE_INDEX_DAILY, read_asof, {"ts_code": benchmark_index})
    if df.empty:
        return np.empty(0), np.empty(0)

    df = df.sort_values(AVAILABLE_AT).reset_index(drop=True)

    if "close" not in df.columns:
        raise InsufficientMarketHistory(
            f"index {benchmark_index!r} bars carry no 'close' column; cannot assess direction/volatility."
        )
    close = pd.to_numeric(df["close"], errors="coerce")
    usable = close.notna()

    turnover_col = next(
        (c for c in _TURNOVER_COLUMNS if c in df.columns and df.loc[usable, c].notna().any()),
        None,
    )
    if turnover_col is None:
        raise InsufficientMarketHistory(
            f"index {benchmark_index!r} bars carry no usable turnover column "
            f"({'/'.join(_TURNOVER_COLUMNS)}); cannot assess stress."
        )
    turnover = pd.to_numeric(df[turnover_col], errors="coerce")

    closes = close[usable].to_numpy(dtype=float)
    turnovers = turnover[usable].to_numpy(dtype=float)
    return closes, turnovers


# ---------------------------------------------------------------------------
# dimension 1 — direction (MA fast/slow cross)
# ---------------------------------------------------------------------------
def _assess_direction(closes: np.ndarray, cfg: MarketStateConfig) -> Direction:
    """Fast/slow MA cross with a neutral band (baseline).

    ``gap = (MA_fast - MA_slow) / MA_slow``. ``gap > band`` → UP; ``gap < -band`` → DOWN;
    within ``±band`` → RANGE (no clear trend). The band (``direction_band_pct``) makes
    RANGE reachable instead of a pure up/down flip.

    Candidate hook (E54, NOT built): a regime-switching / change-point detector would
    replace this cross to lead the rule at turning points; it must be chosen only by
    time-ordered validation, never adopted by default.
    """
    ma_fast = float(np.mean(closes[-cfg.ma_fast_days :]))
    ma_slow = float(np.mean(closes[-cfg.ma_slow_days :]))
    if ma_slow == 0.0:
        return Direction.RANGE
    gap = (ma_fast - ma_slow) / ma_slow
    if gap > cfg.direction_band_pct:
        return Direction.UP
    if gap < -cfg.direction_band_pct:
        return Direction.DOWN
    return Direction.RANGE


# ---------------------------------------------------------------------------
# dimension 2 — volatility (rolling realized vol vs its historical percentiles)
# ---------------------------------------------------------------------------
def _rolling_realized_vol(closes: np.ndarray, window: int) -> np.ndarray:
    """Rolling-``window`` realized vol (sample std of simple daily returns)."""
    returns = pd.Series(closes).pct_change()
    vol = returns.rolling(window=window).std()
    return vol.dropna().to_numpy(dtype=float)


def _assess_volatility(closes: np.ndarray, cfg: MarketStateConfig) -> Volatility:
    """Rolling-20d realized vol placed against its own historical 33rd/67th percentile.

    The *current* realized vol is the latest rolling value; the historical distribution is
    the whole observable rolling-vol series. HIGH iff ``current >= p_high``; LOW iff
    ``current <= p_low``; else NORMAL. HIGH is tested first so a degenerate (constant) vol
    series resolves to the cautious side (fail-closed lean, invariant 6).

    Candidate hook (E59, NOT built): a GARCH-class / realized-vol model would react faster
    to a volatility regime change; adopted only if it demonstrably predicts bumpiness
    better than this rolling baseline.
    """
    vol_series = _rolling_realized_vol(closes, cfg.realized_vol_window_days)
    if vol_series.size == 0:
        raise InsufficientMarketHistory(
            "not enough returns to form a rolling realized-vol series."
        )
    current = float(vol_series[-1])
    p_low = float(np.percentile(vol_series, cfg.vol_low_percentile))
    p_high = float(np.percentile(vol_series, cfg.vol_high_percentile))
    if current >= p_high:
        return Volatility.HIGH
    if current <= p_low:
        return Volatility.LOW
    return Volatility.NORMAL


# ---------------------------------------------------------------------------
# dimension 3 — stress (turnover / "volume" shrink)
# ---------------------------------------------------------------------------
def _index_drawdown_from_peak(closes: np.ndarray) -> float:
    """Current close's drawdown from its trailing observable peak (≤ 0; 0 = at a new peak)."""
    if closes.size == 0:
        return 0.0
    peak = float(np.max(closes))
    if peak <= 0.0:
        return 0.0
    return closes[-1] / peak - 1.0


def _assess_stress(
    turnover: np.ndarray,
    closes: np.ndarray,
    volatility: Volatility,
    cfg: MarketStateConfig,
) -> Stress:
    """Market-stress baseline over three states (§3 sec:ds-marketstate: 正常/流动性紧张/风险压力上升).

    * ``RISING_RISK`` (风险压力上升) — a transparent broad-index proxy (D-MS-1): the index is
      in a drawdown of at least ``drawdown_stress_pct`` below its trailing observable peak
      **AND** realized volatility is in its HIGH band. The doc's true signal is valuation
      (index PE/PB above ``valuation_stress_percentile``) or cross-sectional breadth
      (advance/decline) — neither is in the data layer yet (see ``sources.DATA_GAPS`` and the
      index_dailybasic / index_member TODOs, E40/E41) — so a drawdown+high-vol proxy stands
      in until those sources land. Checked first (it is the more severe / more cautious
      state — uncertainty weights RISING_RISK 0.70 < LIQUIDITY_TIGHT 0.75, fail-closed lean).
    * ``LIQUIDITY_TIGHT`` (流动性紧张) — recent smoothed turnover vs its historical low
      percentile: TIGHT iff ``current < p_turnover_low`` (strict, so a flat/constant-turnover
      market is NORMAL, not falsely "tight").
    * ``NORMAL`` otherwise.
    """
    # --- RISING_RISK proxy: broad-index drawdown + HIGH realized vol (D-MS-1) ---
    drawdown = _index_drawdown_from_peak(closes)
    if volatility is Volatility.HIGH and drawdown <= -cfg.drawdown_stress_pct:
        return Stress.RISING_RISK

    series = pd.Series(turnover).rolling(window=cfg.stress_volume_window_days).mean().dropna()
    if series.empty:
        raise InsufficientMarketHistory(
            "not enough turnover observations to smooth for the stress rule."
        )
    smoothed = series.to_numpy(dtype=float)
    current = float(smoothed[-1])
    threshold = float(np.percentile(smoothed, cfg.turnover_stress_percentile))
    if current < threshold:
        return Stress.LIQUIDITY_TIGHT
    return Stress.NORMAL


# ---------------------------------------------------------------------------
# dimension 4 — event (fed policy/shock flag, point-in-time)
# ---------------------------------------------------------------------------
def _assess_event(
    store: PointInTimeStore, read_asof: datetime, cfg: MarketStateConfig
) -> MarketEvent:
    """Event dimension: NONE unless a policy/shock flag is fed and still active.

    Flags live in the ``market_events`` PIT table (fed via :func:`record_market_event`).
    Only flags observable at the decision moment (``available_at <= read_asof``) and no
    older than ``event_active_days`` count. SHOCK dominates POLICY. Absent any active flag
    (or an empty/missing table) the dimension is NONE (per the doc's default).
    """
    df = store.get_asof(TABLE_MARKET_EVENTS, read_asof)
    if df.empty or "event" not in df.columns:
        return MarketEvent.NONE

    cutoff = read_asof - timedelta(days=cfg.event_active_days)
    available = pd.to_datetime(df[AVAILABLE_AT])
    active = df[available >= cutoff]
    if active.empty:
        return MarketEvent.NONE

    values = set(active["event"].astype(str))
    if MarketEvent.SHOCK.value in values:
        return MarketEvent.SHOCK
    if MarketEvent.POLICY.value in values:
        return MarketEvent.POLICY
    return MarketEvent.NONE


def record_market_event(
    store: PointInTimeStore,
    *,
    event: MarketEvent,
    available_at: datetime,
) -> int:
    """Feed a policy/shock flag into the point-in-time ``market_events`` table.

    ``available_at`` is *when the flag became known* — it is honored by the PIT read path,
    so a flag never affects a decision made before it was observable. Returns the number of
    rows written (0 if ``event`` is NONE — there is nothing to record).
    """
    if event is MarketEvent.NONE:
        return 0
    return store.put(
        TABLE_MARKET_EVENTS,
        [{"event": event.value, AVAILABLE_AT: available_at}],
    )


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def assess_market_state(
    as_of: date,
    store: Optional[PointInTimeStore] = None,
    benchmark_index: str = "000300.SH",
) -> MarketStateVector:
    """Assess the market as four INDEPENDENT dimensions at ``as_of`` (never one label).

    :param as_of: the T-day decision date. Data is read at the 14:30 decision moment, so
        the current day's own EOD close is not visible (point-in-time, invariant 1).
    :param store: point-in-time store to read from; ``None`` uses the default persistent
        cache (which the data layer must have populated). Read-only — never fetched here.
    :param benchmark_index: the broad index whose bars drive direction/volatility/stress
        (default CSI 300, ``000300.SH``).
    :returns: a :class:`MarketStateVector` with independent direction / volatility / stress
        / event fields.
    :raises InsufficientMarketHistory: if the observable history is too short to support
        the baseline rules (fail-closed — no fabricated benign vector).

    Worked example (doc §3 sec:ds-marketstate): 方向向上 / 波动高 / 压力正常 / 事件政策影响
    → ``MarketStateVector(direction=UP, volatility=HIGH, stress=NORMAL, event=POLICY)``.
    """
    cfg = CONFIG.market_state
    pit = _resolve_store(store)
    read_asof = _decision_moment(as_of)

    closes, turnover = _load_index_bars(pit, benchmark_index, read_asof)

    min_bars = cfg.ma_slow_days + 1  # slow MA needs ma_slow_days closes; +1 for one return
    if closes.size < min_bars:
        raise InsufficientMarketHistory(
            f"index {benchmark_index!r}: {closes.size} observable bars < {min_bars} required "
            f"at {as_of.isoformat()} (run the data layer to populate the cache)."
        )

    # Four dimensions assessed independently — never collapsed to a single label.
    # (stress reads the volatility verdict + index level only for its RISING_RISK proxy
    # rule, D-MS-1; the three price/turnover dimensions stay independent judgments.)
    direction = _assess_direction(closes, cfg)
    volatility = _assess_volatility(closes, cfg)
    stress = _assess_stress(turnover, closes, volatility, cfg)
    event = _assess_event(pit, read_asof, cfg)

    return MarketStateVector(
        as_of=as_of,
        direction=direction,
        volatility=volatility,
        stress=stress,
        event=event,
    )
