"""horizon_forecaster — multi-horizon return/risk ranges (§2.2 / §3.3, §4 sec:mod-forecast, module 5).

Responsibility (doc §4 sec:mod-forecast): for each of the three horizons — SHORT
(1--3), MID (5--20), LONG (20--90) trading days — answer the *concrete* questions first
(how much might I gain/lose, the worse / normal / most-optimistic case, the chance of a
positive return, how bumpy it might be, how high the material-drawdown risk is) and emit
a :class:`~qmt.types.HorizonForecast` **range** per horizon. It never emits an
up/down/strong/weak label — that translation to product language is a *separate* concern
(``to_product_view`` / analysis_mode, E57), deliberately not built here.

Doc signature (04-modules.tex, lst:forecast)::

    def forecast(fund_code, as_of, exposure, market) -> dict[str, HorizonForecast]: ...

This module realizes the task-frozen signature (``market_state`` param name, keyed by the
:class:`~qmt.types.Horizon` enum — see DECISIONS D-FC-1).

Baseline (§5 sec:algo-table, "简单历史规律 / 近期动量·均值回复的历史分布"): the empirical
distribution of the fund's own **overlapping forward returns**, computed per horizon from
the ``adj_nav`` series that is *visible at* ``as_of``. ``ret_low/ret_mid/ret_high`` are
empirical quantiles (10 / 50 / 90); ``prob_positive`` and ``vol_est`` and
``drawdown_prob`` are sample statistics of that same distribution. The candidate
(regularized regression / gradient boosting conditioned on estimated exposure, §5) is a
documented hook — this module deliberately implements the simple, transparent baseline
first (§5 sec:algo-principles).

Point-in-time correctness (§6 sec:cc-pit, invariant 1): the NAV series is read **only**
through :meth:`~qmt.pit.PointInTimeStore.get_asof`, so the forecaster can never see a NAV
whose ``available_at`` is after ``as_of``. It does not peek past ``as_of``.

Fail-closed (invariant 6): when too little history is visible to trust the empirical
distribution for a horizon, the forecast for that horizon degrades to an honest, maximally
uncertain "we don't know" (ret_mid 0, a wide symmetric band, prob_positive 0.5, a
conservative drawdown_prob) — never a spuriously narrow/optimistic range.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from qmt.config import CONFIG, HorizonForecasterConfig
from qmt.data.sources import TABLE_FUND_NAV
from qmt.pit import PointInTimeStore
from qmt.types import (
    ExposureEstimate,
    Horizon,
    HorizonForecast,
    MarketEvent,
    MarketStateVector,
    Stress,
    Volatility,
)

__all__ = ["forecast"]


# =============================================================================
# Public entrypoint (task-frozen signature)
# =============================================================================
def forecast(
    fund_code: str,
    as_of: date,
    exposure: list[ExposureEstimate],
    market_state: MarketStateVector,
    store: Optional[PointInTimeStore] = None,
) -> dict[Horizon, HorizonForecast]:
    """Per-horizon return/risk ranges for ``fund_code`` as of ``as_of``.

    Parameters
    ----------
    fund_code:
        6-digit OTC fund code whose ``adj_nav`` history is forecast.
    as_of:
        Decision point. Only NAV with ``available_at <= as_of`` is used (invariant 1).
    exposure:
        Estimated exposure of the fund (three-tier, from ``exposure_estimator``). In the
        baseline this is a point-in-time *consistency guard* (each estimate must belong to
        this fund and must not be dated after ``as_of``); conditioning the return
        distribution on it is the candidate hook (§5, DECISIONS D-FC-6).
    market_state:
        Four-dimensional market state (§3 sec:ds-marketstate). Gates a conservative,
        one-directional risk overlay (§5 顺风/逆风/乱流 designnote): high volatility /
        stress / shock only *widen* ranges and *raise* risk, never inflate optimism.
    store:
        Point-in-time store to read NAV from. ``None`` opens the persistent cache store
        (``settings.data_cache_dir``); tests inject an in-memory / synthetic store.

    Returns
    -------
    dict[Horizon, HorizonForecast]
        One entry per :class:`Horizon` (SHORT / MID / LONG). Always all three horizons;
        a horizon with too little visible history gets a fail-closed degraded range.
    """
    cfg = CONFIG.forecaster
    _guard_exposure_pit(exposure, fund_code, as_of)

    # §5.3 hierarchical candidate (D-HF-5): opt-in only (baseline stays the default). When on,
    # delegate to the market+industry+idiosyncratic decomposition, returning the same
    # dict[Horizon, HorizonForecast] shape so the pipeline is method-agnostic.
    if cfg.use_hierarchical:
        from qmt.pipeline.hierarchical_forecaster import forecast_hierarchical

        return forecast_hierarchical(fund_code, as_of, exposure, market_state, store).forecasts

    src = store if store is not None else _default_store()
    prices = _visible_price_series(src, fund_code, as_of, cfg)

    out: dict[Horizon, HorizonForecast] = {}
    for horizon in (Horizon.SHORT, Horizon.MID, Horizon.LONG):
        _, window_max = CONFIG.horizon.windows[horizon.value]
        out[horizon] = _forecast_one(horizon, window_max, prices, market_state, cfg)
    return out


# =============================================================================
# Point-in-time guards / data access (invariant 1)
# =============================================================================
def _guard_exposure_pit(
    exposure: list[ExposureEstimate], fund_code: str, as_of: date
) -> None:
    """Fail-closed: reject exposure that is for another fund or dated after ``as_of``.

    An exposure estimate produced *after* the decision point would be look-ahead; an
    estimate for a different fund is an inconsistent input. Both must never feed a
    point-in-time forecast (§6 sec:cc-pit).
    """
    for est in exposure:
        if est.fund_code != fund_code:
            raise ValueError(
                f"exposure for {est.fund_code!r} passed to forecast({fund_code!r}) — mismatched fund."
            )
        if est.as_of > as_of:
            raise ValueError(
                f"exposure.as_of {est.as_of} is after decision point {as_of} — look-ahead forbidden."
            )


def _default_store() -> PointInTimeStore:
    """Open the persistent point-in-time cache store (production path)."""
    from qmt.config import get_settings

    return PointInTimeStore(get_settings().data_cache_dir)


def _visible_price_series(
    store: PointInTimeStore,
    fund_code: str,
    as_of: date,
    cfg: HorizonForecasterConfig,
) -> np.ndarray:
    """Return the fund's price series (adj_nav preferred) visible at ``as_of``, ordered by nav_date.

    Reads only via ``get_asof`` (``available_at <= as_of``). Restatements are collapsed to
    the most-recently-disclosed value per ``nav_date``. Prefers ``adj_nav``; falls back to
    ``unit_nav`` only if ``adj_nav`` is entirely unavailable (DECISIONS D-FC-6). Trims to the
    most recent ``lookback_trading_days`` observations.
    """
    df = store.get_asof(TABLE_FUND_NAV, as_of, filters={"fund_code": fund_code})
    if df is None or df.empty or "nav_date" not in df.columns:
        return np.empty(0, dtype=float)

    df = df.sort_values(["nav_date", "available_at"]).drop_duplicates(
        subset="nav_date", keep="last"
    )

    col: Optional[str] = None
    if "adj_nav" in df.columns and df["adj_nav"].notna().any():
        col = "adj_nav"
    elif "unit_nav" in df.columns and df["unit_nav"].notna().any():
        col = "unit_nav"
    if col is None:
        return np.empty(0, dtype=float)

    prices = pd.to_numeric(df[col], errors="coerce")
    prices = prices[prices.notna() & (prices > 0.0)]
    values = prices.to_numpy(dtype=float)
    if cfg.lookback_trading_days > 0:
        values = values[-cfg.lookback_trading_days :]
    return values


# =============================================================================
# Baseline: per-horizon empirical statistics (ranges, not labels — invariant 4)
# =============================================================================
def _forecast_one(
    horizon: Horizon,
    window_max: int,
    prices: np.ndarray,
    market_state: MarketStateVector,
    cfg: HorizonForecasterConfig,
) -> HorizonForecast:
    """Build one horizon's forecast from the empirical forward-return distribution.

    The horizon length is the window's max (SHORT→3, MID→20, LONG→90 trading days), i.e.
    the return over the full window (DECISIONS D-FC-2). Emits a degraded fail-closed forecast
    when fewer than ``min_windows`` overlapping windows are available.
    """
    horizon_len = max(int(window_max), 1)
    fwd_returns, drawdown_hits = _overlapping_windows(prices, horizon_len, cfg)

    if fwd_returns.size < cfg.min_windows:
        stats = _degraded_stats(cfg)
    else:
        q_low = float(np.quantile(fwd_returns, cfg.ret_low_quantile))
        q_mid = float(np.quantile(fwd_returns, 0.5))
        q_high = float(np.quantile(fwd_returns, cfg.ret_high_quantile))
        vol = float(np.std(fwd_returns, ddof=1)) if fwd_returns.size > 1 else 0.0
        stats = _ForecastStats(
            ret_low=q_low,
            ret_mid=q_mid,
            ret_high=q_high,
            prob_positive=float(np.mean(fwd_returns > 0.0)),
            vol_est=vol,
            drawdown_prob=float(np.mean(drawdown_hits)),
        )

    stats = _apply_market_overlay(stats, market_state, cfg)
    return _to_forecast(horizon, stats)


def _overlapping_windows(
    prices: np.ndarray, horizon_len: int, cfg: HorizonForecasterConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (forward returns, drawdown-hit flags) over every overlapping ``horizon_len`` window.

    * forward return of window starting at ``i`` = ``prices[i+h] / prices[i] - 1``.
    * drawdown hit = the worst drop *from entry* anywhere inside the window is at or below
      ``-material_drawdown_threshold`` (a holder who bought at entry was, at some point,
      down more than the threshold). This is genuinely distinct from ``prob_positive``.
    """
    n = prices.size
    if n < horizon_len + 1:
        return np.empty(0, dtype=float), np.empty(0, dtype=bool)

    entries = prices[: n - horizon_len]
    ends = prices[horizon_len:]
    fwd_returns = ends / entries - 1.0

    threshold = cfg.material_drawdown_threshold
    hits = np.empty(entries.size, dtype=bool)
    for i in range(entries.size):
        path = prices[i + 1 : i + 1 + horizon_len]  # points strictly after entry
        worst = path.min() / prices[i] - 1.0
        hits[i] = worst <= -threshold
    return fwd_returns, hits


# =============================================================================
# Market-state risk overlay (§5 gating designnote; one-directional, fail-closed)
# =============================================================================
class _ForecastStats:
    """Mutable carrier for the six numeric outputs before validation into HorizonForecast."""

    __slots__ = ("ret_low", "ret_mid", "ret_high", "prob_positive", "vol_est", "drawdown_prob")

    def __init__(
        self,
        ret_low: float,
        ret_mid: float,
        ret_high: float,
        prob_positive: float,
        vol_est: float,
        drawdown_prob: float,
    ) -> None:
        self.ret_low = ret_low
        self.ret_mid = ret_mid
        self.ret_high = ret_high
        self.prob_positive = prob_positive
        self.vol_est = vol_est
        self.drawdown_prob = drawdown_prob


def _degraded_stats(cfg: HorizonForecasterConfig) -> _ForecastStats:
    """Fail-closed 'we don't know' statistics for a horizon with insufficient history."""
    return _ForecastStats(
        ret_low=-cfg.degraded_ret_band,
        ret_mid=0.0,
        ret_high=cfg.degraded_ret_band,
        prob_positive=0.5,
        vol_est=cfg.degraded_vol_est,
        drawdown_prob=cfg.degraded_drawdown_prob,
    )


def _risk_multiplier(market_state: MarketStateVector, cfg: HorizonForecasterConfig) -> float:
    """Compose a conservative (>= 1.0) risk multiplier from the market-state vector.

    LOW volatility / NONE event never *reduce* the multiplier — the overlay only ever
    widens uncertainty (fail-closed).
    """
    m = 1.0
    if market_state.volatility is Volatility.HIGH:
        m *= cfg.high_vol_mult
    if market_state.stress is not Stress.NORMAL:
        m *= cfg.stress_mult
    if market_state.event is MarketEvent.SHOCK:
        m *= cfg.shock_mult
    elif market_state.event is MarketEvent.POLICY:
        m *= cfg.policy_event_mult
    return m


def _apply_market_overlay(
    stats: _ForecastStats,
    market_state: MarketStateVector,
    cfg: HorizonForecasterConfig,
) -> _ForecastStats:
    """Widen ranges and raise drawdown risk under an adverse market state (never narrow).

    ``ret_mid`` and ``prob_positive`` stay on the pure historical estimate — the baseline
    does not fabricate a directional change from the market vector; it only expresses more
    caution (wider band, higher vol_est, higher drawdown_prob).
    """
    if not cfg.apply_market_overlay:
        return stats
    m = _risk_multiplier(market_state, cfg)
    if m <= 1.0:
        return stats

    stats.ret_low = stats.ret_mid - (stats.ret_mid - stats.ret_low) * m
    stats.ret_high = stats.ret_mid + (stats.ret_high - stats.ret_mid) * m
    stats.vol_est = stats.vol_est * m
    stats.drawdown_prob = min(
        1.0, stats.drawdown_prob + (m - 1.0) * cfg.overlay_drawdown_sensitivity
    )
    return stats


def _to_forecast(horizon: Horizon, stats: _ForecastStats) -> HorizonForecast:
    """Clamp probabilities/vol into their valid domains and build the validated model.

    ``backtested`` stays ``False``: the baseline is not yet proven to hold up in a
    time-ordered backtest (§7 ch:testing); it is set true only once the backtest gate
    passes.
    """
    ret_low = stats.ret_low
    ret_mid = stats.ret_mid
    ret_high = stats.ret_high
    # Numerical hygiene: guarantee ret_low <= ret_mid <= ret_high for the validator.
    ret_low = min(ret_low, ret_mid)
    ret_high = max(ret_high, ret_mid)
    return HorizonForecast(
        horizon=horizon,
        ret_low=ret_low,
        ret_mid=ret_mid,
        ret_high=ret_high,
        prob_positive=_clamp01(stats.prob_positive),
        vol_est=max(0.0, stats.vol_est),
        drawdown_prob=_clamp01(stats.drawdown_prob),
        backtested=False,
    )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
