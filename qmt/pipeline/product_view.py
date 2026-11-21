"""product_view — derive the product-facing view from the underlying numbers.

Stage that turns the numeric outputs of :mod:`qmt.pipeline.horizon_forecaster` (return
ranges), :mod:`qmt.pipeline.uncertainty` (calibrated reliability) and
:mod:`qmt.pipeline.market_state` (the 4-field environment vector) into a
:class:`~qmt.types.ProductView` whose ``direction`` / ``trend_strength`` /
``volatility`` / ``confidence`` are **DERIVED**, never assigned as subjective labels
(§4 sec:mod-forecast lst:forecast ``to_product_view``; §4 designnote «为什么先预测范围再转产品语言»).

The doc's ``to_product_view(fc: HorizonForecast) -> ProductView`` is realized here as
:func:`to_product_view` over the *full* multi-horizon set plus the calibrated
confidence and market state, matching the orchestrator's assembly contract (the
single-``HorizonForecast`` doc signature is a strict projection: it is what
:func:`_primary_forecast` selects). The direction thresholds live in
:class:`qmt.config.ProductViewConfig` (the exact doc口径: UP only when expected
return AND probability-of-positive are both clearly positive; DOWN symmetric; else
RANGE), so the labels are reproducible and calibration-checkable.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from qmt.config import CONFIG, ProductViewConfig
from qmt.types import (
    Direction,
    Horizon,
    HorizonForecast,
    MarketStateVector,
    ProductView,
    Volatility,
)

__all__ = ["to_product_view", "primary_forecast"]


# Order in which a single "headline" horizon is chosen from the multi-horizon set:
# the mid horizon (5–20d) leads the product view (matching the action layer), then the
# short horizon, then the long horizon.
_PRIMARY_HORIZON_ORDER = (Horizon.MID, Horizon.SHORT, Horizon.LONG)

# Escalation ranking for the "more cautious wins" volatility combine (fail-closed lean).
_VOL_RANK = {Volatility.LOW: 0, Volatility.NORMAL: 1, Volatility.HIGH: 2}
_RANK_VOL = {rank: vol for vol, rank in _VOL_RANK.items()}


def primary_forecast(
    forecasts: dict[Horizon, HorizonForecast],
) -> Optional[HorizonForecast]:
    """Select the headline horizon: MID → SHORT → LONG (then any), or ``None`` if empty.

    Accepts either :class:`Horizon` enum keys (the contract) or their string values so a
    caller that keyed by ``"mid"`` still resolves.
    """
    if not forecasts:
        return None
    for h in _PRIMARY_HORIZON_ORDER:
        if h in forecasts:
            return forecasts[h]
        if h.value in forecasts:  # type: ignore[operator]
            return forecasts[h.value]  # type: ignore[index]
    return next(iter(forecasts.values()))


def _derive_direction(fc: HorizonForecast, cfg: ProductViewConfig) -> Direction:
    """UP iff ret_mid AND prob_positive both clearly positive; DOWN symmetric; else RANGE.

    This is the doc口径 verbatim (§4 lst:forecast comment): direction is not "涨还是跌"
    but expected-return-and-probability jointly, so the label is derived from two numbers
    rather than a sign flip.
    """
    if fc.ret_mid > cfg.up_ret_mid_min and fc.prob_positive > cfg.up_prob_positive_min:
        return Direction.UP
    if fc.ret_mid < cfg.down_ret_mid_max and fc.prob_positive < cfg.down_prob_positive_max:
        return Direction.DOWN
    return Direction.RANGE


def _derive_trend_strength(fc: HorizonForecast, cfg: ProductViewConfig) -> float:
    """trend_strength ∈ [0,1] from return magnitude vs bumpiness (D-PV-1).

    ``|ret_mid| / (|ret_mid| + w·vol_est)`` — a large expected move relative to volatility
    is a strong trend; a small move under high volatility is weak ("涨幅不大但波动很大，
    不能说趋势强"). Zero denominator (flat, no vol) → 0.0.
    """
    magnitude = abs(fc.ret_mid)
    denom = magnitude + cfg.trend_strength_vol_weight * max(fc.vol_est, 0.0)
    if denom <= 0.0:
        return 0.0
    return _clamp01(magnitude / denom)


def _derive_volatility(
    fc: HorizonForecast, market_state: MarketStateVector, cfg: ProductViewConfig
) -> Volatility:
    """Volatility derived from the forecast band width, escalated by the environment (D-PV-2).

    The primary horizon's range width (``ret_high - ret_low``) measures how bumpy the path
    could be: ≥ ``vol_range_high_min`` → HIGH, ≤ ``vol_range_low_max`` → LOW, else NORMAL.
    We then take the **more cautious** of that and the market-state volatility, so a
    turbulent environment is never hidden even if the fund's own band looks calm
    (fail-closed lean, invariant 6). "即使方向偏乐观，高波动也不给激进建议."
    """
    width = fc.ret_high - fc.ret_low  # >= 0 by HorizonForecast validator
    if width >= cfg.vol_range_high_min:
        forecast_vol = Volatility.HIGH
    elif width <= cfg.vol_range_low_max:
        forecast_vol = Volatility.LOW
    else:
        forecast_vol = Volatility.NORMAL
    rank = max(_VOL_RANK[forecast_vol], _VOL_RANK[market_state.volatility])
    return _RANK_VOL[rank]


def to_product_view(
    forecasts: dict[Horizon, HorizonForecast],
    confidence: float,
    market_state: MarketStateVector,
    fund_code: str,
    as_of: date,
    cfg: Optional[ProductViewConfig] = None,
) -> ProductView:
    """Derive the product-facing :class:`ProductView` from the numeric forecast set.

    Every product-facing label is DERIVED from the underlying ranges (§4 designnote), so it
    can be traced back to numbers and calibrated:

    * ``direction`` — from the primary horizon's ``ret_mid`` + ``prob_positive`` vs the
      :class:`ProductViewConfig` thresholds (UP/DOWN/RANGE).
    * ``trend_strength`` — from return magnitude vs bumpiness (D-PV-1).
    * ``volatility`` — from the forecast band width, escalated by the market environment
      (D-PV-2, fail-closed lean).
    * ``confidence`` — the already historically-calibrated reliability, passed through
      (clamped to [0,1]); calibration itself happens upstream (uncertainty stage).

    ``action`` is left ``None`` — it is populated only in advice_mode (§3 sec:ds-action),
    never here (analysis_mode gives no personalized advice).

    Fail-closed (invariant 6): an empty forecast set means we have nothing to derive a
    judgment from, so the view is a most-cautious RANGE / zero-trend / (environment)
    volatility with confidence floored — never a fabricated optimistic label.
    """
    pcfg = cfg or CONFIG.product_view
    conf = _clamp01(confidence)

    fc = primary_forecast(forecasts)
    if fc is None:
        return ProductView(
            fund_code=fund_code,
            as_of=as_of,
            direction=Direction.RANGE,
            trend_strength=0.0,
            volatility=market_state.volatility,
            confidence=min(conf, CONFIG.confidence_decay.floor),
            action=None,
        )

    return ProductView(
        fund_code=fund_code,
        as_of=as_of,
        direction=_derive_direction(fc, pcfg),
        trend_strength=_derive_trend_strength(fc, pcfg),
        volatility=_derive_volatility(fc, market_state, pcfg),
        confidence=conf,
        action=None,
    )


def _clamp01(x: float) -> float:
    """Clamp to the closed unit interval."""
    if x != x:  # NaN → fail-closed to 0
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
