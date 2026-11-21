"""uncertainty — reliability calibration of the pipeline's judgments (§4 sec:mod-uncertainty, §1.8-6 / §3.3).

Responsibility (doc): tell the user *how reliable* a judgment is. ``confidence`` must
never be a free score the model happens to emit — it must be **historically
calibrated**: when the system said "high confidence" in the past, similar judgments
must actually have been right more often. This module is pipeline stage 6 (uncertainty),
consuming the multi-horizon :class:`~qmt.types.HorizonForecast` set plus the current
exposure/market context and feeding a *calibrated* confidence into the product view and
the action gate.

Two steps, kept separate so each is testable and the calibration is auditable:

1. :func:`assess_uncertainty` — produce a **raw** confidence in ``[0, 1]`` by combining
   the three reliability signals the doc names: exposure confidence (how well we know
   the fund's holdings), forecast dispersion (how wide the predicted return ranges are),
   and market volatility / stress (how bumpy and tense the environment is). This is a
   score, not yet a calibrated probability.

2. :func:`calibrate` — replace that raw score with the **historical hit-rate** of
   similar past judgments, looked up from a :class:`~qmt.types.CalibrationTable` bucketed
   into ``CONFIG.calibration.bucket_count`` equal-width buckets. A bucket is only trusted
   once it has ``min_samples_per_bucket`` observations; below that we fall back to a
   conservative lower bound (fail-closed, invariant 6), so an uncalibrated system reports
   *low* confidence rather than a confidently wrong one.

Doc interface (04-modules.tex, lst:uncert)::

    def calibrate_confidence(forecast: HorizonForecast,
                             history: CalibrationTable) -> float:
        \"\"\"基于历史上'类似把握程度'的判断实际可靠率，返回校准后的 confidence。
        并结合不同方法之间是否一致。\"\"\"

:func:`calibrate_confidence` realizes that exact signature as the single-forecast
convenience (assess-then-calibrate). The richer :func:`assess_uncertainty` entrypoint
takes the full horizon set plus exposure/market context.

Baseline-first (§5, sec:algo-table row \\UncertaintyM): the baseline is *hit-rate
bucketing* (this module). The candidate direction — checking whether independent methods
agree and shrinking when they disagree ("结合不同方法之间是否一致") — is left as a
documented, inert hook (:pyattr:`assess_uncertainty`'s ``method_agreement`` parameter),
not built here.

Doc-silent constants (recorded in DECISIONS.md): the raw-confidence combination weights
(D57) and the concrete "conservative lower bound" for insufficient samples (D58, which
realizes D07's phrase). The bucketing scheme itself is D07.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

from qmt.config import CONFIG, CalibrationConfig
from qmt.types import (
    CalibrationTable,
    ConfidenceBucket,
    ExposureEstimate,
    Horizon,
    HorizonForecast,
    MarketEvent,
    MarketStateVector,
    Stress,
    Volatility,
)

__all__ = [
    "UncertaintyWeights",
    "DEFAULT_WEIGHTS",
    "assess_uncertainty",
    "build_calibration_table",
    "calibrate",
    "calibrate_confidence",
]


def _clamp01(x: float) -> float:
    """Clamp to the closed unit interval (defensive; all inputs are already there)."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class UncertaintyWeights(BaseModel):
    """Doc-silent reliability weights for :func:`assess_uncertainty` (DECISIONS.md D57).

    The doc (§4 sec:mod-uncertainty, §3.3) requires the raw confidence to *fall* with
    exposure uncertainty, wider forecast dispersion, and rougher markets, but gives no
    functional form. Baseline (simple, transparent, monotone): a **product of three
    reliability factors**, each in ``[0, 1]``::

        raw = exposure_factor * dispersion_factor * market_factor

    * ``exposure_factor``  = mean of the exposure estimates' own ``confidence`` (so this
      module inherits the "confidence falls as disclosure_age rises" invariant from
      :func:`qmt.config.decayed_confidence` through the exposure layer, invariant 2/c).
    * ``dispersion_factor`` = ``exp(-avg_return_range_width / dispersion_scale)`` — wider
      predicted return ranges (``ret_high - ret_low``) mean a less reliable call.
    * ``market_factor``    = ``volatility_reliability * stress_reliability *
      event_reliability`` — high volatility, tight/rising stress, or a policy/shock event
      each discount reliability.

    Kept in this module (not in the frozen ``qmt.config``) so downstream slices can tune
    or inject it via the ``weights`` parameter without touching the frozen contract; the
    numbers are candidates pending the §7 calibration backtest.
    """

    model_config = {"frozen": True}

    dispersion_scale: float = Field(
        0.10,
        gt=0.0,
        description="Return-range width (ret_high-ret_low) that scales the dispersion decay.",
    )
    volatility_reliability: Mapping[Volatility, float] = Field(
        default_factory=lambda: {
            Volatility.LOW: 1.0,
            Volatility.NORMAL: 0.85,
            Volatility.HIGH: 0.60,
        },
        description="Reliability multiplier per market volatility level.",
    )
    stress_reliability: Mapping[Stress, float] = Field(
        default_factory=lambda: {
            Stress.NORMAL: 1.0,
            Stress.LIQUIDITY_TIGHT: 0.75,
            Stress.RISING_RISK: 0.70,
        },
        description="Reliability multiplier per market stress state.",
    )
    event_reliability: Mapping[MarketEvent, float] = Field(
        default_factory=lambda: {
            MarketEvent.NONE: 1.0,
            MarketEvent.POLICY: 0.85,
            MarketEvent.SHOCK: 0.60,
        },
        description="Reliability multiplier per market event.",
    )


# Process-wide default weights; injectable per call via ``weights=``.
DEFAULT_WEIGHTS = UncertaintyWeights()


def _dispersion_factor(forecast: HorizonForecast, weights: UncertaintyWeights) -> float:
    """Reliability from one forecast's return-range width: wider range -> lower factor."""
    width = forecast.ret_high - forecast.ret_low  # >= 0 by HorizonForecast validator
    return math.exp(-width / weights.dispersion_scale)


def _forecasts_dispersion_factor(
    forecasts: Mapping[Horizon, HorizonForecast], weights: UncertaintyWeights
) -> float:
    """Average dispersion factor across the supplied horizons."""
    factors = [_dispersion_factor(f, weights) for f in forecasts.values()]
    return sum(factors) / len(factors)


def _exposure_factor(exposure: list[ExposureEstimate], floor: float) -> float:
    """Reliability from how well we know the holdings: mean exposure confidence.

    Empty exposure means we have *no* holdings signal -> fail-closed to ``floor``
    (invariant 6) rather than assuming full knowledge.
    """
    if not exposure:
        return floor
    return sum(e.confidence for e in exposure) / len(exposure)


def _market_factor(
    market_state: MarketStateVector, weights: UncertaintyWeights
) -> float:
    """Reliability from the environment: rougher/tenser markets discount confidence."""
    vol = weights.volatility_reliability.get(market_state.volatility, 1.0)
    stress = weights.stress_reliability.get(market_state.stress, 1.0)
    event = weights.event_reliability.get(market_state.event, 1.0)
    return vol * stress * event


def assess_uncertainty(
    forecasts: dict[Horizon, HorizonForecast],
    exposure: list[ExposureEstimate],
    market_state: MarketStateVector,
    *,
    weights: Optional[UncertaintyWeights] = None,
    method_agreement: float = 1.0,
) -> float:
    """Raw confidence in ``[0, 1]`` combining the three reliability signals (§4, D57).

    Returns a **raw** score (not yet historically calibrated) as the product of an
    exposure factor, a forecast-dispersion factor, and a market factor — see
    :class:`UncertaintyWeights`. The raw score must still be passed through
    :func:`calibrate` before it is reported as a confidence, because the doc requires
    confidence to be *historically calibrated*, not a free score.

    Args:
        forecasts: multi-horizon forecast set; wider return ranges lower confidence.
        exposure: current exposure estimates; their own ``confidence`` (which decays with
            ``disclosure_age_days``) drives the exposure factor.
        market_state: the 4-field market vector; high volatility / stress / events reduce
            reliability.
        weights: optional override of the doc-silent combination weights (D57).
        method_agreement: documented **candidate hook** (§5 "结合不同方法之间是否一致").
            ``1.0`` (default) is inert; a lower value shrinks the raw score toward 0 to
            reflect disagreement between independent methods. The multi-method estimator
            is not built in this baseline slice.

    Fail-closed: with no forecasts we cannot judge dispersion, so we return the
    conservative floor (``CONFIG.confidence_decay.floor``) rather than a high default.
    """
    w = weights or DEFAULT_WEIGHTS
    floor = CONFIG.confidence_decay.floor

    if not forecasts:
        return floor
    if not 0.0 <= method_agreement <= 1.0:
        raise ValueError("method_agreement must be in [0, 1]")

    exposure_factor = _exposure_factor(exposure, floor)
    dispersion_factor = _forecasts_dispersion_factor(forecasts, w)
    market_factor = _market_factor(market_state, w)

    raw = exposure_factor * dispersion_factor * market_factor * method_agreement
    return _clamp01(raw)


def _bucket_index(raw: float, bucket_count: int) -> int:
    """Index of the equal-width bucket over [0, 1] that ``raw`` falls in (top-inclusive)."""
    r = _clamp01(raw)
    return min(int(r * bucket_count), bucket_count - 1)


def build_calibration_table(
    observations: Iterable[Tuple[float, bool]],
    *,
    cfg: Optional[CalibrationConfig] = None,
) -> CalibrationTable:
    """Bucket historical ``(raw_confidence, was_correct)`` pairs into a calibration table (D07).

    Partitions ``[0, 1]`` into ``cfg.bucket_count`` equal-width buckets and, for each,
    records the empirical hit-rate (fraction of observations that were actually right)
    and the sample count. This is exactly the baseline the doc names — "按历史命中率分桶
    给 confidence" (§5, row \\UncertaintyM) — turning past outcomes into the lookup
    :func:`calibrate` uses.

    Args:
        observations: iterable of ``(raw_confidence, was_correct)``; ``raw_confidence`` is
            clamped into ``[0, 1]``.
        cfg: calibration config (bucket count); defaults to ``CONFIG.calibration``.
    """
    c = cfg or CONFIG.calibration
    n = c.bucket_count
    counts = [0] * n
    hits = [0] * n

    for raw, correct in observations:
        idx = _bucket_index(raw, n)
        counts[idx] += 1
        if correct:
            hits[idx] += 1

    buckets: list[ConfidenceBucket] = []
    for i in range(n):
        cnt = counts[i]
        hit_rate = (hits[i] / cnt) if cnt > 0 else 0.0
        buckets.append(
            ConfidenceBucket(
                lower=i / n,
                upper=(i + 1) / n,
                hit_rate=hit_rate,
                sample_count=cnt,
            )
        )
    return CalibrationTable(buckets=buckets)


def _covering_bucket(
    table: CalibrationTable, raw: float
) -> Optional[ConfidenceBucket]:
    """First bucket whose ``[lower, upper]`` covers ``raw`` (matches ``hit_rate_for``)."""
    for b in table.buckets:
        if b.lower <= raw <= b.upper:
            return b
    return None


def calibrate(
    raw: float,
    table: CalibrationTable,
    *,
    cfg: Optional[CalibrationConfig] = None,
    floor: Optional[float] = None,
) -> float:
    """Map a raw confidence to its historically-calibrated value via ``table`` (D07/D58).

    The calibrated value is the **empirical hit-rate** of the bucket ``raw`` falls in —
    i.e. how often judgments of this stated confidence were actually right — so a system
    that says "0.8" but was historically right only 40% of the time reports ``0.40``.

    Trust rule (fail-closed, D58 — the concrete form of D07's "conservative lower bound"):

    * bucket with ``sample_count >= cfg.min_samples_per_bucket`` -> return its empirical
      ``hit_rate`` (calibration we can trust);
    * bucket with too few samples -> **shrink toward the floor** by sample weight,
      ``w * hit_rate + (1 - w) * floor`` with ``w = sample_count / min_samples`` — a
      thin bucket cannot vouch for a high score, so confidence degrades gracefully;
    * no covering bucket (empty / uncovered table) -> return ``floor`` outright.

    ``floor`` defaults to the system-wide confidence floor ``CONFIG.confidence_decay.floor``
    (0.1); below the action gate's ``min_confidence`` (0.5), so an uncalibrated system
    degrades to "等待新信号" instead of emitting a strong signal.

    A statistically principled lower bound (e.g. a Wilson-score interval) is the natural
    candidate replacement for the linear shrinkage; left for the §7 calibration work.
    """
    c = cfg or CONFIG.calibration
    fl = CONFIG.confidence_decay.floor if floor is None else floor
    r = _clamp01(raw)

    bucket = _covering_bucket(table, r)
    if bucket is None:
        return fl
    if bucket.sample_count >= c.min_samples_per_bucket:
        return _clamp01(bucket.hit_rate)

    w = bucket.sample_count / c.min_samples_per_bucket
    return _clamp01(w * bucket.hit_rate + (1.0 - w) * fl)


def calibrate_confidence(
    forecast: HorizonForecast,
    history: CalibrationTable,
    *,
    weights: Optional[UncertaintyWeights] = None,
    cfg: Optional[CalibrationConfig] = None,
    floor: Optional[float] = None,
) -> float:
    """Doc signature (lst:uncert): calibrated confidence for a **single** forecast.

    Convenience wrapper that scores one forecast by its dispersion and then calibrates it
    against ``history``. The full-context entrypoint (all horizons + exposure + market) is
    :func:`assess_uncertainty`; use this when only one :class:`HorizonForecast` is in hand.
    """
    w = weights or DEFAULT_WEIGHTS
    raw = _dispersion_factor(forecast, w)
    return calibrate(raw, history, cfg=cfg, floor=floor)
