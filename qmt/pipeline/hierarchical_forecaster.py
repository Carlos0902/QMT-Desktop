"""hierarchical_forecaster — the §5.3 分层模型 candidate (源文档 §3.3 / techdesign §5.3).

The doc's model-architecture update says individual-stock prediction should **not** train one
model per stock; instead a stock's forward return decomposes into three independent layers —

    个股预测结果 = 大盘影响 + 行业相对影响 + 个股自身相对影响

with **independent parameters** per layer and **graduated complexity** (market fuller, industry
medium, individual-stock kept deliberately simple because it is noisiest / most overfit-prone).
For fund analysis this is applied at the *fund* level via the estimated industry exposure, so the
fund's expected change can be **attributed** to market environment / industry exposure / a few
idiosyncratic stocks (the product-visible benefit, §5.3 designnote).

Fund-level aggregation (DECISIONS D-HF-1). Writing each holding's return as
``market + (industry − market) + (stock − industry)`` and summing over the fund's holdings with
weights ``w_i`` gives::

    fund_ret ≈ E · market  +  E · Σ_j w_j · (industry_j − market)  +  idiosyncratic

where ``E`` is the fund's **market loading** — the PIT rolling OLS beta of the fund's own NAV to
the benchmark (:func:`qmt.pipeline.exposure_estimator.estimate_market_beta`), with the disclosed
book-sum ``Σ_j w_j`` kept only as the fail-closed fallback (D-HF-7) — and ``w_j`` is the fund's
estimated within-book exposure to industry ``j``. Both the market and industry layers live inside
the equity book, so **both** scale by ``E``. The three layers are:

* **market** — the empirical forward-return distribution of the broad index (``index_daily``),
  scaled by ``E`` (fuller: ≈6-year lookback, D-HF-8);
* **industry (行业相对)** — per exposed industry, the empirical forward-return **deviation**
  ``industry_ret − market_ret`` of its Shenwan index (``sw_daily``), scaled by ``E·w_j`` (medium);
* **idiosyncratic (个股偏差)** — a small, conservative band with a **zero central estimate**
  (never a fabricated idiosyncratic alpha), calibrated at a reference horizon and scaled by
  ``sqrt(h/ref)`` (D-HF-9), widened only when exposure confidence falls below the calibration
  baseline. This IS the doc's "个股偏差模型要保持克制、尽量简单" — a calibrated band, *not* a fitted alpha.

It is a candidate【推断】 alongside the flat NAV-history baseline (:mod:`qmt.pipeline.horizon_forecaster`)
and is **not** the default (``CONFIG.forecaster.use_hierarchical`` is False); per "先基线、后候选、
严格验证" it must beat the baseline under time-ordered validation before replacing it — the
tuning/validation infra (:mod:`qmt.tuning`) can toggle ``use_hierarchical`` and compare.

Point-in-time correctness (invariant 1): every series — fund NAV, the market index, and each
industry index — is read only through :meth:`PointInTimeStore.get_asof` at ``as_of``. The
market-state risk overlay and the degraded / range-hygiene handling are **reused** from the
baseline forecaster (not duplicated), so both methods express caution identically.

Doc-silent decisions: DECISIONS D-HF-1..10.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from qmt.config import CONFIG, HierarchicalForecasterConfig
from qmt.data.sources import TABLE_INDEX_DAILY, TABLE_SW_DAILY
from qmt.pipeline.horizon_forecaster import (
    _apply_market_overlay,
    _guard_exposure_pit,
    _ForecastStats,
    _overlapping_windows,
    _to_forecast,
)
from qmt.pit import AVAILABLE_AT, PointInTimeStore
from qmt.types import (
    ExposureDimension,
    ExposureEstimate,
    ForecastAttribution,
    Horizon,
    HorizonForecast,
    IndustryContribution,
    MarketStateVector,
)

logger = logging.getLogger(__name__)

__all__ = [
    "forecast_hierarchical",
    "HierarchicalForecastResult",
    "SW_L1_INDEX_BY_NAME",
]

# Static Shenwan (申万) 2021 L1 industry-name → index ts_code reference, so the industry layer
# resolves the common exposure labels out of the box against the primed ``sw_daily`` series.
# Injecting ``industry_index_map`` overrides this; an unresolved label fails closed (D-HF-4).
SW_L1_INDEX_BY_NAME: dict[str, str] = {
    "农林牧渔": "801010.SI", "基础化工": "801030.SI", "钢铁": "801040.SI", "有色金属": "801050.SI",
    "电子": "801080.SI", "家用电器": "801110.SI", "食品饮料": "801120.SI", "纺织服饰": "801130.SI",
    "轻工制造": "801140.SI", "医药生物": "801150.SI", "公用事业": "801160.SI", "交通运输": "801170.SI",
    "房地产": "801180.SI", "商贸零售": "801200.SI", "社会服务": "801210.SI", "综合": "801230.SI",
    "建筑材料": "801710.SI", "建筑装饰": "801720.SI", "电力设备": "801730.SI", "国防军工": "801740.SI",
    "计算机": "801750.SI", "传媒": "801760.SI", "通信": "801770.SI", "银行": "801780.SI",
    "非银金融": "801790.SI", "汽车": "801880.SI", "机械设备": "801890.SI", "煤炭": "801950.SI",
    "石油石化": "801960.SI", "环保": "801970.SI", "美容护理": "801980.SI",
}


# =============================================================================
# Result container
# =============================================================================
@dataclass(frozen=True)
class HierarchicalForecastResult:
    """The hierarchical forecast: per-horizon combined ranges + per-horizon attribution.

    ``forecasts`` is drop-in comparable with the baseline ``forecast(...)`` output (same
    ``dict[Horizon, HorizonForecast]`` shape), so the pipeline and the evaluation harness treat
    it identically. ``attribution`` is the extra product-visible decomposition of each horizon's
    ``ret_mid`` into market / industry / idiosyncratic (§5.3).
    """

    forecasts: dict[Horizon, HorizonForecast]
    attribution: dict[Horizon, ForecastAttribution]


# =============================================================================
# per-layer distribution carrier
# =============================================================================
@dataclass
class _LayerDist:
    """A layer's forward-return distribution over one horizon (empirical or degraded)."""

    mid: float
    low: float
    high: float
    vol: float
    drawdown_prob: float
    measurable: bool

    @property
    def hw_down(self) -> float:
        return max(self.mid - self.low, 0.0)

    @property
    def hw_up(self) -> float:
        return max(self.high - self.mid, 0.0)


# =============================================================================
# point-in-time index/industry series (invariant 1)
# =============================================================================
def _index_closes(
    store: PointInTimeStore, ts_code: str, as_of: date, lookback: int
) -> tuple[list[date], np.ndarray]:
    """Return an index's ``(trade_dates, close)`` observable at ``as_of``, oldest→newest.

    Reads only ``available_at <= as_of`` (invariant 1). Restatements collapse to the latest
    value per ``trade_date``; non-positive / NaN closes are dropped; trimmed to ``lookback``.
    Works for both ``index_daily`` (broad market) and ``sw_daily`` (industry) — same schema.
    """
    for table in (TABLE_INDEX_DAILY, TABLE_SW_DAILY):
        df = store.get_asof(table, as_of, {"ts_code": ts_code})
        if df is not None and not df.empty and "close" in df.columns:
            break
    else:
        return [], np.empty(0, dtype=float)
    if "trade_date" not in df.columns:
        return [], np.empty(0, dtype=float)
    df = df.sort_values(["trade_date", AVAILABLE_AT]).drop_duplicates(subset="trade_date", keep="last")
    closes = pd.to_numeric(df["close"], errors="coerce")
    keep = closes.notna() & (closes > 0.0)
    dates = [pd.Timestamp(d).date() for d in df.loc[keep, "trade_date"]]
    vals = closes[keep].to_numpy(dtype=float)
    if lookback > 0:
        dates, vals = dates[-lookback:], vals[-lookback:]
    return dates, vals


def _ewma_vol(ret: np.ndarray, span: int) -> float:
    """EWMA standard deviation of daily returns (most recent weighted most). PIT: ``ret`` is the
    trailing return series ending at the last observable bar."""
    n = ret.size
    if n < 2:
        return float("nan")
    m = min(n, 5 * span)
    seg = ret[-m:]
    w = np.exp(-np.arange(m)[::-1] / float(span))
    sw = float(w.sum())
    mu = float((w * seg).sum() / sw)
    var = float((w * (seg - mu) ** 2).sum() / sw)
    return math.sqrt(max(var, 0.0))


def _vol_regime_multiplier(prices: np.ndarray, cfg: HierarchicalForecasterConfig) -> float:
    """PIT multiplier for the market band: ``clip(1 + gamma*(vol_recent/vol_baseline - 1))``.

    ``> 1`` in a high-vol regime (widen), ``< 1`` in calm (tighten). Uses only the supplied prices
    (read as-of the decision), so it cannot peek. Returns ``1.0`` when disabled or unmeasurable —
    fail-open to the static band, never a surprise widening/narrowing.
    """
    if not cfg.vol_conditioning_enabled or cfg.vol_conditioning_gamma <= 0.0:
        return 1.0
    p = np.asarray(prices, dtype=float)
    if p.size < cfg.vol_conditioning_slow_span + 5:
        return 1.0
    ret = np.diff(np.log(p))
    fast = _ewma_vol(ret, cfg.vol_conditioning_fast_span)
    slow = _ewma_vol(ret, cfg.vol_conditioning_slow_span)
    if not (math.isfinite(fast) and math.isfinite(slow)) or slow <= 0.0:
        return 1.0
    r = 1.0 + cfg.vol_conditioning_gamma * (fast / slow - 1.0)
    return float(min(max(r, cfg.vol_conditioning_clip_lo), cfg.vol_conditioning_clip_hi))


def _empirical_layer(
    prices: np.ndarray, horizon_len: int, q_low: float, q_high: float, min_windows: int
) -> _LayerDist:
    """Empirical forward-return distribution of a single series over ``horizon_len`` (reuses the
    baseline's overlapping-window machinery). ``measurable=False`` when too few windows."""
    from qmt.config import CONFIG as _CFG

    fwd, dd_hits = _overlapping_windows(prices, horizon_len, _CFG.forecaster)
    if fwd.size < min_windows:
        return _LayerDist(mid=0.0, low=0.0, high=0.0, vol=0.0, drawdown_prob=0.0, measurable=False)
    return _LayerDist(
        mid=float(np.quantile(fwd, 0.5)),
        low=float(np.quantile(fwd, q_low)),
        high=float(np.quantile(fwd, q_high)),
        vol=float(np.std(fwd, ddof=1)) if fwd.size > 1 else 0.0,
        drawdown_prob=float(np.mean(dd_hits)),
        measurable=True,
    )


def _deviation_layer(
    industry_prices: tuple[list[date], np.ndarray],
    market_prices: tuple[list[date], np.ndarray],
    horizon_len: int,
    cfg: HierarchicalForecasterConfig,
) -> _LayerDist:
    """Empirical distribution of the industry's forward return **minus** the market's, aligned by
    trade date over overlapping ``horizon_len`` windows (行业相对影响). ``measurable=False`` when
    too little aligned history — fail-closed (a 0 deviation, no fabricated relative strength)."""
    i_dates, i_vals = industry_prices
    m_dates, m_vals = market_prices
    if len(i_dates) < horizon_len + 1 or len(m_dates) < horizon_len + 1:
        return _LayerDist(0.0, 0.0, 0.0, 0.0, 0.0, measurable=False)
    # align on common trade dates
    m_by_date = {d: v for d, v in zip(m_dates, m_vals)}
    common = [(d, iv, m_by_date[d]) for d, iv in zip(i_dates, i_vals) if d in m_by_date]
    if len(common) < horizon_len + 1:
        return _LayerDist(0.0, 0.0, 0.0, 0.0, 0.0, measurable=False)
    iv = np.array([c[1] for c in common], dtype=float)
    mv = np.array([c[2] for c in common], dtype=float)
    n = iv.size
    i_fwd = iv[horizon_len:] / iv[: n - horizon_len] - 1.0
    m_fwd = mv[horizon_len:] / mv[: n - horizon_len] - 1.0
    dev = i_fwd - m_fwd
    if dev.size < cfg.industry_min_windows:
        return _LayerDist(0.0, 0.0, 0.0, 0.0, 0.0, measurable=False)
    return _LayerDist(
        mid=float(np.quantile(dev, 0.5)),
        low=float(np.quantile(dev, cfg.industry_ret_low_quantile)),
        high=float(np.quantile(dev, cfg.industry_ret_high_quantile)),
        vol=float(np.std(dev, ddof=1)) if dev.size > 1 else 0.0,
        drawdown_prob=0.0,  # drawdown is a level concept; the deviation only adjusts the center/spread
        measurable=True,
    )


# =============================================================================
# public entrypoint
# =============================================================================
def forecast_hierarchical(
    fund_code: str,
    as_of: date,
    exposure: list[ExposureEstimate],
    market_state: MarketStateVector,
    store: Optional[PointInTimeStore] = None,
    *,
    industry_index_map: Optional[Mapping[str, str]] = None,
    cfg: Optional[HierarchicalForecasterConfig] = None,
) -> HierarchicalForecastResult:
    """Per-horizon hierarchical forecast + attribution for ``fund_code`` at ``as_of`` (§5.3).

    Parameters mirror the baseline :func:`qmt.pipeline.horizon_forecaster.forecast`, plus:

    * ``industry_index_map`` — ``{industry_label -> sw_ts_code}``; overrides
      :data:`SW_L1_INDEX_BY_NAME`. An unresolved label contributes **zero** industry deviation
      (fail-closed) and feeds the idiosyncratic uncertainty instead (D-HF-4).
    * ``cfg`` — override :class:`HierarchicalForecasterConfig` (independent per-layer params).

    Point-in-time: fund NAV, the market index and every industry index are read only via
    ``get_asof(as_of)``. The market-state overlay + range hygiene are reused from the baseline.
    """
    _guard_exposure_pit(exposure, fund_code, as_of)
    hcfg = cfg or CONFIG.hierarchical
    src = store if store is not None else _default_store()
    imap = dict(industry_index_map) if industry_index_map is not None else SW_L1_INDEX_BY_NAME

    # --- fund's estimated industry exposure (行业暴露) — the aggregation weights ---
    # The MARKET layer scales by the fund's TOTAL equity exposure (all INDUSTRY estimates,
    # incl. the unclassified remainder — those holdings still carry market beta); only the top
    # material industries get an explicit行业相对 deviation, and the un-modeled remainder feeds
    # the idiosyncratic widening instead of silently vanishing (D-HF-1).
    industries = _industry_exposures(exposure, hcfg)
    # Within-disclosed-book share sum (≈1.0 by construction — the top-10 are normalised). Kept ONLY
    # as the fail-closed fallback and as the book-fraction reference for the idio widening below.
    book_exposure = min(
        sum(float(e.point_estimate) for e in exposure if e.dimension is ExposureDimension.INDUSTRY),
        1.0,
    )
    # The market LOADING is the PIT rolling beta of the fund's own NAV to the benchmark (真实暴露
    # tier-2: current effective sensitivity, not a stale quarterly snapshot). The disclosed route is
    # broken on real data — see estimate_market_beta's docstring — and left E pinned at 1.0, which
    # made "scale the market layer by equity exposure" inert and let cash drag leak into the
    # idiosyncratic residual. Fail-closed: no usable history → the old book-sum behaviour.
    from qmt.pipeline.exposure_estimator import estimate_market_beta

    beta = estimate_market_beta(
        fund_code, as_of, src,
        benchmark=hcfg.benchmark_index, window=hcfg.equity_beta_window_days,
        min_obs=hcfg.equity_beta_min_obs, clip_high=hcfg.equity_beta_clip_high,
    )
    equity_exposure = beta if beta is not None else book_exposure
    modeled_weight = sum(w for _lbl, w, _c in industries)
    unmodeled_weight = max(book_exposure - modeled_weight, 0.0)
    avg_conf = (
        float(np.mean([c for _lbl, _w, c in industries])) if industries else 0.0
    )
    # fraction of the book whose industry index could not be resolved → extra idiosyncratic noise
    resolved_labels = {lbl: imap.get(lbl) for lbl, _w, _c in industries}

    # --- market series (大盘) ---
    market_series = _index_closes(src, hcfg.benchmark_index, as_of, hcfg.market_lookback_trading_days)
    # per-industry series (行业), resolved once and reused across horizons
    industry_series = {
        lbl: _index_closes(src, code, as_of, hcfg.industry_lookback_trading_days)
        for lbl, code in resolved_labels.items()
        if code
    }

    forecasts: dict[Horizon, HorizonForecast] = {}
    attribution: dict[Horizon, ForecastAttribution] = {}
    for horizon in (Horizon.SHORT, Horizon.MID, Horizon.LONG):
        _, window_max = CONFIG.horizon.windows[horizon.value]
        horizon_len = max(int(window_max), 1)
        fc, attr = _forecast_one_hierarchical(
            horizon, horizon_len, market_series, industry_series, industries,
            equity_exposure, unmodeled_weight, avg_conf, market_state, hcfg,
        )
        forecasts[horizon] = fc
        attribution[horizon] = attr
    return HierarchicalForecastResult(forecasts=forecasts, attribution=attribution)


def _industry_exposures(
    exposure: list[ExposureEstimate], cfg: HierarchicalForecasterConfig
) -> list[tuple[str, float, float]]:
    """The fund's ``[(industry_label, exposure_weight, confidence)]`` from INDUSTRY estimates.

    Skips the ``unclassified`` sentinel (its weight is genuinely idiosyncratic/unknown), sorts by
    weight desc and caps at ``max_industries`` (the industry layer focuses on the material ones)."""
    from qmt.pipeline.exposure_estimator import UNCLASSIFIED_LABEL

    rows = [
        (e.label, float(e.point_estimate), float(e.confidence))
        for e in exposure
        if e.dimension is ExposureDimension.INDUSTRY and e.label != UNCLASSIFIED_LABEL and e.point_estimate > 0.0
    ]
    rows.sort(key=lambda r: -r[1])
    return rows[: cfg.max_industries]


def _forecast_one_hierarchical(
    horizon: Horizon,
    horizon_len: int,
    market_series: tuple[list[date], np.ndarray],
    industry_series: dict[str, tuple[list[date], np.ndarray]],
    industries: list[tuple[str, float, float]],
    equity_exposure: float,
    unmodeled_weight: float,
    avg_conf: float,
    market_state: MarketStateVector,
    cfg: HierarchicalForecasterConfig,
) -> tuple[HorizonForecast, ForecastAttribution]:
    """Combine the three layers additively for one horizon, then apply the shared overlay."""
    # --- layer 1: market (大盘), scaled by equity exposure ---
    m_prices = market_series[1]
    market = _empirical_layer(
        m_prices, horizon_len, cfg.market_ret_low_quantile, cfg.market_ret_high_quantile, cfg.market_min_windows
    )
    E = equity_exposure
    # De-bias the momentum-biased empirical median: shrink toward 0 (the band shifts with ret_mid).
    # Pure regularization — does not change rank-IC, cuts over-confident excursions (D-CVOL-3).
    market_mid = market.mid * cfg.market_median_shrink
    market_component = market_mid * E

    # --- layer 2: industry deviations (行业相对), exposure-weighted ---
    industry_component = 0.0
    ind_hw_down = ind_hw_up = ind_vol_sq = ind_vol_sum = 0.0
    contribs: list[IndustryContribution] = []
    unresolved_weight = 0.0
    for label, weight, _conf in industries:
        series = industry_series.get(label)
        dev = (
            _deviation_layer(series, market_series, horizon_len, cfg)
            if series is not None
            else _LayerDist(0.0, 0.0, 0.0, 0.0, 0.0, measurable=False)
        )
        resolved = dev.measurable
        if not resolved:
            unresolved_weight += weight
        # The deviation applies to the equity book, and the book's market loading is E — so the
        # fund-level industry loading is E*weight. Leaving it unscaled treats every fund as a
        # 100%-loaded clone of its top-10 industry mix.
        contribution = equity_exposure * weight * dev.mid
        industry_component += contribution
        ind_hw_down += equity_exposure * weight * dev.hw_down
        ind_hw_up += equity_exposure * weight * dev.hw_up
        ind_vol_sq += (equity_exposure * weight * dev.vol) ** 2
        ind_vol_sum += equity_exposure * weight * dev.vol
        contribs.append(
            IndustryContribution(
                label=label, exposure_weight=min(max(weight, 0.0), 1.0),
                deviation=dev.mid, contribution=contribution, resolved=resolved,
            )
        )

    # --- layer 3: idiosyncratic (个股偏差) — simple, restrained, ZERO central estimate ---
    # widen when exposure confidence is low OR part of the book is unmodeled/unresolvable at the
    # industry level (unresolved industry index + industries dropped past max_industries + the
    # unclassified remainder) — that weight's industry tilt is genuinely unknown → more idio noise.
    lowconf = 1.0 - min(max(avg_conf, 0.0), 1.0)
    idio_unknown_frac = min(max(unresolved_weight + unmodeled_weight, 0.0), 1.0)
    # The calibrated band already embeds the calibration sample's TYPICAL unknownness (the residual
    # it was fitted to contains those funds' real exposure error and unmodeled book). Widening for
    # the same unknownness again double-counts it — so scale only the EXCESS above the baseline,
    # renormalised so max unknownness (1.0) still reaches the full multiplier.
    b0 = min(max(cfg.idio_widen_baseline, 0.0), 0.999)
    excess = max(0.0, max(lowconf, idio_unknown_frac) - b0) / (1.0 - b0)
    idio_scale = 1.0 + (cfg.idio_low_confidence_mult - 1.0) * excess
    # The config band/vol are calibrated at idio_reference_horizon_days; residual dispersion grows
    # ~sqrt(h), so other horizons scale accordingly. A flat band measured 3.4x too wide at h=3
    # (drowning the signal) and 1.6x too narrow at h=90 (over-confident — the dangerous direction).
    hscale = math.sqrt(horizon_len / max(1, cfg.idio_reference_horizon_days))
    idio_band = cfg.idio_residual_band * idio_scale * hscale
    idio_vol = cfg.idio_residual_vol * idio_scale * hscale
    idio_component = 0.0  # never fabricate idiosyncratic alpha (§5.3: keep it restrained)

    # --- additive combination (D-HF-1/D-HF-3) ---
    ret_mid = market_component + industry_component + idio_component
    # Conditional-vol multiplier: widen/tighten the market band by the CURRENT vol regime (recent
    # realized vol vs a long baseline), PIT from the market prices. The empirical band adapts only
    # through its slow lookback; this makes it breathe. Validated D-CVOL-1.
    reff = _vol_regime_multiplier(m_prices, cfg)
    m_hw_down = market.hw_down * E * reff
    m_hw_up = market.hw_up * E * reff
    # The SAME regime multiplier must scale the market VOLATILITY too, not only the band half-widths
    # — otherwise vol_est / prob_positive / drawdown_prob (derived from `vol`) would describe the
    # unconditioned regime while the interval describes the conditioned one (an internal
    # contradiction a risk consumer would see). Keep the band and the density consistent.
    m_vol = market.vol * E * reff
    if cfg.combine_vol_independent:
        hw_down = math.sqrt(m_hw_down**2 + ind_hw_down**2 + idio_band**2)
        hw_up = math.sqrt(m_hw_up**2 + ind_hw_up**2 + idio_band**2)
        vol = math.sqrt(m_vol**2 + ind_vol_sq + idio_vol**2)
    else:
        hw_down = m_hw_down + ind_hw_down + idio_band
        hw_up = m_hw_up + ind_hw_up + idio_band
        vol = m_vol + ind_vol_sum + idio_vol

    # prob_positive / drawdown_prob from the combined Normal(ret_mid, vol) approx (D-HF-3).
    prob_positive = _normal_cdf(ret_mid / vol) if vol > 1e-9 else (1.0 if ret_mid > 0 else 0.0)
    threshold = CONFIG.forecaster.material_drawdown_threshold
    drawdown_prob = _normal_cdf((-threshold - ret_mid) / vol) if vol > 1e-9 else (1.0 if ret_mid <= -threshold else 0.0)
    # never optimistic below the market layer's own empirical drawdown risk (fail-closed floor)
    drawdown_prob = max(drawdown_prob, market.drawdown_prob * E)

    stats = _ForecastStats(
        ret_low=ret_mid - hw_down, ret_mid=ret_mid, ret_high=ret_mid + hw_up,
        prob_positive=prob_positive, vol_est=vol, drawdown_prob=drawdown_prob,
    )
    # reuse the baseline's one-directional market-state overlay (widens only; ret_mid unchanged)
    stats = _apply_market_overlay(stats, market_state, CONFIG.forecaster)
    forecast = _to_forecast(horizon, stats)

    attr = ForecastAttribution(
        horizon=horizon, market=market_component, industry=industry_component,
        idiosyncratic=idio_component, equity_exposure=max(E, 0.0), industries=contribs,
    )
    return forecast, attr


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF Φ(x) via the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _default_store() -> PointInTimeStore:
    from qmt.config import get_settings

    return PointInTimeStore(get_settings().data_cache_dir)
