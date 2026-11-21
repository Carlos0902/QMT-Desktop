"""Central configuration for QMT.

Two kinds of configuration live here:

1. ``Settings`` — runtime/secret configuration loaded from the environment / ``.env``
   (the Tushare token, the data-cache directory). Loaded via ``pydantic-settings``.

2. Grouped *calibration constants* — every load-bearing threshold the technical
   design doc (techdesign/chapters/*.tex) leaves **unspecified**. The doc states the
   principle ("confidence decreases as disclosure age rises", "degrade to 等待新信号
   on low confidence"); the concrete numbers are supplied here so they are
   centralized, typed, documented, and tunable without touching business logic.
   Every constant below has a corresponding row in ``DECISIONS.md``.

Nothing in this module hard-codes a secret: ``TUSHARE_TOKEN`` is read from the
environment / ``.env`` only.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from qmt.types import (
    AdviceStrengthCap,
    FundTypeProfile,
    ScopeCategory,
    SupportTier,
)

# Repository root = parent of the ``qmt`` package directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Runtime settings (secrets + paths) — loaded from env / .env
# =============================================================================
class Settings(BaseSettings):
    """Runtime settings. ``TUSHARE_TOKEN`` comes from the environment or ``.env``.

    The token is NEVER hard-coded in source; it lives only in the gitignored
    ``.env`` file (see ``.env.example``).
    """

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    tushare_token: str = Field(
        default="",
        description="Tushare Pro API token. Loaded from env var TUSHARE_TOKEN / .env.",
        alias="TUSHARE_TOKEN",
    )

    # Point-in-time / Parquet / DuckDB data cache. Gitignored (/data_cache/).
    data_cache_dir: Path = Field(
        default=_REPO_ROOT / "data_cache",
        description="Directory for the Parquet/DuckDB point-in-time store and pulled data.",
        alias="QMT_DATA_CACHE_DIR",
    )

    def ensure_data_cache(self) -> Path:
        """Create the data-cache directory if missing and return it."""
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.data_cache_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached ``Settings`` instance."""
    return Settings()


# =============================================================================
# 2. Calibration constants (doc-silent thresholds)
#    Grouped by module. Each group maps to a §/section of the design doc and to a
#    row in DECISIONS.md. Instances are frozen so they read as named constants.
# =============================================================================


class DecisionTimingConfig(BaseModel):
    """Decision timeline constants (§3.1 / §6 sec:cc-pit).

    The doc cites both 14:30 (risknote) and 15:00 (itemize) without reconciling.
    Canonical decision cutoff = T-day 14:30; 15:00 is treated as the separate EOD
    close boundary (T-day close data becomes visible only from T+1).
    """

    model_config = {"frozen": True}

    decision_hour: int = Field(14, description="T-day decision time hour (14:30).")
    decision_minute: int = Field(30, description="T-day decision time minute (14:30).")
    eod_close_hour: int = Field(
        15, description="EOD market close hour; T-day close data visible only from T+1."
    )
    eod_close_minute: int = Field(0, description="EOD market close minute.")
    # Per-fund overrides (e.g. ETF special cutoffs) are keyed by fund_code.
    # Empty by default; populated later. Value = (hour, minute).
    per_fund_cutoff_override: dict[str, tuple[int, int]] = Field(default_factory=dict)


class FreshnessConfig(BaseModel):
    """Freshness (fresh/stale/expired) day cutoffs for data_visibility (§4 sec:mod-visibility).

    The doc requires a freshness verdict but gives no day boundaries. Cutoffs are
    keyed by the kind of data because disclosure cadence differs wildly (daily NAV
    vs quarterly holdings). Boundaries are inclusive on the lower state:
    ``fresh`` iff age <= fresh_max; ``stale`` iff age <= stale_max; else ``expired``.
    """

    model_config = {"frozen": True}

    # Daily NAV / index series (age measured in trading days).
    nav_fresh_max_days: int = Field(3, description="NAV fresh iff age <= 3 trading days.")
    nav_stale_max_days: int = Field(10, description="NAV stale iff age <= 10 trading days.")

    # Quarterly disclosed holdings (age measured in calendar days; aligns with the
    # 15/60/90-day disclosure lags for Q top-10 / semi-annual / annual reports).
    holdings_fresh_max_days: int = Field(45, description="Holdings fresh iff age <= 45 days.")
    holdings_stale_max_days: int = Field(100, description="Holdings stale iff age <= 100 days.")

    # Fallback for any other source without a specific rule.
    default_fresh_max_days: int = Field(7)
    default_stale_max_days: int = Field(30)


class ConfidenceDecayConfig(BaseModel):
    """Confidence-vs-disclosure_age decay (§6 sec:cc-exposure).

    The doc requires only that confidence decrease **monotonically** as
    disclosure_age rises. Baseline functional form: exponential decay
    ``confidence = base * exp(-age / tau)`` clamped to ``floor``.
    """

    model_config = {"frozen": True}

    base: float = Field(1.0, ge=0.0, le=1.0, description="Confidence at age 0.")
    tau_days: float = Field(45.0, gt=0.0, description="Exponential decay scale (≈ one quarter).")
    floor: float = Field(0.1, ge=0.0, le=1.0, description="Lower clamp; confidence never below this.")


class ExposureConfig(BaseModel):
    """True-exposure estimation (§4 sec:mod-expo, §5).

    Returns-based regression of adj_nav returns on industry + style indices.
    Constraints (weights>=0, sum<=1) and the taxonomy are baseline candidates
    pending backtest.
    """

    model_config = {"frozen": True}

    regression_window_trading_days: int = Field(
        60, description="Rolling window (trading days) for the NAV-vs-index regression."
    )
    weights_nonneg: bool = Field(True, description="Constrain regression weights >= 0.")
    weights_sum_max: float = Field(1.0, description="Constrain sum(weights) <= 1 (residual = cash/other).")
    # Baseline low/high band half-width (relative to point_estimate). The doc requires a
    # range (not a label) and that stale disclosures be less certain, but gives no width.
    # Half-width grows linearly with drift_prob: rel = band_min_rel at drift 0 (a
    # just-disclosed snapshot) up to band_max_rel at drift 1 (a very stale one). Calibrated
    # so the doc's worked example (semiconductor 42%, disclosed 68 days ago) reproduces
    # 34%–50%. See DECISIONS.md D-EXP-1. (§3 sec:ds-exposure, §6 sec:cc-exposure)
    band_min_rel: float = Field(
        0.0, ge=0.0, description="Relative low/high half-width at drift_prob=0 (fresh disclosure)."
    )
    band_max_rel: float = Field(
        0.25, ge=0.0, description="Relative low/high half-width at drift_prob=1 (very stale)."
    )
    # Disclosure-lag constants (calendar days) used to set disclosure_age_days context.
    q_report_lag_days: int = Field(15, description="Q1/Q3 top-10 disclosure lag ≈ 15 days.")
    semiannual_report_lag_days: int = Field(60, description="Semi-annual full-holdings lag ≈ 60 days.")
    annual_report_lag_days: int = Field(90, description="Annual full-holdings lag ≈ 90 days.")


class MarketStateConfig(BaseModel):
    """Market-state baseline-rule thresholds (§4 sec:mod-market, §5).

    Direction from a fast/slow MA cross; volatility from historical percentiles of
    rolling realized vol; stress from turnover/valuation percentiles.
    """

    model_config = {"frozen": True}

    ma_fast_days: int = Field(20, description="Fast moving-average window (trading days).")
    ma_slow_days: int = Field(60, description="Slow moving-average window (trading days).")
    direction_band_pct: float = Field(
        0.005,
        description=(
            "Neutral band around the MA cross (relative gap). |ma_fast-ma_slow|/ma_slow "
            "within this band → RANGE, so RANGE is reachable instead of a pure up/down "
            "flip (doc-silent, D46)."
        ),
    )
    realized_vol_window_days: int = Field(20, description="Rolling realized-vol window.")
    vol_low_percentile: float = Field(33.0, description="Below this historical pct → low vol.")
    vol_high_percentile: float = Field(67.0, description="Above this historical pct → high vol.")
    stress_volume_window_days: int = Field(
        5,
        description=(
            "Short window (trading days) for smoothing recent turnover before comparing "
            "it against the historical turnover distribution (doc-silent, D46)."
        ),
    )
    turnover_stress_percentile: float = Field(
        20.0, description="Turnover below this pct → liquidity-tight stress."
    )
    valuation_stress_percentile: float = Field(
        80.0, description="PE/PB above this pct → rising-risk stress."
    )
    # RISING_RISK proxy (doc-silent, D-MS-1). The doc's true rising-risk signal is
    # valuation (PE/PB high percentile) or breadth (advance/decline) — neither is in the
    # data layer yet — so the baseline uses a transparent broad-index proxy: the index is
    # in a drawdown of at least this fraction below its trailing observable peak AND
    # realized volatility is in its HIGH band → risk rising.
    drawdown_stress_pct: float = Field(
        0.10,
        gt=0.0,
        description="Broad-index drawdown-from-peak ≥ this AND HIGH vol → RISING_RISK (proxy, D-MS-1).",
    )
    event_active_days: int = Field(
        5,
        description=(
            "A fed policy/shock flag stays active for this many calendar days after its "
            "available_at; older flags decay to event=NONE (doc-silent, D46)."
        ),
    )


class HorizonConfig(BaseModel):
    """Multi-horizon boundaries (§3 sec:ds-forecast, §4 sec:mod-forecast).

    Trading-day windows. The day-4 gap in the doc's 1-3 / 5-20 / 20-90 layout folds
    into ``mid``. Keys ``short`` / ``mid`` / ``long`` are standardized across modules.
    """

    model_config = {"frozen": True}

    short_min_days: int = Field(1)
    short_max_days: int = Field(3)
    mid_min_days: int = Field(5)  # day-4 folds into mid (mid effectively starts at 4)
    mid_max_days: int = Field(20)
    long_min_days: int = Field(20)
    long_max_days: int = Field(90)

    @property
    def windows(self) -> dict[str, tuple[int, int]]:
        return {
            "short": (self.short_min_days, self.short_max_days),
            "mid": (self.mid_min_days, self.mid_max_days),
            "long": (self.long_min_days, self.long_max_days),
        }


class HorizonForecasterConfig(BaseModel):
    """horizon_forecaster baseline constants (§2.2 / §3.3, §5 sec:algo-table, module 5).

    The doc's §5 baseline for the forecaster is *simple historical regularities*: the
    empirical distribution of the fund's own overlapping forward returns, computed per
    horizon from the ``adj_nav`` series visible at ``as_of``. Everything below is the
    load-bearing calibration the doc leaves unspecified (quantile levels, lookback,
    minimum sample, the material-drawdown threshold, the fail-closed degraded outputs,
    and the conservative market-state risk overlay). The richer candidate (regularized
    regression / gradient boosting conditioned on estimated exposure, §5) is a
    documented hook — not built in the baseline.
    """

    model_config = {"frozen": True}

    # --- empirical return-distribution shape (ranges, not labels) -----------
    ret_low_quantile: float = Field(
        0.10, ge=0.0, le=1.0, description="ret_low = 10th percentile of forward returns (较差情形)."
    )
    ret_high_quantile: float = Field(
        0.90, ge=0.0, le=1.0, description="ret_high = 90th percentile of forward returns (最乐观情形)."
    )
    lookback_trading_days: int = Field(
        500, gt=0, description="Most-recent visible NAV points used (≈2 trading years)."
    )
    min_windows: int = Field(
        20, gt=0, description="Fewer than this many overlapping windows → fail-closed degraded forecast."
    )
    material_drawdown_threshold: float = Field(
        0.10,
        gt=0.0,
        description="'明显回撤': worst drop-from-entry within the window <= -10% counts as a drawdown hit.",
    )

    # --- fail-closed degraded forecast (insufficient visible history) -------
    degraded_ret_band: float = Field(
        0.15, ge=0.0, description="Symmetric +/-band for the 'we don't know' ret_low/high when data is thin."
    )
    degraded_vol_est: float = Field(0.15, ge=0.0, description="Conservative vol_est under insufficient history.")
    degraded_drawdown_prob: float = Field(
        0.5, ge=0.0, le=1.0, description="Conservative drawdown_prob under insufficient history."
    )

    # --- conservative market-state risk overlay (§5 顺风/逆风/乱流 gating) ---
    # One-directional: only widens ranges / raises risk; never narrows or inflates
    # optimism (fail-closed; 'even if direction is optimistic, high vol -> no aggressive
    # read'). ret_mid and prob_positive stay on the pure historical estimate.
    apply_market_overlay: bool = Field(
        True, description="Apply the conservative market-state risk overlay to the baseline."
    )
    high_vol_mult: float = Field(1.30, ge=1.0, description="Range/vol multiplier when volatility=HIGH.")
    stress_mult: float = Field(1.20, ge=1.0, description="Multiplier when stress != normal (liquidity/rising-risk).")
    shock_mult: float = Field(1.30, ge=1.0, description="Multiplier when event=shock.")
    policy_event_mult: float = Field(1.10, ge=1.0, description="Multiplier when event=policy.")
    overlay_drawdown_sensitivity: float = Field(
        0.50, ge=0.0, description="drawdown_prob += (risk_mult - 1) x this (clamped to 1)."
    )

    # --- method switch: baseline (flat NAV history) vs the hierarchical candidate (§5.3) ---
    # The §5.3 分层模型 is a candidate【推断】 alongside the baseline; per "先基线、后候选、
    # 严格验证" it is NOT the default. Flipping this to True routes forecast() through
    # hierarchical_forecaster; a boolean so the tuning/validation infra can toggle and compare
    # it against the baseline (DECISIONS D-HF-5).
    use_hierarchical: bool = Field(
        False, description="Use the §5.3 hierarchical (market+industry+idiosyncratic) forecaster instead of the flat baseline."
    )


class HierarchicalForecasterConfig(BaseModel):
    """Hierarchical individual-stock forecaster (源文档 §3.3 / techdesign §5.3 分层模型).

    Realizes the doc's three-layer decomposition — a stock's forward return is
    ``大盘影响 + 行业相对影响 + 个股自身相对影响`` — aggregated to the fund level via the
    estimated industry exposure (\\ExposureEstimate). The three layers are **independent
    tasks** with independent parameters (each prefixed ``market_`` / ``industry_`` / ``idio_``),
    and complexity is **graduated**: the market layer is fuller (longest lookback), the
    industry layer medium, and the individual-stock (idiosyncratic) layer is kept deliberately
    simple — a small conservative residual band, *not* a fitted per-stock model — because that
    layer is noisiest and most overfit-prone (D-HF-2/D-HF-6). All doc-silent; see DECISIONS D-HF-*.
    """

    model_config = {"frozen": True}

    # broad-market ("大盘") series whose empirical forward-return distribution is the market layer.
    benchmark_index: str = Field("000300.SH", description="Broad-market index ts_code for the 大盘 layer.")

    # --- market layer (fuller: longest lookback) ---
    # Chosen 2026-07 by time-ordered validation (selection <= 2022, holdout 2023+, pre-registered
    # rule): candidates {750, 1500, 3000, full}. No candidate separates on CRPS (all overlap-
    # corrected t's insignificant — the window earns no SKILL claim), but interval COVERAGE does:
    # 750 delivered 0.722 at h=90 on the ≤2022 selection window vs nominal 0.80 (over-confident), while
    # 1500 was closest to nominal at every horizon on BOTH windows. ~6y spans a full regime cycle;
    # full history over-weights the ancient regime and over-covers. See D-HF-8.
    market_lookback_trading_days: int = Field(
        1500, ge=0, description="Market-layer lookback in trading days (≈6 trading years); 0 = 全部可见历史."
    )
    market_ret_low_quantile: float = Field(0.10, ge=0.0, le=1.0)
    market_ret_high_quantile: float = Field(0.90, ge=0.0, le=1.0)
    market_min_windows: int = Field(20, gt=0, description="Min overlapping windows before the market layer is trusted.")

    # --- industry-deviation layer (medium complexity) ---
    industry_lookback_trading_days: int = Field(
        500, gt=0, description="Industry-layer lookback (≈2 trading years — medium)."
    )
    industry_ret_low_quantile: float = Field(0.10, ge=0.0, le=1.0)
    industry_ret_high_quantile: float = Field(0.90, ge=0.0, le=1.0)
    industry_min_windows: int = Field(20, gt=0, description="Min overlapping windows before an industry deviation is trusted.")
    max_industries: int = Field(
        8, gt=0, description="Cap on how many exposed industries the industry layer models (latency / focus)."
    )

    # --- idiosyncratic (individual-stock) layer (SIMPLEST: a fixed conservative residual, no fit) ---
    # Kept restrained by design (§5.3): current true holdings are unobservable externally, and the
    # per-stock layer is the noisiest — so it adds a small, honest uncertainty band with a ZERO
    # central estimate (never a fabricated idiosyncratic alpha), widened when exposure confidence
    # is low. This IS the "个股偏差模型要保持克制、尽量简单" realization (D-HF-2).
    # Calibrated 2026-07 on the PRE-2023 selection window only (holdout untouched): beta-scaled
    # residual std at the h=20 reference measured 0.024-0.037 across funds, mean 0.029. The band is
    # z80 * vol (1.2816σ) so the idio contribution is coherent with the 80% nominal — the previous
    # pair (band 0.03, vol 0.04) implied a different nominal per date. Confirmed on the 2023+
    # holdout before adoption; see D-HF-9 (per-fund level caveats there).
    idio_residual_band: float = Field(
        0.037, ge=0.0, description="Half-width of the idio ret band at the reference horizon (= z80 * idio_residual_vol)."
    )
    idio_residual_vol: float = Field(
        0.029, ge=0.0, description="Idio volatility at the reference horizon (pre-2023 beta-scaled residual std)."
    )
    idio_low_confidence_mult: float = Field(
        2.0, ge=1.0, description="Idiosyncratic band/vol multiplier as exposure confidence → 0 (widen when we know less)."
    )
    # The band/vol above are calibrated on REALISED residuals, which already embed the calibration
    # sample's typical exposure unknownness (measured mean max(lowconf, unresolved) ≈ 0.43, median
    # 0.40 on the pre-2023 window). Widening on top of that double-counts the same uncertainty, so
    # the multiplier applies only to unknownness ABOVE this baseline (renormalised so the extreme
    # max-unknownness case still reaches the full multiplier).
    idio_widen_baseline: float = Field(
        0.4, ge=0.0, le=1.0, description="Typical unknownness already embedded in the calibrated band; widen only above it."
    )

    # --- market loading (真实暴露 tier-2): PIT rolling beta of fund NAV vs the benchmark ---
    # The disclosed route to the fund's market loading is broken twice over: the top-10 holdings
    # normalise to 1.0 (within-disclosed-book shares, not NAV shares), and portfolio mkv is
    # disclosed for the COMBINED fund while fund_nav.net_asset is per share class — Σmkv/net_asset
    # measures 1.5–2.3 on real data. The fund's own returns don't lie: a trailing regression gives
    # the CURRENT effective sensitivity, which is what true-exposure estimation is for. Fail-closed:
    # too little history → fall back to the (pinned) book-sum behaviour.
    equity_beta_window_days: int = Field(
        250, gt=0, description="Trailing common-return observations for the PIT market-beta regression (~1 year)."
    )
    equity_beta_min_obs: int = Field(
        60, gt=0, description="Minimum common observations before the beta estimate is trusted (else fail closed)."
    )
    equity_beta_clip_high: float = Field(
        2.0, gt=0.0, description="Upper bound on estimated NAV beta for high-beta sector funds."
    )

    # The idio band/vol above are calibrated AT this horizon; other horizons scale by sqrt(h/ref),
    # since residual dispersion grows ~sqrt(h). A flat band was measured 3.4x too wide at h=3 and
    # 1.6x too narrow (over-confident — the dangerous direction) at h=90.
    idio_reference_horizon_days: int = Field(
        20, gt=0, description="Horizon (trading days) at which idio_residual_band/vol are calibrated."
    )

    # --- median de-biasing (James-Stein shrinkage of the market central estimate) ---
    # The market median is the 50th percentile of trailing forward returns over a ~6y lookback, so
    # it inherits a MOMENTUM bias: after a bull run it drifts up (it spiked to +0.37 into the 2015
    # crash). Direction is not predictable (validated exhaustively — momentum, mean-reversion AND
    # valuation all fail the out-of-sample gate), so an over-confident median is a wrong bet, not
    # skill. Shrinking it toward 0 (the band shifts with it) is a pure de-biasing: it does not change
    # rank-IC but cuts the worst-case error. Validated non-worse on the 2023+ holdout at every
    # horizon; better on the 2015-containing window. See D-CVOL-3.
    market_median_shrink: float = Field(
        0.85, ge=0.0, le=1.0, description="Shrink factor for the market median toward 0; 1.0 = no shrink (raw empirical median)."
    )

    # --- conditional-volatility band (真实暴露 tier-2 for uncertainty) ---
    # The empirical band adapts to the vol regime only through its long lookback (≈6y), so it is
    # slow: in a calm regime it stays too wide (measured market coverage 0.90 vs nominal 0.80 on
    # 2023+). Scaling the band by the CURRENT vol regime — recent realized vol vs a long baseline —
    # makes it breathe. Forward realized vol is strongly predictable OOS (measured corr 0.41 from
    # persistence alone, 0.48 with a GBM), unlike direction; this is where representation genuinely
    # improves. Tuned on ≤2022 (coverage→nominal without worsening Winkler), confirmed 2023+ (market
    # coverage 0.90→0.84, Winkler flat). See D-CVOL-1.
    vol_conditioning_enabled: bool = Field(
        True, description="Scale the market band half-widths by the current realized-vol regime (PIT)."
    )
    vol_conditioning_gamma: float = Field(
        0.75, ge=0.0, le=1.0, description="Strength of vol conditioning; 0 = off (the old static band)."
    )
    vol_conditioning_clip_lo: float = Field(0.7, gt=0.0, description="Lower clip on the vol-regime multiplier.")
    vol_conditioning_clip_hi: float = Field(1.5, gt=0.0, description="Upper clip on the vol-regime multiplier.")
    vol_conditioning_fast_span: int = Field(20, gt=0, description="EWMA span (days) for recent realized vol.")
    vol_conditioning_slow_span: int = Field(250, gt=0, description="EWMA span (days) for the baseline realized vol.")

    # --- combination ---
    combine_vol_independent: bool = Field(
        True, description="Combine layer volatilities as sqrt(sum of squares) (independent tasks) vs. plain sum (fully correlated)."
    )

    @property
    def market_windows(self) -> tuple[float, float]:
        return (self.market_ret_low_quantile, self.market_ret_high_quantile)

    @property
    def industry_windows(self) -> tuple[float, float]:
        return (self.industry_ret_low_quantile, self.industry_ret_high_quantile)


class CalibrationConfig(BaseModel):
    """Confidence-calibration bucketing (§4 sec:mod-uncertainty, §5)."""

    model_config = {"frozen": True}

    bucket_count: int = Field(10, description="Equal-width confidence buckets over [0, 1].")
    min_samples_per_bucket: int = Field(
        30, description="Below this, fall back to the conservative lower bound."
    )


class RiskBudgetConfig(BaseModel):
    """Risk-budget over-exposure thresholds (§4 sec:mod-riskbudget).

    ``budget_usage`` is normalized to a suitability-derived cap; over-budget when it
    exceeds 1.0. Single-industry concentration has a soft and a hard cap.
    """

    model_config = {"frozen": True}

    budget_usage_cap: float = Field(1.0, description="Over-budget when budget_usage > 1.0.")
    single_industry_soft_cap: float = Field(0.40, description="Warn above 40% single-industry.")
    single_industry_hard_cap: float = Field(0.60, description="Block add-position above 60%.")


class ActionStrategyConfig(BaseModel):
    """Action-strategy gate thresholds (§4 sec:mod-action, §6 sec:cc-compliance).

    Only the 85% position is numeric in the doc; the rest are supplied here. When a
    gate trips on the safety path the action degrades to WAIT_NEW_SIGNAL (等待新信号).
    """

    model_config = {"frozen": True}

    min_confidence: float = Field(
        0.5, description="Calibrated confidence below this → degrade to 等待新信号."
    )
    high_concentration: float = Field(
        0.60, description="Portfolio concentration above this → 'don't chase' trigger."
    )
    short_holding_days: int = Field(
        30, description="Holding shorter than this → 'don't chase' trigger."
    )
    high_roundtrip_cost: float = Field(
        0.01, description="Round-trip cost above 1% → 'don't chase' trigger."
    )
    high_position: float = Field(
        0.85, description="Current single-fund position at/above 85% → do not add (doc example)."
    )
    # Forecast RISK side (预期风险) tempering an add (doc-silent, D-ACT-7). §3.4 lists
    # 预期风险 as an action input: a bullish forecast that is itself very bumpy or carries a
    # high material-drawdown probability should not be chased. A candidate ADD is downgraded
    # to 维持，不追高 when the primary horizon's vol_est ≥ high_forecast_vol OR its
    # drawdown_prob ≥ high_forecast_drawdown_prob.
    high_forecast_vol: float = Field(
        0.20, ge=0.0, description="Primary-horizon vol_est at/above this tempers an add (D-ACT-7)."
    )
    high_forecast_drawdown_prob: float = Field(
        0.50, ge=0.0, le=1.0, description="Primary-horizon drawdown_prob at/above this tempers an add (D-ACT-7)."
    )


class CashAllocationConfig(BaseModel):
    """cash_allocation quality gate + scoring weights (§4 sec:mod-cashalloc, §5).

    All doc-silent (the doc names the six ranking criteria + the protection rule, not the
    numbers). A candidate must clear the quality gate (non-bearish, reliable, not too risky)
    to receive any cash; passers are allocated proportionally to a weighted score under a
    per-fund cap. If none pass — even with must_fully_invest — cash is kept (D-CASH-*).
    """

    model_config = {"frozen": True}

    # quality gate (per candidate) — fail-closed toward keeping cash
    min_expected_return: float = Field(
        0.0, description="Mid-horizon ret_mid at/below this → candidate rejected (bearish/flat)."
    )
    min_confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Calibrated confidence below this → candidate rejected."
    )
    max_drawdown_prob: float = Field(
        0.60, ge=0.0, le=1.0, description="Mid-horizon drawdown_prob above this → candidate rejected."
    )
    # allocation shape
    max_single_fund_weight: float = Field(
        0.40, gt=0.0, le=1.0, description="Cap on any one fund's share of the usable cash."
    )
    max_candidates: int = Field(
        40, gt=0, description="Max FULL-tier funds screened from the universe per request (latency cap)."
    )
    # score weights (reward return / reliability / diversification; penalize drawdown / fee)
    w_return: float = Field(1.0, ge=0.0)
    w_confidence: float = Field(1.0, ge=0.0)
    w_diversification: float = Field(1.0, ge=0.0, description="Reward low overlap with current holdings.")
    w_drawdown: float = Field(1.0, ge=0.0)
    w_fee: float = Field(0.5, ge=0.0)
    return_scale: float = Field(
        0.10, gt=0.0, description="ret_mid normalizer (a 10% mid-horizon return ≈ score 1.0)."
    )


class ProductViewConfig(BaseModel):
    """to_product_view direction thresholds (§4 sec:mod-forecast).

    Direction is UP only when expected return AND probability-of-positive are both
    clearly positive; DOWN only when both clearly negative; otherwise RANGE.
    """

    model_config = {"frozen": True}

    up_ret_mid_min: float = Field(0.005, description="ret_mid must exceed +0.5% for 'up'.")
    up_prob_positive_min: float = Field(0.6, description="prob_positive must exceed 0.6 for 'up'.")
    down_ret_mid_max: float = Field(-0.005, description="ret_mid must be below -0.5% for 'down'.")
    down_prob_positive_max: float = Field(0.4, description="prob_positive must be below 0.4 for 'down'.")

    # --- trend-strength derivation (doc-silent, D-PV-1) ------------------------
    # trend_strength = |ret_mid| / (|ret_mid| + trend_strength_vol_weight * vol_est),
    # so a large return relative to bumpiness → strong trend; a small return under high
    # volatility → weak trend ("涨幅不大但波动很大，不能说趋势强", §4 designnote).
    trend_strength_vol_weight: float = Field(
        1.0, ge=0.0, description="Weight on vol_est vs |ret_mid| in the trend-strength ratio (D-PV-1)."
    )
    # --- volatility derivation from the forecast RANGE (doc-silent, D-PV-2) -----
    # Product-facing volatility is derived from the forecast band width (ret_high-ret_low)
    # of the primary horizon, then escalated to the more-cautious of that and the market
    # environment's volatility (fail-closed lean — never hide bumpiness).
    vol_range_high_min: float = Field(
        0.20, ge=0.0, description="Primary-horizon band width ≥ this → HIGH product volatility (D-PV-2)."
    )
    vol_range_low_max: float = Field(
        0.05, ge=0.0, description="Primary-horizon band width ≤ this → LOW product volatility (D-PV-2)."
    )


# Standard disclaimers for non-FULL types (§3 sec:ds-scope / §6 gate 3).
_LIMITED_DISCLAIMER = (
    "该类型深度分析暂未支持：仅提供已披露信息与风险来源说明，不输出方向性判断或个性化建议。"
)
_DEFERRED_DISCLAIMER = (
    "场内 ETF 交易机制（折溢价 / 盘中定价 / 场内时间轴）v1 暂不处理，仅完成类型识别。"
)


def _default_caliber_table() -> dict[ScopeCategory, FundTypeProfile]:
    """Build the per-category :class:`FundTypeProfile` registry (§3 sec:ds-scope table).

    Equity family → FULL / cap FULL / no disclaimer; every other 场外 type → LIMITED /
    cap NONE / disclaimer; 场内 ETF → DEFERRED. lenses / risk_drivers are the §1 sec:scope
    'main risk source' column in user language.
    """
    FULL, LIMITED, DEFERRED = SupportTier.FULL, SupportTier.LIMITED, SupportTier.DEFERRED
    CAP_FULL, CAP_NONE = AdviceStrengthCap.FULL, AdviceStrengthCap.NONE

    def full(cat, lenses, drivers):
        return FundTypeProfile(category=cat, support_tier=FULL, applicable_lenses=lenses,
                               risk_drivers=drivers, advice_strength_cap=CAP_FULL, disclaimer="")

    def limited(cat, lenses, drivers):
        return FundTypeProfile(category=cat, support_tier=LIMITED, applicable_lenses=lenses,
                               risk_drivers=drivers, advice_strength_cap=CAP_NONE,
                               disclaimer=_LIMITED_DISCLAIMER)

    C = ScopeCategory
    return {
        C.A_SHARE_EQUITY: full(C.A_SHARE_EQUITY, ["industry", "style", "cap"],
                               ["行业", "风格", "市场状态", "波动"]),
        C.SECTOR_THEME:   full(C.SECTOR_THEME, ["industry", "style", "cap"],
                               ["单一行业/主题", "行业景气", "市场状态", "波动"]),
        C.INDEX:          full(C.INDEX, ["index_target", "components", "tracking_error"],
                               ["目标指数", "成分结构", "跟踪误差"]),
        C.EQUITY_HYBRID:  full(C.EQUITY_HYBRID, ["industry", "style", "cap"],
                               ["行业", "风格", "市场状态", "波动（长期高仓位权益）"]),
        C.HYBRID_ACTIVE:  limited(C.HYBRID_ACTIVE, ["asset_allocation", "style"],
                                  ["股债仓位摆动", "资产配置", "择时"]),
        C.BOND:           limited(C.BOND, ["rate", "credit", "duration"],
                                  ["利率", "信用风险", "久期", "债市流动性"]),
        C.MONEY_MARKET:   limited(C.MONEY_MARKET, ["yield", "liquidity"],
                                  ["收益稳定性", "流动性", "底层资产安全边际"]),
        C.QDII_OVERSEAS:  limited(C.QDII_OVERSEAS, ["overseas", "fx"],
                                  ["海外市场", "汇率", "交易时差", "额度"]),
        C.FOF:            limited(C.FOF, ["underlying_fund_lookthrough"],
                                  ["底层是其他基金", "需向下再穿透一层"]),
        C.COMMODITY_ALT:  limited(C.COMMODITY_ALT, ["commodity_price", "valuation"],
                                  ["商品/黄金/REITs 价格", "估值方式特殊"]),
        C.PENSION_TARGET: limited(C.PENSION_TARGET, ["target_date", "risk_level"],
                                  ["投资期限", "风险等级", "下滑轨道"]),
        C.UNCLASSIFIED:   limited(C.UNCLASSIFIED, [], ["类型未能识别（保守降级）"]),
        C.ETF_LISTED:     FundTypeProfile(category=C.ETF_LISTED, support_tier=DEFERRED,
                                          applicable_lenses=["premium_discount", "intraday"],
                                          risk_drivers=["场内折溢价", "盘中定价", "场内交易机制"],
                                          advice_strength_cap=CAP_NONE, disclaimer=_DEFERRED_DISCLAIMER),
    }


class ScopeConfig(BaseModel):
    """Fund-type classification + per-type analysis caliber (§1 sec:scope, §3 sec:ds-scope, §4 sec:mod-scope).

    fund_scope CLASSIFIES every 场外 fund into a :class:`ScopeCategory` and routes it to a
    :class:`FundTypeProfile` (caliber) — it never hard-rejects. The equity family gets FULL
    support; every other 场外 type gets LIMITED (classified + degraded); 场内 ETF is DEFERRED.
    Matched against Tushare fund_basic.fund_type / invest_type / name. The caliber table is
    data (not if/else) so a type's depth/advice-cap is changed by editing config.
    """

    model_config = {"frozen": True}

    # --- equity family (FULL) — style partition (indexed vs active vs 偏股混合) ---
    equity_fund_type: str = Field("股票型")
    hybrid_fund_type: str = Field("混合型")
    in_scope_invest_types: tuple[str, ...] = Field(
        ("普通股票型", "偏股混合型", "被动指数型", "增强指数型", "股票指数")
    )
    equity_hybrid_invest_types: tuple[str, ...] = Field(("偏股混合型",))
    # --- non-equity 场外 types (LIMITED) — matched by fund_type / invest_type substring / name ---
    bond_fund_types: tuple[str, ...] = Field(("债券型",))
    bond_invest_substrings: tuple[str, ...] = Field(("债",))
    money_market_fund_types: tuple[str, ...] = Field(("货币市场型",))
    qdii_fund_types: tuple[str, ...] = Field(("QDII",))
    qdii_invest_substrings: tuple[str, ...] = Field(("QDII", "国际", "海外"))
    fof_fund_types: tuple[str, ...] = Field(("FOF",))
    commodity_fund_types: tuple[str, ...] = Field(("商品型", "另类投资", "REITs"))
    hybrid_active_invest_types: tuple[str, ...] = Field(("平衡混合型", "灵活配置型", "偏债混合型"))
    pension_name_keywords: tuple[str, ...] = Field(("养老", "目标日期", "目标风险"))
    # --- caliber registry: ScopeCategory -> FundTypeProfile ---
    caliber_table: dict[ScopeCategory, FundTypeProfile] = Field(default_factory=_default_caliber_table)

    def profile_for(self, category: ScopeCategory) -> FundTypeProfile:
        """The caliber for ``category`` (falls back to the UNCLASSIFIED profile)."""
        return self.caliber_table.get(category) or self.caliber_table[ScopeCategory.UNCLASSIFIED]


class ComplianceConfig(BaseModel):
    """Compliance-boundary switches (§6 sec:cc-compliance).

    Every boundary is configurable and fails **closed**; the doc defers final
    boundaries to formally effective regulation + legal opinion.
    """

    model_config = {"frozen": True}

    fail_closed: bool = Field(True, description="Ambiguous/missing safety inputs block, never allow.")
    require_suitability_for_advice: bool = Field(True)
    enforce_support_tier_gating: bool = Field(
        True, description="Gate 2: non-FULL support tiers may not run the full strong-judgment pipeline."
    )
    forbid_advice_out_of_scope: bool = Field(
        True, description="Gate 3: forbid strong/personalized advice when caliber.advice_strength_cap==NONE."
    )
    forbid_add_over_risk_budget: bool = Field(True)
    degrade_on_expired_data: bool = Field(True)
    net_of_cost_required: bool = Field(True)
    defer_to_regulation_note: str = Field(
        "所有合规边界最终以正式有效法规和法律意见为准。",
        description="§3.5-9: boundaries ultimately defer to effective regulation.",
    )


class ValidationConfig(BaseModel):
    """'Stably beats baseline' criterion (§5, §7 ch:testing)."""

    model_config = {"frozen": True}

    min_regime_slices_won: int = Field(4, description="Candidate must win >= 4 of 5 regime slices.")
    total_regime_slices: int = Field(5)
    bootstrap_ci_level: float = Field(0.95, description="Bootstrap CI must exclude zero.")


class TushareConfig(BaseModel):
    """Data-layer adapter constants (tushare mapping)."""

    model_config = {"frozen": True}

    otc_market_code: str = Field("O", description="fund_basic market='O' → off-exchange (场外).")
    otc_ts_code_suffix: str = Field(".OF", description="Canonical 6-digit fund_code -> ts_code + '.OF'.")
    fund_basic_page_size: int = Field(
        8000, gt=0, description="fund_basic rows per call; paginate past Tushare's ~15k per-call cap (D44)."
    )
    fund_basic_max_pages: int = Field(
        20, gt=0, description="Safety cap on fund_basic pages per status (avoid an unbounded loop)."
    )


class QmtConfig(BaseModel):
    """Aggregate of all calibration-constant groups. One frozen instance = ``CONFIG``."""

    model_config = {"frozen": True}

    timing: DecisionTimingConfig = Field(default_factory=DecisionTimingConfig)
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)
    confidence_decay: ConfidenceDecayConfig = Field(default_factory=ConfidenceDecayConfig)
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    market_state: MarketStateConfig = Field(default_factory=MarketStateConfig)
    horizon: HorizonConfig = Field(default_factory=HorizonConfig)
    forecaster: HorizonForecasterConfig = Field(default_factory=HorizonForecasterConfig)
    hierarchical: HierarchicalForecasterConfig = Field(default_factory=HierarchicalForecasterConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    risk_budget: RiskBudgetConfig = Field(default_factory=RiskBudgetConfig)
    action: ActionStrategyConfig = Field(default_factory=ActionStrategyConfig)
    cash_allocation: CashAllocationConfig = Field(default_factory=CashAllocationConfig)
    product_view: ProductViewConfig = Field(default_factory=ProductViewConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    tushare: TushareConfig = Field(default_factory=TushareConfig)


# Process-wide singleton of calibration constants.
CONFIG = QmtConfig()


# =============================================================================
# 3. Foundation helper realizing the ThreeTiers confidence-decay invariant.
#    Provided in slice 1 so the "confidence decreases as disclosure_age rises"
#    invariant is concrete and unit-testable from the start (§6 sec:cc-exposure).
# =============================================================================
def decayed_confidence(
    disclosure_age_days: int,
    base: float | None = None,
    cfg: ConfidenceDecayConfig | None = None,
) -> float:
    """Baseline confidence as a monotonically non-increasing function of disclosure age.

    ``confidence = base * exp(-age / tau)`` clamped at ``floor``. Guaranteed
    monotonic non-increasing in ``disclosure_age_days`` (asserted in tests).
    """
    if disclosure_age_days < 0:
        raise ValueError("disclosure_age_days must be >= 0")
    c = cfg or CONFIG.confidence_decay
    b = c.base if base is None else base
    value = b * math.exp(-disclosure_age_days / c.tau_days)
    return max(c.floor, min(b, value))
