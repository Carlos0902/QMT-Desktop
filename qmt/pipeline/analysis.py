"""analysis — the analysis_mode core, import-isolated from advice (§4 sec:mode-split, §6 sec:cc-modes).

This module holds :func:`run_analysis` (and its shared :func:`_run_analysis_core`) plus the
:class:`AnalysisResult` bundle. It is the realization of the doc's code-level mode-isolation
enforcement (§6 sec:cc-modes clause 2): the analysis path lives in a module whose **import
graph never references** :class:`~qmt.types.UserPortfolio` or any advice-only pipeline module
(``action_strategy`` / ``portfolio_lookthrough`` / ``risk_budget``). A static import-guard
test asserts this. The advice orchestrator (:func:`qmt.pipeline.run.run_advice`) imports the
core *from here*; the reverse dependency does not — and cannot — exist.

Signature note (invariant 3): :func:`run_analysis` has **no** :class:`UserPortfolio`
parameter, so the type system makes analysis_mode unable to read user assets. A runtime
belt-and-suspenders backstop (:meth:`ComplianceGuard.gate_no_user_assets_in_analysis` with
``in_analysis_mode=True``) is invoked inside the core as defence-in-depth to the type-level
guard (§3.5-1).

Wiring:  fund_scope → data_visibility → exposure_estimator → market_state →
horizon_forecaster → uncertainty (historically calibrated) → to_product_view → explanation.
fund_scope classifies every 场外 fund; only a ``FULL`` (equity-family) support tier runs the
full pipeline. A ``LIMITED`` / ``DEFERRED`` tier short-circuits to a DEGRADED result
(classified, disclosed facts + risk drivers, disclaimer, no strong judgment, no raise); an
unconfirmable identity fails closed (raise). ``gate_confidence_and_freshness`` still
fail-closed-degrades a FULL run on low confidence / expired data.

Confidence calibration (§4 sec:mod-uncertainty): the raw uncertainty score is replaced by
its historically-calibrated value using a :class:`~qmt.types.CalibrationTable`. A caller may
pass one explicitly; otherwise the cached table persisted by the ``qmt calibrate`` production
path (:mod:`qmt.calibration`, ``data_cache/calibration.json``) is loaded by default. When no
table is available the raw score is reported but clearly flagged *uncalibrated*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

from qmt.calibration import load_cached_calibration_table
from qmt.compliance import ComplianceGuard, ComplianceViolation
from qmt.config import CONFIG, get_settings
from qmt.data.sources import (
    TABLE_FUND_BASIC,
    TABLE_FUND_NAV,
    TABLE_FUND_PORTFOLIO,
    TABLE_INDEX_DAILY,
)
from qmt.pit import AVAILABLE_AT, PointInTimeStore, build_decision_timeline
from qmt.types import (
    AppliesTo,
    CalibrationTable,
    DataField,
    DataSource,
    Direction,
    ExplanationCard,
    ExposureEstimate,
    FundProfile,
    FundStatus,
    Horizon,
    HorizonForecast,
    MarketEvent,
    MarketStateVector,
    ProductView,
    ScopeVerdict,
    Stress,
    SupportTier,
    UpdateFreq,
    VisibilitySet,
    Volatility,
)

from qmt.pipeline.data_visibility import resolve_visibility
from qmt.pipeline.explanation import build_explanation
from qmt.pipeline.exposure_estimator import estimate_exposure
from qmt.pipeline.fund_scope import classify_scope
from qmt.pipeline.horizon_forecaster import forecast
from qmt.pipeline.market_state import InsufficientMarketHistory, assess_market_state
from qmt.pipeline.product_view import to_product_view
from qmt.pipeline.uncertainty import assess_uncertainty, calibrate

__all__ = [
    "AnalysisResult",
    "run_analysis",
    "_run_analysis_core",
    "_resolve_store",
    "_DEFAULT_BENCHMARK",
]

# Default broad benchmark for the market-state stage (matches assess_market_state).
_DEFAULT_BENCHMARK = "000300.SH"


# =============================================================================
# Result container (import the component types; never duplicate them).
# =============================================================================
@dataclass(frozen=True)
class AnalysisResult:
    """Full analysis_mode output bundle (state + explanation, NO personalized advice).

    Deliberately contains **no** :class:`UserPortfolio`, action, look-through or
    risk-budget field — those are advice_mode only. This dataclass being the analysis
    return type is part of the mode-isolation enforcement (§6 sec:cc-modes).

    ``calibrated`` records whether the reported ``confidence`` was passed through a
    historical :class:`CalibrationTable` (True) or is a raw, uncalibrated score (False).
    """

    product_view: ProductView
    explanation: ExplanationCard
    market_state: MarketStateVector
    exposure: list[ExposureEstimate]
    forecasts: dict[Horizon, HorizonForecast]
    visibility: VisibilitySet
    confidence: float
    scope: ScopeVerdict
    calibrated: bool = False


# =============================================================================
# Store / profile resolution helpers
# =============================================================================
def _resolve_store(store: Optional[PointInTimeStore]) -> PointInTimeStore:
    """A caller-supplied store always wins; ``None`` → the persistent cache store.

    Resolved once per orchestrator call and threaded through every stage (and, in advice
    mode, through look-through too) so all stages read the same point-in-time universe.
    """
    if store is not None:
        return store
    return PointInTimeStore(get_settings().data_cache_dir)


def _to_datetime(value: object) -> Optional[datetime]:
    """Best-effort coercion of a stored ``available_at`` cell to a python ``datetime``."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _resolve_profile(
    profile: Union[FundProfile, str], as_of: date, store: PointInTimeStore
) -> Optional[FundProfile]:
    """Return a :class:`FundProfile`, resolving a bare fund code via the PIT ``fund_basic``.

    A :class:`FundProfile` is used as-is. A ``str`` fund code is reconstructed from the
    most-recently-observable ``fund_basic`` row (``available_at <= as_of``); if nothing is
    observable the fund cannot be confirmed in scope, so ``None`` is returned and the caller
    fails closed at :meth:`ComplianceGuard.gate_scope_required`.
    """
    if isinstance(profile, FundProfile):
        return profile

    fund_code = str(profile)
    df = store.get_asof(TABLE_FUND_BASIC, as_of, filters={"fund_code": fund_code})
    if df is None or df.empty:
        return None
    df = df.sort_values(AVAILABLE_AT)
    row = df.iloc[-1].to_dict()

    available_at = _to_datetime(row.get(AVAILABLE_AT)) or datetime.combine(as_of, datetime.min.time())
    status = row.get("status")
    try:
        status_enum = FundStatus(str(status)) if status is not None and str(status) else None
    except ValueError:
        status_enum = None

    def _opt_date(val: object) -> Optional[date]:
        dt = _to_datetime(val)
        return dt.date() if dt is not None else None

    return FundProfile(
        fund_code=fund_code,
        name=str(row.get("name") or fund_code),
        fund_type=(str(row["fund_type"]) if row.get("fund_type") is not None else None),
        invest_type=(str(row["invest_type"]) if row.get("invest_type") is not None else None),
        market=str(row.get("market") or CONFIG.tushare.otc_market_code),
        status=status_enum,
        found_date=_opt_date(row.get("found_date")),
        list_date=_opt_date(row.get("list_date")),
        delist_date=_opt_date(row.get("delist_date")),
        benchmark=(str(row["benchmark"]) if row.get("benchmark") is not None else None),
        available_at=available_at,
    )


def _latest_available_at(
    store: PointInTimeStore, table: str, as_of: date, filters: Optional[dict] = None
) -> Optional[datetime]:
    """Max ``available_at`` among rows of ``table`` observable at ``as_of`` (or ``None``)."""
    df = store.get_asof(table, as_of, filters=filters)
    if df is None or df.empty or AVAILABLE_AT not in df.columns:
        return None
    return _to_datetime(pd.to_datetime(df[AVAILABLE_AT]).max())


def _build_data_fields(
    fund_code: str, as_of: date, store: PointInTimeStore, decision_dt: datetime
) -> list[DataField]:
    """Describe the fund's observable data sources as :class:`DataField`s for visibility.

    One field per populated PIT table (fund NAV, disclosed holdings, benchmark index),
    tagged with its latest observable ``available_at`` so :func:`resolve_visibility` can
    classify freshness. Tables with no observable rows are simply omitted (nothing to show).
    """
    specs = [
        ("fund_nav", TABLE_FUND_NAV, DataSource.FUND_ANNOUNCEMENT, UpdateFreq.DAILY, {"fund_code": fund_code}),
        (
            "fund_portfolio",
            TABLE_FUND_PORTFOLIO,
            DataSource.FUND_ANNOUNCEMENT,
            UpdateFreq.QUARTERLY,
            {"fund_code": fund_code},
        ),
        ("index_daily", TABLE_INDEX_DAILY, DataSource.INDEX, UpdateFreq.DAILY, None),
    ]
    fields: list[DataField] = []
    for name, table, source, freq, filters in specs:
        latest = _latest_available_at(store, table, as_of, filters)
        if latest is None:
            continue
        fields.append(
            DataField(
                name=name,
                source=source,
                update_freq=freq,
                available_at=latest,
                visible_at_decision=latest <= decision_dt,
                applies_to={AppliesTo.EQUITY},
            )
        )
    return fields


def _is_business_day(d: date) -> bool:
    """Whether ``d`` is a Mon–Fri (the default trading-calendar predicate)."""
    return d.weekday() < 5


def _degraded_result(fund_code: str, as_of: date, scope: ScopeVerdict) -> AnalysisResult:
    """Degraded analysis bundle for a classified but non-FULL fund (§4 lst:scope, §3.5-2/3).

    Under the extended scope every 场外 fund is *classified* and none is hard-rejected. A
    ``LIMITED`` type (bond / money / QDII / FOF / commodity / pension / active-hybrid /
    unclassified) or a ``DEFERRED`` type (场内 ETF) does not run the equity pipeline; instead
    it returns a neutral :class:`ProductView` (no directional judgment) and an
    :class:`ExplanationCard` that states what drives this type's risk and carries the caliber's
    mandatory disclaimer ("该类型深度分析暂未支持"). No exposure / forecast stage runs.
    advice_mode still refuses personalized advice (``advice_strength_cap==NONE``, §3.5-3);
    this degraded path is the analysis_mode behaviour.
    """
    caliber = scope.caliber
    deferred = scope.support_tier is SupportTier.DEFERRED
    drivers = caliber.risk_drivers or ["（该类型风险来源待补充）"]
    disclaimer = caliber.disclaimer or "该类型深度分析暂未支持。"

    product_view = ProductView(
        fund_code=fund_code,
        as_of=as_of,
        direction=Direction.RANGE,
        trend_strength=0.0,
        volatility=Volatility.NORMAL,
        confidence=0.0,
        action=None,
    )
    if deferred:
        why_optimistic = "该基金为场内交易品种，v1 仅完成类型识别，不做方向性判断。"
        data_freshness = "未进行深度分析：场内 ETF 交易机制（折溢价 / 盘中定价）暂不处理。"
    else:
        why_optimistic = "该类型已纳入覆盖范围，但 v1 仅提供有限分析，暂不给方向性判断。"
        data_freshness = "已识别基金类型；该类型的深度分析（暴露 / 多周期预测）暂未支持。"
    explanation = ExplanationCard(
        fund_code=fund_code,
        as_of=as_of,
        why_optimistic=why_optimistic,
        why_cautious=disclaimer,
        unknowns=f"该类型的真实暴露与多周期预测 v1 未估计。主要风险来源：{'、'.join(drivers)}。",
        data_freshness=data_freshness,
        what_would_change="若后续版本为该类型实现专用分析口径，或该基金被确认属于权益家族，则升级为完整分析。",
        evidence_seen=[f"基金类型：{scope.category.value}", f"支持档位：{scope.support_tier.value}"],
        driving_factors=[f"该类型主要风险来源：{'、'.join(drivers)}。"],
        counter_evidence=[disclaimer],
        uncertainties=["v1 未对该类型估计行业 / 风格暴露，也未做多周期预测。"],
        change_conditions=["该类型的专用分析口径在后续版本落地，或基金类型被重判为权益家族。"],
        disclosure_age_days=0,
    )
    return AnalysisResult(
        product_view=product_view,
        explanation=explanation,
        market_state=MarketStateVector(
            as_of=as_of,
            direction=Direction.RANGE,
            volatility=Volatility.NORMAL,
            stress=Stress.NORMAL,
            event=MarketEvent.NONE,
        ),
        exposure=[],
        forecasts={},
        visibility=VisibilitySet(as_of=datetime.combine(as_of, datetime.min.time())),
        confidence=0.0,
        scope=scope,
        calibrated=False,
    )


def _load_run_industry_map() -> Optional[dict]:
    """Load the ingest-primed ``symbol -> Shenwan L1`` map, or ``None`` if none was built.

    Best-effort read of ``data_cache/industry_map.json`` (written by
    :func:`qmt.data.ingest.build_industry_map`). Returns ``None`` (not ``{}``) when the map
    is empty/absent so :func:`~qmt.pipeline.exposure_estimator.estimate_exposure` keeps its
    exact prior behaviour (the ``unclassified`` data-gap sentinel) when the quota never
    allowed a map to be built. Never raises — the analysis path stays fail-open on this
    optional reference taxonomy.
    """
    try:
        from qmt.data.ingest import load_cached_industry_map

        cached = load_cached_industry_map()
        return cached or None
    except Exception:  # pragma: no cover - defensive: a bad cache never breaks analysis
        return None


# =============================================================================
# analysis_mode
# =============================================================================
def run_analysis(
    profile: Union[FundProfile, str],
    as_of: date,
    store: Optional[PointInTimeStore] = None,
    *,
    calibration: Optional[CalibrationTable] = None,
) -> AnalysisResult:
    """analysis_mode entrypoint — judge state and explain it; **no** personalized advice.

    Signature note (invariant 3, §6 sec:cc-modes): there is **no** :class:`UserPortfolio`
    parameter here, by design — the type system makes analysis_mode unable to read user
    assets. ``run_advice`` is the sole portfolio-reading entrypoint.

    Pipeline (FULL tier):
    ``fund_scope`` (classify + caliber route, §3.5-2) → ``data_visibility`` → ``exposure_estimator``
    → ``market_state`` → ``horizon_forecaster`` → ``uncertainty`` → ``to_product_view`` →
    ``explanation``, with a fail-closed ``gate_confidence_and_freshness`` degrade.

    A **non-FULL** fund (``LIMITED`` / ``DEFERRED`` support tier) is not an error: the doc's
    fund_scope contract (§4 lst:scope, §3.5-2/3) classifies it and returns a DEGRADED
    :class:`AnalysisResult` (disclosed facts + risk drivers + disclaimer, no strong judgment)
    instead of raising. A fund whose identity cannot be confirmed at all (no observable
    ``fund_basic``) still raises :class:`ComplianceViolation` (fail-closed safety reject, D12).
    ``calibration`` (keyword-only) historically calibrates the raw
    uncertainty when supplied; otherwise the cached table (``data_cache/calibration.json``,
    built by ``qmt calibrate``) is loaded by default. When neither is available the raw
    reliability is reported and flagged *uncalibrated*.
    """
    return _run_analysis_core(profile, as_of, _resolve_store(store), calibration=calibration)


def _run_analysis_core(
    profile: Union[FundProfile, str],
    as_of: date,
    store: PointInTimeStore,
    *,
    calibration: Optional[CalibrationTable] = None,
) -> AnalysisResult:
    """Shared analysis pipeline over an already-resolved store (reused by advice_mode).

    Takes a *resolved* :class:`PointInTimeStore` so advice_mode can thread the same PIT
    universe through look-through. Never accepts a :class:`UserPortfolio` (mode isolation).
    """
    guard = ComplianceGuard()

    # --- gate #1: analysis-mode runtime backstop (belt-and-suspenders to type isolation).
    # This core has no UserPortfolio in scope by construction; invoking the backstop with
    # in_analysis_mode=True asserts that fact at runtime (§3.5-1, §6 sec:cc-modes clause 1).
    gate1 = guard.gate_no_user_assets_in_analysis(None, in_analysis_mode=True)
    if not gate1.passed:  # pragma: no cover - structural: analysis carries no portfolio
        raise ComplianceViolation(gate1)

    # --- stage 1: fund_scope — classify + caliber route (§4 lst:scope, §3.5-2/3) --------
    # Every 场外 fund is classified; the SUPPORT TIER routes the analysis depth:
    #   * unresolvable identity (no observable fund_basic) → cannot confirm the fund → the one
    #     fail-closed HARD raise (D12 safety reject — not a type-based decline);
    #   * FULL (equity family) → run the full pipeline (stages 2-8) below;
    #   * LIMITED / DEFERRED → a DEGRADED result (classified, disclosed facts + risk drivers,
    #     no strong judgment, mandatory disclaimer) — the pipeline does NOT raise (§4 lst:scope).
    #     advice_mode still refuses personalized advice on these (advice_strength_cap==NONE).
    resolved_profile = _resolve_profile(profile, as_of, store)
    if resolved_profile is None:
        raise ComplianceViolation(guard.gate_scope_required(None))
    scope = classify_scope(resolved_profile)
    if scope.support_tier is not SupportTier.FULL:
        return _degraded_result(resolved_profile.fund_code, as_of, scope)
    fund_code = resolved_profile.fund_code

    # --- stage 2: data_visibility — what is observable now vs stale -------------
    decision_dt = build_decision_timeline(as_of).decision_time if _is_business_day(as_of) else datetime.combine(
        as_of, datetime.min.time()
    )
    visibility = resolve_visibility(_build_data_fields(fund_code, as_of, store, decision_dt), as_of)

    # --- stage 3: exposure_estimator — three-tier estimated exposure -----------
    # Best-effort industry map primed by the ingest layer (data_cache/industry_map.json).
    # When present (quota allowed it to be built) holdings map to real Shenwan L1 industries
    # instead of the 'unclassified' data-gap sentinel; absent → None → unchanged behaviour.
    exposure = estimate_exposure(
        fund_code, as_of, store, industry_map=_load_run_industry_map()
    )

    # --- stage 4: market_state — 4 independent dimensions (fail-closed) ---------
    market_fallback = False
    try:
        market_state = assess_market_state(as_of, store, benchmark_index=_DEFAULT_BENCHMARK)
    except InsufficientMarketHistory:
        # Fail-closed (invariant 6): no fabricated benign vector — use the most-cautious
        # environment (RANGE / HIGH vol / RISING_RISK) and mark data expired so confidence
        # degrades. Documented D-RUN-1.
        market_fallback = True
        market_state = MarketStateVector(
            as_of=as_of,
            direction=Direction.RANGE,
            volatility=Volatility.HIGH,
            stress=Stress.RISING_RISK,
            event=MarketEvent.NONE,
        )

    # --- stage 5: horizon_forecaster — return/risk RANGES per horizon ----------
    forecasts = forecast(fund_code, as_of, exposure, market_state, store)

    # --- stage 6: uncertainty — historically-calibrated reliability ------------
    # Default production path: load the cached CalibrationTable (built by `qmt calibrate`,
    # persisted to data_cache/calibration.json). An explicit `calibration` argument wins.
    table = calibration if calibration is not None else load_cached_calibration_table()
    calibrated = table is not None
    raw_conf = assess_uncertainty(forecasts, exposure, market_state)
    confidence = calibrate(raw_conf, table) if table is not None else raw_conf

    # --- gate #4: confidence & freshness (fail-closed degrade) -----------------
    data_expired = market_fallback or not exposure or not forecasts
    gate3 = guard.gate_confidence_and_freshness(confidence, data_expired=data_expired)
    if not gate3.passed:
        # Fail-closed: never let a numerically-high confidence survive an expired-data
        # degrade — floor it so downstream (advice) cannot add on stale ground.
        confidence = min(confidence, CONFIG.confidence_decay.floor)

    # --- stage 7: to_product_view — derive product labels from the numbers -----
    product_view = to_product_view(forecasts, confidence, market_state, fund_code, as_of)

    # --- stage 8: explanation — fixed five sections, non-causal (§3.5-6) --------
    explanation = build_explanation(
        fund_code,
        as_of,
        product_view,
        forecasts,
        market_state,
        exposure,
        confidence,
        guard=guard,
        calibrated=calibrated,
    )

    return AnalysisResult(
        product_view=product_view,
        explanation=explanation,
        market_state=market_state,
        exposure=exposure,
        forecasts=forecasts,
        visibility=visibility,
        confidence=confidence,
        scope=scope,
        calibrated=calibrated,
    )
