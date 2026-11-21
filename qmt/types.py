"""Domain types for QMT — pydantic v2 models for every data structure the design
doc defines (techdesign/chapters/03-data-structures.tex, 04-modules.tex, A1-appendix.tex).

Vocabulary lock: class and field names are the doc's identifiers verbatim (modulo
Python casing). Enum value-sets use the exact strings the doc's module signatures
give (direction up/down/range, freshness fresh/stale/expired, …); where the doc
gives only Chinese product terms (the six ActionAdvice values) the enum member name
is a stable English identifier and the enum *value* is the exact Chinese string.

Cross-cutting invariants enforced at this layer:
  (1) point-in-time correctness — every externally-observed field carries
      ``available_at`` (mandatory; the PIT data-access layer filters
      ``available_at <= as_of``).
  (2) estimated exposure ≠ disclosed holdings — ``ExposureEstimate`` carries the
      three tiers: disclosed facts + ``disclosure_age_days`` / system estimate with
      ``low``/``high`` range / ``confidence``.
  (4) predictions are ranges — ``HorizonForecast`` carries ret_low/mid/high, not a
      label; product labels are *derived* in ``ProductView``.
  (5) market state is a 4-field vector — ``MarketStateVector``.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Optional

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

# --- reusable constrained scalar types -------------------------------------
# fund_code: canonical 6-digit numeric string (doc §3 sec:ds-exposure). Tushare
# ts_code (with .OF/.SH suffix) is mapped only in the data-layer adapter.
FundCode = Annotated[str, StringConstraints(pattern=r"^\d{6}$")]
# probability / ratio in [0, 1].
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
# A fund's market loading (PIT NAV beta): non-negative and may exceed 1 for a high-beta fund, up to
# the estimator's configured clip ceiling. Bounded generously so a valid loading never fails schema.
MarketLoading = Annotated[float, Field(ge=0.0, le=5.0)]


# =============================================================================
# Enums — fixed value-sets (vocabulary lock)
# =============================================================================
class DataSource(str, Enum):
    """DataField.source — where the datum comes from (§3 sec:ds-datafield:
    基金公告 / 行情 / 指数 / 财报 / 资金流 / 平台数据)."""

    FUND_ANNOUNCEMENT = "fund_announcement"  # 基金公告
    QUOTE = "quote"  # 行情
    INDEX = "index"  # 指数
    FINANCIAL_REPORT = "financial_report"  # 财报
    FUND_FLOW = "fund_flow"  # 资金流
    PLATFORM = "platform"  # 平台数据


class UpdateFreq(str, Enum):
    """DataField.update_freq (§3 sec:ds-datafield)."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"


class MissingPolicy(str, Enum):
    """DataField.missing_policy — how missing/delayed/restated data is handled
    (§3 sec:ds-datafield). Doc lists the *situations* (缺失/延迟/后来修正); the
    policy value-set is a doc-silent decision (see DECISIONS.md)."""

    REJECT = "reject"  # fail-closed: drop the field from the as-of read
    FORWARD_FILL = "forward_fill"  # carry the last observable value
    DROP = "drop"  # exclude the missing row only
    RESTATE_AWARE = "restate_aware"  # honor restatements by their own available_at


class AppliesTo(str, Enum):
    """DataField.applies_to / fund category membership (§3 sec:ds-datafield)."""

    EQUITY = "equity"
    INDEX = "index"
    BOND = "bond"
    OVERSEAS = "overseas"
    FOF = "fof"


class Visibility(str, Enum):
    """data_visibility per-field state (§4 sec:mod-visibility: visible / hidden)."""

    VISIBLE = "visible"
    HIDDEN = "hidden"


class Freshness(str, Enum):
    """data_visibility freshness state (§4 sec:mod-visibility: fresh / stale / expired)."""

    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"


class ExposureDimension(str, Enum):
    """ExposureEstimate.dimension (§3 sec:ds-exposure)."""

    INDUSTRY = "industry"
    STYLE = "style"
    CAP = "cap"
    ASSET_CLASS = "asset_class"


class Direction(str, Enum):
    """MarketStateVector.direction / ProductView.direction (§3: up / down / range)."""

    UP = "up"
    DOWN = "down"
    RANGE = "range"


class Volatility(str, Enum):
    """MarketStateVector.volatility / ProductView.volatility (§4 lst:market: low/normal/high)."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Stress(str, Enum):
    """MarketStateVector.stress (§4 lst:market: normal/liquidity_tight/rising_risk)."""

    NORMAL = "normal"
    LIQUIDITY_TIGHT = "liquidity_tight"
    RISING_RISK = "rising_risk"


class MarketEvent(str, Enum):
    """MarketStateVector.event (§4 lst:market: none/policy/shock)."""

    NONE = "none"
    POLICY = "policy"
    SHOCK = "shock"


class Horizon(str, Enum):
    """HorizonForecast.horizon (§3 sec:ds-forecast: short 1-3 / mid 5-20 / long 20-90 trading days)."""

    SHORT = "short"
    MID = "mid"
    LONG = "long"


class DisclosureDepth(str, Enum):
    """Disclosed-holdings depth (tushare mapping need(3)): Q1/Q3 top-10 vs semi/annual full."""

    TOP10 = "top10"
    FULL = "full"


class FundStatus(str, Enum):
    """FundProfile.status — Tushare fund_basic status snapshot (L/I/D)."""

    LISTED = "L"
    ISSUING = "I"
    DELISTED = "D"


class ScopeCategory(str, Enum):
    """ScopeVerdict.category — the fund's TYPE (§1 sec:scope, §3 sec:ds-scope).

    fund_scope now *classifies* every 场外 fund into one of these types and never
    hard-rejects (the old ``OUT_OF_SCOPE`` is retired). The analysis depth for each type
    is carried separately by :class:`SupportTier` via the fund's :class:`FundTypeProfile`.
    """

    # equity family — FULL support (full exposure→market→forecast→advice pipeline)
    A_SHARE_EQUITY = "a_share_equity"
    SECTOR_THEME = "sector_theme"
    INDEX = "index"
    EQUITY_HYBRID = "equity_hybrid"          # 偏股混合型 (high persistent A-share equity)
    # non-equity / harder-to-model — LIMITED support (classified + degraded)
    HYBRID_ACTIVE = "hybrid_active"          # 平衡 / 灵活配置 / 偏债混合
    BOND = "bond"
    MONEY_MARKET = "money_market"
    QDII_OVERSEAS = "qdii_overseas"
    FOF = "fof"
    COMMODITY_ALT = "commodity_alt"          # 商品 / 黄金 / REITs 等
    PENSION_TARGET = "pension_target"        # 养老目标 / 稳健理财
    UNCLASSIFIED = "unclassified"            # fail-closed fallback (still LIMITED, never a reject)
    # exchange-traded — DEFERRED (recognized/classified only; 场内机制 not yet handled)
    ETF_LISTED = "etf_listed"


class SupportTier(str, Enum):
    """Analysis depth a fund type gets in v1 (§3 sec:ds-scope, §4 sec:mod-scope).

    ``FULL`` — equity family: the full exposure→market→forecast→(advice) pipeline.
    ``LIMITED`` — classified but degraded: disclosed facts + the type's risk drivers,
    no strong judgment, no personalized advice, a mandatory disclaimer.
    ``DEFERRED`` — recognized/classified only (场内 ETF trading mechanics not yet built).
    """

    FULL = "full"
    LIMITED = "limited"
    DEFERRED = "deferred"


class AdviceStrengthCap(str, Enum):
    """Strongest advice a fund type may receive (§3 sec:ds-scope, §6 gate 3).

    ``FULL`` — full advice allowed (only FULL-tier types, after the other advice gates).
    ``NONE`` — no strong / personalized advice (all LIMITED and DEFERRED types).
    """

    FULL = "full"
    NONE = "none"


class RiskTolerance(str, Enum):
    """SuitabilityProfile.risk_tolerance — Chinese investor-suitability C1–C5 tiers.

    Value-set is a doc-silent decision (see DECISIONS.md); the doc only requires
    that product risk match the investor category (§6 sec:cc-modes)."""

    C1 = "C1"  # 保守型
    C2 = "C2"  # 稳健型
    C3 = "C3"  # 平衡型
    C4 = "C4"  # 成长型
    C5 = "C5"  # 进取型


class ActionAdvice(str, Enum):
    """The eight action values (§3 sec:ds-action / A1-appendix).

    Member names are stable English identifiers; the enum *value* is the exact
    Chinese product string the doc lists. Six single-fund actions plus two added
    with the usable-cash input (§3.4): ``ALLOCATE_CASH`` / ``HOLD_CASH``.
    """

    CONTINUE_SIP = "维持定投"
    PAUSE_SIP = "暂停定投"
    ADD_MODEST = "适当加仓"
    REDUCE_PARTIAL = "减少部分仓位"
    KEEP_WATCHING = "继续观望"
    WAIT_NEW_SIGNAL = "等待新信号"
    ALLOCATE_CASH = "把可使用现金分配到一组更合适的基金"
    HOLD_CASH = "暂时保留现金"


# =============================================================================
# Data-layer raw records (each carries available_at — invariant 1)
# =============================================================================
class FundMeta(BaseModel):
    """Reference/descriptive metadata for a fund (Tushare fund_basic row).

    Part-B addition beyond the doc's named types (see DECISIONS.md): the doc names
    ``FundProfile`` as the business entity; ``FundMeta`` holds the raw descriptive /
    fee / benchmark fields that feed scope-gating and fee accounting.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    name: str
    management: Optional[str] = Field(None, description="Fund company (基金公司).")
    custodian: Optional[str] = None
    benchmark: Optional[str] = None
    m_fee: Optional[float] = Field(None, description="Management fee rate.")
    c_fee: Optional[float] = Field(None, description="Custodian fee rate.")
    min_amount: Optional[float] = None
    issue_amount: Optional[float] = None


class FundProfile(BaseModel):
    """Business entity gating fund_scope (§3 ch:data, §4 sec:mod-scope).

    Reconstructed from Tushare fund_basic. Survivorship: the point-in-time universe
    is rebuilt from ``found_date``/``delist_date`` (never the current ``status``
    snapshot). ``available_at`` derives from the fund's establishment/listing date —
    a fund is only observable once it exists.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    name: str
    fund_type: Optional[str] = Field(None, description="投资类型: 股票型/混合型/债券型/…")
    invest_type: Optional[str] = Field(None, description="投资风格: 被动指数型/偏股混合型/…")
    market: str = Field("O", description="E=场内 / O=场外; v1 universe is O (OTC).")
    status: Optional[FundStatus] = Field(None, description="Current status snapshot (L/I/D).")
    found_date: Optional[date] = Field(None, description="成立日 (universe reconstruction).")
    list_date: Optional[date] = None
    delist_date: Optional[date] = Field(None, description="清盘/合并日 (survivorship).")
    benchmark: Optional[str] = None
    meta: Optional[FundMeta] = None
    # PIT marker — a fund becomes observable at/after establishment.
    available_at: datetime = Field(
        ..., description="When this fund first became externally observable (invariant 1)."
    )


class NavPoint(BaseModel):
    """One fund NAV observation (Tushare fund_nav row).

    ``nav_date`` = valuation date (end_date, the date the NAV is *for*).
    ``available_at`` = disclosure timestamp (ann_date) — the PIT field. For OTC funds
    NAV is struck after close and published T+1, so available_at > nav_date. Reads
    filter on available_at, NEVER on nav_date (avoids the classic look-ahead leak).
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    nav_date: date = Field(..., description="净值日期 (end_date) — valuation date, do NOT filter on this.")
    unit_nav: float = Field(..., description="单位净值.")
    accum_nav: Optional[float] = Field(None, description="累计净值.")
    adj_nav: Optional[float] = Field(None, description="复权单位净值 — use for return series.")
    net_asset: Optional[float] = Field(None, description="资产净值 (populated at report period-ends only).")
    available_at: datetime = Field(..., description="ann_date — when this NAV was disclosed (invariant 1).")

    @model_validator(mode="after")
    def _check_disclosure_lag(self) -> "NavPoint":
        # available_at must not precede the valuation date it discloses.
        if self.available_at.date() < self.nav_date:
            raise ValueError("NavPoint.available_at (ann_date) precedes nav_date (end_date) — impossible.")
        return self


class HoldingLine(BaseModel):
    """One stock line inside a disclosed-holdings snapshot (Tushare fund_portfolio row)."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Individual stock code.")
    mkv: Optional[float] = Field(None, description="持仓市值 (元).")
    amount: Optional[float] = Field(None, description="持仓数量 (股).")
    stk_mkv_ratio: Optional[float] = Field(None, description="占股票市值比 (%).")
    stk_float_ratio: Optional[float] = Field(None, description="占流通股本比 (%).")


class DisclosedHoldings(BaseModel):
    """A disclosed-holdings snapshot (Tushare fund_portfolio).

    Tier-1 fact only: this is an *input* to exposure_estimator, NEVER emitted as
    'current holdings' (§6 sec:cc-exposure). ``report_end_date`` = report period-end;
    ``available_at`` = ann_date (the PIT field). ``disclosure_age_days`` is computed
    against a decision as_of, not stored raw here.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    report_end_date: date = Field(..., description="报告期截止日 (end_date). NOT the date investors knew it.")
    depth: DisclosureDepth = Field(..., description="top10 (Q1/Q3) vs full (semi/annual).")
    holdings: list[HoldingLine] = Field(default_factory=list)
    available_at: datetime = Field(..., description="ann_date — when the report was published (invariant 1).")

    @model_validator(mode="after")
    def _check_disclosure_lag(self) -> "DisclosedHoldings":
        if self.available_at.date() < self.report_end_date:
            raise ValueError("DisclosedHoldings.available_at precedes report_end_date — impossible.")
        return self


class DataField(BaseModel):
    """Feature/data-field metadata (§3 sec:ds-datafield).

    Every feature must carry its provenance and, above all, ``available_at`` — a
    field lacking it is rejected before backtest (§3 risknote / §6 sec:cc-pit).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Feature name, e.g. '北向资金流入'.")
    source: DataSource
    update_freq: UpdateFreq
    available_at: datetime = Field(..., description="When this datum was truly observable (invariant 1).")
    visible_at_decision: bool = Field(..., description="Whether visible at T-day 14:30.")
    schema_changed: bool = Field(False, description="Whether the statistical caliber/rule ever changed.")
    missing_policy: MissingPolicy = MissingPolicy.REJECT
    applies_to: set[AppliesTo] = Field(default_factory=set)


# =============================================================================
# DecisionTimeline (§3 sec:ds-timeline / A1). Model here; builder in qmt.pit.
# =============================================================================
class DecisionTimeline(BaseModel):
    """The fixed decision → order → confirm → forecast-window chain (§3 sec:ds-timeline).

    Constructed on a trading-day calendar by ``qmt.pit.build_decision_timeline``.
    ``decision_time`` is canonically T-day 14:30 (see config.DecisionTimingConfig).
    """

    model_config = ConfigDict(extra="forbid")

    decision_time: datetime = Field(..., description="When the system produces its judgment (T-day 14:30).")
    order_time: datetime = Field(..., description="User submits subscription/redemption before cutoff.")
    confirm_time: datetime = Field(..., description="Fund confirms shares/redemption per its rules.")
    horizon_start: date = Field(..., description="Forecast window start — first realistically executable point.")
    horizon_end: date = Field(..., description="Forecast window end — set by the horizon.")

    @model_validator(mode="after")
    def _check_order(self) -> "DecisionTimeline":
        if not (self.decision_time <= self.order_time <= self.confirm_time):
            raise ValueError("DecisionTimeline requires decision_time <= order_time <= confirm_time.")
        if self.horizon_end < self.horizon_start:
            raise ValueError("DecisionTimeline requires horizon_end >= horizon_start.")
        return self


# =============================================================================
# Exposure — the ThreeTiers core structure (§3 sec:ds-exposure, invariant 2)
# =============================================================================
class ExposureEstimate(BaseModel):
    """Estimated exposure on one dimension/label (§3 sec:ds-exposure).

    Encodes all three information tiers: (1) disclosed fact age via
    ``disclosure_age_days``; (2) the system estimate as a ``point_estimate`` with a
    ``low``/``high`` range; (3) ``confidence`` in the estimate. Estimated exposure is
    NOT disclosed holdings (§6 sec:cc-exposure); ``confidence`` must decrease as
    ``disclosure_age_days`` rises (enforced in exposure_estimator, baseline in
    ``config.decayed_confidence``).

    Worked example (doc §3): semiconductor 34%–50%, midpoint 42%, disclosed 68 days ago.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    as_of: date = Field(..., description="Decision point the estimate is for; data after this is forbidden.")
    dimension: ExposureDimension
    label: str = Field(..., description="Concrete label, e.g. 'semiconductor'.")
    point_estimate: UnitInterval = Field(..., description="Midpoint in [0, 1] (example: 0.42).")
    low: UnitInterval = Field(..., description="Lower bound (example: 0.34).")
    high: UnitInterval = Field(..., description="Upper bound (example: 0.50), low <= high.")
    disclosure_age_days: int = Field(..., ge=0, description="Days since last disclosure (example: 68).")
    drift_prob: UnitInterval = Field(..., description="Probability holdings have materially drifted.")
    confidence: UnitInterval = Field(..., description="System confidence, historically calibrated.")

    @model_validator(mode="after")
    def _check_range(self) -> "ExposureEstimate":
        if not (self.low <= self.point_estimate <= self.high):
            raise ValueError("ExposureEstimate requires low <= point_estimate <= high.")
        return self


# =============================================================================
# Market state — 4-field vector (§3 sec:ds-marketstate, invariant 5)
# =============================================================================
class MarketStateVector(BaseModel):
    """Four independent dimensions; never collapsed to a single label (§3 sec:ds-marketstate).

    Worked example (doc §3): 方向向上 / 波动高 / 压力正常 / 事件政策影响.
    """

    model_config = ConfigDict(extra="forbid")

    as_of: date = Field(..., description="Decision point this state is for.")
    direction: Direction
    volatility: Volatility
    stress: Stress
    event: MarketEvent


# =============================================================================
# Horizon forecast — ranges, not labels (§3 sec:ds-forecast, invariant 4)
# =============================================================================
class HorizonForecast(BaseModel):
    """Return/risk ranges for one horizon (§3 sec:ds-forecast).

    Emits ranges first (ret_low/mid/high, prob_positive, vol_est, drawdown_prob);
    product-facing labels are DERIVED from these in ProductView, never assigned here.
    """

    model_config = ConfigDict(extra="forbid")

    horizon: Horizon
    ret_low: float = Field(..., description="Worse-case return.")
    ret_mid: float = Field(..., description="Normal-case return.")
    ret_high: float = Field(..., description="Most-optimistic return.")
    prob_positive: UnitInterval = Field(..., description="Probability of positive return.")
    vol_est: float = Field(..., ge=0.0, description="Estimated volatility (how bumpy).")
    drawdown_prob: UnitInterval = Field(..., description="Probability of a material drawdown.")
    backtested: bool = Field(False, description="Whether this judgment holds up in backtest (§7).")

    @model_validator(mode="after")
    def _check_range(self) -> "HorizonForecast":
        if not (self.ret_low <= self.ret_mid <= self.ret_high):
            raise ValueError("HorizonForecast requires ret_low <= ret_mid <= ret_high.")
        return self


# =============================================================================
# Hierarchical-forecast attribution (源文档 §3.3 / techdesign §5.3 分层模型)
# =============================================================================
class IndustryContribution(BaseModel):
    """One industry's contribution to the industry (行业相对) layer of a fund's forecast.

    ``deviation`` is the industry index's expected return **relative to the broad market**
    over the horizon (行业相对影响 per unit exposure); ``exposure_weight`` is the fund's
    estimated exposure to that industry; ``contribution = equity_exposure * exposure_weight *
    deviation`` is its additive share of the fund's ret_mid (the deviation applies to the equity
    book, whose market loading is ``equity_exposure``).
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="Industry label (Shenwan L1 name), e.g. '电子'.")
    exposure_weight: UnitInterval = Field(..., description="Fund's estimated exposure to this industry.")
    deviation: float = Field(..., description="Industry expected return minus market, over the horizon.")
    contribution: float = Field(..., description="equity_exposure * exposure_weight * deviation — additive share of ret_mid.")
    resolved: bool = Field(..., description="Whether an industry index series was resolved (else 0 deviation, fail-closed).")


class ForecastAttribution(BaseModel):
    """Additive attribution of one horizon's ``ret_mid`` to the three §5.3 layers.

    ``ret_mid == market + industry + idiosyncratic`` (up to float rounding): the fund's
    expected change is decomposed into 大盘环境 (market), 行业暴露 (industry), and 少数个股的
    特殊表现 (idiosyncratic). This is the product-visible benefit of the hierarchical model —
    it lets the explanation say *where* an expected move comes from, without writing it as a
    causal conclusion (§5.3 designnote / §4.2). ``industries`` breaks the industry layer down
    per exposed industry.
    """

    model_config = ConfigDict(extra="forbid")

    horizon: Horizon
    market: float = Field(..., description="大盘影响: market_ret * equity_exposure.")
    industry: float = Field(..., description="行业相对影响: equity_exposure * sum of exposure-weighted industry deviations.")
    idiosyncratic: float = Field(..., description="个股自身相对影响: residual (kept small/conservative).")
    equity_exposure: MarketLoading = Field(..., description="Fund's market loading — the PIT rolling NAV beta (fallback: disclosed book sum), scaling BOTH the market and industry layers. May exceed 1 (a high-beta fund); clipped to the configured ceiling.")
    industries: list[IndustryContribution] = Field(default_factory=list, description="Per-industry breakdown of the industry layer.")


# =============================================================================
# Scope / visibility results (§4 sec:mod-scope, sec:mod-visibility)
# =============================================================================
class FundTypeProfile(BaseModel):
    """The analysis *caliber* for one fund type (§3 sec:ds-scope, §4 sec:mod-scope).

    A configurable per-category record (one per :class:`ScopeCategory`) that says how deep
    v1 analyzes this type and how strong its advice may be. Kept as data (a registry in
    ``config.ScopeConfig``) rather than if/else so a type's depth is changed by editing the
    table, and so compliance gate 3 can uniformly read ``advice_strength_cap``.
    """

    model_config = ConfigDict(extra="forbid")

    category: ScopeCategory
    support_tier: SupportTier
    applicable_lenses: list[str] = Field(
        default_factory=list,
        description="Analysis lenses for this type: equity {industry,style,cap}; bond "
        "{rate,credit,duration}; money {yield,liquidity}; QDII {overseas,fx}; FOF "
        "{underlying_fund_lookthrough}; …",
    )
    risk_drivers: list[str] = Field(
        default_factory=list, description="Main risk sources in user language (§1 sec:scope table)."
    )
    advice_strength_cap: AdviceStrengthCap = Field(
        ..., description="Strongest advice allowed; NONE for LIMITED/DEFERRED (gate 3)."
    )
    disclaimer: str = Field(
        "", description="'该类型深度分析暂未支持'-style notice; empty for FULL, mandatory otherwise."
    )


class ScopeVerdict(BaseModel):
    """Result of classify_scope (§4 sec:mod-scope, §3 sec:ds-scope).

    No longer a boolean admission flag: it says *what type* the fund is, *which caliber* it
    gets, and *how deep* v1 supports it. ``in_scope`` is a derived convenience view
    (``support_tier == FULL``) kept for back-compat; downstream gates read ``support_tier``
    and ``caliber.advice_strength_cap`` directly.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    category: ScopeCategory
    support_tier: SupportTier
    caliber: FundTypeProfile = Field(..., description="This type's analysis caliber / profile.")
    market: Optional[str] = Field(
        None, description="Trading venue; confirmed non-'O' (场内) → ETF_LISTED/DEFERRED."
    )
    reason: str = Field("", description="Why this category / support tier; shown to the user.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def in_scope(self) -> bool:
        """Back-compat: True iff the fund gets the FULL analysis pipeline (equity family)."""
        return self.support_tier is SupportTier.FULL


class VisibilitySnapshot(BaseModel):
    """Per-DataField visibility + freshness at an as_of (§4 sec:mod-visibility).

    ``visible`` iff available_at <= as_of; ``freshness`` from the age vs
    config.FreshnessConfig cutoffs. (Fields beyond the doc's prose — doc-silent, see
    DECISIONS.md.)"""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    visible: Visibility
    freshness: Freshness
    available_at: datetime
    age_days: int = Field(..., ge=0, description="as_of - available_at, in days.")


class VisibilitySet(BaseModel):
    """Collection returned by resolve_visibility (§4 lst:vis)."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    snapshots: list[VisibilitySnapshot] = Field(default_factory=list)

    def visible(self) -> list[VisibilitySnapshot]:
        return [s for s in self.snapshots if s.visible is Visibility.VISIBLE]

    def hidden(self) -> list[VisibilitySnapshot]:
        return [s for s in self.snapshots if s.visible is Visibility.HIDDEN]


# =============================================================================
# Advice-mode structures (§3 sec:ds-lookthrough, §4). UserPortfolio + advice-only.
# =============================================================================
class PortfolioHolding(BaseModel):
    """One line of a user's portfolio (§3 sec:ds-lookthrough)."""

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    weight: UnitInterval = Field(..., description="Portfolio weight in [0, 1].")
    hold_days: int = Field(..., ge=0, description="How long the user has held it.")


class UserPortfolio(BaseModel):
    """User portfolio — ONLY ever read in advice_mode (§3 sec:ds-lookthrough, §6 sec:cc-modes).

    analysis_mode code never receives this type (invariant 3, enforced by signatures)."""

    model_config = ConfigDict(extra="forbid")

    holdings: list[PortfolioHolding] = Field(default_factory=list)


class LookthroughResult(BaseModel):
    """Portfolio look-through result (§3 sec:ds-lookthrough). Advice-only.

    Worked example (doc §3): 5 funds → 63% tech-growth concentration.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    industry_conc: dict[str, float] = Field(default_factory=dict, description="label -> merged exposure.")
    style_conc: dict[str, float] = Field(
        default_factory=dict, description="growth/value/dividend/large/small concentration."
    )
    theme_overlap: list[str] = Field(default_factory=list, description="Single-theme repeated exposure.")
    corr_matrix: Optional[np.ndarray] = Field(None, description="Underlying-asset correlation (ndarray).")
    fund_risk_contrib: dict[str, float] = Field(
        default_factory=dict, description="Per-fund contribution to total portfolio risk."
    )
    budget_usage: float = Field(0.0, ge=0.0, description="Risk-budget usage (-> risk_budget).")
    fund_labels: list[str] = Field(
        default_factory=list, description="Fund order for corr_matrix rows/cols (doc-silent aid)."
    )


class RiskBudgetState(BaseModel):
    """Risk-budget assessment (§4 sec:mod-riskbudget). Advice-only.

    Fields beyond ``budget_usage`` are a doc-silent decision (see DECISIONS.md);
    over-budget when budget_usage exceeds config.RiskBudgetConfig.budget_usage_cap."""

    model_config = ConfigDict(extra="forbid")

    budget_usage: float = Field(..., ge=0.0, description="Normalized usage vs suitability cap.")
    over_budget: bool = Field(..., description="True when budget_usage > cap.")
    single_industry_max: float = Field(0.0, ge=0.0, le=1.0, description="Largest single-industry exposure.")
    breaching_dimensions: list[str] = Field(default_factory=list)


# =============================================================================
# Suitability & fees (§4 lst:modes, lst:action)
# =============================================================================
class RedemptionFeeTier(BaseModel):
    """One holding-period redemption-fee tier (short-holding fees handled explicitly, §7 rule 7)."""

    model_config = ConfigDict(extra="forbid")

    min_hold_days: int = Field(..., ge=0)
    max_hold_days: Optional[int] = Field(None, description="None = open-ended upper tier.")
    rate: float = Field(..., ge=0.0, description="Redemption fee rate for this tier.")


class FeeSchedule(BaseModel):
    """Fee schedule feeding action_strategy net-of-cost accounting (§4 lst:action, §6 #4).

    Field-level shape is a doc-silent decision (see DECISIONS.md)."""

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    purchase_fee_rate: float = Field(0.0, ge=0.0, description="申购费率.")
    redemption_tiers: list[RedemptionFeeTier] = Field(
        default_factory=list, description="赎回费率, by holding period (short-hold explicit)."
    )
    sales_service_fee_rate: float = Field(0.0, ge=0.0, description="销售服务费率 (annualized).")

    def redemption_rate(self, hold_days: int) -> float:
        """Return the redemption rate applicable to a given holding period."""
        for tier in self.redemption_tiers:
            upper_ok = tier.max_hold_days is None or hold_days < tier.max_hold_days
            if hold_days >= tier.min_hold_days and upper_ok:
                return tier.rate
        return 0.0


class SuitabilityProfile(BaseModel):
    """Investor-suitability profile — required for personalized advice (§6 #7, sec:cc-modes).

    Field-level shape is a doc-silent decision (see DECISIONS.md); the doc requires
    only that product risk match the investor category and adjust as info changes."""

    model_config = ConfigDict(extra="forbid")

    risk_tolerance: RiskTolerance
    investment_horizon_days: int = Field(..., ge=0, description="Target holding horizon.")
    investment_objective: str = Field("", description="Stated objective (free text for v1).")
    max_single_industry_exposure: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Optional user-specific concentration cap."
    )


# =============================================================================
# Usable cash + cash allocation (§2.8 / §3.4 / §3 sec:ds-cash) — advice-only.
# =============================================================================
class CashStaging(str, Enum):
    """Whether the usable cash is deployed at once or in stages (§2.8)."""

    ONE_TIME = "one_time"   # 一次性买入
    STAGED = "staged"       # 分批买入 (定投节奏)


class UsableCashInput(BaseModel):
    """A lump of cash the user is willing to allocate to funds this round (§2.8).

    NOT all the user's cash and NOT their emergency fund — an explicit constraint input
    to :func:`~qmt.pipeline.cash_allocation.allocate_cash`. Advice-mode only.
    """

    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., gt=0.0, description="Max amount to invest this round (¥).")
    must_fully_invest: bool = Field(False, description="Whether the cash must be fully deployed.")
    allow_hold_cash: bool = Field(True, description="Whether keeping cash is allowed (it is a position).")
    staged: CashStaging = Field(CashStaging.ONE_TIME, description="One-time vs staged (SIP) deployment.")
    min_cash_reserve: float = Field(0.0, ge=0.0, description="Floor of cash to keep uninvested (¥).")


class CashAllocationItem(BaseModel):
    """One fund's slice of a :class:`CashAllocationPlan` (§3 sec:ds-cash)."""

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    amount: float = Field(..., ge=0.0, description="Amount allocated to this fund (¥).")
    weight: UnitInterval = Field(..., description="Share of the usable cash [0, 1].")
    score: float = Field(0.0, description="Candidate risk-return score that drove the allocation.")


class CashAllocationPlan(BaseModel):
    """Output of cash_allocation (§3 sec:ds-cash): invest / how much / which funds / keep how much.

    Protection rule (§2.8): '花光现金' is a conditional goal — when no candidate clears the
    quality bar, ``invest=False`` and ``kept_cash`` retains the cash (cash is a position).
    """

    model_config = ConfigDict(extra="forbid")

    invest: bool = Field(..., description="Whether to deploy cash this round (False → keep cash).")
    allocations: list[CashAllocationItem] = Field(default_factory=list)
    kept_cash: float = Field(..., ge=0.0, description="Cash left uninvested (¥); ≥ min_cash_reserve.")
    rationale: str = Field("", description="Why this allocation / why cash is kept (non-causal).")


# =============================================================================
# Calibration table (§4 lst:uncert)
# =============================================================================
class ConfidenceBucket(BaseModel):
    """One calibration bucket: raw-confidence range -> historical hit-rate (§4 sec:mod-uncertainty)."""

    model_config = ConfigDict(extra="forbid")

    lower: UnitInterval
    upper: UnitInterval
    hit_rate: UnitInterval = Field(..., description="Historical fraction that were actually right.")
    sample_count: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "ConfidenceBucket":
        if self.lower > self.upper:
            raise ValueError("ConfidenceBucket requires lower <= upper.")
        return self


class CalibrationTable(BaseModel):
    """History of confidence -> realized hit-rate, used by calibrate_confidence (§4 lst:uncert)."""

    model_config = ConfigDict(extra="forbid")

    buckets: list[ConfidenceBucket] = Field(default_factory=list)

    def hit_rate_for(self, raw_confidence: float) -> Optional[float]:
        """Look up the calibrated hit-rate for a raw confidence, or None if uncovered."""
        for b in self.buckets:
            if b.lower <= raw_confidence <= b.upper:
                return b.hit_rate
        return None


# =============================================================================
# Outputs — ProductView / ActionRecommendation / ExplanationCard (§3 sec:ds-action, §4 sec:mod-explain)
# =============================================================================
class ProductView(BaseModel):
    """Product-language view DERIVED from the underlying numbers (§3 sec:ds-action).

    direction/trend_strength/volatility/confidence are derived (never subjective
    labels); ``action`` is populated only in advice_mode.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    as_of: date
    direction: Direction = Field(..., description="From expected return + prob_positive.")
    trend_strength: float = Field(..., ge=0.0, le=1.0, description="From return magnitude + volatility.")
    volatility: Volatility = Field(..., description="From forecast range + bumpiness.")
    confidence: UnitInterval = Field(..., description="From historically-calibrated reliability.")
    action: Optional[ActionAdvice] = Field(
        None, description="Only set in advice_mode (§3 sec:ds-action)."
    )


class ActionRecommendation(BaseModel):
    """The output of decide_action / run_advice (doc's ``ActionAdvice`` data structure).

    Naming decision (see DECISIONS.md): the doc's ``ActionAdvice`` *data structure*
    with an ``action`` enum field is realized as this ``ActionRecommendation`` wrapper
    holding an ``action: ActionAdvice`` enum, per the orchestrator task's requested
    two-type split. Advice-only.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    as_of: date
    action: ActionAdvice = Field(..., description="One of the six closed-set action values.")
    rationale: str = Field("", description="Why this action; net-of-cost, non-causal.")
    gated_by: list[str] = Field(
        default_factory=list, description="Names of compliance/strategy gates that fired."
    )


class ExplanationCard(BaseModel):
    """The explanation output (§4 sec:mod-explain, §7 sec:eval-explain).

    The doc describes the card two ways — the product-facing fixed five sections
    (§4) and the evidence structure (§7). Both are modeled: the five product
    sections plus the evidence fields the §7 test asserts non-empty. Must never
    present model attribution as causal fact.
    """

    model_config = ConfigDict(extra="forbid")

    fund_code: FundCode
    as_of: date
    # Five product-facing sections (§4 sec:mod-explain).
    why_optimistic: str = Field(..., description="为什么偏乐观.")
    why_cautious: str = Field(..., description="为什么仍然需要谨慎.")
    unknowns: str = Field(..., description="我们不知道什么.")
    data_freshness: str = Field(..., description="数据有多新.")
    what_would_change: str = Field(..., description="什么情况会改变当前判断.")
    # Evidence structure (§7 sec:eval-explain) — asserted non-empty in tests.
    evidence_seen: list[str] = Field(default_factory=list, description="已经看到的证据.")
    driving_factors: list[str] = Field(default_factory=list, description="推动判断的因素 (must be non-empty).")
    counter_evidence: list[str] = Field(default_factory=list, description="反向证据 (must be non-empty).")
    uncertainties: list[str] = Field(default_factory=list, description="不确定的地方.")
    change_conditions: list[str] = Field(default_factory=list, description="判断改变条件.")
    disclosure_age_days: int = Field(..., ge=0, description="Data-freshness anchor (§7 assertion).")


__all__ = [
    # scalars
    "FundCode",
    "UnitInterval",
    # enums
    "DataSource",
    "UpdateFreq",
    "MissingPolicy",
    "AppliesTo",
    "Visibility",
    "Freshness",
    "ExposureDimension",
    "Direction",
    "Volatility",
    "Stress",
    "MarketEvent",
    "Horizon",
    "DisclosureDepth",
    "FundStatus",
    "ScopeCategory",
    "SupportTier",
    "AdviceStrengthCap",
    "RiskTolerance",
    "ActionAdvice",
    # data-layer records
    "FundMeta",
    "FundProfile",
    "NavPoint",
    "HoldingLine",
    "DisclosedHoldings",
    "DataField",
    # timeline
    "DecisionTimeline",
    # intermediate
    "ExposureEstimate",
    "MarketStateVector",
    "HorizonForecast",
    "FundTypeProfile",
    "ScopeVerdict",
    "VisibilitySnapshot",
    "VisibilitySet",
    # advice-only
    "PortfolioHolding",
    "UserPortfolio",
    "LookthroughResult",
    "RiskBudgetState",
    # suitability & fees
    "RedemptionFeeTier",
    "FeeSchedule",
    "SuitabilityProfile",
    # usable cash + allocation
    "CashStaging",
    "UsableCashInput",
    "CashAllocationItem",
    "CashAllocationPlan",
    # calibration
    "ConfidenceBucket",
    "CalibrationTable",
    # outputs
    "ProductView",
    "ActionRecommendation",
    "ExplanationCard",
]
