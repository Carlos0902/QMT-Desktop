"""Shared data-structure contract for the symbolic-learning factor platform.

This module is the single source of truth every ``qmt.factor`` module imports.
Every structure maps to the technical design (``doc/techdesign/`` ch.3 + appendix
A) and, through it, to ``doc/instruction.md``. Cross-references below cite the
instruction section (``§``) each field implements.

Design choices:
  * Plain ``@dataclass`` (matching the tech doc's ``FactorRecord`` listing) rather
    than pydantic — these are internal research structures, not API payloads.
  * Point-in-time correctness is physical: every observable datum carries an
    ``available_at`` (see :class:`DataField`); the ``safe_data`` layer builds on
    :class:`qmt.pit.PointInTimeStore`, whose ``get_asof`` already enforces it.
  * The nine module ``Protocol``s (ch.4 / appendix A) are declared here so
    implementations and orchestration share one contract.

The AST node classes for :class:`FactorExpr` live in ``qmt.factor.dsl`` to keep the
grammar self-contained; ``FactorExpr.ast`` is typed ``Any`` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataSource(str, Enum):
    """Where a :class:`DataField` came from (instruction §3.1)."""

    FUND_ANNOUNCEMENT = "fund_announcement"
    MARKET = "market"
    INDEX = "index"
    FINANCIAL_REPORT = "financial_report"
    FUND_FLOW = "fund_flow"
    FEE = "fee"
    PLATFORM = "platform"
    ALT = "alt"  # 另类数据：须过数据授权/来源核验


class UpdateFreq(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"


class Freshness(str, Enum):
    """Derived from disclosure age vs. update period — never hand-labelled."""

    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"


class MissingPolicy(str, Enum):
    """How missing/delayed/restated data is handled (instruction §3.6). Never fill 0."""

    DROP = "drop"
    FFILL_CAPPED = "ffill_capped"
    AS_OF_LAG = "as_of_lag"


class FactorStatus(str, Enum):
    """Factor lifecycle (tech doc §3.3 FactorRecord listing)."""

    CANDIDATE = "candidate"  # 刚由 auto_search 产出
    SCREENED = "screened"    # 通过 factor_screening
    COMBINED = "combined"    # 已进入 combination 组合
    ARCHIVED = "archived"    # 留档未启用
    RETIRED = "retired"      # 失效/漂移，已下线


class ValidationScheme(str, Enum):
    ROLLING = "rolling"          # 滚动验证：模拟真实上线
    PURGED_KFOLD = "purged_kfold"  # 去重叠交叉验证
    CPCV = "cpcv"                # 多路径组合，看表现分布


class CostTier(str, Enum):
    """Three cost assumptions (instruction §4.6)."""

    OPT = "opt"  # 乐观档：只计佣金和部分买卖价差
    NEU = "neu"  # 中性档：加滑点 + 简单市场冲击
    CON = "con"  # 保守档：加做空/税费/不利成交/容量折损


class GateName(str, Enum):
    """The four multi-stage gates (instruction §7.5)."""

    PREDICTIVE = "predictive"       # 预测性门槛
    ROBUSTNESS = "robustness"       # 稳健性门槛
    TRADABILITY = "tradability"     # 交易性门槛
    PORTFOLIO_INCREMENT = "portfolio_increment"  # 组合增量门槛


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class UniverseStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    LIQUIDATED = "liquidated"
    MERGED = "merged"


# ---------------------------------------------------------------------------
# safe_data layer: DataField (point-in-time) — instruction §3.2
# ---------------------------------------------------------------------------


@dataclass
class DataField:
    """Point-in-time metadata for one observable datum (tech doc §3.1 / ch.3).

    Two timestamps are mandatory in spirit: ``period_covered`` (what period it
    describes) and ``available_at`` (when investors could actually know it). A
    field missing ``available_at`` MUST be rejected before backtesting — otherwise
    it silently leaks the future (instruction §3.2, §3.5).
    """

    field_name: str
    source: DataSource
    available_at: datetime               # 历史真正可见时刻；缺失即拒收
    period_covered: date | tuple[date, date] | None = None
    update_freq: UpdateFreq = UpdateFreq.DAILY
    freshness: Freshness = Freshness.FRESH
    visible_at_decision: bool = True     # T 日 14:30 是否可见
    schema_changed: bool = False         # 口径是否变过（如北向披露机制调整）
    missing_policy: MissingPolicy = MissingPolicy.DROP
    applies_to: frozenset[str] = field(default_factory=lambda: frozenset({"equity"}))

    def __post_init__(self) -> None:
        if self.available_at is None:  # fail-closed, mirrors PointInTimeStore.put
            raise ValueError(
                f"DataField {self.field_name!r} missing available_at — cannot enter "
                "backtest (would silently leak future data; instruction §3.2)."
            )


@dataclass
class SafePanel:
    """Point-in-time-safe, cleaned wide panel produced by ``safe_data``.

    ``frame`` is a tidy long DataFrame indexed by (fund_code, field) columns with a
    value column, OR a wide (fund_code x field) frame; implementations document
    which. Only fields with ``available_at <= as_of`` are present (hidden fields
    dropped). ``fields`` retains provenance for every column.
    """

    as_of: date
    frame: pd.DataFrame
    fields: dict[str, DataField] = field(default_factory=dict)
    universe: list[str] = field(default_factory=list)  # fund codes visible at as_of


# ---------------------------------------------------------------------------
# base_factor_library: BaseFactorPanel
# ---------------------------------------------------------------------------


@dataclass
class BaseFactorPanel:
    """Base indicators — the raw material for symbolic search (instruction §5.1 L3).

    A 3-D cross-sectional structure: for each ``as_of`` date, a (fund x base_factor)
    matrix. Stored as a dict of wide DataFrames keyed by factor name, each indexed
    by date with fund codes as columns, so DSL evaluation can slice one ``as_of``
    cross-section cheaply. Every base factor inherits point-in-time safety.
    """

    dates: list[date]
    funds: list[str]
    factors: dict[str, pd.DataFrame]     # name -> (date index x fund columns)
    industry_of: dict[str, str] = field(default_factory=dict)  # fund/asset -> industry (neutralization)
    meta: dict[str, DataField] = field(default_factory=dict)

    def cross_section(self, factor: str, as_of: date) -> pd.Series:
        """The (fund -> value) cross-section of one base factor at ``as_of``."""
        frame = self.factors[factor]
        if as_of not in frame.index:
            # nearest prior available row (as-of semantics)
            prior = frame.index[frame.index <= as_of]
            if len(prior) == 0:
                return pd.Series(dtype=float)
            as_of = prior[-1]
        return frame.loc[as_of].dropna()


# ---------------------------------------------------------------------------
# factor_dsl: FactorExpr — instruction §2.1 / §2.3 / §5.3
# ---------------------------------------------------------------------------


@dataclass
class FactorExpr:
    """One human-readable formula factor generated by ``auto_search`` (tech doc §3.2).

    Keeps both the readable ``text`` and the machine ``ast`` (defined in
    ``qmt.factor.dsl``). Records which data fields / time windows / operators it uses
    so point-in-time safety and DSL-whitelist membership are checkable.
    """

    text: str
    ast: Any                                     # dsl.Node tree (serialisable/mutable)
    data_fields: list[str] = field(default_factory=list)
    time_windows: list[int] = field(default_factory=list)
    operators: frozenset[str] = field(default_factory=frozenset)
    complexity: int = 0                          # AST node/depth/op measure; capped
    guarded_ops: list[str] = field(default_factory=list)  # div/log/sqrt protections

    def __hash__(self) -> int:  # allow use as dict key / dedup by text
        return hash(self.text)


@dataclass
class DSLVerdict:
    """Result of ``factor_dsl.validate`` — a hard gate, not a warning (tech doc §4.3)."""

    status: GateStatus
    reasons: list[str] = field(default_factory=list)     # why rejected
    complexity: int = 0

    @property
    def ok(self) -> bool:
        return self.status is GateStatus.PASS


# ---------------------------------------------------------------------------
# Prediction target: Label — instruction §3.3 / §3.4
# ---------------------------------------------------------------------------


@dataclass
class Label:
    """Prediction target for cross-sectional ranking (tech doc §3.4).

    v1 default: future 20–60 trading-day *relative* return (alpha caliber) and
    drawdown, weekly/biweekly rebalance. 1/5/10-day returns are auxiliary
    diagnostics only. ``as_of`` is the executable decision point; nothing after it
    may be used (point-in-time).
    """

    fund_code: str
    as_of: date
    horizon_days: int                     # default range 20..60
    rel_return: float                     # market/peer-adjusted (alpha caliber)
    drawdown: float                       # <= 0, max drawdown within window
    rebalance: str = "weekly"             # weekly | biweekly
    aux_1_5_10: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation split & survivorship universe — instruction §4.2 / §3.5
# ---------------------------------------------------------------------------


@dataclass
class ValidationSplit:
    """Overlap-free validation split (tech doc §3.7).

    ``embargo_days`` must be >= the label window length (default >= 60) so training
    and validation labels never overlap in time (otherwise leakage).
    """

    split_id: str
    scheme: ValidationScheme
    train_span: tuple[date, date]
    valid_span: tuple[date, date]
    embargo_days: int = 60
    n_paths: int = 1
    as_of_series: list[date] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.valid_span[0] <= self.train_span[1]:
            raise ValueError(
                f"valid_span must start strictly after train_span "
                f"({self.valid_span[0]} <= {self.train_span[1]})."
            )


@dataclass
class UniverseMember:
    """One fund's point-in-time membership in the research universe (§3.5)."""

    fund_code: str
    inception_date: date
    end_date: date | None = None          # 清盘/合并/退市日；None=存续
    status: UniverseStatus = UniverseStatus.ACTIVE
    end_reason: str | None = None
    tradable_spans: list[tuple[date, date]] = field(default_factory=list)

    def investable_at(self, as_of: date) -> bool:
        if as_of < self.inception_date:
            return False
        if self.end_date is not None and as_of >= self.end_date:
            return False
        if self.tradable_spans:
            return any(lo <= as_of <= hi for lo, hi in self.tradable_spans)
        return True


@dataclass
class FundUniverse:
    """Survivorship-complete universe: MUST include delisted/liquidated funds (§3.5)."""

    members: dict[str, UniverseMember] = field(default_factory=dict)

    def at(self, as_of: date) -> list[str]:
        """Codes genuinely investable at ``as_of`` (point-in-time membership)."""
        return sorted(c for c, m in self.members.items() if m.investable_at(as_of))

    def add(self, member: UniverseMember) -> None:
        self.members[member.fund_code] = member


# ---------------------------------------------------------------------------
# Cost tiers — instruction §4.4 / §4.6
# ---------------------------------------------------------------------------


@dataclass
class CostComponents:
    """Toggle set + rates for one cost tier (instruction §4.4 component list)."""

    commission: bool = True
    spread: str = "partial"               # "none" | "partial" | "full"
    slippage: bool = False
    market_impact: str = "none"           # "none" | "simple" | "full"
    short_cost: bool = False
    taxes: bool = False
    adverse_fill: bool = False            # 不利成交 / 容量折损
    fund_fees: bool = False               # 申购/赎回费、最低持有期短赎费（基金专属）
    # scalar rates (annualised / per-trade fractions) — tunable, marked 【推断】
    commission_rate: float = 0.0002
    spread_rate: float = 0.0005
    slippage_rate: float = 0.001
    impact_rate: float = 0.0008


@dataclass
class CostTiers:
    """Three cost assumptions; screening's hard gate uses CON, never OPT (§4.6)."""

    opt: CostComponents = field(
        default_factory=lambda: CostComponents(
            spread="partial", slippage=False, market_impact="none",
        )
    )
    neu: CostComponents = field(
        default_factory=lambda: CostComponents(
            spread="full", slippage=True, market_impact="simple", fund_fees=True,
        )
    )
    con: CostComponents = field(
        default_factory=lambda: CostComponents(
            spread="full", slippage=True, market_impact="full", short_cost=True,
            taxes=True, adverse_fill=True, fund_fees=True,
        )
    )

    def get(self, tier: CostTier) -> CostComponents:
        return {CostTier.OPT: self.opt, CostTier.NEU: self.neu, CostTier.CON: self.con}[tier]


# ---------------------------------------------------------------------------
# Search config, ledger, scoring, gates
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    """Search space + budget for one auto_search run (tech doc §3.3, §5.1 L5)."""

    config_id: str
    label_horizon_days: int = 40          # within 20..60
    allowed_fields: list[str] = field(default_factory=list)
    allowed_operators: frozenset[str] = field(default_factory=frozenset)
    allowed_windows: tuple[int, ...] = (5, 20, 60)
    max_complexity: int = 12              # hard cap — never relaxed (§2.3, §7.3)
    max_depth: int = 4
    max_trials: int = 500                 # search budget (recorded in ledger)
    population: int = 60                  # GP population size
    generations: int = 8
    seed: int = 20240703
    seed_from_manual: bool = True         # start near known manual factors (§5.6)
    diversity_max_corr: float | None = None  # opt-in train-only semantic pruning; None = ablation
    diversity_min_dates: int = 3          # shared train cross-sections needed to prune
    min_train_ic_coverage: float = 0.8    # fraction of fixed train dates scorable per fold


@dataclass
class SearchTrial:
    """One recorded search attempt (winner or loser) — overfit governance (§7.3)."""

    trial_id: int
    expr_text: str
    passed_dsl: bool
    select_ic: float | None = None
    parent_id: str | None = None
    generation: int = 0
    note: str = ""


@dataclass
class SearchLedger:
    """Process-level overfit governance (instruction §7.3).

    Records how many formulas were tried, how many times the dataset was
    re-examined, which params were chosen on validation, and enforces a LOCKED
    final test region: once viewed it can no longer inform the DSL, windows,
    screening, or objective. DSR/PBO need this complete record.
    """

    trials: list[SearchTrial] = field(default_factory=list)
    dataset_views: int = 0                        # 同一数据被查看的次数
    params_decided_on_validation: list[str] = field(default_factory=list)
    locked_test_span: tuple[date, date] | None = None
    _test_span_viewed: bool = False

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def record(self, trial: SearchTrial) -> None:
        self.trials.append(trial)

    def view_dataset(self) -> None:
        self.dataset_views += 1

    def lock_test_span(self, span: tuple[date, date]) -> None:
        self.locked_test_span = span

    def mark_test_viewed(self) -> None:
        self._test_span_viewed = True

    def assert_test_untouched(self) -> None:
        """Raise if the locked test region has been consumed by a design decision."""
        if self._test_span_viewed:
            raise RuntimeError(
                "Locked final test region already viewed — it can no longer inform "
                "DSL/windows/screening/objective (instruction §7.3). Test set is "
                "becoming a training set."
            )


@dataclass
class FactorScore:
    """Two-step factor score (instruction §5.4). Fields populated per step."""

    factor_id: str
    # step 1 — standalone usefulness (cost-net, out-of-sample)
    ic: float = 0.0
    rank_ic: float = 0.0
    ic_tstat: float = 0.0                 # overlap-corrected (qmt.stats)
    ic_significant: bool = False          # bootstrap CI + HAC t agree (§4.3)
    monotonic: bool = False               # 分组收益单调
    turnover: float = 0.0
    net_return_by_tier: dict[str, float] = field(default_factory=dict)
    regime_stable: bool = False           # 非仅在某段行业/时间有效
    # step 2 — portfolio increment
    max_corr_existing: float = 0.0        # 与已有因子最大相关
    increment: float = 0.0                # 组合净收益/尾部风险改善
    redundant: bool = False               # 现有因子的复杂重写
    # aggregate
    gate_report: "GateReport | None" = None

    @property
    def standalone_ok(self) -> bool:
        return self.monotonic and self.regime_stable and self.rank_ic > 0


@dataclass
class GateReport:
    """Outcome of the four-gate evaluation (tech doc §5.5 / appendix). Ordered:
    any FAIL eliminates; a high score on one gate cannot buy back a failure on
    another (instruction §7.5)."""

    results: dict[GateName, GateStatus] = field(default_factory=dict)
    reasons: dict[GateName, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(s is GateStatus.PASS for s in self.results.values()) and bool(self.results)

    def first_failure(self) -> GateName | None:
        for g in (GateName.PREDICTIVE, GateName.ROBUSTNESS,
                  GateName.TRADABILITY, GateName.PORTFOLIO_INCREMENT):
            if self.results.get(g) is GateStatus.FAIL:
                return g
        return None


# ---------------------------------------------------------------------------
# factor_archive: FactorRecord — instruction §5.2 (verbatim field mapping)
# ---------------------------------------------------------------------------


@dataclass
class FactorRecord:
    """Full factor archive record (tech doc §3.3; instruction §5.2 nine items).

    Failed candidates are archived too — never keep only winners (§5.2, §7.3),
    otherwise the search itself has survivorship bias and DSR/PBO cannot correct
    for how many attempts produced the winner.
    """

    factor_id: str
    expr: FactorExpr                                   # §5.2-(1)(2)(3)(4)
    parent_id: str | None = None                       # §5.2-(5) lineage
    search_config_id: str = ""
    train_span: tuple[date, date] | None = None        # §5.2-(6)
    valid_span: tuple[date, date] | None = None
    split_id: str = ""
    ic: float = 0.0                                    # §5.2-(7) OOS IC mean
    rank_ic: float = 0.0
    turnover: float = 0.0
    net_return_by_tier: dict[str, float] = field(default_factory=dict)  # opt/neu/con
    capacity_est: float | None = None                  # §4.5 may defer
    industry_exposure: dict[str, float] = field(default_factory=dict)   # §5.2-(8)
    style_exposure: dict[str, float] = field(default_factory=dict)
    market_exposure: float = 0.0
    compliance_passed: bool = False                    # §5.2-(9)
    data_source_checked: bool = False
    status: FactorStatus = FactorStatus.CANDIDATE
    score: float | None = None                         # FactorScore aggregate
    reject_reason: str | None = None                   # why retired/failed (audit)
    created_at: datetime = field(default_factory=lambda: datetime(1970, 1, 1))


# ---------------------------------------------------------------------------
# combination / execution_sim / monitoring auxiliary structures
# ---------------------------------------------------------------------------


@dataclass
class FundScores:
    """Final per-fund composite score at one ``as_of`` (combination output)."""

    as_of: date
    scores: pd.Series                     # fund_code -> composite score
    weights: pd.Series | None = None      # target weights after risk overlay
    by_category: dict[str, pd.Series] = field(default_factory=dict)  # 价量/财务/风险
    risk_notes: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Current holdings entering execution simulation."""

    as_of: date
    holdings: pd.Series                   # fund_code -> current weight
    hold_days: dict[str, int] = field(default_factory=dict)  # for redemption-fee tiers
    cash: float = 1.0


@dataclass
class ExecResult:
    """Net-of-cost outcome per cost tier (execution_sim output)."""

    as_of: date
    target: pd.Series
    executed: pd.Series
    gross_return: float = 0.0
    net_return_by_tier: dict[str, float] = field(default_factory=dict)
    cost_by_tier: dict[str, float] = field(default_factory=dict)
    turnover: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class PerfSnapshot:
    """Live / shadow-run realised performance for monitoring."""

    as_of: date
    realized_return: float
    realized_ic: float | None = None
    exposure: dict[str, float] = field(default_factory=dict)
    net_cost: float = 0.0


@dataclass
class Anomaly:
    """A monitoring finding written back to the archive (§5.1 L10)."""

    as_of: date
    kind: str                             # perf_drop | factor_decay | risk_drift | cost_overrun
    factor_id: str | None
    detail: str
    severity: str = "warn"                # info | warn | critical


# ---------------------------------------------------------------------------
# Module Protocols — the ten-layer contract (tech doc ch.4 / appendix A)
# ---------------------------------------------------------------------------


@runtime_checkable
class SafeDataLayer(Protocol):
    def as_of_view(self, fields: Sequence[DataField], as_of: datetime) -> SafePanel: ...
    def clean(self, panel: SafePanel) -> SafePanel: ...


@runtime_checkable
class BaseFactorLibrary(Protocol):
    def compute_base_factors(self, panel: SafePanel, as_of: date) -> BaseFactorPanel: ...


@runtime_checkable
class FactorDSL(Protocol):
    def parse(self, expr_text: str) -> FactorExpr: ...
    def validate(self, expr: FactorExpr, cfg: SearchConfig) -> DSLVerdict: ...
    def evaluate(self, expr: FactorExpr, panel: BaseFactorPanel, as_of: date) -> pd.Series: ...


@runtime_checkable
class AutoSearch(Protocol):
    def search(self, cfg: SearchConfig, dsl: FactorDSL, panel: BaseFactorPanel,
               labels: Sequence[Label], split: ValidationSplit,
               ledger: SearchLedger) -> list[FactorExpr]: ...


@runtime_checkable
class FactorArchive(Protocol):
    def put(self, record: FactorRecord) -> str: ...
    def get(self, factor_id: str) -> FactorRecord: ...
    def query(self, **filters: Any) -> list[FactorRecord]: ...


@runtime_checkable
class FactorScreening(Protocol):
    def score_standalone(self, expr: FactorExpr, panel: BaseFactorPanel,
                         labels: Sequence[Label], split: ValidationSplit,
                         costs: CostTiers) -> FactorScore: ...
    def score_incremental(self, expr: FactorExpr, existing: Sequence[FactorExpr],
                          panel: BaseFactorPanel, labels: Sequence[Label],
                          split: ValidationSplit) -> FactorScore: ...


@runtime_checkable
class Combination(Protocol):
    def combine(self, factors: Sequence[FactorExpr], scores: Mapping[str, FactorScore],
                panel: BaseFactorPanel, as_of: date) -> FundScores: ...


@runtime_checkable
class ExecutionSim(Protocol):
    def simulate(self, target: FundScores, portfolio_state: PortfolioState,
                 as_of: date, costs: CostTiers) -> ExecResult: ...


@runtime_checkable
class Monitoring(Protocol):
    def check(self, live_perf: PerfSnapshot, expected: ExecResult,
              archive: FactorArchive, as_of: date) -> list[Anomaly]: ...


__all__ = [
    # enums
    "DataSource", "UpdateFreq", "Freshness", "MissingPolicy", "FactorStatus",
    "ValidationScheme", "CostTier", "GateName", "GateStatus", "UniverseStatus",
    # structures
    "DataField", "SafePanel", "BaseFactorPanel", "FactorExpr", "DSLVerdict",
    "Label", "ValidationSplit", "UniverseMember", "FundUniverse",
    "CostComponents", "CostTiers", "SearchConfig", "SearchTrial", "SearchLedger",
    "FactorScore", "GateReport", "FactorRecord", "FundScores", "PortfolioState",
    "ExecResult", "PerfSnapshot", "Anomaly",
    # protocols
    "SafeDataLayer", "BaseFactorLibrary", "FactorDSL", "AutoSearch",
    "FactorArchive", "FactorScreening", "Combination", "ExecutionSim", "Monitoring",
]
