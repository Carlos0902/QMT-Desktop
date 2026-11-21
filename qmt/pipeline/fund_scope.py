"""fund_scope — the classifier + caliber router at the head of the pipeline (§4 sec:mod-scope, §1.8-1).

Responsibility (doc, extended scope): *classify* the fund's type and *route* it to the
right analysis caliber. It takes a :class:`~qmt.types.FundProfile` and returns a
:class:`~qmt.types.ScopeVerdict` carrying the fund's :class:`~qmt.types.ScopeCategory`,
its :class:`~qmt.types.SupportTier`, and its :class:`~qmt.types.FundTypeProfile` caliber.

**Covers every 场外 fund — never hard-rejects (§1 sec:scope).** The equity family
(A 股权益 / 行业主题 / 指数 / 偏股混合) gets ``FULL`` support (the full exposure→market→
forecast→advice pipeline); every other 场外 type (bond / money / QDII / FOF / commodity /
pension / active-hybrid / unclassified) gets ``LIMITED`` — classified, degraded, with a
mandatory disclaimer and no strong/personalized output. A confirmed 场内 (exchange-traded)
fund is recognized as ``ETF_LISTED`` but ``DEFERRED`` — its 场内 mechanics are not yet built.

Doc signature (04-modules.tex, lst:scope)::

    def classify_scope(fund: FundProfile) -> ScopeVerdict:
        \"\"\"把基金判入一个 ScopeCategory，并据类型选择 FundTypeProfile 与 support_tier。
        覆盖全部场外基金，不硬拒绝。\"\"\"

``check_scope`` is kept as a verbatim doc-vocabulary alias (D53).

**Baseline method (this module).** A transparent, fully config-driven mapping over the two
Tushare ``fund_basic`` fields the profile carries — ``fund_type`` (投资类型) and
``invest_type`` (投资风格) — plus the fund ``name``, matched against
:class:`~qmt.config.ScopeConfig` (the type-rule fields + the ``caliber_table``). No holdings,
no NAV, no network. Doc's "simple transparent baseline first" (§5); a richer candidate
(verify type against disclosed holdings / NLP over the legal mandate) can replace
:func:`_classify_category` without changing the contract.

**Never raises; fail-closed default is UNCLASSIFIED (LIMITED).** A type that cannot be
resolved to a specific category falls back to ``UNCLASSIFIED`` (still ``LIMITED`` — no strong
output), *not* a reject. ``classify_scope`` is a pure function and never raises; an
*unresolvable identity* (no observable ``fund_basic``) is a separate upstream failure handled
by the orchestrator (fail-closed raise), not here.

**Documented seam — a *blank* ``market`` is not treated as 场内 (D50).** ``fund_basic`` often
omits ``market``; only a *confirmed* non-OTC market (``!= "O"``) routes to ETF_LISTED/DEFERRED,
a blank market falls through to the type classification.
"""

from __future__ import annotations

from typing import Optional

from qmt.config import CONFIG, ScopeConfig
from qmt.types import FundProfile, ScopeCategory, ScopeVerdict, SupportTier

__all__ = ["classify_scope", "check_scope"]

# --- doc-silent classification constants (see DECISIONS.md D50-D52) ------------------

# The off-exchange (场外) market code. v1 covers only OTC open-end funds; an
# exchange-traded (场内, ``E``) share class is out of the v1 universe (D50).
_OTC_MARKET_CODE = CONFIG.tushare.otc_market_code  # "O"

# Substring that separates the *index* invest-types from the *active-equity* ones inside
# ``ScopeConfig.in_scope_invest_types`` (被动指数型 / 增强指数型 / 股票指数 all contain
# "指数"; 普通股票型 / 偏股混合型 do not). Deriving the partition from the frozen config
# keeps a single source of truth for the in-scope style set (D51).
_INDEX_MARKER = "指数"

# Sector/theme keywords (行业主题). An in-scope *active* equity fund whose name advertises a
# single industry / theme is classified SECTOR_THEME rather than broad A_SHARE_EQUITY. Seed
# list from the doc's own worked examples (§1 sec:scope row 2: 新能源、半导体、医药、消费、
# AI) plus the common A-share sector vocabulary; passive index funds are decided by
# invest_type before this list is consulted, so a "半导体指数" fund stays INDEX (D52).
SECTOR_THEME_KEYWORDS: tuple[str, ...] = (
    "新能源",
    "半导体",
    "芯片",
    "集成电路",
    "医药",
    "医疗",
    "生物",
    "创新药",
    "消费",
    "白酒",
    "食品饮料",
    "科技",
    "人工智能",
    "AI",
    "军工",
    "国防",
    "银行",
    "证券",
    "券商",
    "保险",
    "地产",
    "房地产",
    "有色",
    "钢铁",
    "煤炭",
    "电子",
    "通信",
    "计算机",
    "传媒",
    "互联网",
    "光伏",
    "新能源车",
    "电力",
    "环保",
    "农业",
    "养殖",
    "汽车",
    "5G",
    "云计算",
    "大数据",
    "新材料",
    "高端制造",
    "先进制造",
    "机械",
    "化工",
    "建筑",
    "旅游",
    "游戏",
    "港口",
    "机场",
)


def _norm(value: Optional[str]) -> Optional[str]:
    """Strip a raw type/style string; treat blank as absent (fail-closed friendly)."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _index_invest_types(cfg: ScopeConfig) -> frozenset[str]:
    """The in-scope styles that denote an *index* vehicle (subset of config, D51)."""
    return frozenset(t for t in cfg.in_scope_invest_types if _INDEX_MARKER in t)


def _equity_active_invest_types(cfg: ScopeConfig) -> frozenset[str]:
    """The in-scope styles that denote an *actively managed* equity vehicle (D51)."""
    return frozenset(t for t in cfg.in_scope_invest_types if _INDEX_MARKER not in t)


def _looks_sector_theme(name: str) -> bool:
    """True iff the fund name advertises a single industry/theme (SECTOR_THEME)."""
    return any(kw in name for kw in SECTOR_THEME_KEYWORDS)


def _has_any(value: Optional[str], needles: tuple[str, ...]) -> bool:
    """True iff ``value`` contains any substring in ``needles``."""
    return value is not None and any(n in value for n in needles)


def _classify_category(
    profile: FundProfile, cfg: ScopeConfig
) -> tuple[ScopeCategory, str]:
    """Map ``(fund_type, invest_type, name)`` → (category, reason) — the baseline hook.

    Classifies into exactly one of the 13 :class:`ScopeCategory` members. Never returns a
    reject: an unresolvable type falls back to ``UNCLASSIFIED`` (fail-closed → LIMITED).
    Evaluation order (fail-closed, most-specific first; the 场内 market gate runs earlier in
    :func:`classify_scope`): money → QDII → pension(name) → FOF → commodity → bond →
    index → 偏股混合 → active-equity(theme/broad) → active-hybrid → UNCLASSIFIED. A candidate
    method may override this function (D-SCOPE-3 records the order).
    """
    fund_type = _norm(profile.fund_type)
    invest_type = _norm(profile.invest_type)
    name = profile.name or ""
    C = ScopeCategory

    # --- non-equity 场外 types (LIMITED) — unambiguous fund_type / style first -----------
    if fund_type in cfg.money_market_fund_types or invest_type in cfg.money_market_fund_types:
        return (C.MONEY_MARKET, f"货币市场基金（{fund_type or invest_type}），走有限分析口径。")
    if fund_type in cfg.qdii_fund_types or _has_any(invest_type, cfg.qdii_invest_substrings):
        return (C.QDII_OVERSEAS, f"QDII / 海外基金（{invest_type or fund_type}），走有限分析口径。")
    # Pension/target funds are often structured as FOF or 混合; the name is the strongest
    # user-facing signal, so it is checked before FOF / 混合 (D-SCOPE-4).
    if _has_any(name, cfg.pension_name_keywords):
        return (C.PENSION_TARGET, "养老目标 / 稳健理财基金（按名称识别），走有限分析口径。")
    if fund_type in cfg.fof_fund_types:
        return (C.FOF, "基金中基金（FOF），底层是其他基金，走有限分析口径。")
    if fund_type in cfg.commodity_fund_types:
        return (C.COMMODITY_ALT, f"商品 / 黄金 / REITs 等（{fund_type}），走有限分析口径。")
    # Bond: an explicit 债券型 fund_type, or a pure-bond invest_type ('债' substring) that is
    # NOT a hybrid (偏债混合型 is HYBRID_ACTIVE, handled below).
    if fund_type in cfg.bond_fund_types or (
        _has_any(invest_type, cfg.bond_invest_substrings) and "混合" not in (invest_type or "")
    ):
        return (C.BOND, f"债券型基金（{invest_type or fund_type}），走有限分析口径。")

    # --- equity family (FULL) -----------------------------------------------------------
    # Index / index-linked (被动指数型 / 增强指数型 / 股票指数): the vehicle wins over a
    # sector name, so "半导体指数" stays INDEX (D52).
    if invest_type in _index_invest_types(cfg):
        return (C.INDEX, f"被动 / 指数型基金（{invest_type}），完整分析口径。")
    # Equity-biased hybrid (偏股混合型): long persistent high A-share equity → equity caliber
    # (D-SCOPE-1).
    if invest_type in cfg.equity_hybrid_invest_types:
        return (C.EQUITY_HYBRID, f"偏股混合型基金（{invest_type}），归权益口径，完整分析。")
    # Active equity: 普通股票型, or a bare 股票型 fund_type with no style. Split broad vs
    # sector-theme by the fund name.
    is_active_equity = invest_type in _equity_active_invest_types(cfg) or (
        invest_type is None and fund_type == cfg.equity_fund_type
    )
    if is_active_equity:
        if _looks_sector_theme(name):
            return (C.SECTOR_THEME, f"行业 / 主题股票基金（{invest_type or fund_type}），完整分析口径。")
        return (C.A_SHARE_EQUITY, f"A 股权益基金（{invest_type or fund_type}），完整分析口径。")

    # --- active hybrid (LIMITED) --------------------------------------------------------
    # 平衡 / 灵活配置 / 偏债混合, or a bare 混合型 with no equity style: equity weight swings
    # widely → LIMITED (D-SCOPE-1). Not a reject.
    if invest_type in cfg.hybrid_active_invest_types or fund_type == cfg.hybrid_fund_type:
        return (C.HYBRID_ACTIVE, f"主动混合基金（{invest_type or fund_type}），仓位摆动大，走有限分析口径。")

    # --- fail-closed fallback: classify-all never rejects → UNCLASSIFIED (LIMITED) -------
    signal = invest_type or fund_type or "未知类型"
    return (
        C.UNCLASSIFIED,
        f"无法明确判定基金类型（{signal}），保守归为未分类，走有限分析口径（不给强判断）。",
    )


def classify_scope(
    profile: FundProfile, cfg: Optional[ScopeConfig] = None
) -> ScopeVerdict:
    """Classify ``profile`` and route it to its analysis caliber (§4 sec:mod-scope, lst:scope).

    Covers every 场外 fund and never hard-rejects: the equity family is ``FULL``, every other
    场外 type is ``LIMITED`` (classified + degraded), and a confirmed 场内 (exchange-traded)
    fund is ``ETF_LISTED`` / ``DEFERRED`` (recognized, 场内 mechanics deferred). ``in_scope``
    is the derived convenience view (``support_tier == FULL``).

    A **blank** ``market`` is not treated as 场内 (D50 seam): only a *confirmed* non-OTC market
    routes to ETF_LISTED. Pure over the profile — no network, no NAV, no holdings, no
    ``as_of`` — and **never raises** (an unresolvable identity is handled upstream).
    """
    scfg = cfg or CONFIG.scope

    # Step 0 — 场内 (exchange-traded) gate: v1 recognizes but DEFERS 场内 ETF mechanics. A
    # confirmed non-OTC market → ETF_LISTED (DEFERRED). A blank market is NOT 场内 (D50).
    market = _norm(profile.market)
    if market is not None and market != _OTC_MARKET_CODE:
        return ScopeVerdict(
            fund_code=profile.fund_code,
            category=ScopeCategory.ETF_LISTED,
            support_tier=SupportTier.DEFERRED,
            caliber=scfg.profile_for(ScopeCategory.ETF_LISTED),
            market=market,
            reason=f"场内交易基金（market={market}）；v1 识别归类，场内机制暂不处理（DEFERRED）。",
        )

    category, reason = _classify_category(profile, scfg)
    caliber = scfg.profile_for(category)
    return ScopeVerdict(
        fund_code=profile.fund_code,
        category=category,
        support_tier=caliber.support_tier,
        caliber=caliber,
        market=market,
        reason=reason,
    )


# Doc-vocabulary alias (04-modules.tex lst:scope names it ``check_scope``); the pipeline
# orchestrator calls :func:`classify_scope`. Both resolve to the same gate (D53).
check_scope = classify_scope
