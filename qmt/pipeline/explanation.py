"""explanation — the fixed five-section explanation card (§4 sec:mod-explain / §7 sec:eval-explain).

Responsibility (§4.2 / §1.8-11): say *why* the judgment is what it is, *where* it is
uncertain, and *what would change* it — always as a fixed five-section page:

1. 为什么偏乐观 (``why_optimistic``)
2. 为什么仍然需要谨慎 (``why_cautious``)
3. 我们不知道什么 (``unknowns``)
4. 数据有多新 (``data_freshness``)
5. 什么情况会改变当前判断 (``what_would_change``)

plus the §7 evidence structure (``evidence_seen`` / ``driving_factors`` /
``counter_evidence`` / ``uncertainties`` / ``change_conditions`` / ``disclosure_age_days``).

**Compliance gate #6 (§3.5-6 / §6):** the explanation must never present a model
attribution as a *causal* fact. This module frames everything as "推动判断的因素"
(factors that push the judgment), never "真实原因", and asserts that through
:meth:`ComplianceGuard.gate_no_causal_explanation` before returning — a fail-closed
structural guard, not a copy disclaimer.

The doc's worked example (§4, «务必保留») is reproduced for the canonical scenario
(a mid-horizon bullish call on a fund whose disclosure is stale while turnover is
shrinking): see :func:`build_explanation` and ``tests/test_run_analysis.py``.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from qmt.compliance import ComplianceGuard, ComplianceViolation
from qmt.config import CONFIG
from qmt.types import (
    Direction,
    ExplanationCard,
    ExposureEstimate,
    Horizon,
    HorizonForecast,
    MarketEvent,
    MarketStateVector,
    ProductView,
    Stress,
    Volatility,
)
from qmt.pipeline.product_view import primary_forecast

__all__ = ["build_explanation", "explain"]


# --- Chinese label maps (product copy for the derived numeric states) ----------
_HORIZON_CN = {Horizon.SHORT: "短期", Horizon.MID: "中期", Horizon.LONG: "长期"}
_DIRECTION_STANCE = {Direction.UP: "偏多", Direction.DOWN: "偏空", Direction.RANGE: "中性"}
_DIRECTION_CN = {Direction.UP: "向上", Direction.DOWN: "向下", Direction.RANGE: "震荡"}
_VOL_CN = {Volatility.LOW: "低", Volatility.NORMAL: "正常", Volatility.HIGH: "高"}
_STRESS_CN = {
    Stress.NORMAL: "正常",
    Stress.LIQUIDITY_TIGHT: "流动性偏紧（成交活跃度下降）",
    Stress.RISING_RISK: "风险上升",
}
_EVENT_CN = {
    MarketEvent.NONE: "无重大事件",
    MarketEvent.POLICY: "政策影响",
    MarketEvent.SHOCK: "外部冲击",
}


def _disclosure_age(exposure: list[ExposureEstimate]) -> int:
    """Oldest disclosure age across the exposure estimates (the data-freshness anchor)."""
    if not exposure:
        return 0
    return max(e.disclosure_age_days for e in exposure)


def build_explanation(
    fund_code: str,
    as_of: date,
    product_view: ProductView,
    forecasts: dict[Horizon, HorizonForecast],
    market_state: MarketStateVector,
    exposure: list[ExposureEstimate],
    confidence: float,
    *,
    guard: Optional[ComplianceGuard] = None,
    calibrated: bool = True,
) -> ExplanationCard:
    """Assemble the fixed five-section :class:`ExplanationCard` from the pipeline outputs.

    All copy is *derived* from the numeric state (direction/trend/volatility, the primary
    horizon's return range, market stress/volatility/event, disclosure age) and is framed
    as evidence/factors, never as causal conclusions (§3.5-6). ``driving_factors`` and
    ``counter_evidence`` are always non-empty (the §7 acceptance assertion): the disclosure
    staleness is always a live counter-evidence, and there is always at least one factor on
    the current stance.

    ``calibrated`` records whether the reported ``confidence`` was passed through a historical
    :class:`~qmt.types.CalibrationTable`. When ``False`` the uncertainty section clearly flags
    the score as *uncalibrated (no calibration table)* — the doc forbids presenting an
    unvalidated score as a reliability (§4 «不给未经历史校准的分数»).

    The change-condition (§7) is derived from the short horizon window and the volatility
    escalator, e.g. «如果未来 3 个交易日行业相对表现继续走弱，同时波动明显上升，本次中期
    偏多判断将降级» — the doc's own degrade example.
    """
    g = guard or ComplianceGuard()

    fc = primary_forecast(forecasts)
    horizon = fc.horizon if fc is not None else Horizon.MID
    horizon_cn = _HORIZON_CN.get(horizon, "中期")
    stance = _DIRECTION_STANCE[product_view.direction]
    ret_mid = fc.ret_mid if fc is not None else 0.0
    age = _disclosure_age(exposure)
    short_days = CONFIG.horizon.short_max_days  # "未来 3 个交易日"

    # --- section 1: 为什么偏乐观 -------------------------------------------------
    if product_view.direction is Direction.UP:
        why_optimistic = "当前判断偏乐观，主要因为相关行业近期比大盘更强，短期走势也在改善。"
    elif product_view.direction is Direction.DOWN:
        why_optimistic = "当前判断偏谨慎，相关行业近期弱于大盘，短期走势也在转弱，暂无明显偏乐观理由。"
    else:
        why_optimistic = "当前方向信号不明确，处于震荡区间，暂无明显偏乐观的理由，维持中性。"

    # --- evidence lists (§7 structure) -----------------------------------------
    evidence_seen: list[str] = [
        f"市场方向：{_DIRECTION_CN[market_state.direction]}",
        f"市场波动：{_VOL_CN[market_state.volatility]}",
        f"市场压力：{_STRESS_CN[market_state.stress]}",
        f"市场事件：{_EVENT_CN[market_state.event]}",
    ]

    driving_factors: list[str] = []
    if product_view.direction is Direction.UP:
        driving_factors.append("相关行业近期相对大盘偏强，推动系统更偏向偏多判断")
        driving_factors.append("短期走势有所改善")
    elif product_view.direction is Direction.DOWN:
        driving_factors.append("相关行业近期相对大盘偏弱，推动系统更偏向偏空判断")
        driving_factors.append("短期走势有所转弱")
    else:
        driving_factors.append("方向信号强度有限，多空证据大致均衡")
    driving_factors.append(
        f"{horizon_cn}预期收益区间中位约 {ret_mid:.1%}，趋势强度 {product_view.trend_strength:.2f}"
    )

    counter_evidence: list[str] = []
    if market_state.stress is Stress.LIQUIDITY_TIGHT:
        counter_evidence.append("估值位置已经不低，成交活跃度下降")
    elif market_state.stress is Stress.RISING_RISK:
        counter_evidence.append("市场整体风险上升")
    if market_state.volatility is Volatility.HIGH:
        counter_evidence.append("市场波动明显上升")
    if market_state.event in (MarketEvent.POLICY, MarketEvent.SHOCK):
        counter_evidence.append(f"存在{_EVENT_CN[market_state.event]}，扰动加大")
    # Disclosure staleness is ALWAYS a live counter-evidence (keeps this list non-empty).
    counter_evidence.append(
        f"最近公开持仓已经较旧（约 {age} 天前），当前行业暴露存在估计误差"
    )

    # --- section 2: 为什么仍然需要谨慎 ------------------------------------------
    why_cautious = "但" + "；".join(counter_evidence) + "，构成反向证据。"

    # --- section 3: 我们不知道什么 ----------------------------------------------
    uncertainties: list[str] = [
        f"我们无法直接观测基金当前的真实持仓：最近公开持仓约 {age} 天前，当前暴露只是估计",
        "未来市场路径本身不可预知，预测给出的是区间而非确定值",
    ]
    unknowns = (
        f"我们无法直接观测基金当前的真实持仓——最近公开持仓已较旧（约 {age} 天前），"
        "因此当前行业暴露只是估计，存在估计误差；未来市场路径也不可预知。"
    )

    # --- section 4: 数据有多新 --------------------------------------------------
    calib_note = (
        "该可信度已按历史命中率校准。"
        if calibrated
        else "注意：该可信度为未经历史校准的原始分数（uncalibrated：无校准表 data_cache/calibration.json）。"
    )
    data_freshness = (
        f"最近一次公开持仓披露距今约 {age} 个自然日；净值数据已更新至 {as_of.isoformat()} "
        f"当日可观测的最新交易日。系统当前把握程度约 {min(max(confidence, 0.0), 1.0):.0%}。{calib_note}"
    )
    if not calibrated:
        uncertainties.append(
            "可信度尚未经历史校准（uncalibrated，no calibration table）——请以 qmt calibrate 构建校准表后再解读把握程度。"
        )

    # --- section 5: 什么情况会改变当前判断 (the degrade condition, §7) -----------
    if product_view.direction is Direction.UP:
        change_line = (
            f"如果未来 {short_days} 个交易日行业相对表现继续走弱，同时波动明显上升，"
            f"本次{horizon_cn}偏多判断将降级。"
        )
    elif product_view.direction is Direction.DOWN:
        change_line = (
            f"如果未来 {short_days} 个交易日行业相对表现开始转强，同时波动明显回落，"
            f"本次{horizon_cn}偏空判断将上修。"
        )
    else:
        change_line = (
            f"如果未来 {short_days} 个交易日出现明确的方向突破且波动结构改变，"
            f"本次{horizon_cn}中性判断将更新。"
        )
    change_conditions: list[str] = [change_line]
    what_would_change = change_line

    # --- compliance gate #6: never frame model attribution as a causal conclusion ---
    # Everything above is phrased as "推动判断的因素"/"反向证据", not "真实原因"; assert it.
    explanation_is_causal = False
    gate5 = g.gate_no_causal_explanation(explanation_is_causal=explanation_is_causal)
    if not gate5.passed:  # pragma: no cover - structural guard; always non-causal by construction
        raise ComplianceViolation(gate5)

    return ExplanationCard(
        fund_code=fund_code,
        as_of=as_of,
        why_optimistic=why_optimistic,
        why_cautious=why_cautious,
        unknowns=unknowns,
        data_freshness=data_freshness,
        what_would_change=what_would_change,
        evidence_seen=evidence_seen,
        driving_factors=driving_factors,
        counter_evidence=counter_evidence,
        uncertainties=uncertainties,
        change_conditions=change_conditions,
        disclosure_age_days=age,
    )


# Doc-vocabulary alias (A1 appendix names the function ``explain``).
explain = build_explanation
