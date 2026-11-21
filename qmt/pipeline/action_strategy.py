"""action_strategy — the decision layer that turns a prediction into a *cautious*
action (§4 sec:mod-action, §3.4, §1.8-10; algorithms table §5 sec:algo-table row
``ActionStrategy``). Advice-mode only.

**ACTION != PREDICTION** (cross-cutting invariant 5). The core idea from the doc:

    预测结果只是输入之一。真正的行动建议必须同时考虑风险、成本、持有时间、
    当前仓位、组合集中度和用户风险承受能力。

A bullish, high-confidence, normal-volatility signal does **not** automatically mean
"buy". The doc's worked example (designnote «为什么行动 ≠ 预测»): a bullish /
high-confidence / normal-vol signal that meets *仓位 85% + 高集中 + 短持有 + 高成本*
must output **维持，不追高** (``KEEP_WATCHING``) — never chase. This is enforced here in
the decision layer, not scattered across UI copy.

Baseline method (algorithms table §5): 保守规则 — 阈值触发 + 仓位/集中度硬约束，
net-of-cost. The candidate direction (逐步加入风险预算、成本和持有期限约束的决策层)
is left as a documented extension hook; we ship the transparent baseline first.

Gating (element E81, §6 sec:cc-compliance): confidence < threshold or expired data
degrades to ``WAIT_NEW_SIGNAL`` (等待新信号); over-budget forbids an add-position;
a missing :class:`SuitabilityProfile` refuses personalized advice. Every gate that
fires is recorded in :attr:`ActionRecommendation.gated_by`.

The required entrypoint deliberately restructures the doc's ``lst:action`` signature
(the orchestrator task fixes it): it adds ``confidence``, ``market_state`` and
``position`` as first-class inputs, makes ``lookthrough``/``risk_budget``/
``suitability`` optional, and returns the richer :class:`ActionRecommendation`
wrapper (D16) instead of the bare :class:`ActionAdvice` enum. The doc's ``hold_days``
(§3.4 «用户已经持有多久») is threaded back as a trailing parameter (default 0 =
most-punitive immediate-exit tier) and sizes the short-holding redemption fee; the
forecast's own 预期风险 (``vol_est`` / ``drawdown_prob``) and a truthful
``execution_delay_modeled`` gate-4 input are consumed too (D-ACT-7 / D-ACT-8).
"""

from __future__ import annotations

import math
from typing import Optional

from qmt.compliance import ComplianceGuard
from qmt.config import CONFIG
from qmt.types import (
    ActionAdvice,
    ActionRecommendation,
    FeeSchedule,
    Horizon,
    HorizonForecast,
    LookthroughResult,
    MarketEvent,
    MarketStateVector,
    RiskBudgetState,
    Stress,
    SuitabilityProfile,
    Volatility,
)

__all__ = [
    "decide_action",
    "GATE_POSITION_CAP",
    "GATE_CONCENTRATION_CAP",
    "GATE_COST_CAP",
    "GATE_FORECAST_RISK",
    "GATE_MARKET_TURBULENCE",
]

# --- action-strategy decision-layer hard constraints (the "don't chase" caps). ---
# These are the §5 baseline's 仓位/集中度硬约束 (distinct from the eight §3.5 compliance
# gates, which keep their ``ComplianceGuard`` gate names). Recorded in ``gated_by``.
GATE_POSITION_CAP = "action_position_cap"
GATE_CONCENTRATION_CAP = "action_concentration_cap"
GATE_COST_CAP = "action_roundtrip_cost_cap"
GATE_FORECAST_RISK = "action_forecast_risk"
GATE_MARKET_TURBULENCE = "action_market_turbulence"

# Fallback short-holding redemption probe when the caller threads no real holding period
# (``hold_days=0`` — immediate exit, the most punitive short-hold tier). advice_mode threads
# the *actual* ``hold_days`` from the user's portfolio position; a direct caller that omits it
# defaults to this conservative worst case (D-ACT-1 / D-ACT-8).
_SHORT_HOLD_PROBE_DAYS = 0


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _primary_forecast(
    forecast: dict[Horizon, HorizonForecast],
) -> Optional[HorizonForecast]:
    """Pick the action-driving horizon: MID (5–20d) first, then SHORT, then LONG.

    Accepts either :class:`Horizon` enum keys (the required signature) or their string
    values, so a caller that keyed by ``"mid"`` still resolves.
    """
    if not forecast:
        return None
    for h in (Horizon.MID, Horizon.SHORT, Horizon.LONG):
        if h in forecast:
            return forecast[h]
        if h.value in forecast:  # type: ignore[operator]
            return forecast[h.value]  # type: ignore[index]
    return next(iter(forecast.values()))


def _max_concentration(
    lookthrough: Optional[LookthroughResult],
    risk_budget: Optional[RiskBudgetState],
) -> float:
    """Largest look-through concentration across industry and style dimensions.

    Uses the merged look-through exposures (the doc's «5 funds → 63% tech-growth»
    case is a *style* concentration, so both industry_conc and style_conc count),
    and falls back to the risk-budget's ``single_industry_max`` when look-through is
    absent. Returns 0.0 when nothing is observable (the risk-budget gate, not this
    number, fails an add closed in that case).
    """
    values: list[float] = []
    if lookthrough is not None:
        values.extend(lookthrough.industry_conc.values())
        values.extend(lookthrough.style_conc.values())
    if risk_budget is not None:
        values.append(risk_budget.single_industry_max)
    return max(values) if values else 0.0


def _concentration_cap(suitability: Optional[SuitabilityProfile]) -> float:
    """User-specific concentration cap if the profile sets one, else the config cap."""
    if suitability is not None and suitability.max_single_industry_exposure is not None:
        return suitability.max_single_industry_exposure
    return CONFIG.action.high_concentration


def _roundtrip_cost(fees: FeeSchedule, hold_days: int) -> float:
    """Net round-trip cost: 申购费 + 按实际持有期的赎回费 (§3.5-5 net-of-cost).

    The redemption tier is looked up at the *actual* holding period (``用户已经持有多久``,
    §3.4), so a long-held position is not overcharged the punitive short-hold tier; a caller
    that threads ``hold_days=0`` still gets the most-punitive immediate-exit tier (D-ACT-8)."""
    return fees.purchase_fee_rate + fees.redemption_rate(hold_days)


def _constraint_reasons(fired: list[str]) -> str:
    """Chinese phrases for the hard constraints that fired (for the rationale text)."""
    phrases = {
        GATE_POSITION_CAP: "当前单只基金仓位已达上限（≥85%）",
        GATE_CONCENTRATION_CAP: "组合行业/风格暴露高度集中",
        GATE_COST_CAP: "短持有期申赎成本偏高",
        GATE_FORECAST_RISK: "预测本身的波动/回撤风险偏高",
    }
    parts = [phrases.get(g, g) for g in fired if g in phrases]
    if any(g not in phrases for g in fired):  # the risk-budget compliance gate
        parts.append("风险预算已用尽或无法确认")
    return "、".join(parts)


def decide_action(
    forecast: dict[Horizon, HorizonForecast],
    confidence: float,
    market_state: MarketStateVector,
    fees: FeeSchedule,
    position: float,
    lookthrough: LookthroughResult | None,
    risk_budget: RiskBudgetState | None,
    suitability: SuitabilityProfile | None,
    hold_days: int = 0,
    *,
    execution_delay_modeled: bool = True,
) -> ActionRecommendation:
    """Jointly weigh prediction + confidence + cost + position + concentration + risk
    tolerance → one of the six :class:`ActionAdvice` values, wrapped in an
    :class:`ActionRecommendation` (§4 lst:action; invariant 5 «action ≠ prediction»).

    Prediction is only *one* input. The decision:

    1. **Gates first** (fail-closed, §6). Low/NaN confidence or expired data (an empty
       forecast) → degrade to ``WAIT_NEW_SIGNAL`` (§3.5-4). Missing net-of-cost inputs
       → degrade (§3.5-5). Missing :class:`SuitabilityProfile` → refuse personalized
       advice, degrade to ``WAIT_NEW_SIGNAL`` (§3.5-8). Each firing gate is recorded
       in ``gated_by``.
    2. **Base action** from the prediction + market state (bullish→加仓 candidate,
       bearish→减仓/暂停, turbulent→观望/暂停, neutral→维持定投).
    3. **Hard "don't chase" constraints** (§5 baseline) can only make a *buy* more
       cautious: a candidate ``ADD_MODEST`` meeting 仓位/集中度/成本 caps, a forecast whose
       own 预期风险 (vol_est / drawdown_prob) is high, or an exceeded risk budget (§3.5-7)
       is downgraded to ``KEEP_WATCHING`` — 维持，不追高.

    ``fund_code`` and ``as_of`` for the recommendation come from ``fees.fund_code`` and
    ``market_state.as_of`` (the fixed signature carries no separate identifiers).
    ``hold_days`` (§3.4 «用户已经持有多久») sizes the redemption-fee tier for the net-of-cost
    gate; ``execution_delay_modeled`` is the honest net-of-cost input (§3.5-5, gate 5)
    reflecting whether a T+1 confirmation delay was actually modeled on this path.
    """
    guard = ComplianceGuard()
    fund_code = fees.fund_code
    as_of = market_state.as_of
    gated_by: list[str] = []

    primary = _primary_forecast(forecast)

    # --- Gate 4: confidence & freshness (§3.5-4). Low/NaN conf or no forecast → wait.
    data_expired = primary is None or _is_nan(confidence)
    conf_arg = None if _is_nan(confidence) else confidence
    g4 = guard.gate_confidence_and_freshness(conf_arg, data_expired=data_expired)
    if not g4.passed:
        gated_by.append(g4.gate)
        return ActionRecommendation(
            fund_code=fund_code,
            as_of=as_of,
            action=g4.degraded_action or ActionAdvice.WAIT_NEW_SIGNAL,
            rationale="把握不足或数据过旧，暂不给出强行动建议，建议等待新信号。",
            gated_by=gated_by,
        )

    # --- Gate 8: suitability required for any personalized advice (§3.5-8).
    g8 = guard.gate_suitability_required(suitability)
    if not g8.passed:
        gated_by.append(g8.gate)
        return ActionRecommendation(
            fund_code=fund_code,
            as_of=as_of,
            action=ActionAdvice.WAIT_NEW_SIGNAL,
            rationale="缺少适当性信息（SuitabilityProfile），无法进行个性化建议，等待新信号。",
            gated_by=gated_by,
        )

    # --- Gate 5: net-of-cost accounting must be present (§3.5-5). fees is required,
    # so this normally passes; if a future config still blocks, degrade closed.
    g5 = guard.gate_net_of_cost(
        fees_provided=fees is not None, execution_delay_modeled=execution_delay_modeled
    )
    if not g5.passed:
        gated_by.append(g5.gate)
        return ActionRecommendation(
            fund_code=fund_code,
            as_of=as_of,
            action=ActionAdvice.WAIT_NEW_SIGNAL,
            rationale="缺少净成本核算（费用/执行延迟），暂不给出行动建议。",
            gated_by=gated_by,
        )

    # --- Prediction signal (derived from ranges, never a bare label; §5 ProductView
    # direction thresholds are reused so 判定口径 stays consistent across modules).
    assert primary is not None  # data_expired above guarantees this
    pv = CONFIG.product_view
    bullish = primary.ret_mid > pv.up_ret_mid_min and primary.prob_positive > pv.up_prob_positive_min
    bearish = primary.ret_mid < pv.down_ret_mid_max and primary.prob_positive < pv.down_prob_positive_max

    # --- Market state contributes turbulence, not the direction call.
    turbulent = (
        market_state.volatility is Volatility.HIGH
        or market_state.stress is not Stress.NORMAL
        or market_state.event is MarketEvent.SHOCK
    )

    # --- Hard-constraint factors (the §5 baseline's 仓位/集中度 caps, net-of-cost).
    acfg = CONFIG.action
    pos_high = position >= acfg.high_position
    concentration = _max_concentration(lookthrough, risk_budget)
    # over-budget is NOT folded in here: §3.5-7 handles it in its own gate below, so the
    # two signals stay separable in ``gated_by``.
    conc_high = concentration >= _concentration_cap(suitability)
    roundtrip_cost = _roundtrip_cost(fees, hold_days)
    cost_high = roundtrip_cost >= acfg.high_roundtrip_cost
    # --- Forecast RISK side (§3.4 预期风险): a bullish call that is itself very bumpy or
    # carries a high material-drawdown probability must not be chased (D-ACT-7).
    forecast_risk_high = (
        primary.vol_est >= acfg.high_forecast_vol
        or primary.drawdown_prob >= acfg.high_forecast_drawdown_prob
    )

    # --- Base action from prediction + market.
    if bearish:
        base = ActionAdvice.REDUCE_PARTIAL if pos_high else ActionAdvice.PAUSE_SIP
    elif turbulent and not bullish:
        base = ActionAdvice.PAUSE_SIP
    elif bullish and not turbulent:
        base = ActionAdvice.ADD_MODEST
    elif bullish and turbulent:
        base = ActionAdvice.KEEP_WATCHING
        gated_by.append(GATE_MARKET_TURBULENCE)
    else:  # neutral / range in a calm market
        base = ActionAdvice.CONTINUE_SIP

    # --- «Don't chase»: hard constraints only bite a candidate ADD (§3.4 designnote).
    # A bullish/high-confidence/normal-vol signal at 高仓位/高集中/高成本/超预算 becomes
    # KEEP_WATCHING — 维持，不追高. This is what makes 行动 ≠ 预测 enforceable.
    if base is ActionAdvice.ADD_MODEST:
        fired: list[str] = []
        if pos_high:
            fired.append(GATE_POSITION_CAP)
        if conc_high:
            fired.append(GATE_CONCENTRATION_CAP)
        if cost_high:
            fired.append(GATE_COST_CAP)
        if forecast_risk_high:
            fired.append(GATE_FORECAST_RISK)
        # §3.5-7: never add over the risk budget. Missing budget fails closed → no add.
        g6 = guard.gate_risk_budget(risk_budget, is_add_position=True)
        if not g6.passed:
            fired.append(g6.gate)
        if fired:
            gated_by.extend(fired)
            reason = _constraint_reasons(fired)
            return ActionRecommendation(
                fund_code=fund_code,
                as_of=as_of,
                action=ActionAdvice.KEEP_WATCHING,
                rationale=f"预测虽偏乐观，但{reason}，追高性价比不足，建议维持，不追高。",
                gated_by=gated_by,
            )

    rationale = _rationale_for(base, roundtrip_cost)
    return ActionRecommendation(
        fund_code=fund_code,
        as_of=as_of,
        action=base,
        rationale=rationale,
        gated_by=gated_by,
    )


def _rationale_for(action: ActionAdvice, roundtrip_cost: float) -> str:
    """Non-causal Chinese rationale for a settled (non-degraded) action."""
    if action is ActionAdvice.ADD_MODEST:
        return "预测偏乐观、把握较高且市场平稳，仓位与集中度仍在预算内，扣除费用后仍有改善空间，可适当加仓。"
    if action is ActionAdvice.REDUCE_PARTIAL:
        return "预测偏谨慎且当前仓位较重，可考虑减少部分仓位以控制回撤。"
    if action is ActionAdvice.PAUSE_SIP:
        return "当前预测偏谨慎或市场承压，建议暂停定投，等待环境改善。"
    if action is ActionAdvice.KEEP_WATCHING:
        return "预测虽偏乐观，但当前波动或压力偏高，建议观望，等待市场企稳。"
    return "预测中性、无明显追高或减仓理由，维持原有定投节奏。"
