"""run — the advice_mode orchestrator + re-export of the analysis core (§4 sec:mode-split, §6, A1).

The two mode-isolated public entrypoints are:

* :func:`run_analysis` — analysis_mode. **Defined in the import-isolated
  :mod:`qmt.pipeline.analysis` module** (whose dependency graph never references
  :class:`UserPortfolio` or any advice module) and re-exported here for back-compat.
* :func:`run_advice` — advice_mode. Defined here: the **sole** holder of
  :class:`UserPortfolio`. It reuses the analysis core, then penetrates the portfolio and
  applies the suitability / risk-budget / net-of-cost compliance gates before emitting an
  action.

Keeping ``run_advice`` (which imports the advice modules + ``UserPortfolio``) in a *separate*
module from the analysis core is the doc's code-level mode-isolation enforcement (§6
sec:cc-modes clause 2): a static import-guard test asserts the analysis modules — including
``qmt.pipeline.analysis`` — never import the advice package.

Wiring (advice):  analysis core → portfolio_lookthrough → risk_budget → action_strategy,
with ``gate_suitability_required`` / ``gate_net_of_cost`` / ``gate_risk_budget``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from qmt.compliance import ComplianceGuard, ComplianceViolation
from qmt.pit import confirm_lag_modeled
from qmt.types import (
    ActionRecommendation,
    CalibrationTable,
    ExplanationCard,
    FeeSchedule,
    FundProfile,
    Horizon,
    HorizonForecast,
    LookthroughResult,
    ProductView,
    RiskBudgetState,
    SuitabilityProfile,
    UserPortfolio,
)

# Analysis core (import-isolated module) — re-exported so existing callers keep importing
# ``run_analysis`` / ``AnalysisResult`` / ``_run_analysis_core`` from ``qmt.pipeline.run``.
from qmt.pipeline.analysis import (
    AnalysisResult,
    _resolve_store,
    _run_analysis_core,
    run_analysis,
)
from qmt.pipeline.action_strategy import decide_action
from qmt.pipeline.portfolio_lookthrough import look_through
from qmt.pipeline.risk_budget import assess_risk_budget

from typing import Optional, Union

__all__ = ["AnalysisResult", "AdviceResult", "run_analysis", "run_advice"]


@dataclass(frozen=True)
class AdviceResult:
    """Full advice_mode output bundle. Produced only by :func:`run_advice`."""

    product_view: ProductView
    action: ActionRecommendation
    explanation: ExplanationCard
    lookthrough: LookthroughResult
    risk_budget: RiskBudgetState
    # Per-fund forecasts + calibrated confidence — reused by the portfolio-level cash_allocation
    # (§4 sec:mod-cashalloc) so it needs no second analysis pass.
    forecasts: dict[Horizon, HorizonForecast] = field(default_factory=dict)
    confidence: float = 0.0


def _hold_days_for(portfolio: UserPortfolio, fund_code: str) -> int:
    """Actual holding period of ``fund_code`` in the user's portfolio (§3.4 «用户已经持有多久»).

    Returns the matching position's ``hold_days``; ``0`` (the most-punitive immediate-exit
    short-hold tier) when the advised fund is not among the holdings — fail-closed on an
    unknown holding period rather than assuming a long, cheap-to-exit hold (D-ACT-8).
    """
    return next((h.hold_days for h in portfolio.holdings if h.fund_code == fund_code), 0)


# =============================================================================
# advice_mode — the SOLE UserPortfolio holder
# =============================================================================
def run_advice(
    profile: Union[FundProfile, str],
    as_of: date,
    portfolio: UserPortfolio,
    suitability: SuitabilityProfile,
    fees: FeeSchedule,
    position: float = 0.0,
    store: Optional[object] = None,
    *,
    calibration: Optional[CalibrationTable] = None,
) -> AdviceResult:
    """advice_mode entrypoint — the ONLY function that reads a :class:`UserPortfolio`.

    Reuses the analysis core (which never sees ``portfolio``), then penetrates the portfolio
    and applies the advice-only compliance gates before deciding an action:

    * ``gate_advice_strength_cap`` (§3.5-3) — refuse personalized advice on a non-FULL type
      (``advice_strength_cap==NONE``: bond / money / QDII / FOF / commodity / pension / 场内 ETF).
    * ``gate_suitability_required`` (§3.5-8) — refuse personalized advice with no suitability.
    * ``gate_net_of_cost`` (§3.5-5) — an action must carry a fee schedule + a **truthfully
      modeled** execution/confirmation delay (derived from the T+1 decision timeline via
      :func:`qmt.pit.confirm_lag_modeled`, never hardcoded).
    * ``gate_risk_budget`` (§3.5-7) — enforced inside :func:`decide_action` for the
      add-position path (surfaced in ``ActionRecommendation.gated_by``).

    Reproduces the doc's «行动 ≠ 预测» example: a bullish, high-confidence, normal-volatility
    signal against an 85%-position / highly-concentrated book resolves to 继续观望
    (KEEP_WATCHING) — «维持，不追高» — not an add.
    """
    resolved_store = _resolve_store(store)
    core = _run_analysis_core(profile, as_of, resolved_store, calibration=calibration)
    fund_code = core.product_view.fund_code

    guard = ComplianceGuard()
    # --- gate #3: no personalized advice when the type's advice cap is NONE ------
    # analysis_mode DEGRADES a non-FULL fund (a neutral AnalysisResult); advice_mode must
    # HARD-refuse it — a LIMITED/DEFERRED type has advice_strength_cap==NONE, so the doc forbids
    # strong/personalized output (§3.5-3). Only the equity family (FULL / cap==FULL) proceeds.
    gate3 = guard.gate_advice_strength_cap(core.scope)
    if not gate3.passed:
        raise ComplianceViolation(gate3)
    # --- gate #8: suitability required (admission for any personalized advice) --
    gate8 = guard.gate_suitability_required(suitability)
    if not gate8.passed:
        raise ComplianceViolation(gate8)
    # --- gate #5: net-of-cost accounting must be present (fees + a real modeled delay) --
    delay_modeled = confirm_lag_modeled(as_of)
    gate5 = guard.gate_net_of_cost(fees_provided=fees is not None, execution_delay_modeled=delay_modeled)
    if not gate5.passed:
        raise ComplianceViolation(gate5)

    # --- portfolio_lookthrough → risk_budget (advice-only chain) ---------------
    lookthrough = look_through(portfolio, as_of, resolved_store)
    risk_budget = assess_risk_budget(lookthrough, suitability)
    # gate #7 marker; add-position enforcement lives in decide_action (is_add_position=True).
    guard.gate_risk_budget(risk_budget, is_add_position=False)

    # --- action_strategy — action ≠ prediction, jointly gated ------------------
    # Thread the position's ACTUAL holding period and the truthful execution-delay flag.
    hold_days = _hold_days_for(portfolio, fund_code)
    action = decide_action(
        core.forecasts,
        core.confidence,
        core.market_state,
        fees,
        position,
        lookthrough,
        risk_budget,
        suitability,
        hold_days,
        execution_delay_modeled=delay_modeled,
    )

    # In advice_mode the product view carries the decided action (§3 sec:ds-action).
    product_view = core.product_view.model_copy(update={"action": action.action})

    return AdviceResult(
        product_view=product_view,
        action=action,
        explanation=core.explanation,
        lookthrough=lookthrough,
        risk_budget=risk_budget,
        forecasts=core.forecasts,
        confidence=core.confidence,
    )
