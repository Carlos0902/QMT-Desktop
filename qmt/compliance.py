"""Compliance guard — the §3.5 / §6 sec:cc-compliance **nine** boundary gates as a
configurable, fail-closed skeleton.

The design doc is emphatic that compliance is a **model-layer hard constraint**, not
a disclaimer. Under the extended scope (§1 sec:scope) the old "no advice on out-of-scope
funds" constraint splits into two: gate 2 (support-tier gating — a non-FULL type may not
run the full strong-judgment pipeline) and gate 3 (advice-strength-cap — LIMITED/DEFERRED
types get no strong/personalized advice, and carry a disclaimer). The remaining constraints
(low-confidence/expired degradation, net-of-cost, no causal explanations, risk-budget cap,
suitability-required, regulation deference) renumber to 4..9.

Each gate is a named method returning a ``GateResult``. Every gate fails **closed**: when a
required input is missing or ambiguous and ``ComplianceConfig.fail_closed`` is set, it blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qmt.config import CONFIG, ComplianceConfig
from qmt.types import (
    ActionAdvice,
    AdviceStrengthCap,
    RiskBudgetState,
    ScopeVerdict,
    SuitabilityProfile,
    SupportTier,
    UserPortfolio,
)

__all__ = ["GateResult", "ComplianceViolation", "ComplianceGuard"]


@dataclass(frozen=True)
class GateResult:
    """Outcome of one compliance gate.

    ``passed`` True → the action may proceed. ``passed`` False → blocked; ``reason``
    explains why and ``degraded_action`` (when set) is the safe fallback the caller
    must emit instead (typically WAIT_NEW_SIGNAL / 等待新信号).
    """

    gate: str
    passed: bool
    reason: str = ""
    degraded_action: Optional[ActionAdvice] = None


class ComplianceViolation(RuntimeError):
    """Raised by ``require`` helpers when a hard gate blocks and the caller opted to raise."""

    def __init__(self, result: GateResult) -> None:
        super().__init__(f"[{result.gate}] {result.reason}")
        self.result = result


class ComplianceGuard:
    """Nine-gate compliance boundary (§6 sec:cc-compliance). Configurable, fail-closed.

    Gate numbering matches the doc's §3.5 table (1..9). Each method encodes *which* input
    decides the boundary and the fail-closed default.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None) -> None:
        self.config = config or CONFIG.compliance

    # --- 1. analysis_mode must never read user assets / give personalized advice
    def gate_no_user_assets_in_analysis(
        self, portfolio: Optional[UserPortfolio], *, in_analysis_mode: bool
    ) -> GateResult:
        """§3.5-1: enforced primarily by type isolation (run_analysis has no
        UserPortfolio param). This gate is the runtime backstop: a UserPortfolio
        present while in analysis mode is a hard violation."""
        if in_analysis_mode and portfolio is not None:
            return GateResult(
                "gate_1_no_user_assets_in_analysis",
                passed=False,
                reason="analysis_mode received a UserPortfolio — forbidden (§3.5-1).",
            )
        return GateResult("gate_1_no_user_assets_in_analysis", passed=True)

    # --- 2. cover all funds, but not with equal strength — support-tier gating
    def gate_scope_required(self, verdict: Optional[ScopeVerdict]) -> GateResult:
        """§3.5-2: a non-FULL support tier may not run the full strong-judgment pipeline.

        A resolved but non-FULL fund does NOT hard-fail analysis (it *degrades*); this gate
        reports that the full pipeline is not admitted. A **missing** verdict (unresolvable
        identity) is the one fail-closed case that the orchestrator raises on."""
        if verdict is None:
            return self._fail_closed(
                "gate_2_scope_required", "missing ScopeVerdict — cannot confirm fund identity/type."
            )
        if verdict.support_tier is not SupportTier.FULL and self.config.enforce_support_tier_gating:
            return GateResult(
                "gate_2_scope_required",
                passed=False,
                reason=f"support_tier={verdict.support_tier.value}（非 FULL）：不启用完整强判断链路，降级为有限分析 (§3.5-2).",
            )
        return GateResult("gate_2_scope_required", passed=True)

    # --- 3. no strong / personalized advice when the type's advice cap is NONE
    def gate_advice_strength_cap(self, verdict: Optional[ScopeVerdict]) -> GateResult:
        """§3.5-3: LIMITED/DEFERRED types (advice_strength_cap==NONE) get no strong advice.

        Reads ``verdict.caliber.advice_strength_cap``. When NONE, strong/personalized output
        is forbidden and the caliber's disclaimer must be surfaced (advice_mode refuses)."""
        if verdict is None:
            return self._fail_closed(
                "gate_3_advice_strength_cap", "missing ScopeVerdict — cannot confirm advice cap."
            )
        if (
            verdict.caliber.advice_strength_cap is AdviceStrengthCap.NONE
            and self.config.forbid_advice_out_of_scope
        ):
            return GateResult(
                "gate_3_advice_strength_cap",
                passed=False,
                reason=f"该类型 advice 上限=NONE：禁止强/个性化建议 (§3.5-3)。{verdict.caliber.disclaimer}",
            )
        return GateResult("gate_3_advice_strength_cap", passed=True)

    # --- 4. no strong action when confidence low / uncertainty high / data expired
    def gate_confidence_and_freshness(
        self, confidence: Optional[float], *, data_expired: bool
    ) -> GateResult:
        """§3.5-4: degrade to '等待新信号' when confidence < threshold or data expired."""
        if confidence is None:
            return self._fail_closed(
                "gate_4_confidence_and_freshness",
                "missing calibrated confidence — cannot issue a strong action.",
                degraded_action=ActionAdvice.WAIT_NEW_SIGNAL,
            )
        if data_expired and self.config.degrade_on_expired_data:
            return GateResult(
                "gate_4_confidence_and_freshness",
                passed=False,
                reason="required data is expired — degrade to 等待新信号 (§3.5-4).",
                degraded_action=ActionAdvice.WAIT_NEW_SIGNAL,
            )
        if confidence < CONFIG.action.min_confidence:
            return GateResult(
                "gate_4_confidence_and_freshness",
                passed=False,
                reason=f"confidence {confidence:.2f} < {CONFIG.action.min_confidence} — 等待新信号 (§3.5-4).",
                degraded_action=ActionAdvice.WAIT_NEW_SIGNAL,
            )
        return GateResult("gate_4_confidence_and_freshness", passed=True)

    # --- 5. never ignore short-holding fees & execution delay (net-of-cost)
    def gate_net_of_cost(self, *, fees_provided: bool, execution_delay_modeled: bool) -> GateResult:
        """§3.5-5: an action decision must carry a FeeSchedule and model delay."""
        if not self.config.net_of_cost_required:
            return GateResult("gate_5_net_of_cost", passed=True)
        if not fees_provided or not execution_delay_modeled:
            return self._fail_closed(
                "gate_5_net_of_cost",
                "action decided without net-of-cost accounting (fees/delay) (§3.5-5).",
            )
        return GateResult("gate_5_net_of_cost", passed=True)

    # --- 6. never present model attribution as a causal conclusion
    def gate_no_causal_explanation(self, *, explanation_is_causal: bool) -> GateResult:
        """§3.5-6: explanations distinguish '推动判断的因素' from '真实原因'."""
        if explanation_is_causal:
            return GateResult(
                "gate_6_no_causal_explanation",
                passed=False,
                reason="explanation framed model contribution as a causal conclusion (§3.5-6).",
            )
        return GateResult("gate_6_no_causal_explanation", passed=True)

    # --- 7. never breach the user's risk budget
    def gate_risk_budget(self, budget: Optional[RiskBudgetState], *, is_add_position: bool) -> GateResult:
        """§3.5-7: forbid 'add-position' outputs when the risk budget is exceeded."""
        if budget is None:
            return self._fail_closed(
                "gate_7_risk_budget", "missing RiskBudgetState — cannot confirm within budget."
            )
        if is_add_position and budget.over_budget and self.config.forbid_add_over_risk_budget:
            return GateResult(
                "gate_7_risk_budget",
                passed=False,
                reason="risk budget exceeded — 'add-position' forbidden (§3.5-7).",
            )
        return GateResult("gate_7_risk_budget", passed=True)

    # --- 8. no personalized add/reduce advice without suitability info
    def gate_suitability_required(self, suitability: Optional[SuitabilityProfile]) -> GateResult:
        """§3.5-8: refuse personalized add/reduce advice absent a SuitabilityProfile."""
        if not self.config.require_suitability_for_advice:
            return GateResult("gate_8_suitability_required", passed=True)
        if suitability is None:
            return GateResult(
                "gate_8_suitability_required",
                passed=False,
                reason="no SuitabilityProfile — refuse personalized advice (§3.5-8).",
            )
        return GateResult("gate_8_suitability_required", passed=True)

    # --- 9. all boundaries ultimately defer to effective regulation + legal opinion
    def gate_regulation_deference(self) -> GateResult:
        """§3.5-9: a marker gate — boundaries are configurable and defer to law.

        Always 'passes' but carries the deference note so callers surface it; the
        real enforcement is that every other gate reads from ``ComplianceConfig``."""
        return GateResult(
            "gate_9_regulation_deference",
            passed=True,
            reason=self.config.defer_to_regulation_note,
        )

    # --- helpers ------------------------------------------------------------
    def _fail_closed(
        self, gate: str, reason: str, degraded_action: Optional[ActionAdvice] = None
    ) -> GateResult:
        """Return a blocking result when fail_closed is on; otherwise pass (opt-in only)."""
        if self.config.fail_closed:
            return GateResult(gate, passed=False, reason=f"fail-closed: {reason}", degraded_action=degraded_action)
        return GateResult(gate, passed=True, reason=f"fail-open (disabled): {reason}")

    def all_gates(self) -> list[str]:
        """Names of the nine gate methods, for traceability / test coverage."""
        return [
            "gate_no_user_assets_in_analysis",
            "gate_scope_required",
            "gate_advice_strength_cap",
            "gate_confidence_and_freshness",
            "gate_net_of_cost",
            "gate_no_causal_explanation",
            "gate_risk_budget",
            "gate_suitability_required",
            "gate_regulation_deference",
        ]
