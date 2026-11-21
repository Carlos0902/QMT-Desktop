"""risk_budget — is the user already carrying too much risk? (§4 sec:mod-riskbudget, §1.8-9).

Responsibility (doc): judge whether the user has already taken on **too much** risk and
emit a :class:`~qmt.types.RiskBudgetState` that feeds ``action_strategy``. This is
**advice_mode only** — it consumes a :class:`~qmt.types.LookthroughResult` (which only
exists once a :class:`~qmt.types.UserPortfolio` has been penetrated) and a
:class:`~qmt.types.SuitabilityProfile`. Neither type is ever held by the analysis_mode
chain, so the signature itself keeps this module out of analysis_mode (invariant 3). The
**mandatory** ``suitability`` parameter is the type-level realization of §3.5 gate 8
("refuse personalized advice without a SuitabilityProfile"): a caller structurally cannot
invoke it without one.

Doc signature (A1-appendix.tex, lst:appendix-sigs)::

    def assess_risk_budget(look: LookthroughResult, profile) -> RiskBudgetState: ...

Baseline — a transparent rule (§5 model-selection principle "simple, transparent baseline
first"): compare the portfolio's single-bucket concentration (largest industry, largest
style) and the look-through's aggregate risk-budget usage against a *suitability-derived*
cap. Usage is normalized so ``1.0`` means "exactly at the cap"; ``over_budget`` is true
when usage exceeds :attr:`~qmt.config.RiskBudgetConfig.budget_usage_cap` (strictly
``> 1.0``). fail-closed (invariant 6): per-dimension usages combine by ``max`` (the most
cautious dimension wins), and a non-positive cap reads as breached.

Candidate hook (deferred, baseline-first rule): a volatility-based risk budget that uses
:attr:`LookthroughResult.corr_matrix` / :attr:`LookthroughResult.fund_risk_contrib`
(risk-parity / marginal-risk style) rather than a pure concentration cap. Not built first.

Doc-silent calibration lives in :class:`~qmt.config.RiskBudgetConfig` and is logged in
DECISIONS.md (D54 tier→cap mapping, D55 style reuses the single-bucket cap, D56 max
combination + the look-through aggregate as a ``portfolio`` dimension).
"""

from __future__ import annotations

from typing import Optional

from qmt.config import CONFIG, RiskBudgetConfig
from qmt.types import (
    ExposureDimension,
    LookthroughResult,
    RiskBudgetState,
    RiskTolerance,
    SuitabilityProfile,
)

__all__ = ["assess_risk_budget", "derive_single_bucket_cap"]

# Order of the Chinese investor-suitability tiers, most-conservative → most-aggressive.
# C1 (保守型) tolerates the least single-bucket concentration, C5 (进取型) the most.
_TOLERANCE_ORDER: tuple[RiskTolerance, ...] = (
    RiskTolerance.C1,
    RiskTolerance.C2,
    RiskTolerance.C3,
    RiskTolerance.C4,
    RiskTolerance.C5,
)

# breaching_dimensions identifier for the look-through's aggregate usage. There is no
# ExposureDimension member for the whole-portfolio aggregate (industry/style are the
# per-bucket ones), so it gets its own stable string (D56).
_DIM_PORTFOLIO = "portfolio"

# Pure numerical guard against a zero/negative cap (NOT a business threshold): a
# suitability cap of 0 means "tolerate no single-bucket concentration", so any positive
# concentration must read as over-budget rather than divide by zero.
_CAP_FLOOR = 1e-9


def derive_single_bucket_cap(
    suitability: SuitabilityProfile,
    cfg: Optional[RiskBudgetConfig] = None,
) -> float:
    """Return the suitability-derived cap on any single industry/style bucket.

    * An explicit :attr:`SuitabilityProfile.max_single_industry_exposure` always wins
      (the user's own concentration limit).
    * Otherwise the cap is a linear interpolation across the C1–C5 tiers between the
      configured soft cap (C1) and hard cap (C5) — no new magic numbers, only the two
      config endpoints (D54): C1→0.40, C2→0.45, C3→0.50, C4→0.55, C5→0.60 at defaults.
    """
    rcfg = cfg or CONFIG.risk_budget
    override = suitability.max_single_industry_exposure
    if override is not None:
        return override
    soft = rcfg.single_industry_soft_cap
    hard = rcfg.single_industry_hard_cap
    idx = _TOLERANCE_ORDER.index(suitability.risk_tolerance)
    span = len(_TOLERANCE_ORDER) - 1  # 4
    return soft + (hard - soft) * (idx / span)


def _largest(conc: dict[str, float]) -> float:
    """Largest exposure in a concentration map (0.0 if empty). Negatives are ignored."""
    return max((v for v in conc.values()), default=0.0)


def _usage(value: float, cap: float) -> float:
    """Normalized usage = value / cap, fail-closed on a non-positive cap.

    A non-positive concentration contributes no usage; a non-positive cap floors the
    denominator so any positive concentration reads as (very) over-budget.
    """
    if value <= 0.0:
        return 0.0
    denom = cap if cap > 0.0 else _CAP_FLOOR
    return value / denom


def assess_risk_budget(
    lookthrough: LookthroughResult,
    suitability: SuitabilityProfile,
    cfg: Optional[RiskBudgetConfig] = None,
) -> RiskBudgetState:
    """Assess whether the user is over their risk budget (advice_mode only).

    Compares the penetrated portfolio's single-bucket concentration and the look-through's
    aggregate usage against the suitability-derived cap, and returns a
    :class:`RiskBudgetState`:

    * ``budget_usage`` — ``max`` of the per-dimension usages (fail-closed): largest
      industry / cap, largest style / cap, and the look-through's own ``budget_usage``
      normalized against ``budget_usage_cap``. ``0..1+`` (open-ended above the cap).
    * ``over_budget`` — ``budget_usage > budget_usage_cap`` (strictly; at-cap is *not*
      over). This is what §3.5 gate 7 reads to forbid "add-position" outputs.
    * ``single_industry_max`` — the largest single-industry exposure (clamped to [0, 1]).
    * ``breaching_dimensions`` — the dimensions whose own usage exceeds the cap, in a
      stable order: ``industry``, ``style``, ``portfolio``.

    ``cfg`` is injectable for testing; it defaults to the frozen :data:`qmt.config.CONFIG`
    ``risk_budget`` group.
    """
    rcfg = cfg or CONFIG.risk_budget
    cap = derive_single_bucket_cap(suitability, rcfg)

    single_industry_max = _largest(lookthrough.industry_conc)
    single_style_max = _largest(lookthrough.style_conc)

    industry_usage = _usage(single_industry_max, cap)
    style_usage = _usage(single_style_max, cap)
    # The look_through step precomputes an aggregate risk-budget usage (LookthroughResult
    # .budget_usage, documented "-> risk_budget"); fold it in as a whole-portfolio
    # dimension so its signal is never discarded, normalized vs the budget_usage cap.
    portfolio_usage = _usage(lookthrough.budget_usage, rcfg.budget_usage_cap)

    breaching: list[str] = []
    if industry_usage > rcfg.budget_usage_cap:
        breaching.append(ExposureDimension.INDUSTRY.value)
    if style_usage > rcfg.budget_usage_cap:
        breaching.append(ExposureDimension.STYLE.value)
    if portfolio_usage > rcfg.budget_usage_cap:
        breaching.append(_DIM_PORTFOLIO)

    budget_usage = max(industry_usage, style_usage, portfolio_usage)
    over_budget = budget_usage > rcfg.budget_usage_cap

    return RiskBudgetState(
        budget_usage=budget_usage,
        over_budget=over_budget,
        single_industry_max=min(1.0, max(0.0, single_industry_max)),
        breaching_dimensions=breaching,
    )
