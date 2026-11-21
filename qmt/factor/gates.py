"""Multi-stage gate evaluation (instruction §7.5; tech doc §5.5 / appendix A).

The search objective is four **ordered** gates, not a single weighted score: any
FAIL eliminates the factor, and a high score on one gate cannot buy back a failure
on another. This stops the objective from being hijacked by one metric (e.g. high IC
with runaway turnover). Gates map to layers: predictive/robustness at
``factor_screening``, tradability at ``cost_model``, portfolio-increment at
``combination``.
"""

from __future__ import annotations

from dataclasses import dataclass

from qmt.factor.types import (
    CostTier,
    FactorScore,
    GateName,
    GateReport,
    GateStatus,
)

__all__ = ["GateConfig", "evaluate_gates"]


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the four gates (defaults marked 【推断】, tune per universe)."""

    min_rank_ic: float = 0.02            # predictive: OOS rank-IC direction
    min_dates: int = 8                   # not driven by a handful of dates
    require_significance: bool = True    # predictive: overlap-corrected IC must be significant (§4.3)
    turnover_cap: float = 0.6            # tradability: churn ceiling
    tradable_tier: CostTier = CostTier.NEU  # neutral-cost must stay positive
    max_corr_existing: float = 0.7       # portfolio-increment: redundancy ceiling
    min_increment: float = 0.0           # must improve net return / tail risk


def evaluate_gates(score: FactorScore, cfg: GateConfig | None = None) -> GateReport:
    """Run the four gates in order and record pass/fail + reason for each."""
    cfg = cfg or GateConfig()
    report = GateReport()

    # 1) predictive — OOS rank-IC direction, monotone buckets, and (if required)
    #    statistically significant after overlap correction. A high raw rank-IC that
    #    is not significant is exactly the spurious signal to reject (§4.3, §7.3).
    sig_ok = (not cfg.require_significance) or score.ic_significant
    pred_ok = (
        score.rank_ic >= cfg.min_rank_ic
        and score.monotonic
        and sig_ok
    )
    report.results[GateName.PREDICTIVE] = _status(pred_ok)
    report.reasons[GateName.PREDICTIVE] = (
        f"rank_ic={score.rank_ic:.3f} (>= {cfg.min_rank_ic}), monotonic={score.monotonic}, "
        f"significant={score.ic_significant} (t={score.ic_tstat:.2f})"
    )

    # 2) robustness — sign consistent across regimes / window perturbation
    rob_ok = score.regime_stable
    report.results[GateName.ROBUSTNESS] = _status(rob_ok)
    report.reasons[GateName.ROBUSTNESS] = f"regime_stable={score.regime_stable}"

    # 3) tradability — neutral-cost net return positive and turnover under cap
    neu_net = score.net_return_by_tier.get(cfg.tradable_tier.value, 0.0)
    trad_ok = (neu_net > 0.0) and (score.turnover <= cfg.turnover_cap)
    report.results[GateName.TRADABILITY] = _status(trad_ok)
    report.reasons[GateName.TRADABILITY] = (
        f"net@{cfg.tradable_tier.value}={neu_net:.4f}>0, "
        f"turnover={score.turnover:.3f}<= {cfg.turnover_cap}"
    )

    # 4) portfolio increment — not redundant with existing, adds net value
    inc_ok = (
        (score.max_corr_existing < cfg.max_corr_existing)
        and (not score.redundant)
        and (score.increment >= cfg.min_increment)
    )
    report.results[GateName.PORTFOLIO_INCREMENT] = _status(inc_ok)
    report.reasons[GateName.PORTFOLIO_INCREMENT] = (
        f"max_corr={score.max_corr_existing:.2f} (< {cfg.max_corr_existing}), "
        f"redundant={score.redundant}, increment={score.increment:.4f}"
    )
    return report


def _status(ok: bool) -> GateStatus:
    return GateStatus.PASS if ok else GateStatus.FAIL
