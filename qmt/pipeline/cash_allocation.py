"""cash_allocation — allocate a user's usable cash across candidate funds (§4 sec:mod-cashalloc).

ADVICE-MODE ONLY. When the user supplies a :class:`~qmt.types.UsableCashInput` (§2.8), this
module answers the concrete question the doc poses (§3.4): *这笔钱现在应该买哪些基金？每只买多少？
有没有必要暂时不买？* It scores each candidate fund on the doc's six criteria, keeps only those
that clear a quality gate, and allocates the cash proportionally to score under a per-fund cap.

**Protection rule (§2.8, the load-bearing invariant).** "花光现金" is a *conditional* goal, not
a command: when no candidate clears the quality bar (broadly down / high-vol / low-confidence /
unsuitable), the plan is ``invest=False`` and the cash is kept — **even if
``must_fully_invest=True``**. Cash is itself a position; the system never buys just to spend.

**Baseline method (this module).** A transparent, config-driven score
(``w_return·ret + w_confidence·conf + w_diversification·(1-overlap) − w_drawdown·dd − w_fee·fee``)
with a hard quality gate; passers get a score-proportional slice capped at
``max_single_fund_weight``. A candidate (constrained mean-variance / risk-parity) can replace
:func:`allocate_cash` later. Candidate set is the caller's responsibility — v1 passes the
user's held **FULL**-tier funds (± a watchlist); non-FULL types never enter (§4 sec:mod-scope,
D-CASH-1).

Inputs are pre-computed per-fund signals (forecasts + calibrated confidence + exposure overlap
+ purchase fee), so this module imports no analysis/scope module and stays advice-neutral.
"""

from __future__ import annotations

from typing import Mapping, Optional

from qmt.config import CONFIG, CashAllocationConfig
from qmt.types import (
    CashAllocationItem,
    CashAllocationPlan,
    Horizon,
    HorizonForecast,
    RiskBudgetState,
    UsableCashInput,
)

__all__ = ["allocate_cash"]

# Fee normalizer: a ~1.5% purchase fee ≈ one unit of penalty (matches the default tiers).
_FEE_SCALE = 0.015


def _mid_forecast(fc_map: Optional[Mapping[Horizon, HorizonForecast]]) -> Optional[HorizonForecast]:
    """The mid-horizon (5–20d) forecast used for cash-allocation scoring, else any available."""
    if not fc_map:
        return None
    return fc_map.get(Horizon.MID) or next(iter(fc_map.values()), None)


def allocate_cash(
    cash: UsableCashInput,
    candidates: list[str],
    forecasts: Mapping[str, Mapping[Horizon, HorizonForecast]],
    confidence: Mapping[str, float],
    budget: Optional[RiskBudgetState] = None,
    *,
    overlap: Optional[Mapping[str, float]] = None,
    fees: Optional[Mapping[str, float]] = None,
    cfg: Optional[CashAllocationConfig] = None,
) -> CashAllocationPlan:
    """Allocate ``cash`` across ``candidates`` by risk-return, or keep it (§4 sec:mod-cashalloc).

    ``forecasts`` / ``confidence`` / ``overlap`` / ``fees`` are per-fund signals keyed by
    ``fund_code`` (overlap = exposure overlap with current holdings ∈ [0,1]; fees = purchase
    fee rate). Returns a :class:`CashAllocationPlan`. Never raises; conservative by construction.
    """
    cfg = cfg or CONFIG.cash_allocation
    overlap = overlap or {}
    fees = fees or {}
    amount = cash.amount
    min_reserve = min(cash.min_cash_reserve, amount)

    # --- 1) per-fund quality gate + score (fail-closed toward keeping cash) ------------
    scored: list[tuple[str, float]] = []
    for fc in candidates:
        mid = _mid_forecast(forecasts.get(fc))
        if mid is None:
            continue
        ret = float(mid.ret_mid)
        dd = float(mid.drawdown_prob)
        conf = float(confidence.get(fc, 0.0))
        # gate: must be non-bearish, reliable enough, and not too likely to draw down.
        if ret <= cfg.min_expected_return or conf < cfg.min_confidence or dd > cfg.max_drawdown_prob:
            continue
        ov = float(overlap.get(fc, 0.0))
        fee = float(fees.get(fc, 0.0))
        score = (
            cfg.w_return * (ret / cfg.return_scale)
            + cfg.w_confidence * conf
            + cfg.w_diversification * (1.0 - ov)
            - cfg.w_drawdown * dd
            - cfg.w_fee * (fee / _FEE_SCALE)
        )
        if score > 0:
            scored.append((fc, score))

    # --- 2) no passer → keep cash (protection rule; overrides must_fully_invest) --------
    if not scored:
        return CashAllocationPlan(
            invest=False,
            allocations=[],
            kept_cash=amount,
            rationale="当前候选基金普遍不满足质量门槛（预期偏弱 / 回撤概率高 / 把握不足），"
            "建议暂时保留现金——现金也是一种仓位，不为买入而买入。",
        )

    # --- 3) investable amount (conservative when already over risk budget) --------------
    investable = amount - min_reserve
    over_budget = bool(budget is not None and budget.over_budget)
    if over_budget:
        investable *= 0.5  # already over budget → deploy less, prefer keeping cash
    investable = max(investable, 0.0)

    # --- 4) score-proportional allocation under the per-fund cap ------------------------
    total_score = sum(s for _, s in scored)
    cap = cfg.max_single_fund_weight * amount
    items: list[CashAllocationItem] = []
    for fc, s in scored:
        amt = min(investable * (s / total_score), cap) if total_score > 0 else 0.0
        if amt > 0:
            items.append(
                CashAllocationItem(
                    fund_code=fc, amount=round(amt, 2), weight=min(amt / amount, 1.0), score=round(s, 4)
                )
            )
    items.sort(key=lambda x: -x.amount)
    invested = sum(i.amount for i in items)

    if not items or invested <= 0:
        return CashAllocationPlan(
            invest=False, allocations=[], kept_cash=amount,
            rationale="未找到足够合适、风险可承受的配置，暂时保留现金。",
        )

    kept = max(round(amount - invested, 2), min_reserve)
    note = f"在 {len(scored)} 只合格候选中按风险收益比分配，保留现金 ¥{kept:,.0f}。"
    if over_budget:
        note += "组合已接近 / 超出风险预算，投入更保守、优先保留现金。"
    if cash.must_fully_invest and kept > min_reserve:
        note += "（尽管希望尽量用完，但受单基金上限与风险约束，剩余部分保留。）"
    return CashAllocationPlan(invest=True, allocations=items, kept_cash=kept, rationale=note)
