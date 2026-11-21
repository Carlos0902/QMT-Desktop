"""Shared amount conventions and explicit scenario defaults for OTC fund fees.

Subscription uses paid/(1+rate); redemption applies a calendar-holding-period
rate to confirmed units times executable unit NAV. Default rates are historical
research assumptions, not verified prospectus/channel fees for every fund.
Date-specific disclosed schedules must override them. Management, custodian and
sales-service accruals are normally already in published NAV and must not be
subtracted again. The service-fee field is descriptive and separate from m_fee
and c_fee; separately disclosed service-fee rebates need separate cash modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

from qmt.types import FeeSchedule, RedemptionFeeTier

__all__ = [
    "DEFAULT_PURCHASE_FEE_RATE",
    "DEFAULT_SALES_SERVICE_FEE_RATE",
    "DEFAULT_REDEMPTION_TIERS",
    "default_redemption_tiers",
    "default_fee_schedule",
    "TradeAmounts", "subscription_amounts", "redemption_amounts",
]


@dataclass(frozen=True)
class TradeAmounts:
    gross: float
    net: float
    fee: float


def subscription_amounts(gross_amount: float, fee_rate: float) -> TradeAmounts:
    """External subscription fee: net subscription = paid amount / (1 + rate)."""
    if not math.isfinite(gross_amount) or gross_amount < 0:
        raise ValueError("gross_amount must be finite and nonnegative")
    if not math.isfinite(fee_rate) or fee_rate < 0:
        raise ValueError("fee_rate must be finite and nonnegative")
    net = gross_amount / (1.0 + fee_rate)
    return TradeAmounts(gross_amount, net, gross_amount - net)


def redemption_amounts(gross_amount: float, fee_rate: float) -> TradeAmounts:
    """Redemption: confirmed units × fill unit NAV, less the holding-period fee."""
    if not math.isfinite(gross_amount) or gross_amount < 0:
        raise ValueError("gross_amount must be finite and nonnegative")
    if not math.isfinite(fee_rate) or not 0 <= fee_rate <= 1:
        raise ValueError("redemption fee_rate must lie in [0, 1]")
    fee = gross_amount * fee_rate
    return TradeAmounts(gross_amount, gross_amount - fee, fee)

# Documented defaults (D-BT-7): a discounted retail off-exchange A-share equity fund.
DEFAULT_PURCHASE_FEE_RATE: float = 0.0015  # 0.15% discounted subscription
DEFAULT_SALES_SERVICE_FEE_RATE: float = 0.004  # descriptive legacy assumption, already in NAV

# (min_hold_days, max_hold_days|None, rate) — short-hold tier is punitive & explicit (rule 7).
DEFAULT_REDEMPTION_TIERS: tuple[tuple[int, Optional[int], float], ...] = (
    (0, 7, 0.015),  # < 7 calendar days; actual prospectus/channel schedules override
    (7, 30, 0.005),  # < 30 days
    (30, 365, 0.0025),  # < 1 year
    (365, None, 0.0),  # >= 1 year: waived
)


def _fee_attr(meta: Optional[Any], name: str) -> Optional[float]:
    """Read a non-negative float fee attribute from ``meta`` (object attr or dict key), else None."""
    if meta is None:
        return None
    raw = meta.get(name) if isinstance(meta, dict) else getattr(meta, name, None)
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return val if val >= 0.0 else None


def default_redemption_tiers() -> list[RedemptionFeeTier]:
    """Build the documented short-hold-aware redemption tiers (fresh list per call)."""
    return [
        RedemptionFeeTier(min_hold_days=lo, max_hold_days=hi, rate=rate)
        for lo, hi, rate in DEFAULT_REDEMPTION_TIERS
    ]


def default_fee_schedule(fund_code: str, meta: Optional[Any] = None) -> FeeSchedule:
    """Research default; replace with disclosed date/channel-specific schedules.

    meta.sales_service_fee_rate may describe an actual service accrual. Management
    and custodian fees never populate that field; NAV already includes accruals.
    """
    # Management/custodian fees are distinct from sales-service fees. All are
    # normally accrued within published NAV and must not be deducted again by
    # NAV-based backtests. This field remains descriptive for legacy callers.
    disclosed_service = _fee_attr(meta, "sales_service_fee_rate")
    sales_service = (disclosed_service if disclosed_service is not None
                     else DEFAULT_SALES_SERVICE_FEE_RATE)
    return FeeSchedule(
        fund_code=fund_code,
        purchase_fee_rate=DEFAULT_PURCHASE_FEE_RATE,
        redemption_tiers=default_redemption_tiers(),
        sales_service_fee_rate=sales_service,
    )
