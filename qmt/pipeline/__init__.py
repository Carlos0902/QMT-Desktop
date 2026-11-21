"""Pipeline modules — the doc's conceptual stages, each with a descriptive name.

Modules use descriptive names, never pipeline numbers (invariant 7). The admission gate
:mod:`qmt.pipeline.fund_scope` heads the chain, followed by the first point-in-time gate
:mod:`qmt.pipeline.data_visibility`; later slices add ``exposure_estimator``,
``market_state``, ``horizon_forecaster``, etc.
"""

from __future__ import annotations

from qmt.pipeline.action_strategy import decide_action
from qmt.pipeline.data_visibility import classify_freshness, resolve_visibility
from qmt.pipeline.explanation import build_explanation, explain
from qmt.pipeline.exposure_estimator import (
    estimate_exposure,
    latest_visible_holdings,
)
from qmt.pipeline.fund_scope import check_scope, classify_scope
from qmt.pipeline.horizon_forecaster import forecast
from qmt.pipeline.cash_allocation import allocate_cash
from qmt.pipeline.market_state import assess_market_state, record_market_event
from qmt.pipeline.portfolio_lookthrough import look_through
from qmt.pipeline.product_view import primary_forecast, to_product_view
from qmt.pipeline.risk_budget import assess_risk_budget, derive_single_bucket_cap
from qmt.pipeline.run import (
    AdviceResult,
    AnalysisResult,
    run_advice,
    run_analysis,
)
from qmt.pipeline.uncertainty import (
    assess_uncertainty,
    build_calibration_table,
    calibrate,
    calibrate_confidence,
)

__all__ = [
    "decide_action",
    "resolve_visibility",
    "classify_freshness",
    "estimate_exposure",
    "latest_visible_holdings",
    "classify_scope",
    "check_scope",
    "forecast",
    "assess_market_state",
    "record_market_event",
    "look_through",
    "assess_risk_budget",
    "derive_single_bucket_cap",
    "allocate_cash",
    "assess_uncertainty",
    "build_calibration_table",
    "calibrate",
    "calibrate_confidence",
    "to_product_view",
    "primary_forecast",
    "build_explanation",
    "explain",
    "run_analysis",
    "run_advice",
    "AnalysisResult",
    "AdviceResult",
]
