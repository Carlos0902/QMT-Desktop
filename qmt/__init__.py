"""QMT — decision-support system for off-exchange (场外) Chinese public equity funds.

Implements the 11-stage pipeline defined in ``techdesign/`` (the binding contract).
This is slice 1 (foundation): domain types, config/calibration constants, the
point-in-time enforcement layer, and the compliance-guard skeleton. Downstream module
authors import against the contract re-exported here.

Cross-cutting invariants (see techdesign §6 sec:cc-*):
  1. point-in-time correctness — every externally-observed field carries
     ``available_at``; all reads filter ``available_at <= as_of`` (qmt.pit).
  2. estimated exposure ≠ disclosed holdings — three tiers on ExposureEstimate.
  3. analysis/advice mode isolation — analysis code never receives UserPortfolio.
  4. predictions are ranges, not labels — HorizonForecast.
  5. market state is a 4-field vector — MarketStateVector.
"""

from __future__ import annotations

from qmt import backtest, compliance, config, data, pit, pipeline, types
from qmt.backtest import BacktestResult, run_backtest
from qmt.config import CONFIG, get_settings

__all__ = [
    "types",
    "config",
    "pit",
    "compliance",
    "data",
    "pipeline",
    "backtest",
    "run_backtest",
    "BacktestResult",
    "CONFIG",
    "get_settings",
]

__version__ = "0.1.0"
