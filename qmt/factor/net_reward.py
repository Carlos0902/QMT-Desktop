"""Fee-net operator reward on an isolated, earlier training episode.

Every formula is tested as an equal-rank addition to a frozen low-volatility
anchor. No inner-selection or outer outcomes enter the evaluator. This is a
one-factor marginal reward, not a reproduction of AlphaGen's changing pool.
"""
from __future__ import annotations

from dataclasses import replace
import copy

import numpy as np
import pandas as pd

from qmt.factor.incremental import (
    InnerFold, _run_pool, _training_panel, _truncate_frame,
)
from qmt.factor.dsl import FormulaDSL
from qmt.factor.metrics import scores_by_date_from_frame
from qmt.factor.rl_search import RLGuidedSearch
from qmt.factor.rl_operators import OperatorLearningConfig


class FeeNetReward:
    """A physically clipped training ledger shared by matched search arms.

    A new instance is required per seed; only the immutable execution histories
    may be shared. Insufficient signals or incomplete economic marks cannot earn
    a reward by leaving the simulated account in cash.
    """

    def __init__(self, data, fold: InnerFold, *, anchor, selection_start,
                 execution_config, fees, dividends, min_usable_fraction=.8):
        self.fold = copy.deepcopy(fold)
        self.selection_start = pd.Timestamp(selection_start).date()
        if not fold.decision_dates or fold.end_date >= self.selection_start:
            raise ValueError("reward episode must end before independent selection")
        if execution_config.seal_date is not None and fold.end_date >= execution_config.seal_date:
            raise ValueError("reward episode reaches the seal")
        self.anchor = copy.deepcopy(anchor)
        self.panel = _training_panel(data.panel, fold.end_date)
        self.unit = _truncate_frame(data.unit_nav, fold.end_date)
        self.adjusted = _truncate_frame(data.adjusted_nav, fold.end_date)
        self.availability = _truncate_frame(data.nav_available_at, fold.end_date)
        # Outcomes of earlier trades may be disclosed after episode end, but
        # must be known before the independent selection stage begins. Signals
        # still use the separate point-in-time visible panels at each decision.
        known = self.availability.apply(lambda c: pd.to_datetime(c) < pd.Timestamp(self.selection_start))
        self.unit = self.unit.where(known)
        self.adjusted = self.adjusted.where(known)
        self.visible = {n: _truncate_frame(getattr(data, n), fold.end_date) for n in
                        ("visible_unit_nav", "visible_adjusted_nav", "visible_nav_dates")}
        self.execution = replace(execution_config, end_date=fold.end_date)
        self.fees, self.dividends = fees, dividends
        self.min_usable_fraction = min_usable_fraction
        self.records = {}
        self.baseline = self._run((self.anchor,))
        if not self._usable(self.baseline):
            raise ValueError("reward anchor lacks usable signals or economic valuations")

    def _run(self, exprs):
        scores = {str(i): scores_by_date_from_frame(FormulaDSL().evaluate_frame(e, self.panel),
                                                  self.fold.decision_dates)
                  for i, e in enumerate(exprs)}
        return _run_pool(tuple(scores), scores, self.fold, self.unit, self.adjusted,
                         self.execution, self.fees, self.visible, None,
                         self.dividends)

    def _usable(self, metrics):
        return (metrics["usable_fraction"] >= self.min_usable_fraction
                and metrics["traded_gross_money"] > 0
                and metrics["stale_valuation_days"] == 0)

    def evaluate(self, expr):
        if expr.text not in self.records:
            if expr.text == self.anchor.text:
                metrics = self.baseline
            else:
                metrics = self._run((self.anchor, copy.deepcopy(expr)))
            usable = self._usable(metrics)
            delta = metrics["final_equity"] / self.baseline["final_equity"] - 1
            self.records[expr.text] = {"formula": expr.text, "usable": usable,
                                      "paired_net_delta": float(delta) if usable else None,
                                      "proposed": metrics}
        return self.records[expr.text]

    def audit(self):
        return {"reward_fold": self.fold.to_dict(), "selection_start": str(self.selection_start),
                "anchor_formula": self.anchor.text, "anchor_metrics": self.baseline,
                "candidate_metrics": list(self.records.values()),
                "reward_evaluations": len(self.records),
                "objective": "equity(anchor+candidate)/equity(anchor)-1, frozen equal ranks",
                "selection_or_outer_feedback_used": False,
                "outcome_publication_cutoff_exclusive": str(self.selection_start),
                "limitations": ["adaptive reuse of one training episode can overfit",
                                "marginal gain against a fixed anchor, not an evolving factor pool"]}


class FeeNetGuidedSearch(RLGuidedSearch):
    """Same GP and Q controller, with executable marginal wealth as fitness.

    IC coverage remains a discovery quality gate; viable formulas are ranked by
    reward-episode net marginal wealth. Uniform is the matched objective control.
    """

    def __init__(self, dsl, *, evaluator: FeeNetReward, mode="q"):
        super().__init__(dsl, mode=mode,
                         learning_config=OperatorLearningConfig(novelty_bonus=0.0))
        self.evaluator = evaluator
        self.discovery_fitness = {}

    def _fitness(self, expr, panel, labmap, select_dates, ledger):
        if expr.text in self._fit_cache:
            return self._fit_cache[expr.text]
        ic = super()._fitness(expr, panel, labmap, select_dates, ledger)
        self.discovery_fitness[expr.text] = float(ic) if np.isfinite(ic) else None
        if np.isfinite(ic):
            record = self.evaluator.evaluate(expr)
            value = record["paired_net_delta"] if record["usable"] else -np.inf
            self._fit_cache[expr.text] = value
        return self._fit_cache[expr.text]

    def _proposal_feedback(self, **kwargs):
        # The shared controller uses improvement in incumbent fitness. Here that
        # fitness is fee-net marginal wealth, not IC; novelty bonus is zero.
        super()._proposal_feedback(**kwargs)
        self._trace[-1]["fitness_kind"] = "fee_net_marginal_wealth"

    def policy_audit(self):
        audit = super().policy_audit()
        audit.update(reward="clipped positive improvement in best fee-net marginal wealth; waste penalties",
                     reward_is_net_alpha=True, fitness_kind="fee_net_marginal_wealth",
                     discovery_ic_fitness=self.discovery_fitness,
                     reward_episode=self.evaluator.audit())
        return audit
