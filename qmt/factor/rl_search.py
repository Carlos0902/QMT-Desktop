"""Train-only RL guidance for an unchanged, explainable GP formula search.

The policy chooses mutation, crossover, or a fresh random AST. All three modes
share the GP backbone, initial formulas, proposal budget and fitness evaluator;
only operator selection changes. Each search starts with a new policy, and the
downstream fee-net incremental selector remains responsible for accepting alpha.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from typing import Literal

import numpy as np

from qmt.factor.auto_search import SymbolicSearch
from qmt.factor.rl_operators import (
    OperatorLearningConfig,
    OperatorState,
    TabularQController,
    UniformOperatorController,
)
from qmt.factor.types import FactorDSL, SearchConfig, SearchLedger

__all__ = ["RLGuidedSearch"]


class RLGuidedSearch(SymbolicSearch):
    """Matched uniform, contextual-bandit, or tabular-Q operator control.

    State aggregates training-search stagnation, positive incumbent fitness and
    recent usable novelty. A proposal changes the formula population and search
    trajectory; ``gamma>0`` values its later rewards. The aggregation is only a
    proxy for the complete GP population, not a fully observed Markov state.
    """

    def __init__(self, dsl: FactorDSL, industry_of: dict[str, str] | None = None,
                 *, mode: Literal["uniform", "bandit", "q"] = "q",
                 learning_config: OperatorLearningConfig | None = None) -> None:
        super().__init__(dsl, industry_of)
        if mode not in {"uniform", "bandit", "q"}:
            raise ValueError("operator mode must be uniform, bandit, or q")
        self.mode = mode
        config = learning_config or OperatorLearningConfig()
        self.learning_config = replace(config, gamma=0.0) if mode == "bandit" else config
        if mode == "q" and self.learning_config.gamma == 0:
            raise ValueError("q mode requires gamma>0; use bandit for gamma=0")
        self._controller = None
        self._trace: list[dict] = []
        self._stagnation = 0
        self._recent_novel = deque(maxlen=self.learning_config.novelty_window)
        self._pending: tuple[OperatorState, str] | None = None

    def _before_search(self, cfg: SearchConfig, ledger: SearchLedger) -> None:
        # Independent from the AST RNG: policy exploration cannot consume random
        # numbers that would otherwise define initial formulas or AST edits.
        seed = int(np.random.SeedSequence([cfg.seed, 0x51474F50]).generate_state(1)[0])
        self._controller = (UniformOperatorController(seed=seed) if self.mode == "uniform"
                            else TabularQController(seed=seed, config=self.learning_config))
        self._trace = []
        self._stagnation = 0
        self._recent_novel.clear()
        self._pending = None

    def _state(self) -> OperatorState:
        lo, hi = self.learning_config.stagnation_thresholds
        stagnation_bin = 0 if self._stagnation < lo else 1 if self._stagnation < hi else 2
        finite = [f for f in self._fit_cache.values() if np.isfinite(f)]
        positive_incumbent = int(bool(finite) and max(finite) > 0)
        recent_novelty = int(bool(self._recent_novel)
                             and sum(self._recent_novel) / len(self._recent_novel) >= 0.5)
        return stagnation_bin, positive_incumbent, recent_novelty

    def _choose_operator(self, rng, population, cfg, generation) -> str:
        if self._controller is None:
            raise RuntimeError("operator policy must be initialized before proposals")
        state = self._state()
        action = self._controller.choose(state)
        self._pending = state, action
        return action

    def _proposal_feedback(self, *, action, before_population, child, trial, cfg,
                           generation, terminal, before_best, was_known) -> None:
        if self._controller is None or self._pending is None:
            raise RuntimeError("proposal feedback requires a preceding policy decision")
        state, selected = self._pending
        if action != selected:
            raise RuntimeError("proposal feedback action differs from the policy decision")
        self._pending = None
        usable = (child is not None and trial.passed_dsl and trial.select_ic is not None
                  and np.isfinite(trial.select_ic))
        novel = bool(usable and not was_known)
        gain = (max(0.0, float(trial.select_ic) - before_best)
                if usable and before_best is not None and np.isfinite(before_best) else 0.0)
        gain = min(gain, 1.0)
        if not usable:
            reward = -self.learning_config.invalid_penalty
            reason = "invalid_or_unscorable_proposal"
        elif was_known:
            reward = -self.learning_config.duplicate_penalty
            reason = "duplicate_training_formula"
        else:
            reward = gain + self.learning_config.novelty_bonus
            reason = "new_usable_training_formula"
        self._stagnation = 0 if gain > 1e-12 else self._stagnation + 1
        self._recent_novel.append(novel)
        next_state = self._state()
        update = self._controller.update(state, action, reward, next_state, terminal=terminal)
        self._trace.append({"trial_id": trial.trial_id, "generation": generation,
                            "operator": action, "state": list(state),
                            "next_state": list(next_state), "terminal": bool(terminal),
                            "reward": float(reward), "reward_reason": reason,
                            "training_best_fitness_gain": float(gain),
                            "novel_usable_formula": novel, "was_known": bool(was_known),
                            **update})

    def policy_audit(self) -> dict:
        """JSON-compatible complete policy configuration and transition trace."""
        return {"mode": self.mode,
                "learning_config": asdict(self.learning_config),
                "policy": self._controller.snapshot() if self._controller is not None else None,
                "state": ["training_stagnation_bucket (configured thresholds)",
                          "training_incumbent_fitness_positive",
                          "recent_usable_novel_formula_rate_at_least_half"],
                "reward": "clipped discovery-training best-fitness gain + usable novelty bonus; invalid/duplicate waste penalties",
                "reward_is_net_alpha": False,
                "training_only": True, "validation_or_outer_feedback_used": False,
                "policy_reset_each_search": True, "policy_rng_independent_of_ast_rng": True,
                "state_aggregation_limitation": "complete GP population is only partly observed; no MDP convergence claim",
                "transitions": list(self._trace)}
