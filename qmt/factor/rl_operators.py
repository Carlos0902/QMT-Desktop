"""Small, auditable learners for controlling genetic-programming operators.

The environment is *formula search*, not portfolio trading. A transition is one
budgeted formula proposal followed by its discovery-training fitness result.
The Q learner bootstraps later search rewards; setting ``gamma=0`` gives the
matched contextual-bandit ablation. Neither controller consumes fund returns,
validation outcomes, or execution state itself.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "OPERATORS", "OperatorState", "OperatorLearningConfig",
    "UniformOperatorController", "TabularQController",
]

OPERATORS = ("mutate", "crossover", "restart")
OperatorState = tuple[int, ...]


@dataclass(frozen=True)
class OperatorLearningConfig:
    """Pre-registered operator-learning controls; never fitted on outer results."""

    alpha: float = 0.2
    epsilon: float = 0.2
    gamma: float = 0.8
    novelty_bonus: float = 0.01
    duplicate_penalty: float = 0.05
    invalid_penalty: float = 0.1
    stagnation_thresholds: tuple[int, int] = (3, 10)
    novelty_window: int = 12

    def __post_init__(self) -> None:
        for name in ("alpha", "epsilon", "gamma", "novelty_bonus",
                     "duplicate_penalty", "invalid_penalty"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0 <= self.epsilon <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("epsilon must lie in [0, 1] and gamma in [0, 1)")
        if min(self.novelty_bonus, self.duplicate_penalty, self.invalid_penalty) < 0:
            raise ValueError("bonuses and penalties must be nonnegative")
        lo, hi = self.stagnation_thresholds
        if lo < 1 or hi <= lo or self.novelty_window < 1:
            raise ValueError("invalid stagnation thresholds or novelty window")


class UniformOperatorController:
    """Matched fixed-probability control with all three RL operators available."""

    def __init__(self, *, seed: int, actions: Sequence[str] = OPERATORS) -> None:
        self.actions = tuple(actions)
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be nonempty and unique")
        self._rng = np.random.default_rng(seed)
        self.action_counts: Counter[str] = Counter()
        self.updates = 0

    def choose(self, state: OperatorState) -> str:
        action = self.actions[int(self._rng.integers(len(self.actions)))]
        self.action_counts[action] += 1
        return action

    def update(self, state: OperatorState, action: str, reward: float,
               next_state: OperatorState, *, terminal: bool = False) -> dict:
        if action not in self.actions or not np.isfinite(reward):
            raise ValueError("unknown action or nonfinite reward")
        self.updates += 1
        return {"target": float(reward), "q_before": None, "q_after": None}

    def snapshot(self) -> dict:
        return {"kind": "uniform", "actions": list(self.actions),
                "probabilities": {a: 1 / len(self.actions) for a in self.actions},
                "action_counts": {a: self.action_counts[a] for a in self.actions},
                "updates": self.updates}


class TabularQController(UniformOperatorController):
    """Epsilon-greedy tabular Q learning on an aggregated search state.

    ``Q(s,a) += alpha * (reward + gamma * max Q(next_s,.) - Q(s,a))``;
    terminal transitions omit the continuation value. Search-state aggregation
    is an approximation, so standard fully observed MDP guarantees do not apply.
    """

    def __init__(self, *, seed: int, config: OperatorLearningConfig | None = None,
                 actions: Sequence[str] = OPERATORS) -> None:
        super().__init__(seed=seed, actions=actions)
        self.config = config or OperatorLearningConfig()
        self.q_values: dict[OperatorState, dict[str, float]] = {}
        self.visit_counts: Counter[tuple[OperatorState, str]] = Counter()

    def _row(self, state: OperatorState) -> dict[str, float]:
        state = tuple(state)
        return self.q_values.setdefault(state, {a: 0.0 for a in self.actions})

    def choose(self, state: OperatorState) -> str:
        row = self._row(state)
        if self._rng.random() < self.config.epsilon:
            candidates = self.actions
        else:
            best = max(row.values())
            candidates = tuple(a for a in self.actions if row[a] == best)
        action = candidates[int(self._rng.integers(len(candidates)))]
        self.action_counts[action] += 1
        return action

    def update(self, state: OperatorState, action: str, reward: float,
               next_state: OperatorState, *, terminal: bool = False) -> dict:
        if action not in self.actions or not np.isfinite(reward):
            raise ValueError("unknown action or nonfinite reward")
        row = self._row(state)
        before = row[action]
        future = 0.0 if terminal or self.config.gamma == 0 else max(self._row(next_state).values())
        target = float(reward) + self.config.gamma * future
        row[action] = before + self.config.alpha * (target - before)
        self.visit_counts[(tuple(state), action)] += 1
        self.updates += 1
        return {"target": target, "q_before": before, "q_after": row[action]}

    def snapshot(self) -> dict:
        return {"kind": "contextual_bandit" if self.config.gamma == 0 else "tabular_q",
                "actions": list(self.actions), "alpha": self.config.alpha,
                "epsilon": self.config.epsilon, "gamma": self.config.gamma,
                "action_counts": {a: self.action_counts[a] for a in self.actions},
                "updates": self.updates,
                "q_table": [{"state": list(s), "values": dict(self.q_values[s]),
                             "visits": {a: self.visit_counts[(s, a)] for a in self.actions}}
                            for s in sorted(self.q_values)]}
