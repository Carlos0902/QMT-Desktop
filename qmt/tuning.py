"""Parameter-search ("training") over the pipeline's tunable models (§5, §7 ch:testing).

The design doc is emphatic that this is **not** a "predict tomorrow's price" model zoo: every
stage ships a *simple, transparent baseline* whose load-bearing thresholds live in
:mod:`qmt.config`, and a better setting is chosen "only by strict time-ordered validation —
never start with the most complex model" (§5). This module is that chooser. What a user calls
"training" here is a **walk-forward hyper-parameter search**:

1. Enumerate a curated grid of candidate settings for one model's tunable parameters
   (:data:`TUNABLE_MODELS` / :data:`WHOLE_MARKET_SPEC`), guaranteeing every level of every axis is
   actually tried (:meth:`TuningSpec.candidate_configs`).
2. Divide the user's month range into expanding **walk-forward folds** plus a trailing **holdout**
   that selection never touches (:func:`plan_walk_forward`).
3. Score every candidate on **every fold's out-of-sample window**, point-in-time and net of a real
   cost (:func:`market_training_fees`), and **select by the mean out-of-sample fold score**.
4. Confirm the winner on the untouched holdout and put it through the doc's "stably beats baseline"
   gate — cross-validation, holdout, per-regime slices and a bootstrap CI that must exclude zero
   (:func:`_judge_promotion`). Only a candidate that passes all of them is ``promotable``.

**Why selection moved off the training window.** Scoring every candidate out-of-sample on
CSI300 2022-2024 showed the rank correlation between the in-sample objective and the out-of-sample
objective was **negative** (Spearman ~ -0.36): ranking by in-sample performance picked the *worst*
generalisers. The old "rank on train, confirm the top-k" procedure therefore selected the wrong
candidate by construction. It survives as :func:`stream_parameter_search` (a single-split
diagnostic used by the CLI and by tests) but is no longer how the product trains. The search now
also *reports* :func:`spearman` of (in-sample, out-of-sample) so that misleadingness stays visible.

Point-in-time discipline (invariant 1) is inherited from the evaluators: every candidate's
backtest / replay reads only ``get_asof(<= decision_moment)``, and selection never sees the
holdout. The only global side effect is that scoring a candidate temporarily swaps the calibration
constants on the shared :data:`qmt.config.CONFIG` singleton (via
:func:`qmt.controlplane.api.apply_config_override`); the search snapshots CONFIG up front and
restores it in a ``finally`` so a run leaves the process exactly as it found it. Finished runs are
persisted (:func:`persist_run`) so a training result outlives the browser tab.

Doc-silent decisions (DECISIONS.md D-TUNE-*): "training" == walk-forward parameter search
(D-TUNE-1); the per-model grids (D-TUNE-2); the objective-metric menu incl. the Sharpe / turnover
definitions (D-TUNE-3); the overfitting guard (D-TUNE-4); the coverage-preserving grid subsample +
latency cap (D-TUNE-5); auto-selected whole-market target + real fees (D-TUNE-7); the month-range
split (D-TUNE-8); walk-forward selection + promotion gate (D-TUNE-9); run persistence (D-TUNE-10).
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import product
from typing import Callable, Iterator, Optional, Union

import numpy as np
import pandas as pd

from qmt.backtest import BacktestResult, run_backtest
from qmt.data.sources import TABLE_FUND_NAV, TABLE_INDEX_DAILY
from qmt.fees import default_fee_schedule
from qmt.pit import PointInTimeStore
from qmt.types import FeeSchedule

logger = logging.getLogger(__name__)

__all__ = [
    "BacktestMetrics",
    "DirectionMetrics",
    "ParamGrid",
    "TuningSpec",
    "TUNABLE_MODELS",
    "tuning_spec",
    "tunable_models_catalog",
    "compute_backtest_metrics",
    "compute_direction_metrics",
    "Trial",
    "SearchResult",
    "stream_parameter_search",
    "run_parameter_search",
    "MAX_TRIALS",
    "REFINE_MAX_TRIALS",
    "TradingDaysPerYear",
]

# Annualisation factor for daily metrics (trading days/year). Matches the backtest engine.
TradingDaysPerYear = 252.0

# Latency cap: a search evaluates at most this many candidate settings. Raised from 48 to 64 once
# the engine's price-tape (D-BT-15) made a backtest ~80x cheaper, so every curated grid here is now
# searched EXHAUSTIVELY. Larger grids fall back to the coverage-preserving sampler below (D-TUNE-5).
MAX_TRIALS = 64
# Coarse-to-fine: extra candidates spent probing *between* the coarse levels around the incumbent.
# Linear in the number of parameters (2 per axis), so it is a small, bounded addition to the run.
REFINE_MAX_TRIALS = 8
# Deterministic seed for the coverage-preserving subsample (reproducible runs).
SUBSAMPLE_SEED = 20240701


# =============================================================================
# Metric containers — everything derivable from one point-in-time evaluation
# =============================================================================
@dataclass(frozen=True)
class BacktestMetrics:
    """Net-of-cost performance of one candidate over one window (backtest objective).

    All fields are derived from a :class:`~qmt.backtest.BacktestResult`: the equity curve
    (Sharpe / volatility / CAGR / win-rate), the per-decision log (turnover), and the
    already-net headline numbers (net return / max drawdown / cost / baseline deltas). Sharpe
    and volatility are annualised from the net-of-cost per-interval portfolio returns with a
    zero risk-free rate (D-TUNE-3).
    """

    net_return: float
    cagr: float
    sharpe: float
    volatility: float
    max_drawdown: float
    calmar: float
    turnover: float          # annualised one-way target-weight turnover (D-TUNE-3)
    win_rate: float
    total_cost: float
    n_decisions: int
    measurable: bool         # False when the window had no observable data (fail-closed)
    vs_baseline: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "net_return": self.net_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "turnover": self.turnover,
            "win_rate": self.win_rate,
            "total_cost": self.total_cost,
            "n_decisions": self.n_decisions,
            "measurable": self.measurable,
            "vs_baseline": {k: round(v, 6) for k, v in self.vs_baseline.items()},
        }

    def objective(self, metric: str) -> float:
        """The value of a selectable, higher-is-better objective metric.

        Unmeasurable windows (no data) return ``-inf`` so a failed candidate always sorts last.
        """
        if not self.measurable:
            return float("-inf")
        if metric == "sharpe":
            return self.sharpe
        if metric == "net_return":
            return self.net_return
        if metric == "cagr":
            return self.cagr
        if metric == "calmar":
            return self.calmar
        if metric == "vs_buy_and_hold":
            return self.vs_baseline.get("buy_and_hold", float("-inf"))
        raise ValueError(f"unknown backtest objective metric {metric!r}.")


@dataclass(frozen=True)
class DirectionMetrics:
    """Direction-call quality of one candidate over one window (direction objective).

    Reuses the confidence-calibration replay (:func:`qmt.calibration.replay_confidence_observations`)
    which already pairs each point-in-time forecast with whether its product-view **direction**
    call was subsequently right over the primary horizon; ``hit_rate`` is the mean of those.
    """

    hit_rate: float
    n_obs: int
    measurable: bool

    def as_dict(self) -> dict[str, object]:
        return {"hit_rate": self.hit_rate, "n_obs": self.n_obs, "measurable": self.measurable}

    def objective(self, metric: str) -> float:
        if not self.measurable:
            return float("-inf")
        if metric == "hit_rate":
            return self.hit_rate
        raise ValueError(f"unknown direction objective metric {metric!r}.")


Metrics = Union[BacktestMetrics, DirectionMetrics]


# =============================================================================
# Metric computation
# =============================================================================
def _equity_daily_returns(curve: pd.DataFrame) -> np.ndarray:
    """Per-interval net portfolio returns implied by a ``date, equity`` curve (oldest→newest)."""
    if curve is None or curve.empty or "equity" not in curve.columns:
        return np.empty(0, dtype=float)
    eq = pd.to_numeric(curve["equity"], errors="coerce").to_numpy(dtype=float)
    eq = eq[np.isfinite(eq) & (eq > 0.0)]
    if eq.size < 2:
        return np.empty(0, dtype=float)
    return eq[1:] / eq[:-1] - 1.0


def _curve_years(curve: pd.DataFrame) -> float:
    """Elapsed years spanned by a ``date, equity`` curve (for annualising CAGR / turnover)."""
    if curve is None or curve.empty or "date" not in curve.columns or len(curve) < 2:
        return 0.0
    dates = pd.to_datetime(curve["date"])
    span_days = (dates.iloc[-1] - dates.iloc[0]).days
    return span_days / 365.25 if span_days > 0 else 0.0


def _annualized_turnover(result: BacktestResult, years: float) -> float:
    """Average annual one-way target-weight turnover across the funds (D-TUNE-3).

    For each fund, sum ``|signal_t - signal_{t-1}|`` over its consecutive decision dates (how
    much the target invested fraction moved), average across funds, and annualise by the window
    length. A pure buy-and-hold (constant signal) → ~0; a signal that flips fully in and out
    every rebalance → high. A transparent trading-intensity proxy, not a lot-level fill count.
    """
    if years <= 0.0:
        return 0.0
    by_fund: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for d in result.decision_log:
        by_fund[d.fund_code].append((d.as_of, float(d.signal)))
    if not by_fund:
        return 0.0
    per_fund_abs_change: list[float] = []
    for _code, seq in by_fund.items():
        seq.sort(key=lambda kv: kv[0])
        change = sum(abs(seq[i][1] - seq[i - 1][1]) for i in range(1, len(seq)))
        per_fund_abs_change.append(change)
    avg_change = float(np.mean(per_fund_abs_change)) if per_fund_abs_change else 0.0
    return avg_change / years


def compute_backtest_metrics(result: BacktestResult) -> BacktestMetrics:
    """Derive the full metric suite (Sharpe / turnover / drawdown / …) from a backtest result."""
    rets = _equity_daily_returns(result.equity_curve)
    years = _curve_years(result.equity_curve)

    if rets.size >= 2:
        mean = float(np.mean(rets))
        std = float(np.std(rets, ddof=1))
        sharpe = (mean / std) * math.sqrt(TradingDaysPerYear) if std > 0.0 else 0.0
        volatility = std * math.sqrt(TradingDaysPerYear)
        win_rate = float(np.mean(rets > 0.0))
    else:
        sharpe = volatility = win_rate = 0.0

    net_return = float(result.net_return)
    cagr = ((1.0 + net_return) ** (1.0 / years) - 1.0) if (years > 0.0 and net_return > -1.0) else 0.0
    max_dd = float(result.max_drawdown)
    calmar = (cagr / max_dd) if max_dd > 1e-9 else 0.0
    turnover = _annualized_turnover(result, years)

    return BacktestMetrics(
        net_return=net_return,
        cagr=cagr,
        sharpe=sharpe,
        volatility=volatility,
        max_drawdown=max_dd,
        calmar=calmar,
        turnover=turnover,
        win_rate=win_rate,
        total_cost=float(result.total_cost),
        n_decisions=int(result.n_decisions),
        measurable=True,
        vs_baseline=dict(result.vs_baseline or {}),
    )


def compute_direction_metrics(
    code: str, start: date, end: date, store: PointInTimeStore
) -> DirectionMetrics:
    """Point-in-time direction hit-rate over ``[start, end]`` (reuses the calibration replay)."""
    from qmt.calibration import replay_confidence_observations  # local: avoid import cost/cycle

    obs = replay_confidence_observations(code, start, end, store)
    if not obs:
        return DirectionMetrics(hit_rate=0.0, n_obs=0, measurable=False)
    hits = sum(1 for _raw, correct in obs if correct)
    return DirectionMetrics(hit_rate=hits / len(obs), n_obs=len(obs), measurable=True)


# =============================================================================
# Search space — the curated per-model candidate grids
# =============================================================================
@dataclass(frozen=True)
class ParamGrid:
    """One tunable parameter and the candidate values a search sweeps over it."""

    path: str                       # dotted CONFIG path, e.g. 'market_state.ma_fast_days'
    label: str                      # short human label for the UI
    values: tuple[Union[int, float], ...]


@dataclass(frozen=True)
class TuningSpec:
    """A tunable model: its parameter grid + how a candidate is scored (objective)."""

    id: str
    name_zh: str
    name_en: str
    config_group: str
    objective: str                  # 'backtest' | 'direction'
    metrics: tuple[str, ...]        # selectable higher-is-better objective metrics (first = default)
    params: tuple[ParamGrid, ...]
    doc_ref: str
    summary: str
    strategy: str = "baseline"      # backtest strategy driven by these params (backtest objective)
    whole_market: bool = False      # trains on the broad market index (000300.SH), not a fund (D-TUNE-7)
    # Rejects parameter combinations that are meaningless rather than merely bad. The coarse grid is
    # curated so nothing violates this, but refinement *invents* new levels between the coarse ones,
    # and that is exactly where an inverted ma_fast/ma_slow pair would first appear.
    constraint: Optional[Callable[[dict], bool]] = None

    def feasible(self, cand: dict) -> bool:
        return True if self.constraint is None else bool(self.constraint(cand))

    def candidate_configs(self) -> tuple[list[dict[str, Union[int, float]]], bool]:
        """Full Cartesian product of the grids, coverage-preserving-subsampled to ``MAX_TRIALS``.

        Returns ``(candidates, capped)``; ``capped`` is True when the raw grid exceeded
        :data:`MAX_TRIALS` and was thinned (D-TUNE-5, no silent truncation).

        **Why not an even stride** (the bug this replaces): the Cartesian product varies the LAST
        axis fastest, so a fixed stride *aliases* with that axis's period and can delete one of its
        levels entirely. With the 4x4x4 timing grid (64) and a 48-cap, stride 4/3 dropped **every**
        candidate with ``direction_band_pct = 0.02`` — a whole parameter value was never tried,
        while the flag still claimed the survivors "span the grid". Instead we take a seeded shuffle
        and greedily pick candidates that introduce an uncovered ``(axis, level)`` pair first, so
        **every level of every axis is guaranteed to appear**; the remaining slots are filled from
        the shuffled order. Deterministic (``SUBSAMPLE_SEED``) and verified by tests.
        """
        axes = [[(g.path, v) for v in g.values] for g in self.params]
        full = [dict(combo) for combo in product(*axes)] if axes else []
        full = [c for c in full if self.feasible(c)]
        if len(full) <= MAX_TRIALS:
            return full, False

        rng = random.Random(SUBSAMPLE_SEED)
        order = list(full)
        rng.shuffle(order)

        needed: set[tuple[str, Union[int, float]]] = {
            (g.path, v) for g in self.params for v in g.values
        }
        picked: list[dict] = []
        rest: list[dict] = []
        for cand in order:
            if len(picked) >= MAX_TRIALS:
                rest.append(cand)
                continue
            pairs = set(cand.items())
            if pairs & needed:  # introduces at least one still-uncovered (axis, level)
                needed -= pairs
                picked.append(cand)
            else:
                rest.append(cand)
        if needed:
            # Only reachable when the number of (axis, level) pairs exceeds MAX_TRIALS, which no
            # curated grid does. Never silently claim coverage we do not have (D-TUNE-5).
            logger.warning(
                "grid subsample for %s could not cover %d parameter level(s) within MAX_TRIALS=%d: %s",
                self.id, len(needed), MAX_TRIALS, sorted(needed),
            )
        for cand in rest:  # fill the remaining budget
            if len(picked) >= MAX_TRIALS:
                break
            picked.append(cand)
        return picked, True

    def refine_around(self, winner: dict, *, seen: list[dict], budget: int) -> list[dict]:
        """Local **coarse-to-fine** refinement: new levels between the winner and its neighbours.

        The coarse grid can only ever return one of its own levels, so a genuine optimum sitting
        between two levels is unreachable no matter how many folds we score. Here we walk one axis
        at a time away from the incumbent (a coordinate refinement, not a second Cartesian product)
        and propose the **midpoint** to each adjacent coarse level. That is ``2 * n_axes``
        candidates at most, so the extra cost is linear in the number of parameters rather than
        exponential, and every proposal is one interpretable step from a configuration we already
        scored out-of-sample.

        Refinement candidates are scored on the **same folds** as the coarse ones and re-enter the
        *same* paired-LCB selection alongside them. They get no privilege for being newer, and the
        holdout is still touched exactly once, after selection closes.
        """
        if not winner or budget <= 0:
            return []
        seen_keys = {tuple(sorted(c.items())) for c in seen}
        out: list[dict] = []
        for g in self.params:
            levels = list(g.values)
            if g.path not in winner or len(levels) < 2:
                continue
            try:
                pos = levels.index(winner[g.path])
            except ValueError:
                continue
            is_int = all(isinstance(v, int) for v in levels)
            for nb in (pos - 1, pos + 1):
                if not (0 <= nb < len(levels)):
                    continue
                mid = (levels[pos] + levels[nb]) / 2.0
                mid = int(round(mid)) if is_int else round(float(mid), 6)
                if mid in levels or mid == winner[g.path]:
                    continue                      # midpoint collapsed onto an existing level
                cand = dict(winner)
                cand[g.path] = mid
                key = tuple(sorted(cand.items()))
                if key in seen_keys or not self.feasible(cand):
                    continue
                seen_keys.add(key)
                out.append(cand)
                if len(out) >= budget:
                    logger.warning(
                        "refinement budget %d exhausted for %s; axes after %r were not refined "
                        "(no silent caps: this run probed only a prefix of the parameter axes)",
                        budget, self.id, g.path,
                    )
                    return out
        return out

    def grid_size(self) -> int:
        n = 1
        for g in self.params:
            n *= len(g.values)
        return n


# Curated tunable models. Each grid stays inside the field's declared CONFIG bounds and small
# enough that the walk-forward search is interactive; a candidate hook (larger grids / Bayesian
# search) is deferred (D-TUNE-2).
def _ma_ordering_holds(cand: dict) -> bool:
    """A fast MA must be strictly faster than the slow MA, else the cross signal inverts silently."""
    f = cand.get("market_state.ma_fast_days")
    sl = cand.get("market_state.ma_slow_days")
    return True if (f is None or sl is None) else f < sl


TUNABLE_MODELS: list[TuningSpec] = [
    TuningSpec(
        id="timing_ma",
        name_zh="均线择时", name_en="MA-cross timing",
        config_group="market_state", objective="backtest", strategy="ma_cross",
        metrics=("sharpe", "net_return", "calmar", "vs_buy_and_hold"),
        params=(
            ParamGrid("market_state.ma_fast_days", "快线窗口", (10, 15, 20, 30)),
            ParamGrid("market_state.ma_slow_days", "慢线窗口", (45, 60, 90, 120)),
            ParamGrid("market_state.direction_band_pct", "中性带", (0.0, 0.005, 0.01, 0.02)),
        ),
        doc_ref="§5 sec:algo-table / §7 sec:test-baselines (基线3) / D46",
        summary="搜索快/慢均线窗口与中性带，用点位回测（净费后）的夏普比率择优——择时基线的调参。",
        constraint=_ma_ordering_holds,
    ),
    TuningSpec(
        id="vol_control",
        name_zh="波动控制", name_en="Volatility control",
        config_group="market_state", objective="backtest", strategy="vol_control",
        metrics=("sharpe", "calmar", "net_return"),
        params=(
            ParamGrid("market_state.realized_vol_window_days", "已实现波动窗口", (10, 15, 20, 30, 45, 60)),
        ),
        doc_ref="§5 / §7 sec:test-baselines (基线5)",
        summary="搜索已实现波动的回看窗口，用目标波动/已实现波动的仓位基线择优（夏普/卡玛）。",
    ),
    TuningSpec(
        id="direction",
        name_zh="方向阈值", name_en="Direction thresholds",
        config_group="product_view", objective="direction",
        metrics=("hit_rate",),
        params=(
            ParamGrid("product_view.up_ret_mid_min", "看多收益阈", (0.002, 0.005, 0.01)),
            ParamGrid("product_view.up_prob_positive_min", "看多概率阈", (0.55, 0.6, 0.65)),
            ParamGrid("product_view.down_ret_mid_max", "看空收益阈", (-0.002, -0.005, -0.01)),
            ParamGrid("product_view.down_prob_positive_max", "看空概率阈", (0.35, 0.4, 0.45)),
        ),
        doc_ref="§4 sec:mod-forecast / §4 sec:mod-uncertainty (校准) / D-CAL-1",
        summary="搜索产品视图的方向判定阈值，用历史『方向判断是否正确』的命中率（点位回放）择优。",
    ),
]

# The broad market index that stands for "the whole market" (源文档 §3.3 / §5.3 大盘层：
# 大盘模型学的是大盘规律). Training the market-timing model on THIS — not a single fund — is the
# only way to learn the market layer of the hierarchical architecture (D-TUNE-7).
# The only baseline a search reports (`vs_buy_and_hold`). Re-simulating `ma`/`trend`/`vol_control`/
# `fixed_sip` for every candidate is pure waste: `buy_and_hold`/`fixed_sip`/`vol_control` are
# invariant to the swept timing params, and `ma` IS the strategy (a self-comparison, always 0).
SEARCH_BASELINES: tuple[str, ...] = ("buy_and_hold",)

WHOLE_MARKET_BENCHMARK = "000300.SH"
# The pseudo fund_code the market index is reshaped under so the fund-shaped backtest engine can
# run on it unchanged (6 digits = the CSI 300 code, so it also reads as a valid code).
PSEUDO_MARKET_CODE = "000300"

# The auto-selected whole-market training: the market-timing (大盘择时) model, tuned on the broad
# index. The user selects NEITHER the method NOR the target — the system fixes both to align with
# the hierarchical model's market layer.
WHOLE_MARKET_SPEC = TuningSpec(
    id="whole_market",
    name_zh="市场层择时（全市场）", name_en="Market-layer timing (whole market)",
    config_group="market_state", objective="backtest", strategy="ma_cross", whole_market=True,
    metrics=("sharpe", "net_return", "calmar", "vs_buy_and_hold"),
    params=(
        ParamGrid("market_state.ma_fast_days", "快线窗口", (10, 15, 20, 30)),
        ParamGrid("market_state.ma_slow_days", "慢线窗口", (45, 60, 90, 120)),
        ParamGrid("market_state.direction_band_pct", "中性带", (0.0, 0.005, 0.01, 0.02)),
    ),
    doc_ref="源文档 §3.3 / techdesign §5.3 sec:algo-hierarchical（大盘层）/ §7 sec:test-baselines",
    summary="面向整个市场（沪深300）训练大盘择时模型：搜索快/慢均线窗口与中性带，用点位、净费口径的"
            "回测夏普比率择优。训练目标是全市场而非单只基金——这样学到的才是分层模型架构里的『大盘规律』，"
            "而不是某只基金里混杂的大盘/行业/个股信号。",
    constraint=_ma_ordering_holds,
)

_SPEC_BY_ID = {s.id: s for s in TUNABLE_MODELS + [WHOLE_MARKET_SPEC]}


def tuning_spec(model_id: str) -> TuningSpec:
    """The :class:`TuningSpec` for ``model_id`` (raises :class:`KeyError` if unknown)."""
    if model_id not in _SPEC_BY_ID:
        raise KeyError(f"unknown tunable model {model_id!r}; known: {sorted(_SPEC_BY_ID)}.")
    return _SPEC_BY_ID[model_id]


def auto_selected_spec() -> TuningSpec:
    """The training the system auto-selects (the user picks neither method nor fund, D-TUNE-7).

    Always the whole-market market-timing model — the market layer of the §5.3 hierarchical
    architecture, which by construction learns market-wide rules from the broad index.
    """
    return WHOLE_MARKET_SPEC


def build_market_store(store: PointInTimeStore, benchmark: str = WHOLE_MARKET_BENCHMARK) -> PointInTimeStore:
    """Reshape the broad index (``index_daily``) into a fund-shaped in-memory PIT store.

    The Timeline backtest engine reads ``fund_nav`` (``fund_code`` / ``nav_date`` / ``adj_nav``);
    the market index lives in ``index_daily`` (``ts_code`` / ``trade_date`` / ``close``). Copying
    each bar verbatim — **including its ``available_at``** — into a ``fund_nav`` row keyed by
    :data:`PSEUDO_MARKET_CODE` lets the whole-market training reuse the entire engine (metrics /
    execution lag / regime slices) unchanged, with point-in-time correctness preserved (a bar
    disclosed after a decision moment stays invisible). Returns an empty store if the index is
    not primed (the caller then reports "nothing measurable", never a fabricated result).
    """
    df = store.get_asof(TABLE_INDEX_DAILY, datetime(2200, 1, 1), {"ts_code": benchmark})
    tmp = PointInTimeStore()
    if df is None or df.empty or "trade_date" not in df.columns or "close" not in df.columns:
        return tmp
    rows: list[dict] = []
    for r in df.to_dict("records"):
        close = r.get("close")
        if close is None or (isinstance(close, float) and close != close) or float(close) <= 0.0:
            continue
        rows.append({
            "fund_code": PSEUDO_MARKET_CODE,
            "nav_date": r.get("trade_date"),
            "adj_nav": float(close),
            "unit_nav": float(close),
            "available_at": r.get("available_at"),
        })
    if rows:
        tmp.put(TABLE_FUND_NAV, rows)
    return tmp


# Fraction of the user-chosen month range spent on TRAINING; the remainder is the strictly-later
# out-of-sample VALIDATION window. The user picks only a start and an end month — the system
# decides the split (D-TUNE-8). 70/30 is the conventional walk-forward split: enough training
# history to fit the grid, enough held-out tail to detect overfitting.
TRAIN_FRACTION = 0.7


def _add_months(d: date, n: int) -> date:
    """First day of the month ``n`` months after ``d``'s month."""
    total = (d.year * 12 + (d.month - 1)) + n
    return date(total // 12, total % 12 + 1, 1)


def _month_end(d: date) -> date:
    """Last calendar day of ``d``'s month."""
    return _add_months(d, 1) - timedelta(days=1)


def split_train_validation(
    start_month: date, end_month: date, train_fraction: float = TRAIN_FRACTION
) -> tuple[date, date, date, date]:
    """Divide an inclusive [start_month, end_month] span into train + validation windows (D-TUNE-8).

    The user selects only the overall range (month granularity — a training window is a
    macro-scale choice, and day precision is false precision on a walk-forward split). The system
    puts the first ``train_fraction`` of the months in TRAINING and the strictly-later remainder in
    OUT-OF-SAMPLE VALIDATION, month-aligned, guaranteeing ``train_end < valid_start`` (the
    overfitting guard, D-TUNE-4).

    Returns ``(train_start, train_end, valid_start, valid_end)``. Raises :class:`ValueError` when
    the span is under 2 months (both windows must be non-empty) or inverted.
    """
    s = date(start_month.year, start_month.month, 1)
    e = date(end_month.year, end_month.month, 1)
    if e < s:
        raise ValueError(f"end month {e:%Y-%m} precedes start month {s:%Y-%m}.")
    months = (e.year - s.year) * 12 + (e.month - s.month) + 1
    if months < 2:
        raise ValueError(
            f"range spans {months} month(s); need at least 2 so training and validation are both non-empty."
        )
    n_train = max(1, min(months - 1, round(months * train_fraction)))
    train_start = s
    train_end = _month_end(_add_months(s, n_train - 1))
    valid_start = _add_months(s, n_train)
    valid_end = _month_end(e)
    assert train_end < valid_start  # month-aligned split can never overlap
    return train_start, train_end, valid_start, valid_end


def market_training_fees(code: str = PSEUDO_MARKET_CODE) -> FeeSchedule:
    """The **realistic, non-zero** cost of expressing the market-timing signal (D-TUNE-7 rev).

    The whole-market training previously charged **zero** fees ("an index isn't a tradeable fund"),
    which (a) made the "净费口径 / net-of-cost" claim false and (b) left **turnover completely
    unpenalised** in the objective — the search could prefer a churning setting for free, in direct
    violation of the doc's protocol rule 6 («纳入交易成本，不能假设零成本»).

    The tuned timing model is ultimately executed by buying and redeeming a real off-exchange
    equity fund, so the honest proxy is that fund's cost: :func:`qmt.fees.default_fee_schedule`
    (subscription + the punitive short-holding redemption tiers + the annual sales-service drag).
    Every candidate is therefore scored net of a genuine round-trip cost, and a high-turnover
    candidate now pays for its churn.
    """
    return default_fee_schedule(code)


def _catalog_entry(spec: TuningSpec) -> dict:
    """Serialisable display record for one spec (params merged with live CONFIG values)."""
    from qmt.config import CONFIG

    _, capped = spec.candidate_configs()
    params = []
    for g in spec.params:
        group, field_name = g.path.split(".", 1)
        current = getattr(getattr(CONFIG, group), field_name, None)
        params.append({"path": g.path, "label": g.label, "values": list(g.values), "current": current})
    return {
        "id": spec.id, "name_zh": spec.name_zh, "name_en": spec.name_en,
        "config_group": spec.config_group, "objective": spec.objective,
        "strategy": spec.strategy, "metrics": list(spec.metrics),
        "whole_market": spec.whole_market,
        "params": params, "grid_size": spec.grid_size(),
        "n_candidates": min(spec.grid_size(), MAX_TRIALS), "capped": capped,
        "doc_ref": spec.doc_ref, "summary": spec.summary,
    }


def tunable_models_catalog() -> dict:
    """Serialisable catalog of the fund-based tunable models (CLI / advanced use)."""
    return {"models": [_catalog_entry(s) for s in TUNABLE_MODELS], "max_trials": MAX_TRIALS}


def whole_market_catalog(store: Optional[PointInTimeStore] = None) -> dict:
    """The single **auto-selected** whole-market training the page shows (no picker, D-TUNE-7).

    When a ``store`` is supplied, also reports ``data_range`` — the full observable month span of the
    benchmark index (CSI300 ``index_daily``). The page defaults the training window to this whole
    range so training uses ALL available history by default (the benchmark goes back to 2002, far
    earlier than the industry-intersection grid's 2014 start, which does not constrain whole-market
    timing since it reads the index alone).
    """
    cat = {
        "model": _catalog_entry(auto_selected_spec()),
        "benchmark": WHOLE_MARKET_BENCHMARK,
        "metric": auto_selected_spec().metrics[0],
        "max_trials": MAX_TRIALS,
    }
    if store is not None:
        rng = _benchmark_month_range(store, WHOLE_MARKET_BENCHMARK)
        if rng is not None:
            cat["data_range"] = {"start_month": rng[0], "end_month": rng[1]}
    return cat


def _benchmark_month_range(store: PointInTimeStore, benchmark: str) -> Optional[tuple[str, str]]:
    """Full observable ``(start_month, end_month)`` (``YYYY-MM``) of the benchmark's index_daily."""
    from qmt.data.sources import TABLE_INDEX_DAILY

    df = store.get_asof(TABLE_INDEX_DAILY, datetime(2200, 1, 1), {"ts_code": benchmark})
    if df is None or df.empty or "trade_date" not in df.columns:
        return None
    td = pd.to_datetime(df["trade_date"])
    lo, hi = td.min(), td.max()
    if pd.isna(lo) or pd.isna(hi):
        return None
    return (f"{lo.year:04d}-{lo.month:02d}", f"{hi.year:04d}-{hi.month:02d}")


# =============================================================================
# Trial + result containers
# =============================================================================
@dataclass
class Trial:
    """One evaluated candidate: its overrides, training metrics and (top-k) validation metrics."""

    index: int
    params: dict[str, Union[int, float]]
    train: Metrics
    objective_value: float
    valid: Optional[Metrics] = None

    def as_dict(self, metric: Optional[str] = None) -> dict:
        # ``objective_value`` is already the TRAINING objective scalar; ``valid_objective`` is
        # the matching OUT-OF-SAMPLE scalar, computed via ``Metrics.objective`` so the metric→
        # field mapping (incl. the nested ``vs_buy_and_hold`` = vs_baseline.buy_and_hold) lives
        # in one place on the server — the frontend never has to know the layout (D-TUNE-3).
        valid_objective = None
        if self.valid is not None and metric:
            vo = self.valid.objective(metric)
            valid_objective = None if vo == float("-inf") else round(vo, 6)
        return {
            "index": self.index,
            "params": self.params,
            "objective_value": (None if self.objective_value == float("-inf") else round(self.objective_value, 6)),
            "valid_objective": valid_objective,
            "train": self.train.as_dict(),
            "valid": self.valid.as_dict() if self.valid is not None else None,
        }


@dataclass
class SearchResult:
    """The full outcome of one walk-forward parameter search."""

    spec_id: str
    fund_code: str
    objective: str
    metric: str
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    trials: list[Trial]
    best: Optional[Trial]
    baseline: Optional[Trial]           # the current live CONFIG, scored the same way
    total_candidates: int
    capped: bool

    def as_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "fund_code": self.fund_code,
            "objective": self.objective,
            "metric": self.metric,
            "train": f"{self.train_start.isoformat()}..{self.train_end.isoformat()}",
            "valid": f"{self.valid_start.isoformat()}..{self.valid_end.isoformat()}",
            "trials": [t.as_dict(self.metric) for t in self.trials],
            "best": self.best.as_dict(self.metric) if self.best else None,
            "baseline": self.baseline.as_dict(self.metric) if self.baseline else None,
            "total_candidates": self.total_candidates,
            "capped": self.capped,
        }


# =============================================================================
# The walk-forward search — streaming generator (the SSE/CLI drive it)
# =============================================================================
def _evaluate(spec: TuningSpec, fund_code: str, start: date, end: date, store: PointInTimeStore) -> Metrics:
    """Score the CURRENT CONFIG for ``spec`` on ``[start, end]`` (point-in-time, net of cost)."""
    if spec.objective == "backtest":
        fees = market_training_fees(fund_code) if spec.whole_market else default_fee_schedule(fund_code)
        try:
            result = run_backtest(
                fund_code, start, end, strategy=spec.strategy,
                fees=fees, store=store, baselines=SEARCH_BASELINES,
            )
        except ValueError:
            # No observable data in the window → fail-closed unmeasurable (never a fabricated 0).
            return BacktestMetrics(
                net_return=0.0, cagr=0.0, sharpe=0.0, volatility=0.0, max_drawdown=0.0,
                calmar=0.0, turnover=0.0, win_rate=0.0, total_cost=0.0, n_decisions=0,
                measurable=False, vs_baseline={},
            )
        return compute_backtest_metrics(result)
    if spec.objective == "direction":
        return compute_direction_metrics(fund_code, start, end, store)
    raise ValueError(f"unknown objective {spec.objective!r} on spec {spec.id!r}.")


def stream_parameter_search(
    spec: TuningSpec,
    fund_code: str,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    store: PointInTimeStore,
    *,
    metric: Optional[str] = None,
    top_k: int = 5,
    cancel: Optional[Callable[[], bool]] = None,
    _result_sink: Optional[list] = None,
) -> Iterator[dict]:
    """LEGACY single-split search: rank on the training window, confirm the top-k out-of-sample.

    .. warning::
       **Do not use this to select parameters.** Measured on the broad market, ranking candidates
       by their *training* objective is anti-predictive (Spearman(train, OOS) ~ -0.36): the winner
       generalises worse than the baseline and worse than a median candidate. The product path is
       :func:`stream_walk_forward_search`, which selects on out-of-sample folds and gates the
       winner behind :func:`_judge_promotion`. This function is kept as a single-split *diagnostic*
       (CLI ``qmt tune`` and tests) and to document the procedure that was replaced.

    Yields dict events in order: one ``start``; a ``baseline`` (the current live CONFIG scored on
    train + valid); one ``trial`` per candidate (training metrics, streamed as they complete —
    the visualised "training process"); one ``validation`` per top-``k`` candidate (out-of-sample
    confirmation on the strictly-later window); and a final ``done`` carrying the whole
    :class:`SearchResult`.

    Overfitting guard (D-TUNE-4): the best is chosen by the **training** objective; the validation
    window is only *reported*, never selected on — and ``train_end < valid_start`` is required
    (fail-closed) so validation is genuinely out-of-sample. The shared CONFIG is snapshotted up
    front and restored in ``finally`` (generator ``.close()`` triggers it), so the search leaves
    the process exactly as it found it.
    """
    from qmt.controlplane.api import apply_config_override, restore_config, snapshot_config

    if valid_start <= train_end:
        raise ValueError(
            f"validation window must start strictly after the training window "
            f"(train_end={train_end}, valid_start={valid_start}) — else selection peeks at the "
            f"validation data (overfitting guard, D-TUNE-4)."
        )
    if train_end < train_start or valid_end < valid_start:
        raise ValueError("each window's end must be on/after its start.")

    metric = metric or spec.metrics[0]
    if metric not in spec.metrics:
        raise ValueError(f"metric {metric!r} not offered by model {spec.id!r} (choose {list(spec.metrics)}).")

    candidates, capped = spec.candidate_configs()
    snapshot = snapshot_config()

    def _apply(cand: dict) -> None:
        restore_config(snapshot)            # reset to the clean base first, so un-swept params stay baseline
        for path, val in cand.items():
            apply_config_override(path, val)

    try:
        yield {
            "event": "start",
            "spec_id": spec.id, "fund_code": fund_code, "objective": spec.objective, "metric": metric,
            "total": len(candidates), "capped": capped, "grid_size": spec.grid_size(),
            "train": f"{train_start.isoformat()}..{train_end.isoformat()}",
            "valid": f"{valid_start.isoformat()}..{valid_end.isoformat()}",
        }

        # --- baseline: the current live CONFIG, scored the same way (the bar to beat) --------
        baseline_train = _evaluate(spec, fund_code, train_start, train_end, store)
        baseline_valid = _evaluate(spec, fund_code, valid_start, valid_end, store)
        baseline_trial = Trial(
            index=-1, params={}, train=baseline_train,
            objective_value=baseline_train.objective(metric), valid=baseline_valid,
        )
        yield {"event": "baseline", "trial": baseline_trial.as_dict(metric)}

        # --- training sweep: score every candidate on the training window -------------------
        trials: list[Trial] = []
        for i, cand in enumerate(candidates):
            if cancel is not None and cancel():
                yield {"event": "cancelled", "completed": i, "total": len(candidates)}
                return
            _apply(cand)
            m = _evaluate(spec, fund_code, train_start, train_end, store)
            trial = Trial(index=i, params=cand, train=m, objective_value=m.objective(metric))
            trials.append(trial)
            yield {"event": "trial", "index": i, "total": len(candidates), "trial": trial.as_dict(metric)}

        # --- rank on training, confirm the top-k out-of-sample on the validation window -----
        trials.sort(key=lambda t: t.objective_value, reverse=True)
        for t in trials[: max(top_k, 1)]:
            if cancel is not None and cancel():
                yield {"event": "cancelled", "completed": len(trials), "total": len(candidates)}
                return
            if t.objective_value == float("-inf"):
                continue
            _apply(t.params)
            t.valid = _evaluate(spec, fund_code, valid_start, valid_end, store)
            yield {"event": "validation", "trial": t.as_dict(metric)}

        best = next((t for t in trials if t.objective_value != float("-inf")), None)
        result = SearchResult(
            spec_id=spec.id, fund_code=fund_code, objective=spec.objective, metric=metric,
            train_start=train_start, train_end=train_end, valid_start=valid_start, valid_end=valid_end,
            trials=trials, best=best, baseline=baseline_trial,
            total_candidates=len(candidates), capped=capped,
        )
        if _result_sink is not None:
            _result_sink.append(result)
        yield {"event": "done", "result": result.as_dict()}
    finally:
        restore_config(snapshot)


# =============================================================================
# Walk-forward validation (D-TUNE-9) — the selection procedure that replaces the
# single-split "rank on train, confirm top-k" search, which was measurably
# anti-predictive (Spearman(train, OOS) ~ -0.36 on CSI300 2022-2024).
# =============================================================================
MAX_FOLDS = 3               # selection folds (expanding train, next segment validates)
HOLDOUT_FRACTION = 0.25     # trailing months reserved, never touched during selection
MIN_MONTHS = 4              # at minimum: a training segment, an OOS validation segment, a holdout
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20240702


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: an expanding training window and the segment right after it."""

    index: int
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "train": f"{self.train_start.isoformat()}..{self.train_end.isoformat()}",
            "valid": f"{self.valid_start.isoformat()}..{self.valid_end.isoformat()}",
        }


@dataclass(frozen=True)
class WalkForwardPlan:
    """Folds used for SELECTION plus a trailing holdout used only to CONFIRM the winner."""

    folds: list[Fold]
    holdout_start: date
    holdout_end: date

    @property
    def insample_start(self) -> date:
        return self.folds[-1].train_start

    @property
    def insample_end(self) -> date:
        return self.folds[-1].train_end

    def as_dict(self) -> dict:
        return {
            "folds": [f.as_dict() for f in self.folds],
            "holdout": f"{self.holdout_start.isoformat()}..{self.holdout_end.isoformat()}",
            "insample": f"{self.insample_start.isoformat()}..{self.insample_end.isoformat()}",
        }


def plan_walk_forward(
    start_month: date, end_month: date, *, max_folds: int = MAX_FOLDS
) -> WalkForwardPlan:
    """Divide a month range into expanding-window folds + an untouched trailing holdout.

    The trailing :data:`HOLDOUT_FRACTION` of months is reserved (never scored during selection).
    The remaining months are cut into ``k+1`` contiguous segments; fold ``i`` trains on segments
    ``0..i`` and validates on segment ``i+1``. Selection uses the **mean fold-validation** score —
    each fold's validation window is strictly later than its own training window, so the selection
    signal is genuinely out-of-sample (D-TUNE-9). Raises :class:`ValueError` when the span is too
    short to give at least one fold plus a holdout month.
    """
    s = date(start_month.year, start_month.month, 1)
    e = date(end_month.year, end_month.month, 1)
    if e < s:
        raise ValueError(f"end month {e:%Y-%m} precedes start month {s:%Y-%m}.")
    n = (e.year - s.year) * 12 + (e.month - s.month) + 1
    if n < MIN_MONTHS:
        raise ValueError(
            f"range spans {n} month(s); need at least {MIN_MONTHS} so the system can build a "
            f"training fold, an out-of-sample validation fold and a held-out confirmation window."
        )
    holdout_months = max(1, round(n * HOLDOUT_FRACTION))
    sel = n - holdout_months
    if sel < 2:  # pragma: no cover - unreachable given MIN_MONTHS, kept as a fail-closed guard
        raise ValueError(f"range spans {n} month(s); not enough for a fold plus a holdout.")
    k = max(1, min(max_folds, sel - 1))

    # cut `sel` months into k+1 contiguous segments, largest-first remainder
    base, extra = divmod(sel, k + 1)
    sizes = [base + (1 if i < extra else 0) for i in range(k + 1)]
    bounds, cur = [], 0
    for sz in sizes:
        bounds.append((cur, cur + sz - 1))  # inclusive month offsets
        cur += sz

    folds: list[Fold] = []
    for i in range(k):
        train_end_off = bounds[i][1]
        v_lo, v_hi = bounds[i + 1]
        folds.append(Fold(
            index=i,
            train_start=s,
            train_end=_month_end(_add_months(s, train_end_off)),
            valid_start=_add_months(s, v_lo),
            valid_end=_month_end(_add_months(s, v_hi)),
        ))
    return WalkForwardPlan(
        folds=folds,
        holdout_start=_add_months(s, sel),
        holdout_end=_month_end(e),
    )


# --- statistics -------------------------------------------------------------
def _curve_returns(curve: pd.DataFrame) -> pd.Series:
    """Per-interval net returns of a ``date, equity`` curve, indexed by date."""
    if curve is None or curve.empty or len(curve) < 2:
        return pd.Series(dtype=float)
    eq = pd.Series(
        pd.to_numeric(curve["equity"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(curve["date"]),
    ).dropna()
    return eq.pct_change().dropna()


def paired_bootstrap_ci(
    a: pd.Series, b: pd.Series, level: float, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, int]:
    """CI for the mean of the **paired** difference ``a - b`` (candidate minus baseline).

    Pairs the two per-interval return series on their common dates (they share a trading grid),
    then bootstraps the mean difference. Returns ``(lo, hi, n_pairs)``. The doc's "stably beats
    baseline" gate (§7, ``ValidationConfig.bootstrap_ci_level``) requires this CI to **exclude
    zero** — i.e. ``lo > 0`` for a genuine improvement.
    """
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 5:
        return (float("nan"), float("nan"), len(joined))
    diff = (joined["a"] - joined["b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi), int(diff.size))


def regime_wins(cand: BacktestResult, base: BacktestResult) -> tuple[int, int]:
    """``(regimes_won, regimes_present)`` comparing per-regime net return (protocol rule 8).

    A regime counts as *present* only when **both** runs actually traded in it. Unioning the key
    sets and defaulting a missing regime to ``0.0`` fabricates evidence: a regime only the
    candidate has would be "won" against a baseline that never traded there, and a regime only the
    baseline has (where it lost money) would be "won" by a candidate with no data at all. Both
    inflate the win count *and* the number of observed regimes, which is exactly what the doc's
    "wins >= k of n regimes" gate counts — so it must be an intersection (D-BT-17).
    """
    keys = {
        k for k in (set(cand.net_return_by_regime) & set(base.net_return_by_regime)) if k != "warmup"
    }
    won = sum(1 for k in keys if cand.net_return_by_regime[k] > base.net_return_by_regime[k])
    return won, len(keys)


def _tied_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks with **average ranks for ties** (the definition Spearman requires)."""
    order = a.argsort(kind="mergesort")
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(a.size, dtype=float)
    s = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no SciPy), tie-aware.

    Returns ``nan`` when either input is constant — a naive implementation ranks a constant array
    ``0..n-1`` by argsort order and reports a spurious +/-1 correlation. Real objective values tie
    often (e.g. many candidates score exactly 0.0), so ties get average ranks.
    """
    if x.size < 3 or y.size != x.size:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    rx, ry = _tied_ranks(x), _tied_ranks(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run_parameter_search(
    spec: TuningSpec,
    fund_code: str,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    store: PointInTimeStore,
    *,
    metric: Optional[str] = None,
    top_k: int = 5,
) -> SearchResult:
    """Drain :func:`stream_parameter_search` and return the final :class:`SearchResult`.

    The non-streaming entrypoint (CLI / tests). Runs the search exactly once and captures the
    live :class:`SearchResult` via the generator's result sink (no double computation).
    """
    sink: list[SearchResult] = []
    for _ev in stream_parameter_search(
        spec, fund_code, train_start, train_end, valid_start, valid_end, store,
        metric=metric, top_k=top_k, _result_sink=sink,
    ):
        pass
    if not sink:  # pragma: no cover - the generator always sinks a result before 'done'
        raise RuntimeError("parameter search did not produce a result.")
    return sink[0]


# =============================================================================
# Selection under noise (D-TUNE-11)
# =============================================================================
# Measured: the winner had cv_mean=+0.234 with cv_std=1.426 across folds (1.762 / -1.060 / 0.000).
# Most of that spread is the *market regime of the fold*, which the baseline experiences too — it is
# common-mode noise, not evidence about the candidate. Pairing each candidate against the baseline
# fold-by-fold cancels it, and turns "which candidate had the luckiest fold" into "which candidate
# beat the baseline consistently".
SELECT_LAMBDA = 1.0        # LCB = mean(delta) - lambda * SE(delta): a pessimistic, robust score
SELECT_MIN_PAIRED_T = 1.0  # a candidate must beat the baseline with at least this paired t


@dataclass(frozen=True)
class PairedEdge:
    """A candidate's per-fold edge over the baseline, and its noise-aware summary."""

    mean: float          # mean(candidate_fold - baseline_fold)
    se: float            # standard error of that mean across folds
    t: float             # paired t-statistic
    lcb: float           # mean - lambda*se  (the score we actually maximise)
    n_folds: int

    def as_dict(self) -> dict:
        def f(v):
            return None if not math.isfinite(v) else round(float(v), 6)
        return {"mean": f(self.mean), "se": f(self.se), "t": f(self.t), "lcb": f(self.lcb),
                "n_folds": self.n_folds}


def paired_edge(candidate: "WFCandidate", baseline: "WFCandidate") -> PairedEdge:
    """Fold-paired edge of ``candidate`` over ``baseline``. Fail-closed on any unmeasurable fold."""
    a = np.asarray(candidate.fold_objectives, dtype=float)
    b = np.asarray(baseline.fold_objectives, dtype=float)
    if a.size != b.size or a.size == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return PairedEdge(float("-inf"), float("inf"), float("nan"), float("-inf"), 0)
    d = a - b
    mean = float(d.mean())
    if d.size < 2:
        return PairedEdge(mean, float("inf"), float("nan"), float("-inf"), d.size)
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    t = mean / se if se > 0 else (float("inf") if mean > 0 else float("-inf"))
    lcb = mean - SELECT_LAMBDA * se
    return PairedEdge(mean, se, t, lcb, d.size)


def select_winner(
    scored: list["WFCandidate"], baseline: "WFCandidate"
) -> tuple[Optional["WFCandidate"], dict[int, PairedEdge], str]:
    """Pick the winner by the **paired lower-confidence bound**, then apply the one-SE rule.

    1. Fail-closed eligibility: measurable on **every** fold, a positive mean paired edge, and a
       paired ``t >= SELECT_MIN_PAIRED_T``. A candidate that merely won the loudest fold is out.
    2. Maximise ``LCB = mean(delta) - SE(delta)`` — a pessimistic score that punishes a candidate
       whose edge swings across folds, which is exactly the ``cv_std=1.43`` pathology.
    3. **One-SE rule**: among candidates within one standard error of the best LCB (statistically
       indistinguishable from it), take the most *parsimonious* one — here the lowest turnover,
       since turnover is the thing that costs real money and is the usual overfitting tell.

    Returns ``(winner_or_None, edges_by_index, reason)``. ``None`` means no candidate beat the
    baseline convincingly — the honest outcome, and the baseline is kept.
    """
    edges = {c.index: paired_edge(c, baseline) for c in scored}

    def _passes_t(e: PairedEdge) -> bool:
        # A candidate that beats the baseline by the SAME margin on every fold has zero spread and
        # therefore t=+inf — the strongest evidence there is, not a rejectable non-finite value.
        if e.mean <= 0.0:
            return False
        if math.isinf(e.t) and e.t > 0:
            return True
        return math.isfinite(e.t) and e.t >= SELECT_MIN_PAIRED_T

    eligible = [c for c in scored if math.isfinite(c.cv_mean) and _passes_t(edges[c.index])]
    if not eligible:
        n_pos = sum(1 for c in scored if math.isfinite(edges[c.index].mean) and edges[c.index].mean > 0)
        n_folds = edges[scored[0].index].n_folds if scored else 0
        extra = (
            "（本次只有 1 折，无法估计逐折方差，按 fail-closed 处理）" if n_folds < 2 else
            "（逐折优势波动太大，无法与噪声区分）"
        )
        return None, edges, (
            f"没有候选稳定优于基线：{n_pos}/{len(scored)} 个候选的逐折平均优势为正，"
            f"但没有一个达到配对 t≥{SELECT_MIN_PAIRED_T}{extra}。"
        )
    best = max(eligible, key=lambda c: edges[c.index].lcb)
    # The one-SE band is ONE standard error below the leader's mean paired edge. Because
    # `lcb = mean - SELECT_LAMBDA * se` with lambda = 1, that threshold is exactly the leader's LCB.
    # Subtracting `se` from the *LCB* would make the band `mean - 2*se` — twice as wide as intended,
    # which (with an SE estimated from only 2-3 folds) swallowed nearly every eligible candidate and
    # promoted the tie-break to be the real selector.
    threshold = edges[best.index].lcb
    pool = [c for c in eligible if edges[c.index].mean >= threshold]
    # Parsimony tie-break among candidates statistically indistinguishable from the leader. Turnover
    # is what costs real money and is the usual overfitting tell — measured OUT OF SAMPLE.
    winner = min(pool, key=lambda c: (c.oos_turnover, c.index))
    reason = (
        f"在 {len(eligible)} 个通过配对显著性的候选中，按下置信界 LCB 择优（最优 LCB "
        f"{edges[best.index].lcb:+.4f}）；再用 1-SE 规则在与最优统计上无法区分的 {len(pool)} 个候选里"
        f"选样本外换手最低者（样本外换手 {winner.oos_turnover:.2f}）。"
    )
    return winner, edges, reason


# =============================================================================
# Walk-forward search: evaluate, select out-of-sample, confirm on a holdout,
# and only then declare a candidate promotable (D-TUNE-9).
# =============================================================================
def _evaluate_full(
    spec: TuningSpec,
    fund_code: str,
    start: date,
    end: date,
    store: PointInTimeStore,
    regime_labeler: Optional[Callable] = None,
) -> tuple[BacktestMetrics, Optional[BacktestResult]]:
    """Score the CURRENT CONFIG on ``[start, end]`` and also return the raw backtest result.

    The raw result carries the per-regime attribution and the equity curve that the regime-slice
    and bootstrap gates need. ``regime_labeler`` pins the regime partition so two runs of the same
    window stay comparable slice-by-slice (D-BT-17). Unmeasurable window → fail-closed metrics.
    """
    if spec.objective != "backtest":
        raise ValueError("walk-forward validation currently supports only the backtest objective.")
    fees = market_training_fees(fund_code) if spec.whole_market else default_fee_schedule(fund_code)
    try:
        result = run_backtest(
            fund_code, start, end, strategy=spec.strategy, fees=fees, store=store,
            baselines=SEARCH_BASELINES, regime_labeler=regime_labeler,
        )
    except ValueError:
        return (
            BacktestMetrics(
                net_return=0.0, cagr=0.0, sharpe=0.0, volatility=0.0, max_drawdown=0.0,
                calmar=0.0, turnover=0.0, win_rate=0.0, total_cost=0.0, n_decisions=0,
                measurable=False, vs_baseline={},
            ),
            None,
        )
    return compute_backtest_metrics(result), result


@dataclass
class WFCandidate:
    """One candidate scored in-sample and on every walk-forward validation fold."""

    index: int
    params: dict[str, Union[int, float]]
    insample: BacktestMetrics
    fold_metrics: list[BacktestMetrics]
    metric: str

    @property
    def insample_objective(self) -> float:
        return self.insample.objective(self.metric)

    @property
    def fold_objectives(self) -> list[float]:
        return [m.objective(self.metric) for m in self.fold_metrics]

    @property
    def oos_turnover(self) -> float:
        """Mean turnover across the **validation** folds — the parsimony tie-break.

        Deliberately not ``insample.turnover``: selection must never read an in-sample quantity,
        and an in-sample turnover was in practice not a last-resort tie-break but the *dominant*
        selector, because the one-SE band was accidentally twice as wide as intended.
        """
        vals = [m.turnover for m in self.fold_metrics if math.isfinite(m.turnover)]
        return float(np.mean(vals)) if vals else float("inf")

    @property
    def cv_mean(self) -> float:
        """Mean out-of-sample fold objective — the selection score.

        **Fail-closed**: a candidate that is unmeasurable on *any* fold scores ``-inf``. Averaging
        only the folds it survived would let a candidate that failed 2 of 3 windows win on the one
        it happened to be measurable in — selecting on a biased subset of the evidence.
        """
        vals = self.fold_objectives
        if not vals or any(not math.isfinite(v) for v in vals):
            return float("-inf")
        return float(np.mean(vals))

    @property
    def cv_std(self) -> float:
        vals = self.fold_objectives
        if not vals or any(not math.isfinite(v) for v in vals):
            return 0.0
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    def as_dict(self) -> dict:
        def _f(v: float) -> Optional[float]:
            return None if not math.isfinite(v) else round(v, 6)
        return {
            "index": self.index,
            "params": self.params,
            "insample_objective": _f(self.insample_objective),
            "fold_objectives": [_f(v) for v in self.fold_objectives],
            "cv_mean": _f(self.cv_mean),
            "cv_std": _f(self.cv_std),
            "insample": self.insample.as_dict(),
        }


@dataclass
class PromotionCriterion:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class WalkForwardResult:
    """The full outcome: candidates, diagnostics, holdout confirmation, promotion verdict."""

    run_id: str
    spec_id: str
    fund_code: str
    metric: str
    benchmark: str
    plan: WalkForwardPlan
    candidates: list[WFCandidate]
    baseline: WFCandidate
    best: Optional[WFCandidate]
    holdout_best: Optional[BacktestMetrics]
    holdout_baseline: Optional[BacktestMetrics]
    rank_correlation: float          # Spearman(in-sample objective, CV objective) across candidates
    n_beating_baseline_cv: int
    total_candidates: int
    capped: bool
    criteria: list[PromotionCriterion]
    created_at: str
    selection_reason: str = ""
    edges: dict = field(default_factory=dict)   # candidate index -> PairedEdge.as_dict()

    @property
    def promotable(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    def as_dict(self) -> dict:
        def _f(v: Optional[float]) -> Optional[float]:
            return None if v is None or not math.isfinite(v) else round(v, 6)
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "spec_id": self.spec_id,
            "fund_code": self.fund_code,
            "metric": self.metric,
            "benchmark": self.benchmark,
            "plan": self.plan.as_dict(),
            "candidates": [c.as_dict() for c in self.candidates],
            "baseline": self.baseline.as_dict(),
            "best": self.best.as_dict() if self.best else None,
            "holdout": {
                "best": self.holdout_best.as_dict() if self.holdout_best else None,
                "baseline": self.holdout_baseline.as_dict() if self.holdout_baseline else None,
                "best_objective": _f(self.holdout_best.objective(self.metric)) if self.holdout_best else None,
                "baseline_objective": _f(self.holdout_baseline.objective(self.metric)) if self.holdout_baseline else None,
            },
            "selection": {"reason": self.selection_reason, "edges": self.edges},
            "diagnostics": {
                "rank_correlation": _f(self.rank_correlation),
                "n_beating_baseline_cv": self.n_beating_baseline_cv,
                "n_candidates": len(self.candidates),
                "total_candidates": self.total_candidates,
                "capped": self.capped,
            },
            "promotion": {
                "promotable": self.promotable,
                "criteria": [c.as_dict() for c in self.criteria],
            },
        }


def _judge_promotion(
    metric: str,
    best: Optional[WFCandidate],
    baseline: WFCandidate,
    holdout_best: Optional[BacktestMetrics],
    holdout_baseline: Optional[BacktestMetrics],
    best_res: Optional[BacktestResult],
    base_res: Optional[BacktestResult],
) -> list[PromotionCriterion]:
    """The doc's "stably beats baseline" gate (§5 / §7 ch:testing), evaluated fail-closed.

    A candidate is promotable ONLY if it (1) beats the baseline on the mean out-of-sample fold
    score, (2) still beats it on the untouched holdout, (3) wins enough per-regime slices with
    enough regimes actually observed, and (4) its holdout edge over the baseline is statistically
    distinguishable from zero (bootstrap CI excludes 0). Anything missing → not promotable.
    """
    from qmt.config import CONFIG

    v = CONFIG.validation
    out: list[PromotionCriterion] = []
    if best is None or holdout_best is None or holdout_baseline is None or best_res is None or base_res is None:
        return [PromotionCriterion("可测量性", False, "没有可测量的候选或留出窗口无数据（fail-closed）。")]

    # 1. out-of-sample cross-validation
    ok1 = math.isfinite(best.cv_mean) and best.cv_mean > baseline.cv_mean
    out.append(PromotionCriterion(
        "样本外交叉验证优于基线",
        ok1,
        f"候选 CV 均值 {best.cv_mean:.3f} vs 基线 {baseline.cv_mean:.3f}",
    ))

    # 2. untouched holdout
    hb, hbase = holdout_best.objective(metric), holdout_baseline.objective(metric)
    ok2 = math.isfinite(hb) and math.isfinite(hbase) and hb > hbase
    out.append(PromotionCriterion(
        "留出窗口（从未参与择优）优于基线",
        ok2,
        f"候选 {hb:.3f} vs 基线 {hbase:.3f}",
    ))

    # 3. regime slices
    won, present = regime_wins(best_res, base_res)
    ok3 = present >= v.total_regime_slices and won >= v.min_regime_slices_won
    detail = f"在 {present} 个已观测市场环境中赢 {won} 个（要求：≥{v.total_regime_slices} 个环境中赢 ≥{v.min_regime_slices_won} 个）"
    if present < v.total_regime_slices:
        detail += "；观测到的环境不足，证据不充分"
    out.append(PromotionCriterion("跨市场环境稳定", ok3, detail))

    # 4. bootstrap significance on the holdout
    lo, hi, n = paired_bootstrap_ci(
        _curve_returns(best_res.equity_curve), _curve_returns(base_res.equity_curve), v.bootstrap_ci_level
    )
    ok4 = math.isfinite(lo) and lo > 0.0
    out.append(PromotionCriterion(
        f"留出优势的 {int(v.bootstrap_ci_level * 100)}% 自助置信区间不含 0",
        ok4,
        (f"CI [{lo:+.5f}, {hi:+.5f}]（n={n}）" if math.isfinite(lo) else "配对样本不足，无法检验"),
    ))
    return out


def stream_walk_forward_search(
    spec: TuningSpec,
    fund_code: str,
    plan: WalkForwardPlan,
    store: PointInTimeStore,
    *,
    metric: Optional[str] = None,
    benchmark: str = WHOLE_MARKET_BENCHMARK,
    cancel: Optional[Callable[[], bool]] = None,
    _result_sink: Optional[list] = None,
) -> Iterator[dict]:
    """Walk-forward parameter search as a stream of progress events.

    For every candidate: score it in-sample (diagnostic only) and on **each** validation fold.
    Select by the mean fold score — never by the in-sample score, which was shown to be
    anti-predictive. Confirm the winner on the untouched holdout and judge it against the doc's
    promotion gate. Also reports Spearman(in-sample, CV) so the anti-predictiveness is *visible*.
    The shared CONFIG is snapshotted and restored in a ``finally``.
    """
    from qmt.controlplane.api import apply_config_override, restore_config, snapshot_config

    metric = metric or spec.metrics[0]
    if metric not in spec.metrics:
        raise ValueError(f"metric {metric!r} not offered by model {spec.id!r} (choose {list(spec.metrics)}).")

    candidates, capped = spec.candidate_configs()
    snapshot = snapshot_config()
    # +1 baseline, +the exact refinement upper bound (2 midpoints per axis, capped), +2 holdouts
    max_refine = min(REFINE_MAX_TRIALS, 2 * len(spec.params))
    total_evals = (len(candidates) + 1 + max_refine) * (len(plan.folds) + 1) + 2

    def _apply(cand: dict) -> None:
        restore_config(snapshot)
        for path, val in cand.items():
            apply_config_override(path, val)

    def _score(cand: dict) -> tuple[BacktestMetrics, list[BacktestMetrics]]:
        _apply(cand)
        ins, _ = _evaluate_full(spec, fund_code, plan.insample_start, plan.insample_end, store)
        folds = [
            _evaluate_full(spec, fund_code, f.valid_start, f.valid_end, store)[0] for f in plan.folds
        ]
        return ins, folds

    try:
        yield {
            "event": "start", "spec_id": spec.id, "fund_code": fund_code, "benchmark": benchmark,
            "objective": spec.objective, "metric": metric, "total": len(candidates),
            "capped": capped, "grid_size": spec.grid_size(), "total_evals": total_evals,
            "plan": plan.as_dict(),
        }

        # --- baseline: the live CONFIG, scored exactly the same way -----------------------
        restore_config(snapshot)
        b_ins, b_folds = _score({})
        baseline = WFCandidate(index=-1, params={}, insample=b_ins, fold_metrics=b_folds, metric=metric)
        yield {"event": "baseline", "candidate": baseline.as_dict()}

        # --- every candidate, scored on every fold (selection is out-of-sample) -----------
        scored: list[WFCandidate] = []
        for i, cand in enumerate(candidates):
            if cancel is not None and cancel():
                yield {"event": "cancelled", "completed": i, "total": len(candidates)}
                return
            ins, folds = _score(cand)
            wf = WFCandidate(index=i, params=cand, insample=ins, fold_metrics=folds, metric=metric)
            scored.append(wf)
            yield {"event": "trial", "index": i, "total": len(candidates), "candidate": wf.as_dict()}

        # Selection is by the fold-PAIRED lower-confidence bound + one-SE rule (D-TUNE-11), not by
        # raw cv_mean: the fold-to-fold spread is mostly the fold's own market regime, which the
        # baseline lives through too. Pairing cancels that common-mode noise.
        best, edges, select_reason = select_winner(scored, baseline)
        yield {"event": "selection", "stage": "coarse",
               "winner_index": (best.index if best else None),
               "reason": select_reason,
               "edges": {str(i): e.as_dict() for i, e in edges.items()}}

        # --- coarse-to-fine: probe BETWEEN the coarse levels around the incumbent --------
        # Only meaningful once a coarse winner exists; refined candidates are scored on the same
        # folds and re-enter the same paired-LCB contest with no privilege. The holdout stays shut.
        refined = spec.refine_around(best.params, seen=candidates, budget=REFINE_MAX_TRIALS) if best else []
        if refined:
            yield {"event": "refine_start", "n": len(refined), "around": best.params}
            for j, cand in enumerate(refined):
                if cancel is not None and cancel():
                    yield {"event": "cancelled", "completed": len(candidates) + j,
                           "total": len(candidates) + len(refined)}
                    return
                idx = len(candidates) + j
                ins, folds = _score(cand)
                wf = WFCandidate(index=idx, params=cand, insample=ins, fold_metrics=folds, metric=metric)
                scored.append(wf)
                yield {"event": "trial", "stage": "refine", "index": idx,
                       "total": len(candidates) + len(refined), "candidate": wf.as_dict()}
            best, edges, select_reason = select_winner(scored, baseline)
            yield {"event": "selection", "stage": "refine",
                   "winner_index": (best.index if best else None),
                   "reason": select_reason,
                   "edges": {str(i): e.as_dict() for i, e in edges.items()}}

        scored.sort(key=lambda c: (edges[c.index].lcb if math.isfinite(edges[c.index].lcb) else -1e18), reverse=True)

        # --- diagnostics: would in-sample ranking have helped at all? ---------------------
        xs = np.array([c.insample_objective for c in scored if math.isfinite(c.insample_objective) and math.isfinite(c.cv_mean)])
        ys = np.array([c.cv_mean for c in scored if math.isfinite(c.insample_objective) and math.isfinite(c.cv_mean)])
        rho = spearman(xs, ys)
        n_beat = sum(1 for c in scored if math.isfinite(c.cv_mean) and c.cv_mean > baseline.cv_mean)
        yield {"event": "diagnostics", "rank_correlation": (None if not math.isfinite(rho) else round(rho, 6)),
               "n_beating_baseline_cv": n_beat, "n_candidates": len(scored)}

        # --- confirm the winner on the untouched holdout ----------------------------------
        holdout_best = holdout_baseline = None
        best_res = base_res = None
        if best is not None:
            # Pin the regime partition to the BASELINE config so the candidate and the baseline are
            # attributed under one identical set of regime buckets — otherwise the swept
            # `ma_fast_days` moves the labeller and the slice comparison is meaningless (D-BT-17).
            restore_config(snapshot)
            from qmt.backtest import make_frozen_regime_labeler
            from qmt.config import CONFIG as _CFG

            ho_labeler = make_frozen_regime_labeler(store, plan.holdout_end, _CFG.market_state.ma_fast_days)

            _apply(best.params)
            holdout_best, best_res = _evaluate_full(
                spec, fund_code, plan.holdout_start, plan.holdout_end, store, regime_labeler=ho_labeler
            )
            restore_config(snapshot)
            holdout_baseline, base_res = _evaluate_full(
                spec, fund_code, plan.holdout_start, plan.holdout_end, store, regime_labeler=ho_labeler
            )
            yield {"event": "holdout",
                   "best": holdout_best.as_dict(), "baseline": holdout_baseline.as_dict()}

        criteria = _judge_promotion(metric, best, baseline, holdout_best, holdout_baseline, best_res, base_res)

        result = WalkForwardResult(
            run_id=_new_run_id(), spec_id=spec.id, fund_code=fund_code, metric=metric,
            benchmark=benchmark, plan=plan, candidates=scored, baseline=baseline, best=best,
            holdout_best=holdout_best, holdout_baseline=holdout_baseline,
            rank_correlation=rho, n_beating_baseline_cv=n_beat,
            total_candidates=spec.grid_size(), capped=capped, criteria=criteria,
            created_at=datetime.now().isoformat(timespec="seconds"),
            selection_reason=select_reason,
            edges={str(i): e.as_dict() for i, e in edges.items()},
        )
        if _result_sink is not None:
            _result_sink.append(result)
        yield {"event": "done", "result": result.as_dict()}
    finally:
        restore_config(snapshot)


def run_walk_forward_search(
    spec: TuningSpec, fund_code: str, plan: WalkForwardPlan, store: PointInTimeStore,
    *, metric: Optional[str] = None, benchmark: str = WHOLE_MARKET_BENCHMARK,
) -> WalkForwardResult:
    """Drain :func:`stream_walk_forward_search` and return the :class:`WalkForwardResult`."""
    sink: list[WalkForwardResult] = []
    for _ev in stream_walk_forward_search(
        spec, fund_code, plan, store, metric=metric, benchmark=benchmark, _result_sink=sink
    ):
        pass
    if not sink:
        raise RuntimeError("walk-forward search did not produce a result.")
    return sink[0]


# =============================================================================
# Persistence — a training run must be reviewable after the browser tab closes
# =============================================================================
TUNING_RUNS_DIRNAME = "tuning_runs"


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def tuning_runs_dir() -> "Path":
    from pathlib import Path

    from qmt.config import get_settings

    d: Path = get_settings().data_cache_dir / TUNING_RUNS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def persist_run(result: WalkForwardResult) -> str:
    """Write a finished run to ``data_cache/tuning_runs/<run_id>.json``; return the path."""
    path = tuning_runs_dir() / f"{result.run_id}.json"
    path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    logger.info("persisted tuning run %s -> %s", result.run_id, path)
    return str(path)


def list_runs(limit: int = 50) -> list[dict]:
    """Newest-first summaries of the persisted training runs."""
    out: list[dict] = []
    for p in sorted(tuning_runs_dir().glob("*.json"), reverse=True)[:limit]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - a corrupt file never breaks the list
            continue
        out.append({
            "run_id": d.get("run_id"), "created_at": d.get("created_at"),
            "spec_id": d.get("spec_id"), "benchmark": d.get("benchmark"), "metric": d.get("metric"),
            "promotable": d.get("promotion", {}).get("promotable"),
            "rank_correlation": d.get("diagnostics", {}).get("rank_correlation"),
            "best_params": (d.get("best") or {}).get("params"),
            "plan": d.get("plan", {}),
        })
    return out


# A run id is exactly the timestamp `persist_run` writes: 8 digits, 'T', 6 digits. Anchoring the
# pattern this tightly (no '.', no separators) makes path traversal structurally impossible.
_RUN_ID_RE = re.compile(r"\A\d{8}T\d{6}\Z")


def load_run(run_id: str) -> dict:
    """Load one persisted run by id (raises :class:`FileNotFoundError` when absent/invalid)."""
    if not _RUN_ID_RE.match(run_id or ""):
        raise FileNotFoundError(f"invalid run id {run_id!r}")
    path = tuning_runs_dir() / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
