"""Training-only selection of complementary factors by executable portfolio gain.

Candidates must be frozen in a discovery subsection before every inner validation
label. This module never generates or fits an expression. It compares equal-rank
pools using the same fund execution engine, dates, fees and starting cash on each
side. The paired statistic is terminal net equity(new) / net equity(current) - 1,
not compounded demeaned forward returns. Outer validation cannot enter selection.

The pool-increment principle is inspired by AlphaGen
(https://arxiv.org/abs/2306.12964); this is a conservative fund-specific greedy
selector, not a reproduction of its RL search or optimizer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import copy
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from qmt.factor.dsl import FormulaDSL
from qmt.factor.metrics import scores_by_date_from_frame
from qmt.factor.types import BaseFactorPanel, FactorExpr

if TYPE_CHECKING:
    from qmt.factor.fund_execution import ExecutionConfig

__all__ = [
    "IncrementalConfig", "InnerFold", "IncrementalSelection",
    "build_inner_folds", "select_incremental_pool", "make_frozen_pool_score_callback",
]

_EPS = 1e-12


@dataclass(frozen=True)
class IncrementalConfig:
    """Pre-registered controls; never tuned on the outer validation period."""

    n_folds: int = 3
    max_pool_size: int = 3
    min_mean_delta: float = 0.0
    min_positive_folds: int = 2
    worst_fold_tolerance: float = 0.0
    horizon_rows: int = 20
    stride_rows: int = 20
    min_usable_fraction: float = 0.8

    def __post_init__(self):
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if self.max_pool_size < 1:
            raise ValueError("max_pool_size must be positive")
        if not 1 <= self.min_positive_folds <= self.n_folds:
            raise ValueError("min_positive_folds must lie between one and n_folds")
        if self.horizon_rows < 1 or self.stride_rows < 1:
            raise ValueError("horizon_rows and stride_rows must be positive")
        if not np.isfinite(self.min_mean_delta) or self.min_mean_delta < 0:
            raise ValueError("min_mean_delta must be finite and nonnegative")
        if not np.isfinite(self.worst_fold_tolerance) or self.worst_fold_tolerance < 0:
            raise ValueError("worst_fold_tolerance must be finite and nonnegative")
        if not np.isfinite(self.min_usable_fraction) or not 0 < self.min_usable_fraction <= 1:
            raise ValueError("min_usable_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class InnerFold:
    fold_id: str
    decision_dates: tuple[date, ...]
    end_date: date

    def to_dict(self) -> dict:
        return {"fold_id": self.fold_id,
                "decision_dates": [str(d) for d in self.decision_dates],
                "end_date": str(self.end_date)}


@dataclass(frozen=True)
class IncrementalSelection:
    selected_names: tuple[str, ...]
    selected_exprs: tuple[FactorExpr, ...]
    baseline_name: str
    folds: tuple[InnerFold, ...]
    baseline_fold_metrics: Mapping[str, tuple[dict, ...]]
    paired_trial_ledger: tuple[dict, ...]
    audit: Mapping[str, Any]

    def to_dict(self) -> dict:
        return {"selected_names": list(self.selected_names),
                "selected_formulas": [e.text for e in self.selected_exprs],
                "frozen_weights": {n: 1.0 / len(self.selected_names)
                                   for n in self.selected_names},
                "baseline_name": self.baseline_name,
                "folds": [f.to_dict() for f in self.folds],
                "baseline_fold_metrics": dict(self.baseline_fold_metrics),
                "paired_trial_ledger": list(self.paired_trial_ledger),
                "audit": dict(self.audit)}


def _date(d) -> date:
    return pd.Timestamp(d).date()


def _observations(dates: Sequence[date]) -> tuple[date, ...]:
    return tuple(sorted({_date(d) for d in dates}))


def _position(observed: tuple[date, ...], d: date) -> int:
    return int(np.searchsorted(np.asarray(observed, dtype="datetime64[D]"),
                               np.datetime64(d), side="right")) - 1


def build_inner_folds(
    observation_dates: Sequence[date], *, discovery_end: date,
    outer_train_end: date, horizon_rows: int, stride_rows: int,
    n_folds: int = 3,
) -> tuple[InnerFold, ...]:
    """Three chronological inner blocks with exact observed-row label purges.

    ``discovery_end`` is the last decision whose forward label could have been
    used to produce candidates. Its complete label end must precede the first
    inner decision. Each fold leaves a horizon-sized tail for label maturation;
    folds' outcome periods are disjoint and stay at/before ``outer_train_end``.
    """
    if horizon_rows < 1 or stride_rows < 1 or n_folds < 2:
        raise ValueError("positive horizon/stride and at least two folds required")
    observed = _observations(observation_dates)
    discovery_end, outer_train_end = _date(discovery_end), _date(outer_train_end)
    if not observed or discovery_end < observed[0] or discovery_end >= outer_train_end:
        raise ValueError("invalid discovery/outer training boundaries")
    first = _position(observed, discovery_end) + horizon_rows + 1
    last = _position(observed, outer_train_end)
    if first > last:
        raise ValueError("insufficient observations after discovery label purge")
    blocks = np.array_split(np.asarray(observed[first:last + 1], dtype=object), n_folds)
    folds = []
    for i, block in enumerate(blocks):
        if len(block) <= horizon_rows:
            raise ValueError("insufficient observations for complete inner validation horizons")
        ds = tuple(block[:len(block) - horizon_rows:stride_rows])
        folds.append(InnerFold(f"inner{i + 1}", ds, block[-1]))
    return tuple(folds)


def _validate_folds(folds, observed, discovery_end, outer_train_end, cfg):
    if len(folds) != cfg.n_folds:
        raise ValueError("fold count must equal the pre-registered n_folds")
    discovery_pos = _position(observed, discovery_end)
    if discovery_pos < 0 or discovery_pos + cfg.horizon_rows >= len(observed):
        raise ValueError("discovery labels have no complete observed horizon")
    label_end = observed[discovery_pos + cfg.horizon_rows]
    previous_end = label_end
    ids = set()
    for f in folds:
        if f.fold_id in ids:
            raise ValueError("fold_id values must be unique")
        ids.add(f.fold_id)
        if not f.decision_dates or tuple(sorted(set(f.decision_dates))) != f.decision_dates:
            raise ValueError("inner decision dates must be nonempty, sorted and unique")
        if f.decision_dates[0] <= previous_end:
            raise ValueError("inner fold overlaps prior label/outcome region")
        if f.end_date > outer_train_end or f.end_date < f.decision_dates[-1]:
            raise ValueError("inner fold extends beyond outer training")
        if f.end_date not in observed:
            raise ValueError("inner end_date must be an observed trading date")
        end_pos = _position(observed, f.end_date)
        for d in f.decision_dates:
            if d not in observed or _position(observed, d) + cfg.horizon_rows > end_pos:
                raise ValueError("inner decision lacks a complete future-label horizon")
        previous_end = f.end_date


def _truncate_frame(frame: pd.DataFrame, end: date) -> pd.DataFrame:
    cutoff = pd.Timestamp(end) if isinstance(frame.index, pd.DatetimeIndex) else end
    return frame.loc[frame.index <= cutoff].copy()


def _training_panel(panel: BaseFactorPanel, end: date) -> BaseFactorPanel:
    return BaseFactorPanel(
        dates=[_date(d) for d in panel.dates if _date(d) <= end],
        funds=list(panel.funds),
        factors={n: _truncate_frame(f, end) for n, f in panel.factors.items()},
        industry_of=dict(panel.industry_of), meta=dict(panel.meta),
    )


def _mask_unavailable_nav(unit_nav, adjusted_nav, availability, end_date):
    """Exclude outcome quotes whose publication is beyond this inner cutoff."""
    source = availability.copy()
    source.index = [_date(d) for d in source.index]
    if source.index.has_duplicates or source.columns.has_duplicates:
        raise ValueError("nav_available_at must have unique valuation dates and fund columns")
    source = source.reindex(index=[_date(d) for d in unit_nav.index], columns=unit_nav.columns)
    source.index = unit_nav.index
    cutoff = pd.Timestamp(_date(end_date)) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    known = pd.DataFrame(False, index=unit_nav.index, columns=unit_nav.columns)
    for code in source:
        published = pd.to_datetime(source[code], errors="coerce")
        local_cutoff = cutoff
        if published.dt.tz is not None:
            local_cutoff = cutoff.tz_localize("Asia/Shanghai").tz_convert(published.dt.tz)
        known[code] = published.notna() & (published <= local_cutoff)
    return unit_nav.where(known), adjusted_nav.where(known.reindex(
        index=adjusted_nav.index, columns=adjusted_nav.columns, fill_value=False))


def _blend_sections(sections: Sequence[pd.Series | None], held: pd.Series):
    """Fixed equal ranks; missing components or held signals retain holdings."""
    if not sections or any(s is None or s.empty for s in sections):
        return None
    ranks = [s.replace([np.inf, -np.inf], np.nan).dropna().rank(pct=True) - 0.5
             for s in sections]
    if any(s.empty for s in ranks):
        return None
    complete = pd.concat(ranks, axis=1).dropna()
    held_names = held[held > _EPS].index
    if complete.empty or not held_names.isin(complete.index).all():
        return None
    # dropna before averaging is mandatory: formula weights cannot vary by fund.
    composite = complete.sum(axis=1) / len(ranks)
    # A cancelled/constant blend has no ordering information. Never let code
    # order manufacture a portfolio, or treat missing information as a fit.
    return composite if float(composite.max() - composite.min()) > _EPS else None


class _FrozenPoolCallback:
    """Auditable scoring-only callback; it cannot fit or change pool weights."""

    def __init__(self, sections, formula_errors=None):
        self.sections = tuple(sections)
        self.formula_errors = dict(formula_errors or {})
        self.attempted_dates = []
        self.formula_present_dates = []
        self.usable_dates = []
        self.missing_dates = []

    def __call__(self, d, held):
        d = _date(d)
        self.attempted_dates.append(d)
        sections = [s.get(d) for s in self.sections]
        if sections and all(s is not None and not s.empty
                            and np.isfinite(s.to_numpy(float)).any() for s in sections):
            self.formula_present_dates.append(d)
        blend = _blend_sections(sections, held)
        if blend is None:
            self.missing_dates.append(d)
        else:
            self.usable_dates.append(d)
        return blend


def make_frozen_pool_score_callback(
    exprs: Sequence[FactorExpr], panel: BaseFactorPanel,
    decision_dates: Sequence[date], *, end_date: date,
):
    """Score frozen equal-rank expressions with the same inner/outer policy.

    No labels or fitting enter this function. Feature frames are physically
    truncated to ``end_date`` before causal expression evaluation. None scores
    request no new orders, retaining the whole held portfolio. An outer reporting
    run may use this callback without passing its outcomes back to selection.
    Callback counters distinguish formula presence, complete held signals and
    actual attempted decisions; they do not change the trading policy.
    """
    if not exprs:
        raise ValueError("a frozen pool must contain at least one expression")
    frozen = tuple(copy.deepcopy(e) for e in exprs)
    if len({e.text for e in frozen}) != len(frozen):
        raise ValueError("frozen pool contains duplicate formulas")
    end_date = _date(end_date)
    dates = [_date(d) for d in decision_dates]
    if any(d > end_date for d in dates):
        raise ValueError("scoring decisions extend beyond the feature cutoff")
    frozen_panel = _training_panel(panel, end_date)
    dsl, sections, errors = FormulaDSL(), [], {}
    for e in frozen:
        try:
            frame = dsl.evaluate_frame(e, frozen_panel)
            sections.append(scores_by_date_from_frame(frame, dates))
        except (KeyError, TypeError, ValueError) as exc:
            sections.append({})
            errors[e.text] = str(exc)
    return _FrozenPoolCallback(sections, errors)


def select_incremental_pool(
    candidates: Mapping[str, FactorExpr], baselines: Mapping[str, FactorExpr],
    panel: BaseFactorPanel, unit_nav: pd.DataFrame, adjusted_nav: pd.DataFrame,
    *, discovery_end: date, outer_train_end: date,
    execution_config: "ExecutionConfig", fees_resolver=None,
    visible_unit_nav: pd.DataFrame | None = None,
    visible_adjusted_nav: pd.DataFrame | None = None,
    visible_nav_dates: pd.DataFrame | None = None,
    nav_available_at: pd.DataFrame | None = None,
    dividend_events=None,
    folds: Sequence[InnerFold] | None = None,
    config: IncrementalConfig | None = None,
) -> IncrementalSelection:
    """Select a frozen pool using inner-training fee-net paired wealth gains.

    The caller must produce ``candidates`` using only discovery-period labels.
    This function enforces the subsequent label purge, physically excludes all
    outer-validation NAV/features, and records every evaluated addition. The
    strongest manual singleton is chosen by mean log net wealth across inner
    folds; no weights, gates or execution controls are fitted on outer data.
    """
    cfg = config or IncrementalConfig()
    discovery_end, outer_train_end = _date(discovery_end), _date(outer_train_end)
    observed = _observations(execution_config.trading_dates)
    seal = getattr(execution_config, "seal_date", None)
    if seal is not None and outer_train_end >= _date(seal):
        raise ValueError("outer training reaches the sealed region")
    if not observed or discovery_end < observed[0] or discovery_end >= outer_train_end:
        raise ValueError("invalid discovery/outer training boundaries")
    if not baselines:
        raise ValueError("at least one pre-registered manual baseline is required")
    if set(candidates) & set(baselines):
        raise ValueError("candidate and baseline names must be distinct")
    if getattr(execution_config, "initial_cash", 0) <= 0:
        raise ValueError("execution initial cash must be positive")

    folds = tuple(InnerFold(f.fold_id, tuple(_date(d) for d in f.decision_dates),
                            _date(f.end_date)) for f in folds) if folds is not None else build_inner_folds(
        observed, discovery_end=discovery_end, outer_train_end=outer_train_end,
        horizon_rows=cfg.horizon_rows, stride_rows=cfg.stride_rows, n_folds=cfg.n_folds,
    )
    _validate_folds(folds, observed, discovery_end, outer_train_end, cfg)
    frozen = {n: copy.deepcopy(e) for n, e in {**baselines, **candidates}.items()}
    train_panel = _training_panel(panel, outer_train_end)
    nav = _truncate_frame(unit_nav, outer_train_end)
    adjusted = _truncate_frame(adjusted_nav, outer_train_end)
    availability = (_truncate_frame(nav_available_at, outer_train_end)
                    if nav_available_at is not None else None)
    visible = {n: _truncate_frame(f, outer_train_end) for n, f in
               {"visible_unit_nav": visible_unit_nav,
                "visible_adjusted_nav": visible_adjusted_nav,
                "visible_nav_dates": visible_nav_dates}.items() if f is not None}
    dsl = FormulaDSL()
    decisions = sorted({d for f in folds for d in f.decision_dates})
    scores = {}
    formula_errors = {}
    for n, e in frozen.items():
        try:
            scores[n] = scores_by_date_from_frame(dsl.evaluate_frame(e, train_panel), decisions)
        except (KeyError, TypeError, ValueError) as exc:
            scores[n] = {}
            formula_errors[n] = str(exc)

    metrics_cache = {}

    def evaluate(names):
        key = tuple(names)
        if key not in metrics_cache:
            metrics_cache[key] = tuple(
                _run_pool(key, scores, f, nav, adjusted, execution_config, fees_resolver,
                          visible, availability, dividend_events)
                for f in folds
            )
        return metrics_cache[key]

    baseline_metrics = {n: evaluate((n,)) for n in sorted(baselines)}
    usable_baselines = [n for n in baselines if n not in formula_errors
                        and all(m["usable_fraction"] >= cfg.min_usable_fraction
                                and m["traded_gross_money"] > _EPS for m in baseline_metrics[n])]
    if not usable_baselines:
        raise ValueError("no manual baseline has usable signals in every inner fold")
    baseline_name = min(
        usable_baselines,
        key=lambda n: (-float(np.mean([m["log_net_wealth"] for m in baseline_metrics[n]])), n),
    )
    pool = [baseline_name]
    remaining = sorted(candidates)
    ledger = []
    round_id = 0
    while remaining and len(pool) < cfg.max_pool_size:
        round_id += 1
        current = evaluate(pool)
        eligible = []
        for n in remaining:
            if any(frozen[n].text == frozen[k].text for k in pool):
                ledger.append({"round": round_id, "candidate": n,
                               "pool_before": list(pool), "eligible": False,
                               "accepted": False, "reason": "duplicate_formula", "folds": []})
                continue
            proposed = evaluate((*pool, n))
            paired = []
            for old, new in zip(current, proposed):
                paired.append({"fold_id": old["fold_id"],
                               "paired_net_delta": new["final_equity"] / old["final_equity"] - 1.0,
                               "turnover_delta": new["turnover"] - old["turnover"],
                               "cost_money_delta": new["cost_money"] - old["cost_money"],
                               "current": old, "proposed": new})
            deltas = [f["paired_net_delta"] for f in paired]
            mean_delta = float(np.mean(deltas))
            positive = sum(d > _EPS for d in deltas)
            reasons = []
            if n in formula_errors or any(m["usable_fraction"] < cfg.min_usable_fraction for m in proposed):
                reasons.append("missing_candidate_signals")
            if any(m["traded_gross_money"] <= _EPS for m in proposed):
                reasons.append("no_executed_trades")
            if mean_delta <= cfg.min_mean_delta + _EPS:
                reasons.append("no_fee_net_increment")
            if positive < cfg.min_positive_folds:
                reasons.append("insufficient_positive_folds")
            if min(deltas) < -cfg.worst_fold_tolerance - _EPS:
                reasons.append("negative_inner_fold")
            row = {"round": round_id, "candidate": n, "formula": frozen[n].text,
                   "pool_before": list(pool), "pool_after": [*pool, n],
                   "mean_paired_net_delta": mean_delta,
                   "worst_paired_net_delta": min(deltas), "positive_folds": positive,
                   "worst_proposed_drawdown": min(m["max_drawdown"] for m in proposed),
                   "eligible": not reasons, "accepted": False,
                   "reason": ";".join(reasons) if reasons else "passed_not_selected",
                   "folds": paired}
            ledger.append(row)
            if not reasons:
                eligible.append(row)
        if not eligible:
            break
        chosen = min(eligible, key=lambda r: (-r["mean_paired_net_delta"], r["candidate"]))
        chosen.update(accepted=True, reason="selected_largest_training_increment")
        pool.append(chosen["candidate"])
        remaining.remove(chosen["candidate"])

    audit = {
        "selection_scope": "discovery plus chronological inner folds, all within outer training",
        "discovery_end": str(discovery_end), "outer_train_end": str(outer_train_end),
        "last_selection_outcome_date": str(max(f.end_date for f in folds)),
        "outer_validation_values_used": False,
        "candidate_generation": "caller must freeze candidates using discovery labels only",
        "config": asdict(cfg), "formula_errors": formula_errors,
        "publication_visibility_matrices": sorted(visible),
        "inner_nav_availability_guard": availability is not None,
        "observed_dividend_history_supplied": dividend_events is not None,
        "inner_outcome_publication_cutoff": "each fold end at 23:59:59 Asia/Shanghai",
        "missing_nav_availability": "reject quote" if availability is not None else "caller must supply PIT-clipped outcome quotes",
        "execution_controls": {n: getattr(execution_config, n, None) for n in
                               ("initial_cash", "top_k", "sticky_rank_buffer",
                                "fill_lag_sessions", "purchase_confirm_sessions",
                                "redemption_confirm_sessions", "redemption_settlement_sessions",
                                "reinvest_dividends", "liquidate_at_end", "require_status")},
        "fee_source": ("default assumed schedule" if fees_resolver is None
                       else "provided mapping" if isinstance(fees_resolver, Mapping)
                       else "provided deterministic as-of resolver"),
        "pool_rule": "fixed equal percentile ranks; complete cases; missing component/held score retains holdings",
        "paired_statistic": "final net equity(new) / final net equity(current) - 1",
        "manual_selection": "highest mean log net wealth across the same inner folds",
        "execution_comparability": "same engine, cash, fees, topK, sticky rule and fold dates",
        "distinct_pools_evaluated": len(metrics_cache),
        "selection_trials": len(ledger),
        "limitations": ["inner-fold selection still requires a later untouched validation",
                        "multiple candidate additions are research trials, not adjusted significance tests",
                        "terminal wealth follows configured liquidation policy; unsold holdings are marked",
                        "fee/status callbacks must be deterministic across paired runs"],
    }
    return IncrementalSelection(tuple(pool), tuple(frozen[n] for n in pool), baseline_name,
                                folds, baseline_metrics, tuple(ledger), audit)


def _run_pool(names, scores, fold, unit_nav, adjusted_nav, execution_config, fees_resolver,
              visible=None, availability=None, dividend_events=None):
    """Bridge to the fund execution engine; callbacks only read frozen signals."""
    from qmt.factor.fund_execution import run_fund_backtest

    score_callback = _FrozenPoolCallback([scores[n] for n in names])

    cfg = replace(execution_config, end_date=fold.end_date)
    unit = _truncate_frame(unit_nav, fold.end_date)
    adjusted = _truncate_frame(adjusted_nav, fold.end_date)
    if availability is not None:
        unit, adjusted = _mask_unavailable_nav(unit, adjusted, availability, fold.end_date)
    result = run_fund_backtest(
        unit, adjusted,
        fold.decision_dates, score_callback, fees_resolver=fees_resolver, config=cfg,
        **({"dividend_events": dividend_events} if dividend_events is not None else {}),
        **{n: _truncate_frame(f, fold.end_date) for n, f in (visible or {}).items()},
    )
    daily = result.daily
    equity = daily["equity"].to_numpy(float)
    if not len(equity) or not np.isfinite(equity).all() or equity[-1] <= 0:
        raise ValueError("fund engine produced no finite positive terminal equity")
    wealth = float(equity[-1] / cfg.initial_cash)
    path = np.r_[cfg.initial_cash, equity]
    drawdown = float(np.min(path / np.maximum.accumulate(path) - 1.0))
    return {"fold_id": fold.fold_id, "first_decision": str(fold.decision_dates[0]),
            "end_date": str(fold.end_date), "n_decisions": len(fold.decision_dates),
            "final_equity": float(equity[-1]), "net_wealth_ratio": wealth,
            "log_net_wealth": float(np.log(wealth)), "max_drawdown": drawdown,
            "turnover": float(result.turnover), "cost_money": float(result.cost_money),
            "traded_gross_money": float(result.total_traded_money),
            "execution_assumptions": list(result.assumptions),
            "stale_valuation_days": int((daily["stale_holdings"] > 0).sum()),
            "attempted_strategy_decisions": len(set(score_callback.attempted_dates)),
            "formula_present_decisions": len(set(score_callback.formula_present_dates)),
            "usable_signal_decisions": len(set(score_callback.usable_dates)),
            "usable_fraction": len(set(score_callback.usable_dates)) / len(fold.decision_dates),
            "missing_signal_decisions": [str(d) for d in score_callback.missing_dates]}
