"""Strict, independently re-searched chronological fund research segments.

The runner consumes a caller-loaded ExecutionData bundle and audited historical
fee, dealing-status and dividend histories. It never fetches data or substitutes
assumed fees, OPEN status, or adjusted/unit dividend proxies. Incomplete coverage
produces blocked records with no test returns. Each segment freezes its own pool
using earlier discovery and inner-training folds before reporting outer outcomes.

Products of segment wealth ratios describe independently cash-initialized
research episodes, not the NAV of a continuous account across year boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from qmt.factor.auto_search import SymbolicSearch
from qmt.factor.baselines import manual_exprs
from qmt.factor.cost_benchmark import ExecutionData, executable_labels, paired_active_log_returns
from qmt.factor.dsl import FormulaDSL
from qmt.factor.event_history import DividendHistory, StatusHistory
from qmt.factor.fee_history import FeeHistory
from qmt.factor.fund_execution import ExecutionConfig, run_fund_backtest
from qmt.factor.incremental import (
    IncrementalConfig, build_inner_folds, make_frozen_pool_score_callback,
    select_incremental_pool, _truncate_frame, _training_panel,
)
from qmt.factor.types import SearchConfig, SearchLedger, ValidationScheme, ValidationSplit
from qmt.stats import newey_west_tstat

__all__ = ["OuterFold", "WalkForwardConfig", "CoverageRequest", "default_outer_folds",
           "aggregate_segment_gates", "joint_family_test", "audit_nav_quality", "run_walk_forward"]


def _date(value):
    return pd.Timestamp(value).date()


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    train_start: date
    discovery_end: date
    train_end: date
    test_start: date
    test_end: date

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.fold_id):
            raise ValueError("fold_id must be a safe short identifier")
        if not self.train_start < self.discovery_end < self.train_end < self.test_start <= self.test_end:
            raise ValueError("outer discovery/training/test boundaries must be chronological")


@dataclass(frozen=True)
class WalkForwardConfig:
    seeds: tuple[int, ...] = (7, 19, 43)
    budget: int = 120
    horizon_rows: int = 40
    stride_rows: int = 20
    warmup_rows: int = 320
    initial_cash: float = 100_000.0
    top_k: int = 5
    sticky_rank_buffer: int = 0
    min_funds: int = 5
    min_winning_segments: int = 2
    worst_relative_tolerance: float = 0.0
    drawdown_tolerance: float = 0.0
    min_usable_fraction: float = 0.8
    bootstrap_repetitions: int = 999
    bootstrap_block_rows: int = 20
    family_alpha: float = 0.05
    bootstrap_seed: int = 20261014
    seal_date: date = date(2024, 12, 10)
    strict: bool = True

    def __post_init__(self):
        if not self.strict:
            raise ValueError("this runner supports strict historical inputs only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.budget < 1 or self.horizon_rows < 1 or self.stride_rows < 1 or self.warmup_rows < 0:
            raise ValueError("positive budget, horizon and stride required")
        if self.min_funds < 5 or self.top_k < 1 or self.sticky_rank_buffer < 0:
            raise ValueError("strict cross-section requires at least five funds and valid top K")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive and finite")
        if not 1 <= self.min_winning_segments <= 3:
            raise ValueError("min_winning_segments must lie between one and three")
        if self.worst_relative_tolerance < 0 or self.drawdown_tolerance < 0:
            raise ValueError("loss/drawdown tolerances must be nonnegative")
        if not 0 < self.min_usable_fraction <= 1:
            raise ValueError("min_usable_fraction must lie in (0,1]")
        if self.bootstrap_repetitions < 99 or self.bootstrap_block_rows < 1 or not 0 < self.family_alpha < 1:
            raise ValueError("invalid pre-registered family bootstrap controls")


@dataclass(frozen=True)
class CoverageRequest:
    fold_id: str
    fund_codes: tuple[str, ...]
    start_date: date
    end_date: date
    selection_cutoff: pd.Timestamp
    initial_cash: float
    top_k: int
    feature_start_date: date
    discovery_start_date: date
    discovery_end_date: date
    covered_daily_buy_money_exclusive: float | None = None
    research_minimum_remaining_units: float = 0.0
    research_minimum_redemption_units: float = 0.0
    dividend_mode: str = "cash"
    required_components: tuple[str, ...] = ("nav", "fees", "status", "dividends")


def default_outer_folds(seal_date=date(2024, 12, 10)) -> tuple[OuterFold, ...]:
    final_end = min(date(2024, 12, 6), _date(seal_date) - timedelta(days=1))
    return tuple(OuterFold(str(year), date(year - 4, 1, 1), date(year - 3, 12, 31),
                           date(year - 1, 12, 31), date(year, 1, 1),
                           date(year, 12, 31) if year < 2024 else final_end)
                 for year in (2022, 2023, 2024))


def _clean(value):
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    return value


def _persist(path, payload):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2,
                                   allow_nan=False, default=str) + "\n")


def aggregate_segment_gates(records: Sequence[dict], config: WalkForwardConfig) -> dict:
    """Pre-registered descriptive gates against the fixed low-volatility factor."""
    if len(records) != 3 or any(r.get("status") != "COMPLETE" for r in records):
        return {"status": "BLOCKED_INCOMPLETE_SEGMENTS", "passed": False,
                "winning_segments": None, "compounded_relative_segment_statistic": None,
                "worst_relative_gain": None, "drawdown_not_worse": None}
    gains = [float(r["relative_gain_vs_fixed_low_vol"]) for r in records]
    wins = sum(g > 1e-12 for g in gains)
    compounded = float(np.prod([1 + g for g in gains]) - 1)
    dd_ok = all(r["selected"]["max_drawdown"] >= r["fixed_low_vol"]["max_drawdown"]
                - config.drawdown_tolerance - 1e-12 for r in records)
    checks = {"stable_wins": wins >= config.min_winning_segments,
              "positive_compounded_relative_statistic": compounded > 1e-12,
              "worst_relative_fold_within_tolerance": min(gains) >= -config.worst_relative_tolerance - 1e-12,
              "drawdown_not_worse": dd_ok}
    return {"status": "COMPLETE", "passed": all(checks.values()), "checks": checks,
            "winning_segments": wins, "compounded_relative_segment_statistic": compounded,
            "worst_relative_gain": min(gains), "drawdown_not_worse": dd_ok,
            "aggregation": "product of independently cash-initialized segment wealth ratios; not continuous live NAV"}


def joint_family_test(segments: Sequence[pd.DataFrame], config: WalkForwardConfig) -> dict:
    """Joint one-sided max-t block bootstrap over pre-registered paired arms.

    Resampling indices are identical for all arms and remain within each outer
    segment. Seeds are correlated search arms, not additional market histories.
    This is exploratory family reporting, not a proof of alpha or stationarity.
    """
    if len(segments) != 3 or any(s.empty for s in segments):
        return {"status": "BLOCKED_INCOMPLETE_SEGMENTS", "adjusted_pvalues": None}
    names = list(segments[0].columns)
    if not names or any(list(s.columns) != names for s in segments):
        return {"status": "BLOCKED_UNPAIRED_ARMS", "adjusted_pvalues": None}
    arrays = [s.to_numpy(float) for s in segments]
    if any(not np.isfinite(a).all() or len(a) < 2 * config.bootstrap_block_rows for a in arrays):
        return {"status": "INSUFFICIENT_PAIRED_OBSERVATIONS", "adjusted_pvalues": None}
    full = np.concatenate(arrays)
    lag = config.bootstrap_block_rows - 1
    observed = np.array([newey_west_tstat(full[:, j], lag) for j in range(len(names))])
    valid = np.isfinite(observed)
    if not valid.any():
        return {"status": "DEGENERATE_PAIRED_RETURNS", "adjusted_pvalues": {n: None for n in names}}
    means = full.mean(axis=0)
    centered = [a - means for a in arrays]
    rng = np.random.default_rng(config.bootstrap_seed)
    maxima = []
    for _ in range(config.bootstrap_repetitions):
        draw = []
        for a in centered:
            starts = rng.integers(len(a), size=int(np.ceil(len(a) / config.bootstrap_block_rows)))
            idx = ((starts[:, None] + np.arange(config.bootstrap_block_rows)) % len(a)).reshape(-1)[:len(a)]
            draw.append(a[idx])
        sample = np.concatenate(draw)
        stats = np.array([newey_west_tstat(sample[:, j], lag) for j in np.flatnonzero(valid)])
        maxima.append(float(np.max(stats[np.isfinite(stats)])) if np.isfinite(stats).any() else np.inf)
    pvalues = {n: (float((1 + sum(t >= observed[j] for t in maxima)) /
                        (config.bootstrap_repetitions + 1)) if valid[j] else None)
               for j, n in enumerate(names)}
    return {"status": "COMPLETE", "method": "joint one-sided max-t circular block bootstrap within outer segments",
            "adjusted_pvalues": pvalues,
            "reject_positive_null": {n: p is not None and observed[j] > 0 and p <= config.family_alpha
                                      for j, (n, p) in enumerate(pvalues.items())},
            "observed_hac_t": {n: float(observed[j]) if valid[j] else None for j, n in enumerate(names)},
            "block_rows": config.bootstrap_block_rows, "repetitions": config.bootstrap_repetitions,
            "alpha": config.family_alpha, "n_outer_segments": 3,
            "limitations": ["only three chronological market segments", "stationarity and block-length sensitivity remain",
                            "search trials recorded separately; seeds do not multiply independent market evidence"]}


def _local_publication(value):
    if pd.isna(value):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert("Asia/Shanghai").tz_localize(None) if stamp.tzinfo else stamp
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def audit_nav_quality(data: ExecutionData, request: CoverageRequest, *, seal_date: date) -> dict:
    """Audit declared execution quotes before any search or test-return calculation.

    All funds and supplied valuation sessions need positive finite unit/adjusted
    NAV and a publication timestamp no earlier than that NAV's closing valuation.
    The timestamp is the supplied conservative source marker, not evidence of
    first public dissemination. Warmup feature usability remains a separate gate.
    """
    sessions = tuple(sorted({_date(d) for d in data.trading_dates
                             if request.start_date <= _date(d) <= request.end_date}))
    codes = request.fund_codes
    missing = []
    normalized = {}
    for name in ("unit_nav", "adjusted_nav", "nav_available_at"):
        frame = getattr(data, name, None)
        if frame is None:
            missing.append({"component": "nav", "field": name, "reason": "required quote matrix absent"})
            continue
        idx = [_date(d) for d in frame.index]
        if len(set(idx)) != len(idx) or frame.columns.has_duplicates:
            missing.append({"component": "nav", "field": name, "reason": "duplicate valuation rows or fund columns"})
            continue
        # Reindex to the declared unsealed window before inspecting values.
        normalized[name] = frame.set_axis(idx).rename(columns=str).reindex(index=sessions, columns=codes)
    if len(normalized) != 3 or not sessions:
        return {"complete": False, "missing": missing or [{"component": "nav", "reason": "no execution sessions"}],
                "start_date": str(request.start_date), "end_date": str(request.end_date)}
    closing = pd.Series(pd.to_datetime(sessions) + pd.Timedelta(hours=15), index=sessions)
    for code in codes:
        unit = pd.to_numeric(normalized["unit_nav"][code], errors="coerce")
        adj = pd.to_numeric(normalized["adjusted_nav"][code], errors="coerce")
        pub = pd.to_datetime(normalized["nav_available_at"][code].map(_local_publication), errors="coerce")
        invalid = {
            "missing_nonpositive_or_nonfinite_unit_nav": ~(np.isfinite(unit) & (unit > 0)),
            "missing_nonpositive_or_nonfinite_adjusted_nav": ~(np.isfinite(adj) & (adj > 0)),
            "missing_publication_marker": pub.isna(),
            "publication_before_valuation_close": pub.notna() & (pub < closing),
            "quote_disclosed_in_sealed_region": pub.notna() & (pub >= pd.Timestamp(seal_date)),
        }
        for reason, mask in invalid.items():
            for year in sorted({d.year for d in sessions}):
                bad = [d for d in sessions if d.year == year and bool(mask.loc[d])]
                if bad:
                    missing.append({"fund_code": code, "year": year, "component": "nav",
                                    "reason": reason, "invalid_sessions": len(bad),
                                    "first_invalid": str(bad[0]), "last_invalid": str(bad[-1])})
    return {"complete": not missing, "missing": missing, "n_funds": len(codes),
            "n_execution_sessions": len(sessions), "start_date": str(request.start_date),
            "end_date": str(request.end_date),
            "publication_scope": "provided conservative source marker, not observed first-publication time",
            "warmup_scope": "source provenance requested separately; causal feature usability still checked"}


def _history_coverage(data, fold, cfg, fees, statuses, dividends, validator, execution_config=None):
    codes = tuple(sorted(map(str, data.unit_nav.columns)))
    observed = tuple(sorted({_date(d) for d in data.trading_dates}))
    discovery = [d for d in observed if fold.train_start <= d <= fold.discovery_end][::cfg.stride_rows]
    if not discovery:
        raise ValueError("pre-registered discovery interval has no observed sessions")
    inner = build_inner_folds(observed, discovery_end=discovery[-1], outer_train_end=fold.train_end,
                             horizon_rows=cfg.horizon_rows, stride_rows=cfg.stride_rows)
    execution_start = inner[0].decision_dates[0]
    first_discovery_pos = observed.index(discovery[0])
    feature_start = observed[max(0, first_discovery_pos - cfg.warmup_rows)]
    sessions = [d for d in observed if execution_start <= d <= fold.test_end]
    first_test = next((d for d in observed if fold.test_start <= d <= fold.test_end), None)
    if first_test is None:
        raise ValueError("pre-registered outer test interval has no observed sessions")
    cutoff = pd.Timestamp(first_test) + pd.Timedelta(hours=14, minutes=30) - pd.Timedelta(nanoseconds=1)
    request = CoverageRequest(
        fold.fold_id, codes, execution_start, fold.test_end, cutoff, cfg.initial_cash, cfg.top_k,
        feature_start, discovery[0], discovery[-1],
        covered_daily_buy_money_exclusive=getattr(execution_config, "covered_daily_buy_money_exclusive", None),
        research_minimum_remaining_units=getattr(execution_config, "research_minimum_remaining_units", 0.0),
        research_minimum_redemption_units=getattr(execution_config, "research_minimum_redemption_units", 0.0),
    )
    seal = min(cfg.seal_date, _date(data.metadata.get("sealed_from", cfg.seal_date)))
    nav_quality = audit_nav_quality(data, request, seal_date=seal)
    missing = []
    if not nav_quality["complete"]:
        missing.extend(nav_quality["missing"])
    if len(codes) < cfg.min_funds:
        missing.append({"component": "universe", "reason": "fewer than five verified funds"})
    for name, history, kind in (("fees", fees, FeeHistory), ("status", statuses, StatusHistory),
                               ("dividends", dividends, DividendHistory)):
        if not isinstance(history, kind):
            missing.append({"component": name, "reason": f"explicit {kind.__name__} required"})
            for code in codes:
                for year in range(execution_start.year, fold.test_end.year + 1):
                    missing.append({"fund_code": code, "year": year, "component": name,
                                    "reason": "historical input absent or unverified"})
    if validator is None:
        missing.append({"component": "manifest", "reason": "audited full-coverage validator required"})
        external = None
    else:
        external = validator(request)
        if not isinstance(external, Mapping) or external.get("complete") is not True:
            missing.append({"component": "manifest", "reason": "coverage manifest is incomplete",
                            "details": external})
    # A false external manifest blocks before any numerical test/label evaluation.
    if missing:
        return {"complete": False, "missing": missing, "request": asdict(request), "external": external,
                "nav_quality": nav_quality}
    years = sorted({d.year for d in sessions})
    for code in codes:
        for year in years:
            dates = [d for d in sessions if d.year == year]
            start, end = max(execution_start, date(year, 1, 1)), min(fold.test_end, date(year, 12, 31))
            for name, history in (("status", statuses), ("dividends", dividends)):
                try:
                    history.assert_coverage([code], start, end)
                except ValueError as exc:
                    missing.append({"fund_code": code, "year": year, "component": name, "reason": str(exc)})
            bad_fees = [d for d in dates if fees(code, d) is None]
            if bad_fees:
                missing.append({"fund_code": code, "year": year, "component": "fees",
                                "missing_sessions": len(bad_fees), "first_missing": str(bad_fees[0]),
                                "last_missing": str(bad_fees[-1])})
            bad_status = [d for d in dates if any(statuses(code, d, side) is None for side in ("buy", "sell"))]
            if bad_status:
                missing.append({"fund_code": code, "year": year, "component": "status",
                                "missing_sessions": len(bad_status), "first_missing": str(bad_status[0]),
                                "last_missing": str(bad_status[-1])})
    for event in dividends.events:
        if event.fund_code in codes and execution_start <= event.ex_date <= fold.test_end:
            if event.available_at is None or event.cash_pay_date is None:
                missing.append({"component": "dividends", "fund_code": event.fund_code,
                                "year": event.ex_date.year, "event_id": event.event_id,
                                "reason": "cash mode requires publication and actual payment dates"})
    return {"complete": not missing, "missing": missing, "request": asdict(request), "external": external,
            "nav_quality": nav_quality}


def _clip_training_data(data, end, cutoff):
    unit, adjusted = _truncate_frame(data.unit_nav, end), _truncate_frame(data.adjusted_nav, end)
    availability = _truncate_frame(data.nav_available_at, end)
    unit = unit.where(availability.notna() & (availability < cutoff))
    adjusted = adjusted.where(availability.notna() & (availability < cutoff))
    return replace(data, panel=_training_panel(data.panel, end), unit_nav=unit, adjusted_nav=adjusted,
                   nav_available_at=availability,
                   visible_unit_nav=_truncate_frame(data.visible_unit_nav, end),
                   visible_adjusted_nav=_truncate_frame(data.visible_adjusted_nav, end),
                   visible_nav_dates=_truncate_frame(data.visible_nav_dates, end))


def _metrics(result):
    daily = result.daily
    return {"final_equity": float(result.final_equity), "net_return": float(result.net_return),
            "max_drawdown": float(result.max_drawdown), "cost_money": float(result.cost_money),
            "turnover": float(result.turnover), "traded_gross_money": float(result.total_traded_money),
            "stale_valuation_days": int((daily["stale_holdings"] > 0).sum()),
            "stale_adjustment_days": int((daily["stale_adjustments"] > 0).sum()),
            "terminal_stale_holdings": int(daily["stale_holdings"].iloc[-1]),
            "terminal_stale_adjustments": int(daily["stale_adjustments"].iloc[-1]),
            "first_valuation_date": str(daily.index[0]), "last_valuation_date": str(daily.index[-1]),
            "assumptions": list(result.assumptions)}


def run_walk_forward(
    data: ExecutionData, *, fee_history=None, status_history=None, dividend_history=None,
    coverage_validator: Callable[[CoverageRequest], Mapping] | None = None,
    folds: Sequence[OuterFold] | None = None, config: WalkForwardConfig | None = None,
    execution_config: ExecutionConfig | None = None, execution_runner: Callable | None = None,
    search_factory: Callable | None = None, search_method: str = "gp",
    fee_reward_stage: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """Run strict rolling research, or return coverage blocks without test returns.

    ``execution_runner`` may adapt the engine but must retain its call signature
    and historical-input requirements. The default is ``run_fund_backtest``.
    Fund membership comes solely from the caller's pre-registered source coverage
    bundle, never from the results produced here.
    """
    cfg = config or WalkForwardConfig()
    seal = min(cfg.seal_date, _date(data.metadata.get("sealed_from", cfg.seal_date)))
    folds = tuple(folds) if folds is not None else default_outer_folds(seal)
    if len(folds) != 3 or len({f.fold_id for f in folds}) != 3:
        raise ValueError("exactly three uniquely named outer segments are required")
    if any(f.test_end >= seal for f in folds) or any(a.test_end >= b.test_start for a, b in zip(folds, folds[1:])):
        raise ValueError("outer tests must be ordered, nonoverlapping and strictly before the seal")
    for frame in (data.unit_nav, data.adjusted_nav, *data.panel.factors.values()):
        if len(frame) and max(_date(d) for d in frame.index) >= seal:
            raise ValueError("input bundle contains sealed valuation/feature rows")
    coverage = [_history_coverage(data, f, cfg, fee_history, status_history,
                                  dividend_history, coverage_validator, execution_config) for f in folds]
    report = {"protocol": {"config": asdict(cfg), "sealed_from": str(seal),
                           "outer_folds": [asdict(f) for f in folds],
                           "fixed_baseline": "low_vol_60", "dividend_mode": "observed cash",
                           "search_method": search_method,
                           "fee_reward_stage": fee_reward_stage,
                           "status_required": True, "fee_fallback": False,
                           "universe": list(map(str, data.unit_nav.columns)),
                           "universe_selection": data.metadata,
                           "pool_refit": "independent discovery/inner selection before each outer segment",
                           "seed_interpretation": "correlated search arms, not independent market samples",
                           "aggregate_interpretation": "independently cash-initialized segment statistic; not continuous account NAV"},
              "segments": [], "aggregate_by_seed": {}, "family_test": None,
              "alpha_approved": False}
    report["protocol"]["historical_sources"] = {
        "fees": fee_history.provenance() if isinstance(fee_history, FeeHistory) else None,
        "status": sorted({s.source for s in status_history.coverage} |
                         {e.source for e in status_history.events}) if isinstance(status_history, StatusHistory) else None,
        "dividends": sorted({s.source for s in dividend_history.coverage} |
                            {e.source for e in dividend_history.events}) if isinstance(dividend_history, DividendHistory) else None,
        "authentication": "source completeness is supplied by the audited coverage validator; no fallback inference",
    }
    executor = execution_runner or run_fund_backtest
    paired_frames = []
    by_seed = {seed: [] for seed in cfg.seeds}
    dsl = FormulaDSL()
    for fold, manifest in zip(folds, coverage):
        blocked_status = ("BLOCKED_NAV_QUALITY" if not manifest["nav_quality"]["complete"]
                          else "BLOCKED_COVERAGE")
        segment = {"fold_id": fold.fold_id, "coverage": manifest,
                   "status": blocked_status if not manifest["complete"] else "READY",
                   "searches": [], "test_results": None}
        report["segments"].append(segment)
        if not manifest["complete"]:
            for seed in cfg.seeds:
                by_seed[seed].append({"status": blocked_status})
            continue
        observed = tuple(_date(d) for d in data.trading_dates if _date(d) < seal)
        discovery = [d for d in observed if fold.train_start <= d <= fold.discovery_end][::cfg.stride_rows]
        test_dates = [d for d in observed if fold.test_start <= d <= fold.test_end][::cfg.stride_rows]
        if not discovery or not test_dates:
            raise ValueError("pre-registered fold has no observed discovery/test decisions")
        actual_discovery_end = discovery[-1]
        inner = build_inner_folds(observed, discovery_end=actual_discovery_end,
                                  outer_train_end=fold.train_end, horizon_rows=cfg.horizon_rows,
                                  stride_rows=cfg.stride_rows)
        reward_fold = inner[0] if fee_reward_stage else None
        selection_discovery_end = reward_fold.end_date if reward_fold else actual_discovery_end
        if reward_fold:
            inner = build_inner_folds(observed, discovery_end=selection_discovery_end,
                                      outer_train_end=fold.train_end, horizon_rows=cfg.horizon_rows,
                                      stride_rows=cfg.stride_rows)
        cutoff = pd.Timestamp(test_dates[0]) + pd.Timedelta(hours=14, minutes=30)
        training = _clip_training_data(data, fold.train_end, cutoff)
        labels = executable_labels(training.adjusted_nav, training.nav_available_at, discovery,
                                   horizon=cfg.horizon_rows,
                                   known_before=pd.Timestamp(reward_fold.decision_dates[0] if reward_fold else inner[0].decision_dates[0]) + pd.Timedelta(hours=14, minutes=30))
        split = ValidationSplit(f"{fold.fold_id}:discovery", ValidationScheme.ROLLING,
                                (discovery[0], actual_discovery_end),
                                (inner[0].decision_dates[0], fold.train_end),
                                embargo_days=cfg.horizon_rows,
                                as_of_series=[d for f in inner for d in f.decision_dates])
        manuals = manual_exprs(dsl, set(training.panel.factors), price_only=True)
        if "low_vol_60" not in manuals:
            raise ValueError("the pre-registered low_vol_60 baseline cannot be evaluated")
        base_execution = execution_config or ExecutionConfig(observed, initial_cash=cfg.initial_cash,
                                                              top_k=cfg.top_k, fill_lag_sessions=0)
        base_execution = replace(base_execution, initial_cash=cfg.initial_cash, top_k=cfg.top_k,
                                 sticky_rank_buffer=cfg.sticky_rank_buffer, seal_date=seal,
                                 status_callback=status_history, require_status=True,
                                 dividend_mode="cash", reinvest_dividends=False)
        visible_train = {n: getattr(training, n) for n in
                         ("visible_unit_nav", "visible_adjusted_nav", "visible_nav_dates")}
        outer_metrics, outer_paths, paired_columns = {}, {}, {}

        def evaluate(name, exprs):
            callback = make_frozen_pool_score_callback(exprs, data.panel, test_dates, end_date=fold.test_end)
            local_sessions = tuple(d for d in observed if test_dates[0] <= d <= fold.test_end)
            exec_cfg = replace(base_execution, trading_dates=local_sessions, end_date=local_sessions[-1])
            visible = {n: _truncate_frame(getattr(data, n), fold.test_end) for n in
                       ("visible_unit_nav", "visible_adjusted_nav", "visible_nav_dates")}
            result = executor(_truncate_frame(data.unit_nav, fold.test_end),
                              _truncate_frame(data.adjusted_nav, fold.test_end), test_dates, callback,
                              config=exec_cfg, fees_resolver=fee_history, dividend_events=dividend_history,
                              **visible)
            actual_sessions = tuple(_date(d) for d in result.daily.index)
            if actual_sessions != local_sessions:
                raise ValueError(f"strict NAV quality: incomplete or misaligned daily valuation path: {name}")
            equity_values = result.daily["equity"].to_numpy(float)
            if not np.isfinite(equity_values).all() or (equity_values <= 0).any():
                raise ValueError(f"strict NAV quality: nonfinite/nonpositive account equity: {name}")
            if ((result.daily["stale_holdings"] > 0).any()
                    or (result.daily["stale_adjustments"] > 0).any()):
                raise ValueError(f"strict NAV quality: stale held-fund valuation path: {name}")
            usable = len(set(callback.usable_dates)) / len(test_dates)
            if usable < cfg.min_usable_fraction or result.total_traded_money <= 0:
                raise ValueError(f"strict outer execution lacks usable signals/trades: {name}")
            outer_metrics[name] = {**_metrics(result), "usable_signal_fraction": usable}
            outer_paths[name] = result.daily.equity
            if output_dir is not None:
                _persist(Path(output_dir) / fold.fold_id / (name + "_execution.json"),
                         {"metrics": outer_metrics[name], "events": [asdict(e) for e in result.events]})

        frozen_selections = {}
        for seed in cfg.seeds:
            search_cfg = SearchConfig(f"walk_forward:{fold.fold_id}:{seed}",
                                      label_horizon_days=cfg.horizon_rows,
                                      allowed_fields=[n for n in training.panel.factors if n != "px"],
                                      max_trials=cfg.budget, population=min(20, cfg.budget),
                                      generations=cfg.budget, max_complexity=10, seed=seed,
                                      diversity_max_corr=0.95)
            ledger = SearchLedger(locked_test_span=(seal, date(2100, 1, 1)))
            if search_method in {"q_net", "uniform_net"}:
                if reward_fold is None:
                    raise ValueError("net reward search requires a separate reward episode")
                from qmt.factor.net_reward import FeeNetReward, FeeNetGuidedSearch
                evaluator = FeeNetReward(training, reward_fold, anchor=manuals["low_vol_60"],
                                         selection_start=inner[0].decision_dates[0],
                                         execution_config=base_execution, fees=fee_history,
                                         dividends=dividend_history,
                                         min_usable_fraction=cfg.min_usable_fraction)
                searcher = FeeNetGuidedSearch(dsl, evaluator=evaluator,
                                              mode="q" if search_method == "q_net" else "uniform")
            else:
                searcher = (search_factory or SymbolicSearch)(dsl)
            exprs = searcher.search(search_cfg, dsl, training.panel, labels, split, ledger)
            selected = select_incremental_pool(
                {f"searched_{i}": e for i, e in enumerate(exprs)}, manuals, training.panel,
                training.unit_nav, training.adjusted_nav, discovery_end=selection_discovery_end,
                outer_train_end=fold.train_end, execution_config=base_execution,
                fees_resolver=fee_history, nav_available_at=training.nav_available_at,
                dividend_events=dividend_history, folds=inner,
                config=IncrementalConfig(horizon_rows=cfg.horizon_rows, stride_rows=cfg.stride_rows,
                                         min_usable_fraction=cfg.min_usable_fraction), **visible_train)
            audit = {"seed": seed, "search_config": asdict(search_cfg), "ledger": asdict(ledger),
                     "search_method": search_method,
                     "search_diagnostics": {
                         "best_discovery_fitness": max((t.select_ic for t in ledger.trials
                                                        if t.select_ic is not None), default=None),
                         "unique_scored_formulas": ledger.dataset_views,
                         "distinct_recorded_asts": len({t.expr_text for t in ledger.trials
                                                        if t.expr_text != "<no_candidate>"}),
                         "syntax_or_construction_rejections": sum(not t.passed_dsl for t in ledger.trials),
                         "unique_usable_formulas": len({t.expr_text for t in ledger.trials
                                                        if t.select_ic is not None}),
                         "returned_candidates": len(exprs)},
                     "operator_policy": searcher.policy_audit() if hasattr(searcher, "policy_audit") else None,
                     "discovery_span": [str(discovery[0]), str(actual_discovery_end)],
                     "reward_fold": reward_fold.to_dict() if reward_fold else None,
                     "selection_discovery_end": str(selection_discovery_end),
                     "label_known_before": str(pd.Timestamp(reward_fold.decision_dates[0] if reward_fold else inner[0].decision_dates[0]) + pd.Timedelta(hours=14, minutes=30)),
                     "training_cutoff": str(cutoff), "train_end": str(fold.train_end),
                     "inner_folds": [f.to_dict() for f in inner], "selection": selected.to_dict(),
                     "outer_outcomes_used_for_search": False}
            segment["searches"].append(audit)
            if output_dir is not None:
                root = Path(output_dir) / fold.fold_id / f"seed_{seed}"
                _persist(root / "search_ledger.json", audit)
                _persist(root / "selection.json", selected.to_dict())
            frozen_selections[seed] = selected
        # Every search/pool is frozen before any outer outcome is evaluated.
        evaluate("fixed_low_vol_60", (manuals["low_vol_60"],))
        for seed in cfg.seeds:
            selected = frozen_selections[seed]
            own_name, chosen_name = f"selected_{seed}", f"chosen_manual_{seed}"
            evaluate(own_name, selected.selected_exprs)
            evaluate(chosen_name, (manuals[selected.baseline_name],))
            own, fixed, chosen = outer_metrics[own_name], outer_metrics["fixed_low_vol_60"], outer_metrics[chosen_name]
            row = {"status": "COMPLETE", "fold_id": fold.fold_id, "seed": seed,
                   "selected": own, "fixed_low_vol": fixed, "chosen_manual": chosen,
                   "chosen_manual_name": selected.baseline_name,
                   "relative_gain_vs_fixed_low_vol": own["final_equity"] / fixed["final_equity"] - 1,
                   "relative_gain_vs_chosen_manual": own["final_equity"] / chosen["final_equity"] - 1}
            by_seed[seed].append(row)
            for baseline_name, path in (("fixed_low_vol", outer_paths["fixed_low_vol_60"]),
                                        ("chosen_manual", outer_paths[chosen_name])):
                values = paired_active_log_returns(outer_paths[own_name], path, cfg.initial_cash)
                paired_columns[f"seed_{seed}_vs_{baseline_name}"] = values
        segment.update(status="COMPLETE", test_results=outer_metrics)
        paired = pd.DataFrame(paired_columns, index=outer_paths["fixed_low_vol_60"].index)
        paired_frames.append(paired)
        segment["paired_daily_active_log_returns"] = {
            "dates": [str(d) for d in paired.index],
            "columns": {c: paired[c].tolist() for c in paired.columns}}
    report["aggregate_by_seed"] = {str(seed): aggregate_segment_gates(rows, cfg) for seed, rows in by_seed.items()}
    report["paired_segment_records"] = by_seed
    complete = len(paired_frames) == 3
    report["family_test"] = joint_family_test(paired_frames, cfg) if complete else {
        "status": "BLOCKED_INCOMPLETE_COVERAGE", "adjusted_pvalues": None}
    report["status"] = ("COMPLETE_STRICT_RESEARCH" if complete else
                        "BLOCKED_NAV_QUALITY" if any(s["status"] == "BLOCKED_NAV_QUALITY"
                                                     for s in report["segments"]) else "BLOCKED_COVERAGE")
    report["verdict"] = ("Strict historical research report; no automatic alpha approval."
                         if complete else "Missing verified histories/coverage; test returns were not computed for blocked segments.")
    if output_dir is not None:
        _persist(Path(output_dir) / "walk_forward_report.json", report)
    return _clean(report)
