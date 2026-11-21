"""Execution-aware fund research: discovery -> inner pool selection -> outer test.

All cached values are physically filtered before the pinned seal. NAV publication
and valuation dates are distinct: features use published snapshots at 14:30,
while fills and outcome marks use unit NAV. Fee/settlement assumptions are printed
and sensitivity tested; absent historical prospectus/status data are not inferred
to be verified execution conditions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date
import json
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qmt.factor.auto_search import SymbolicSearch
from qmt.factor.base_factors import BaseFactors
from qmt.factor.baselines import manual_exprs
from qmt.factor.dsl import FormulaDSL
from qmt.factor.types import BaseFactorPanel, Label, SafePanel, SearchConfig, SearchLedger, ValidationScheme, ValidationSplit
from qmt.fees import default_fee_schedule


@dataclass
class ExecutionData:
    panel: BaseFactorPanel
    unit_nav: pd.DataFrame
    adjusted_nav: pd.DataFrame
    visible_unit_nav: pd.DataFrame
    visible_adjusted_nav: pd.DataFrame
    visible_nav_dates: pd.DataFrame
    nav_available_at: pd.DataFrame
    trading_dates: tuple[date, ...]
    share_classes: dict[str, str]
    metadata: dict


def _visible_panel(raw, sessions, column, *, max_stale_sessions=5):
    """Latest known valuation, never latest publication of an older valuation."""
    decision_times = pd.to_datetime(sessions) + pd.Timedelta(hours=14, minutes=30)
    closing_times = pd.to_datetime(raw.nav_date) + pd.Timedelta(hours=15)
    availability = pd.to_datetime(raw.available_at)
    effective = pd.concat([closing_times, availability], axis=1).max(axis=1)
    positions = np.searchsorted(decision_times.to_numpy(), effective.to_numpy(), side="left")
    known = raw.copy()
    known["visible_position"] = positions
    known = known[known.visible_position < len(sessions)]
    output = pd.DataFrame(index=sessions, columns=sorted(raw.fund_code.unique()), dtype=float)
    source_dates = pd.DataFrame(index=sessions, columns=output.columns, dtype=object)
    # An older NAV disclosed late must not roll a fund's signal back in time.
    for code, group in known.groupby("fund_code", sort=True):
        latest_date = pd.Timestamp.min
        for pos, updates in group.sort_values(["visible_position", "nav_date", "available_at"]).groupby("visible_position"):
            record = updates.iloc[-1]
            if pd.Timestamp(record.nav_date) < latest_date:
                continue
            latest_date = pd.Timestamp(record.nav_date)
            d = sessions[int(pos)]
            output.at[d, code] = float(record[column])
            source_dates.at[d, code] = pd.Timestamp(record.nav_date).date()
    return output.ffill(limit=max_stale_sessions), source_dates.ffill(limit=max_stale_sessions)


def domestic_fund_mask(names):
    """Exclude products requiring FX/foreign calendars absent from this engine."""
    return ~names.fillna("").str.contains(
        r"QDII|USD|HKD|现汇|现钞|美元|港币|境外|海外|恒生|纳斯达克|美国|全球", case=False, regex=True)


def conservative_date_disclosures(values):
    """Treat midnight cache markers derived from announcement dates as day-only.

    The cache does not establish the publication hour for these observations.
    End-of-day availability avoids presuming that they were public by 14:30.
    Actual intraday timestamps retain their supplied time.
    """
    stamps = pd.to_datetime(values)
    day_only = stamps.notna() & stamps.eq(stamps.dt.normalize())
    return stamps.where(~day_only, stamps + pd.Timedelta(hours=23, minutes=59, seconds=59))


def load_execution_data(cache: Path, *, max_funds=80, start=date(2016, 1, 1),
                        seal=date(2024, 12, 10), codes=None, domestic_only=True,
                        conservative_nav_publication=False):
    boundary = pd.Timestamp(seal).to_pydatetime()
    basic = pq.read_table(cache / "fund_basic.parquet", filters=[("available_at", "<", boundary)],
                          columns=["fund_code", "name", "fund_type", "market", "found_date"]).to_pandas()
    universe = basic[(basic.fund_type == "股票型") & (basic.market == "O")
                     & (basic.found_date < pd.Timestamp(start))]
    excluded = universe[~domestic_fund_mask(universe.name)]
    if domestic_only:
        universe = universe[domestic_fund_mask(universe.name)]
    if codes is None:
        codes = sorted(universe.fund_code.dropna().unique())[:max_funds]
    else:
        codes = sorted(set(codes))
        if set(codes) - set(universe.fund_code):
            raise ValueError("explicit codes include unavailable/out-of-scope/foreign funds")
    if len(codes) < 5:
        raise ValueError("need at least five pre-existing OTC equity funds")
    raw = pq.read_table(cache / "fund_nav.parquet",
                        filters=[("available_at", "<", boundary), ("nav_date", "<", boundary),
                                 ("nav_date", ">=", pd.Timestamp(start).to_pydatetime()),
                                 ("fund_code", "in", codes)],
                        columns=["fund_code", "nav_date", "unit_nav", "adj_nav", "available_at"]).to_pandas()
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["unit_nav", "adj_nav", "available_at"])
    raw = raw[(raw.unit_nav > 0) & (raw.adj_nav > 0)]
    if conservative_nav_publication:
        raw["available_at"] = conservative_date_disclosures(raw.available_at)
    calendar = pq.read_table(cache / "index_daily.parquet",
                             filters=[("ts_code", "=", "000300.SH"), ("trade_date", "<", boundary),
                                      ("trade_date", ">=", pd.Timestamp(start).to_pydatetime())],
                             columns=["trade_date"]).to_pandas()
    sessions = tuple(sorted(set(pd.to_datetime(calendar.trade_date).dt.date)))
    if not sessions:
        raise ValueError("no pre-seal mainland trading calendar in index cache")
    # Observations off the declared trading calendar cannot be simulated fills.
    raw = raw[pd.to_datetime(raw.nav_date).dt.date.isin(sessions)].copy()
    # Fingerprint only the already filtered pre-seal subset, never entire cache
    # files that may also hold sealed observations.
    subset = raw.sort_values(["fund_code", "nav_date", "available_at"])
    nav_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
        subset[["fund_code", "nav_date", "unit_nav", "adj_nav", "available_at"]],
        index=False).to_numpy().tobytes()).hexdigest()
    raw["valuation"] = pd.to_datetime(raw.nav_date).dt.date
    unit = raw.pivot_table(index="valuation", columns="fund_code", values="unit_nav", aggfunc="last").reindex(sessions)
    adjusted = raw.pivot_table(index="valuation", columns="fund_code", values="adj_nav", aggfunc="last").reindex(sessions)
    available = raw.pivot_table(index="valuation", columns="fund_code", values="available_at", aggfunc="last").reindex(sessions)
    visible_unit, sources = _visible_panel(raw, sessions, "unit_nav")
    visible_adjusted, _ = _visible_panel(raw, sessions, "adj_nav")
    panel = BaseFactors().compute_base_factors(SafePanel(sessions[-1], visible_adjusted), sessions[-1])
    classes = {}
    for _, row in universe[universe.fund_code.isin(codes)].iterrows():
        name = str(row["name"])
        classes[row.fund_code] = "C" if re.search(r"C(?:类)?$", name) else "A_or_unknown"
    return ExecutionData(panel, unit, adjusted, visible_unit, visible_adjusted, sources, available, sessions, classes,
                         {"nav_rows": len(raw), "n_funds": len(unit.columns), "sealed_from": str(seal),
                          "pre_seal_nav_subset_sha256": nav_fingerprint,
                          "sealed_values_read": False, "calendar_source": "000300.SH valuation dates",
                          "decision_cutoff": "14:30 Asia/Shanghai; available_at and prior closing time",
                          "nav_publication_precision": ("date-only midnight cache markers shifted to end of announcement day"
                                                        if conservative_nav_publication else "cached announcement-date midnight assumption"),
                          "staleness_limit_sessions": 5,
                          "share_class_C_inferred": sum(v == "C" for v in classes.values()),
                          "domestic_only": domestic_only,
                          "currency_calendar_scope": "CNY domestic OTC; foreign/QDII excluded" if domestic_only else "legacy mixed scope",
                          "scope_excluded_codes": sorted(excluded.fund_code.tolist()),
                          "limitations": ["metadata revisions and survivorship completeness unverified",
                                          "share classes inferred from cached names, not historical prospectuses",
                                          "dividends reconstructed from adj/unit ratios, not official event records",
                                          "actual historical subscription/redemption status unavailable",
                                          "historical fund/channel fee schedules unavailable; scenario rates used"]})


def executable_labels(adjusted, available_at, decisions, *, horizon, known_before):
    """Full-horizon valuation return labels published before the next research stage.

    Missing endpoint quotes do not settle at an arbitrary stale NAV. Adjusted
    NAV is a total-return target; the execution engine uses unit NAV and lots.
    """
    quotes = adjusted.where(available_at.notna() & (available_at <= pd.Timestamp(known_before)))
    labs = []
    dates = list(quotes.index)
    positions = {d: i for i, d in enumerate(dates)}
    for d in decisions:
        i = positions[d]
        j = i + horizon
        if j >= len(dates):
            continue
        entry, exit = quotes.iloc[i], quotes.iloc[j]
        returns = (exit / entry - 1).replace([np.inf, -np.inf], np.nan).dropna()
        returns -= returns.mean()
        window = quotes.iloc[i:j + 1]
        dd = (window / window.cummax() - 1).min()
        for code, value in returns.items():
            if pd.notna(dd[code]):
                labs.append(Label(str(code), d, horizon, float(value), float(min(0, dd[code]))))
    return labs


def _fees(data, scenario):
    purchase = {"opt": 0.0012, "neu": 0.0015, "con": 0.015}[scenario]
    def resolver(code, as_of):
        sched = default_fee_schedule(code)
        rate = 0.0 if data.share_classes.get(code) == "C" else purchase
        return sched.model_copy(update={"purchase_fee_rate": rate})
    return resolver


def _metrics(result, initial_cash):
    equity = result.daily.equity.to_numpy(float)
    path = np.r_[initial_cash, equity]
    days = max(1, (result.daily.index[-1] - result.daily.index[0]).days)
    wealth = equity[-1] / initial_cash
    stale_days = int((result.daily.stale_holdings > 0).sum())
    stale_fraction = stale_days / len(result.daily)
    return {"final_equity": float(equity[-1]), "net_return": float(wealth - 1),
            "annualized_return": float(wealth ** (365.25 / days) - 1),
            "max_drawdown": float(np.min(path / np.maximum.accumulate(path) - 1)),
            "cost_money": float(result.cost_money), "turnover": float(result.turnover),
            "traded_gross_money": float(result.total_traded_money),
            "stale_valuation_days": stale_days, "stale_valuation_fraction": stale_fraction,
            "valuation_quality": ("stale valuation affects comparison" if stale_fraction > .05
                                  else "some stale marks" if stale_days else "all held marks observed"),
            "blocked_orders": int(result.daily.blocked_orders.sum()),
            "terminal_estimated_redemption_fees": int(result.daily.pending_estimated_redemption_fees.iloc[-1]),
            "assumptions": list(result.assumptions),
            "n_valuation_days": result.daily.shape[0], "n_events": len(result.events)}


def paired_active_log_returns(own: pd.Series, baseline: pd.Series, initial_cash: float):
    shared = pd.concat([own.rename("own"), baseline.rename("base")], axis=1).dropna()
    if shared.empty or (shared <= 0).any().any() or initial_cash <= 0:
        raise ValueError("paired equity paths must be nonempty and positive")
    logs = np.log(shared).diff()
    logs.iloc[0] = np.log(shared.iloc[0] / initial_cash)
    return (logs.own - logs.base).dropna().to_numpy(float)


def json_ready(value):
    """Undefined statistics are null, never nonstandard JSON NaN/Infinity."""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def run_cost_benchmark(cache: Path, *, max_funds=80, budget=120, seeds=(7, 19, 43), output_dir=None, fee_history=None):
    from qmt.factor.fund_execution import ExecutionConfig, run_fund_backtest
    from qmt.factor.incremental import IncrementalConfig, build_inner_folds, select_incremental_pool, make_frozen_pool_score_callback
    cfg_path = cache / "shadow" / "config.json"
    persisted = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    seal = min(date.fromisoformat(persisted.get("locked_start", "2024-12-10")), date(2024, 12, 10))
    data = load_execution_data(cache, max_funds=max_funds, seal=seal)
    if fee_history is not None:
        from qmt.factor.fee_history import FeeHistory
        actual_fees = FeeHistory.load(Path(fee_history))
    else:
        actual_fees = None
    sessions = data.trading_dates
    discovery_end, outer_train_end = date(2019, 12, 31), date(2021, 12, 31)
    horizon, stride = 40, 20
    inner_cfg = IncrementalConfig(horizon_rows=horizon, stride_rows=stride)
    folds = build_inner_folds(sessions, discovery_end=discovery_end, outer_train_end=outer_train_end,
                              horizon_rows=horizon, stride_rows=stride)
    first = next(i for i, d in enumerate(sessions) if d >= date(2018, 1, 1))
    discovery_dates = [d for d in sessions[first::stride] if d <= discovery_end]
    discovery_end = discovery_dates[-1]
    folds = build_inner_folds(sessions, discovery_end=discovery_end, outer_train_end=outer_train_end,
                              horizon_rows=horizon, stride_rows=stride)
    search_split = ValidationSplit("discovery", ValidationScheme.ROLLING,
                                    (discovery_dates[0], discovery_end),
                                    (folds[0].decision_dates[0], outer_train_end),
                                    embargo_days=horizon,
                                    as_of_series=[d for f in folds for d in f.decision_dates])
    # Labels use executable valuation NAV, while features use published NAV.
    labels = executable_labels(data.adjusted_nav, data.nav_available_at, discovery_dates,
                               horizon=horizon,
                               known_before=pd.Timestamp(folds[0].decision_dates[0]) + pd.Timedelta(hours=14, minutes=30))
    dsl = FormulaDSL()
    manuals = manual_exprs(dsl, set(data.panel.factors), price_only=True)
    selections = []
    searches = []
    execution = ExecutionConfig(trading_dates=sessions, initial_cash=100000, top_k=5,
                                fill_lag_sessions=0, purchase_confirm_sessions=1,
                                redemption_confirm_sessions=1, redemption_settlement_sessions=3,
                                seal_date=seal)
    visible = {n: getattr(data, n) for n in ("visible_unit_nav", "visible_adjusted_nav", "visible_nav_dates")}
    for seed in seeds:
        search_cfg = SearchConfig(f"fee-net-discovery:{seed}", label_horizon_days=horizon,
                                   allowed_fields=[n for n in data.panel.factors if n != "px"],
                                   max_trials=budget, population=20, generations=budget,
                                   max_complexity=10, seed=seed, diversity_max_corr=0.95)
        ledger = SearchLedger(locked_test_span=(seal, date(2100, 1, 1)))
        exprs = SymbolicSearch(dsl).search(search_cfg, dsl, data.panel, labels, search_split, ledger)
        candidates = {f"searched_{i}": e for i, e in enumerate(exprs)}
        selected = select_incremental_pool(candidates, manuals, data.panel, data.unit_nav, data.adjusted_nav,
                                            discovery_end=discovery_end, outer_train_end=outer_train_end,
                                            execution_config=execution, fees_resolver=(actual_fees or _fees(data, "neu")),
                                            nav_available_at=data.nav_available_at,
                                            folds=folds, config=inner_cfg, **visible)
        selections.append((seed, selected, exprs))
        searches.append({"seed": seed, "recorded_trials": ledger.n_trials,
                         "dataset_views": ledger.dataset_views,
                         "trial_ledger": [asdict(t) for t in ledger.trials],
                         "candidate_formulas": [e.text for e in exprs], "selection": selected.to_dict()})
        print(f"seed={seed}, inner baseline={selected.baseline_name}, selected={selected.selected_names}", flush=True)
    eval_first = next(i for i, d in enumerate(sessions) if d >= date(2022, 4, 1))
    # Five-session tail accommodates all pre-registered settlement scenarios.
    eval_dates = sessions[eval_first:len(sessions) - 5:stride]
    if not eval_dates:
        raise ValueError("no outer validation decisions before the seal")
    rows = []
    paths = {}
    neutral_results = {}
    def evaluate(name, scenario, exprs=None, weight_callback=None):
        scenario_cfg = replace(execution, fill_lag_sessions=1 if scenario == "con" else 0,
                                redemption_settlement_sessions={"opt": 2, "neu": 3, "con": 5}[scenario])
        callback = (make_frozen_pool_score_callback(exprs, data.panel, eval_dates, end_date=sessions[-1])
                    if exprs is not None else None)
        result = run_fund_backtest(data.unit_nav, data.adjusted_nav, eval_dates,
                                   callback, weight_callback=weight_callback,
                                   config=scenario_cfg, fees_resolver=(actual_fees or _fees(data, scenario)),
                                   end_date=sessions[-1], **visible)
        row = {"method": name, "scenario": scenario, **_metrics(result, execution.initial_cash)}
        if callback is not None:
            row["usable_signal_decisions"] = len(set(callback.usable_dates))
        rows.append(row)
        paths[(name, scenario)] = result.daily.equity
        if scenario == "neu":
            neutral_results[name] = result
        if output_dir is not None:
            p = Path(output_dir) / (name + "_" + scenario + ".json")
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"metrics": row,
                       "daily": [{"date": str(d), **v} for d, v in result.daily.to_dict("index").items()],
                       "events": [e.to_dict() if hasattr(e, "to_dict") else vars(e) for e in result.events]}
            p.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")
    # Buy-and-hold equal weights are chosen at the first cutoff, not by future returns.
    def equal_weight(d, held):
        if d != eval_dates[0]:
            return None
        available = data.visible_unit_nav.loc[d].dropna().index
        return pd.Series(1.0 / len(available), index=available) if len(available) else pd.Series(dtype=float)
    for scenario in ("opt", "neu", "con"):
        evaluate("cash", scenario, weight_callback=lambda d, held: pd.Series(dtype=float))
        evaluate("equal_weight_hold", scenario, weight_callback=equal_weight)
        for name, expr in manuals.items():
            evaluate("manual_" + name, scenario, [expr])
        for seed, selected, exprs in selections:
            evaluate(f"selected_{seed}", scenario, selected.selected_exprs)
            evaluate(f"gp_equal_rank_{seed}", scenario, exprs)
    from qmt.stats import assess_significance
    paired = []
    for seed, selected, _ in selections:
        method, baseline = f"selected_{seed}", "manual_" + selected.baseline_name
        for scenario in ("opt", "neu", "con"):
            own, base = paths[(method, scenario)], paths[(baseline, scenario)]
            delta = paired_active_log_returns(own, base, execution.initial_cash)
            sig = assess_significance(delta, overlap=20, min_abs_t=2.0)
            paired.append({"seed": seed, "scenario": scenario, "baseline": baseline,
                           "relative_net_wealth_gain": float(own.iloc[-1] / base.iloc[-1] - 1),
                           "mean_daily_active_log_return": float(np.mean(delta)),
                           "hac_t": float(sig.t_newey_west), "significant_unadjusted": bool(sig.significant),
                           "n_daily_observations": len(delta),
                           "multiple_testing_adjusted": False})
    added = {seed for seed, selected, _ in selections if len(selected.selected_names) > 1}
    added_gains = [x["relative_net_wealth_gain"] for x in paired if x["seed"] in added]
    verdict = ("No candidate accepted; retained frozen manual baseline." if not added else
               "Added factors failed to improve over frozen manual in every tested scenario." if all(g <= 0 for g in added_gains)
               else "Mixed/positive development evidence requires further preregistered evaluation.")
    return {"protocol": {"research_date": "2026-10-14", **data.metadata,
                          "execution_version": "cash_units_fifo_with_funding_intents",
                          "initial_cash": execution.initial_cash, "top_k": execution.top_k,
                          "seed_list": list(seeds), "search_budget": budget,
                          "discovery_decisions": [str(discovery_dates[0]), str(discovery_end)],
                          "outer_train_end": str(outer_train_end),
                          "outer_validation": [str(eval_dates[0]), str(sessions[-1])],
                          "candidate_labels": "40 sessions from executable valuation adjusted NAV",
                          "selection_rule": inner_cfg.__dict__,
                          "terminal_valuation": "marked holdings, not forced redemption",
                          "fee_assumptions": "A/unknown subscription 0.12%/0.15%/1.5%, C 0%; common assumed historical redemption ladder",
                          "provided_fee_history": actual_fees.provenance() if actual_fees else None,
                          "scenario_fee_rule": "provided fee history fixed across scenarios" if actual_fees else "scenario defaults",
                          "fill_sessions": {"opt": 0, "neu": 0, "con": 1},
                          "ratio_tolerance": execution.adjustment_ratio_tolerance,
                          "settlement_sessions": {"opt": 2, "neu": 3, "con": 5},
                          "validation_feedback_used_for_selection": False,
                          "validation_status": "previously used development period, not untouched final test",
                          "raw_nav_pre_filtered_at_seal": True},
            "searches": searches, "results": rows, "paired_vs_frozen_manual": paired,
            "verdict": verdict + " Historical fee/status/dividend completeness prevents certification; no live portfolio change."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--max-funds", type=int, default=80)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cost_benchmark.json"))
    parser.add_argument("--fees-file", type=Path, help="date/channel fee history JSON; missing coverage blocks execution")
    args = parser.parse_args()
    if args.max_funds < 5 or args.budget < 1:
        parser.error("need max-funds >=5 and budget>=1")
    report = run_cost_benchmark(args.cache, max_funds=args.max_funds, budget=args.budget,
                                seeds=tuple(int(s) for s in args.seeds.split(",")),
                                fee_history=args.fees_file,
                                output_dir=args.output.parent / "cost_paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
