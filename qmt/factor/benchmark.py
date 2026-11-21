"""Offline, fixed-budget research benchmark on the unsealed NAV cache.

Run ``python -m qmt.factor.benchmark --output artifacts/search_benchmark.json``.
The configured seal is read before any NAV table; Arrow predicates physically
exclude its dates. This is a research ablation, not a live portfolio or a final
test. The cached universe/metadata are not certified survivorship-complete.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qmt.factor.auto_search import SymbolicSearch
from qmt.factor.base_factors import BaseFactors
from qmt.factor.baselines import manual_exprs
from qmt.factor.diversity import mean_abs_rank_corr, rank_signature
from qmt.factor.dsl import FormulaDSL
from qmt.factor.metrics import ic_summary, rank_turnover, scores_by_date_from_frame
from qmt.factor.panel import build_labels, labels_by_date, usable_price_frame
from qmt.factor.types import SafePanel, SearchConfig, SearchLedger
from qmt.factor.validation import train_valid_split
from qmt.pit import PointInTimeStore


def load_research_panel(cache: Path, *, max_funds: int, start: date, seal: date):
    """Read pre-seal equity-fund NAVs; choose codes before examining returns."""
    boundary = pd.Timestamp(seal).to_pydatetime()
    basic = pq.read_table(
        cache / "fund_basic.parquet",
        filters=[("available_at", "<", boundary)],
        columns=["fund_code", "fund_type", "market", "found_date"],
    ).to_pandas()
    universe = basic[(basic.fund_type == "股票型") & (basic.market == "O")
                     & (basic.found_date < pd.Timestamp(start))]
    codes = sorted(universe.fund_code.dropna().unique())[:max_funds]
    if not codes:
        raise ValueError("no eligible pre-seal fund metadata")
    raw = pq.read_table(
        cache / "fund_nav.parquet",
        filters=[("available_at", "<", boundary), ("nav_date", "<", boundary),
                 ("nav_date", ">=", pd.Timestamp(start).to_pydatetime()),
                 ("fund_code", "in", codes)],
        columns=["fund_code", "nav_date", "adj_nav", "available_at"],
    ).to_pandas()
    store = PointInTimeStore()
    try:
        store.put("benchmark_nav", raw)
        prices = usable_price_frame(store, "benchmark_nav", value_col="adj_nav",
                                    code_col="fund_code", date_col="nav_date")
    finally:
        store.close()
    # Disclosure or the one-row decision lag must also stay outside the seal.
    prices = prices.loc[prices.index < seal]
    if prices.empty:
        raise ValueError("no usable pre-seal NAVs")
    panel = BaseFactors().compute_base_factors(SafePanel(prices.index[-1], prices), prices.index[-1])
    return panel, prices, len(raw)


def evaluate_pool(exprs, dsl, panel, labels, split, *, horizon: int, stride: int, pool_sink=None):
    """Frozen equal-rank pool; validation is used for reporting only."""
    dates = list(split.as_of_series)
    scores = {e.text: scores_by_date_from_frame(dsl.evaluate_frame(e, panel), dates) for e in exprs}
    pool = {}
    for d in dates:
        if scores and all(d in s for s in scores.values()):
            sections = [s[d].rank(pct=True) - 0.5 for s in scores.values()]
            # Complete-case pooling avoids changing weights with missing signals.
            joined = pd.concat(sections, axis=1).dropna()
            if len(joined) >= 5:
                pool[d] = joined.mean(axis=1)
    if pool_sink is not None:
        pool_sink.update(pool)
    ic = ic_summary(pool, labels_by_date(labels), horizon_days=horizon, stride_days=stride)
    signatures = [rank_signature(s) for s in scores.values()]
    corrs = [mean_abs_rank_corr(a, b) for a, b in itertools.combinations(signatures, 2)]
    valid_corrs = [c for c in corrs if c is not None]
    return {
        "pool_valid_rank_ic": ic.mean_rank_ic,
        "pool_valid_hac_t": ic.tstat,
        "pool_valid_n_dates": ic.n_dates,
        "pool_coverage_dates": len(pool),
        "validation_decisions": len(dates),
        "pool_rank_turnover": rank_turnover(pool),
        "valid_mean_abs_pair_corr": float(np.mean(valid_corrs)) if valid_corrs else None,
        "valid_max_abs_pair_corr": max(valid_corrs) if valid_corrs else None,
    }


def run_benchmark(cache: Path, *, max_funds=80, budget=120, seeds=(7, 19, 43),
                  start=date(2016, 1, 1), horizon=40, stride=20) -> dict:
    config_path = cache / "shadow" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    # Never loosen a pinned seal; use the earlier project anchor as fallback.
    seal = min(date.fromisoformat(config.get("locked_start", "2024-12-10")), date(2024, 12, 10))
    panel, prices, n_rows = load_research_panel(cache, max_funds=max_funds, start=start, seal=seal)
    # Fixed absolute warmup + observed stride: appending later data cannot move the grid.
    decision_start = date(start.year + 2, 1, 1)
    observations = list(prices.index)
    first = next(i for i, d in enumerate(observations) if d >= decision_start)
    decisions = observations[first:len(observations) - horizon:stride]
    split = train_valid_split(decisions, horizon_days=horizon, train_frac=0.6,
                              embargo_days=60, observation_dates=observations)
    labels = build_labels(prices, decisions, horizon_days=horizon)
    dsl = FormulaDSL()
    base = SearchConfig(
        "benchmark", label_horizon_days=horizon,
        allowed_fields=[f for f in panel.factors if f != "px"],
        allowed_windows=(5, 20, 60), max_complexity=10, max_depth=4,
        max_trials=budget, population=20, generations=budget, seed_from_manual=True,
    )
    rows = []
    pools = {}
    for seed in seeds:
        for method in ("budgeted_gp", "diverse_gp", "random"):
            cfg = replace(base, seed=seed, config_id=f"{method}:{seed}")
            if method == "diverse_gp":
                cfg = replace(cfg, diversity_max_corr=0.95)
            elif method == "random":
                cfg = replace(cfg, population=budget, generations=0, seed_from_manual=False)
            ledger = SearchLedger(locked_test_span=(seal, date(2100, 1, 1)))
            t0 = time.perf_counter()
            searcher = SymbolicSearch(dsl)
            exprs = searcher.search(cfg, dsl, panel, labels, split, ledger)
            # Search does not retain signatures in the no-diversity ablation.
            train_panel = searcher._training_panel(panel, split.train_span[1])
            train_dates = sorted(d for d in decisions if split.train_span[0] <= d <= split.train_span[1])
            signatures = {e.text: searcher._train_signatures.get(e.text) or rank_signature(
                scores_by_date_from_frame(dsl.evaluate_frame(e, train_panel), train_dates)) for e in exprs}
            train_correlations = [
                mean_abs_rank_corr(signatures[a.text], signatures[b.text])
                for a, b in itertools.combinations(exprs, 2)
            ]
            train_correlations = [c for c in train_correlations if c is not None]
            pools[(seed, method)] = {}
            row = {"method": method, "seed": seed, "recorded_trials": ledger.n_trials,
                   "dataset_views": ledger.dataset_views, "n_candidates": len(exprs),
                   "search_seconds": time.perf_counter() - t0,
                   "train_max_abs_pair_corr": max(train_correlations) if train_correlations else None,
                   "formulas": [e.text for e in exprs],
                   **evaluate_pool(exprs, dsl, panel, labels, split, horizon=horizon, stride=stride,
                                   pool_sink=pools[(seed, method)])}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "formulas"}, ensure_ascii=False), flush=True)
    # Paired reporting only: same supported decisions, same fund names and same
    # labels for all three methods. No feedback from these metrics enters search.
    labmap = labels_by_date(labels)
    methods = ("budgeted_gp", "diverse_gp", "random")
    for seed in seeds:
        common_dates = set.intersection(*(set(pools[(seed, m)]) for m in methods))
        paired = {m: {} for m in methods}
        paired_labels = {}
        for d in sorted(common_dates):
            names = sorted(set.intersection(*(set(pools[(seed, m)][d].index) for m in methods),
                                            set(labmap.get(d, pd.Series(dtype=float)).index)))
            if len(names) < 5:
                continue
            y = labmap[d].reindex(names)
            sections = {m: pools[(seed, m)][d].reindex(names) for m in methods}
            if y.std() == 0 or any(s.std() == 0 for s in sections.values()):
                continue
            paired_labels[d] = y
            for m, s in sections.items():
                paired[m][d] = s
        for row in [r for r in rows if r["seed"] == seed]:
            ic = ic_summary(paired[row["method"]], paired_labels, horizon_days=horizon, stride_days=stride)
            row.update(paired_valid_rank_ic=ic.mean_rank_ic, paired_valid_hac_t=ic.tstat,
                       paired_valid_n_dates=ic.n_dates, paired_rank_turnover=rank_turnover(paired[row["method"]]))
    manual = manual_exprs(dsl, set(panel.factors))
    manual_rows = [{"name": n, "formula": e.text,
                    **evaluate_pool([e], dsl, panel, labels, split, horizon=horizon, stride=stride)}
                   for n, e in manual.items()]
    summary = []
    for method in ("budgeted_gp", "diverse_gp", "random"):
        group = [r for r in rows if r["method"] == method]
        summary.append({"method": method,
                        **{key: float(np.mean([r[key] for r in group])) for key in
                           ("pool_valid_rank_ic", "paired_valid_rank_ic", "pool_rank_turnover",
                            "paired_rank_turnover", "dataset_views", "n_candidates", "search_seconds")}})
    return {
        "protocol": {"research_date": "2026-10-13", "seed_list": list(seeds), "budget": budget,
                     "algorithm_version": "fixed_fold_coverage_v2",
                     "min_train_ic_coverage": base.min_train_ic_coverage,
                     "semantic_threshold": 0.95, "threshold_tuned_on_validation": False,
                     "horizon_rows": horizon, "stride_rows": stride,
                     "nav_rows_read": n_rows, "n_funds": len(panel.funds),
                     "price_span": [str(observations[0]), str(observations[-1])],
                     "train_span": [str(d) for d in split.train_span],
                     "valid_span": [str(d) for d in split.valid_span], "sealed_from": str(seal),
                     "sealed_values_read": False, "pool_weights": "equal ranks; every formula present; common finite names",
                     "paired_evaluation": "per-seed intersection of dates and names supported by all three methods",
                     "scope": "price-only cached OTC equity funds; lexicographic code sample",
                     "limitations": ["cached metadata can contain unrecorded revisions",
                                     "universe is not certified survivorship-complete",
                                     "one development split; repeated seeds are not independent market samples",
                                     "ranking metrics are not realistic executable net portfolio returns",
                                     "budgeted_gp already includes safety fixes and candidate archive; not legacy code",
                                     "random has no manual seeds; same recorded trial budget, different prior",
                                     "budget counts recorded candidates, not label-free AST-generation retries"]},
        "rows": rows, "summary": summary, "manual_baselines": manual_rows,
        "verdict": "Research ablation only; no factor approval or live portfolio change.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/search_benchmark.json"))
    parser.add_argument("--max-funds", type=int, default=80)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seeds", default="7,19,43")
    args = parser.parse_args()
    if args.max_funds < 5 or args.budget < 1:
        parser.error("need max-funds >= 5 and budget >= 1")
    result = run_benchmark(args.cache, max_funds=args.max_funds, budget=args.budget,
                           seeds=tuple(int(x) for x in args.seeds.split(",")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
