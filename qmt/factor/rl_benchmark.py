"""Predeclared GP/operator-RL comparison on reviewed historical fund segments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from qmt.factor.rl_operators import OperatorLearningConfig
from qmt.factor.rolling_cli import run_reviewed_walk_forward
from qmt.factor.walk_forward import WalkForwardConfig, joint_family_test, _clean

METHODS = ("gp", "uniform", "bandit", "q")


def _finite_mean(values):
    usable = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(usable)) if usable else None


def prepare_study(output_dir, *, budget=120, seeds=(7, 19, 43)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment_id": str(uuid.uuid4()),
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "methods": list(METHODS), "budget": budget, "seeds": list(seeds),
                "operator_learning": vars(OperatorLearningConfig()),
                "control": "uniform uses identical mutation/crossover/restart actions and backbone",
                "budget_unit": "ledger proposals including seeds, invalid/None and duplicates; fitness cache misses may differ",
                "fitness": "discovery robust Rank-IC fitness; fee-net incremental gating remains independent inner stage",
                "hyperparameters_fitted_on_outer": False,
                "historical_periods_previously_inspected": True,
                "sealed_from": "2024-12-10", "alpha_approved": False}
    (output_dir / "study_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with (output_dir / "experiment_registry.jsonl").open("a") as stream:
        stream.write(json.dumps(protocol) + "\n")
    return protocol


def summarize_reports(reports, *, methods=METHODS):
    """Reject unmatched inputs; report outcomes jointly without choosing a winner."""
    if tuple(reports) != tuple(methods):
        raise ValueError("all four predeclared methods are required in fixed order")
    first = reports["gp"]
    if any(r["status"] != "COMPLETE_STRICT_RESEARCH" for r in reports.values()):
        raise ValueError("all arms must have complete certified historical coverage")
    common_code = first["protocol"]["universe_selection"]["research_code_sha256"]
    rl_code = None
    for method, report in reports.items():
        if (report["protocol"].get("search_method") != method or
                any(a.get("search_method") != method for s in report["segments"] for a in s["searches"])):
            raise ValueError("report method differs from its declared comparison arm")
        code = report["protocol"]["universe_selection"]["research_code_sha256"]
        if any(code.get(name) != sha for name, sha in common_code.items()):
            raise ValueError("method reports use different common execution/search code")
        if method != "gp":
            current_rl = {name: code.get(name) for name in ("rl_search.py", "rl_operators.py")}
            if not all(current_rl.values()) or (rl_code is not None and current_rl != rl_code):
                raise ValueError("method reports use different or missing operator-controller code")
            rl_code = current_rl
    def signature(report):
        p = report["protocol"]
        d = p["universe_selection"]
        return (p["config"], p["universe"], p["sales_channel"], p.get("fee_reward_stage", False),
                d["historical_file_fingerprints"], d["pre_seal_nav_subset_sha256"],
                p["outer_folds"], p["settlement_interpretation"], p["nav_publication_interpretation"])
    if any(signature(r) != signature(first) for r in reports.values()):
        raise ValueError("method comparison has different budgets, sources, universe or execution assumptions")
    rows, diagnostics, family_segments = [], {}, []
    for method, report in reports.items():
        audits = [a for s in report["segments"] for a in s["searches"]]
        ledger = [t for a in audits for t in a["ledger"]["trials"]]
        diagnostics[method] = {
            "searches": len(audits), "proposal_attempts": len(ledger),
            "mean_new_formula_fitness_evaluations": float(np.mean([a["ledger"]["dataset_views"] for a in audits])),
            "mean_unique_usable_formulas": float(np.mean([a["search_diagnostics"]["unique_usable_formulas"] for a in audits])),
            "mean_best_discovery_fitness": _finite_mean([a["search_diagnostics"]["best_discovery_fitness"] for a in audits]),
            "syntax_or_construction_rejections": sum(not t["passed_dsl"] for t in ledger),
            "paired_pool_addition_trials": sum(len(a["selection"]["paired_trial_ledger"]) for a in audits),
            "accepted_searched_factors": sum(sum(n.startswith("searched_") for n in a["selection"]["selected_names"]) for a in audits),
            "operator_counts": {op: sum(t.get("generation", 0) > 0 and
                                        t.get("note", "").split(":")[0] == op for t in ledger)
                                for op in ("mutate", "crossover", "restart")}}
        for s in report["segments"]:
            for a in s["searches"]:
                seed = a["seed"]
                own = s["test_results"][f"selected_{seed}"]
                base = s["test_results"]["fixed_low_vol_60"]
                rows.append({"method": method, "year": s["fold_id"], "seed": seed,
                             "net_return": own["net_return"], "max_drawdown": own["max_drawdown"],
                             "cost_money": own["cost_money"],
                             "selected_names": a["selection"]["selected_names"],
                             "selected_formulas": a["selection"]["selected_formulas"],
                             "relative_gain_vs_low_vol": own["final_equity"] / base["final_equity"] - 1})
    for i in range(3):
        paired = {}
        reference_dates = first["segments"][i]["paired_daily_active_log_returns"]["dates"]
        raw = {}
        for method, report in reports.items():
            s = report["segments"][i]
            paths = s["paired_daily_active_log_returns"]
            if paths["dates"] != reference_dates:
                raise ValueError("method paths use different paired valuation dates")
            raw[method] = paths["columns"]
        for method in methods:
            for seed in first["protocol"]["config"]["seeds"]:
                key = f"seed_{seed}_vs_fixed_low_vol"
                active = np.array(raw[method][key], dtype=float)
                paired[f"{method}_seed_{seed}_vs_low_vol"] = active
                for ref in (("gp", "uniform", "bandit") if tuple(methods) == METHODS else methods):
                    if method != ref:
                        paired[f"{method}_seed_{seed}_vs_{ref}"] = active - np.array(raw[ref][key], dtype=float)
        family_segments.append(pd.DataFrame(paired, index=reference_dates))
    cfg = first["protocol"]["config"]
    family_config = WalkForwardConfig(seeds=tuple(cfg["seeds"]), budget=cfg["budget"], top_k=cfg["top_k"],
                                      bootstrap_repetitions=cfg["bootstrap_repetitions"],
                                      bootstrap_block_rows=cfg["bootstrap_block_rows"], family_alpha=cfg["family_alpha"],
                                      bootstrap_seed=cfg["bootstrap_seed"])
    return _clean({"status": "COMPLETE_EXPLORATORY_COMPARISON", "methods": list(methods),
            "diagnostics": diagnostics, "rows": rows,
            "gates_by_method": {m: reports[m]["aggregate_by_seed"] for m in methods},
            "joint_method_family_test": joint_family_test(family_segments, family_config),
            "alpha_approved": False,
            "limitations": ["source-accessible subset, not a survivorship-complete fund universe", "historical development years previously viewed",
                            "operator rewards and fitness differ by predeclared method", "aggregate search state is not fully Markov",
                            "joint bootstrap conditions on these observed annual regimes, not future regimes",
                            "this family excludes earlier historical research; repeated runs are recorded separately"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data_cache/execution_history/paper_research"))
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl_gp_benchmark"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if not args.summarize_only:
        prepare_study(args.output_dir, budget=args.budget, seeds=seeds)
    if args.prepare_only:
        return
    if not args.summarize_only:
        for method in METHODS:
            print(f"Running {method}", flush=True)
            run_reviewed_walk_forward(args.history, args.cache, budget=args.budget, seeds=seeds,
                                       search_method=method, output_dir=args.output_dir / method)
    reports = {m: json.loads((args.output_dir / m / "walk_forward_report.json").read_text()) for m in METHODS}
    registered = json.loads((args.output_dir / "study_protocol.json").read_text())
    for report in reports.values():
        cfg = report["protocol"]["config"]
        if cfg["budget"] != registered["budget"] or list(cfg["seeds"]) != registered["seeds"]:
            raise ValueError("executed budgets/seeds differ from the predeclared study")
    summary = summarize_reports(reports)
    summary["study_protocol"] = registered
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "alpha_approved": False,
                      "diagnostics": summary["diagnostics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
