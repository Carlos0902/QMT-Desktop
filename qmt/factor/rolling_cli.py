"""Runnable strict rolling audit/research on reviewed historical source files."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from qmt.factor.cost_benchmark import load_execution_data, json_ready
from qmt.factor.historical_bundle import HistoricalBundle
from qmt.factor.fund_execution import ExecutionConfig
from qmt.factor.walk_forward import WalkForwardConfig, run_walk_forward


def run_reviewed_walk_forward(history_dir: Path, cache: Path, *, budget=120, seeds=(7, 19, 43),
                              output_dir=None, search_method="gp", fee_reward_stage=False):
    if search_method not in {"gp", "uniform", "bandit", "q", "q_net", "uniform_net"}:
        raise ValueError("unsupported search method")
    search_factory = None
    if search_method in {"uniform", "bandit", "q"}:
        from qmt.factor.rl_search import RLGuidedSearch
        search_factory = lambda dsl: RLGuidedSearch(dsl, mode=search_method)
    histories = HistoricalBundle.load(history_dir)
    if len(histories.codes) < 5:
        raise ValueError("five pre-registered domestic funds required; do not pick by test returns")
    data = load_execution_data(cache, codes=histories.codes, domestic_only=True,
                               conservative_nav_publication=True)
    data.metadata.update(universe_selection_basis=histories.manifest.get("universe_selection_basis"),
                         historical_file_fingerprints=histories.fingerprints,
                         share_class_source="reviewed prospectuses: class A / single-class domestic funds",
                         dividend_source="reviewed official dividend events; no adjusted/unit ratio reconstruction",
                         managers=histories.manifest.get("managers", {}),
                         public_source_exposure=histories.manifest.get("public_source_exposure"),
                         limitations=["source-accessibility selected subset; full domestic universe not certified",
                                      "metadata revisions and complete historical survivorship unverified",
                                      "provider NAV announcement dates are conservative information-delay scenarios; earliest daily publication unverified",
                                      "normal T+1 broker confirmations and T+7 contractual payment bound, not investor records"])
    code_files = ["walk_forward.py", "rolling_cli.py", "fund_execution.py", "incremental.py",
                  "auto_search.py", "historical_bundle.py", "event_history.py", "fee_history.py", "cost_benchmark.py"]
    if search_method != "gp":
        code_files += ["rl_search.py", "rl_operators.py"]
    if fee_reward_stage:
        code_files += ["net_reward.py"]
    data.metadata["research_code_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in code_files}
    # Five fully sourced names need a ranking portfolio smaller than all five.
    top_k = min(5, max(2, len(histories.codes) // 3))
    config = WalkForwardConfig(budget=budget, seeds=tuple(seeds), top_k=top_k)
    # Prospectuses document payment within T+7, not actual individual settlement.
    execution = ExecutionConfig(data.trading_dates, initial_cash=config.initial_cash,
                                 top_k=top_k, fill_lag_sessions=0,
                                 purchase_confirm_sessions=1, redemption_confirm_sessions=1,
                                 redemption_settlement_sessions=7,
                                 dividend_mode="cash", reinvest_dividends=False,
                                 covered_daily_buy_money_exclusive=500000.0,
                                 research_minimum_remaining_units=1.0,
                                 research_minimum_redemption_units=1.0,
                                 confirmation_requires_public_nav=False,
                                 fee_assumptions=("Reviewed ordinary direct-counter historical fee tables.",
                                                  "Cash payment uses documented T+7 contractual upper-bound scenario, not investor settlement observations."))
    report = run_walk_forward(data, fee_history=histories.fees, status_history=histories.statuses,
                              dividend_history=histories.dividends, coverage_validator=histories.validate,
                              config=config, execution_config=execution, output_dir=output_dir,
                              search_factory=search_factory, search_method=search_method,
                              fee_reward_stage=fee_reward_stage)
    report["protocol"].update(sales_channel=histories.fees.channel,
                               verified_daily_subscription_scope="per fund gross daily subscription <500000 CNY; outside scope raises",
                               redemption_policy="partial redemptions leave >=1 unit; full redemption may leave zero; common admissible source intersection",
                               confirmation_interpretation="normal T+1 contractual broker confirmations assumed; not delayed by public NAV disclosure",
                               nav_publication_interpretation="provider date-only NAV cache markers available after announcement day end; earliest daily publication times unverified, so this is a conservative delay scenario",
                               settlement_interpretation="contractual upper bound 7 sessions; actual investor settlement dates unavailable",
                               portfolio_top_k_basis="pre-registered fraction of source-supported universe, no test return tuning",
                               validation_status="historical development periods previously inspected, not untouched final test")
    if output_dir is not None:
        path = Path(output_dir) / "walk_forward_report.json"
        path.write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data_cache/execution_history/paper_research"))
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--search-method", choices=("gp", "uniform", "bandit", "q"), default="gp")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/walk_forward"))
    args = parser.parse_args()
    report = run_reviewed_walk_forward(args.history, args.cache, budget=args.budget,
                                       seeds=tuple(int(x) for x in args.seeds.split(",")), output_dir=args.output_dir,
                                       search_method=args.search_method)
    print(json.dumps({"status": report["status"], "alpha_approved": report["alpha_approved"],
                      "segments": [{"fold_id": f["fold_id"], "status": f["status"]} for f in report["segments"]],
                      "output": str(args.output_dir / "walk_forward_report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
