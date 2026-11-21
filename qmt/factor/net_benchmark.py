"""Pre-register and compare isolated fee-net rewards against matched controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from qmt.factor.historical_bundle import HistoricalBundle
from qmt.factor.rolling_cli import run_reviewed_walk_forward
from qmt.factor.rl_benchmark import summarize_reports

METHODS = ("gp", "q", "uniform_net", "q_net")


def prepare(output, history, *, budget=120, seeds=(7, 19, 43)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = HistoricalBundle.load(Path(history))
    protocol = {"experiment_id": str(uuid.uuid4()),
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "methods": list(METHODS), "budget": budget, "seeds": list(seeds),
                "universe": list(history.codes), "managers": history.manifest.get("managers", {}),
                "history_sha256": history.fingerprints,
                "source_audit_sha256": hashlib.sha256((Path(history.directory) / "source_audit.json").read_bytes()).hexdigest()
                    if (Path(history.directory) / "source_audit.json").exists() else None,
                "public_source_exposure": history.manifest.get("public_source_exposure"),
                "reward_stage": "first original inner block; independent 3-fold selection after 40-session purge",
                "formula_discovery": "original earlier discovery labels; full label maturity before reward start",
                "net_fitness": "paired terminal net wealth gain of adding candidate to frozen low_vol_60 anchor",
                "net_controller_reward": "positive incumbent net-fitness improvement, clipped at 1; novelty bonus 0; shared waste penalties",
                "net_objective_control": "uniform_net has identical objective, actions, proposal cap and execution model",
                "ic_control": "gp and q use IC objective but identical shortened selection folds",
                "budget_unit": "proposals including manual seeds, invalid attempts and duplicates; evaluation counts reported separately",
                "stability_gate": "win >=2/3 segments, positive segment relative-wealth product, no losing segment, drawdown not worse",
                "significance": "joint max-t 20-session block bootstrap, 999 repetitions, family alpha .05",
                "selection_frozen_before_each_outer": True,
                "historical_periods_previously_inspected": True, "sealed_from": "2024-12-10",
                "alpha_approved": False,
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("net_reward.py"))}}
    path = output / "study_protocol.json"
    if path.exists():
        raise ValueError("study already registered; use --method/--summarize or a new output directory")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    with (output / "experiment_registry.jsonl").open("a") as stream:
        stream.write(json.dumps(protocol) + "\n")
    return protocol


def summarize(output):
    output = Path(output)
    registered = json.loads((output / "study_protocol.json").read_text())
    reports = {m: json.loads((output / m / "walk_forward_report.json").read_text()) for m in METHODS}
    for r in reports.values():
        p = r["protocol"]
        if (p["config"]["budget"] != registered["budget"] or
                p["config"]["seeds"] != registered["seeds"] or
                p["universe"] != registered["universe"] or
                p["universe_selection"]["historical_file_fingerprints"] != registered["history_sha256"] or
                not p.get("fee_reward_stage")):
            raise ValueError("executed study differs from registered inputs or partitions")
    if any(r["status"] != "COMPLETE_STRICT_RESEARCH" for r in reports.values()):
        result = {"status": "BLOCKED_COMPARISON", "alpha_approved": False,
                  "method_status": {m: r["status"] for m, r in reports.items()}}
    else:
        result = summarize_reports(reports, methods=METHODS)
    result["study_protocol"] = registered
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def record_arm_start(output, registered, method):
    """Never silently erase completed searches; account for partial retries."""
    output = Path(output)
    if (output / method / "walk_forward_report.json").exists():
        raise ValueError("completed arm already exists; register a new output directory for another run")
    receipt = {"experiment_id": registered["experiment_id"], "event": "arm_started",
               "method": method, "run_id": str(uuid.uuid4()),
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "partial_retry": (output / method).exists()}
    if receipt["partial_retry"]:
        archive = output / "partial_attempts" / method / receipt["run_id"]
        shutil.copytree(output / method, archive)
        receipt["preserved_partial_attempt"] = str(archive)
    with (output / "experiment_registry.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/net_reward_benchmark"))
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.output_dir)
        print(json.dumps({"status": result["status"], "alpha_approved": False}))
        return
    if not (args.output_dir / "study_protocol.json").exists():
        prepare(args.output_dir, args.history, budget=args.budget,
                seeds=tuple(int(s) for s in args.seeds.split(",")))
    registered = json.loads((args.output_dir / "study_protocol.json").read_text())
    histories = HistoricalBundle.load(args.history)
    if histories.fingerprints != registered["history_sha256"]:
        raise ValueError("registered history files changed")
    if args.prepare_only:
        return
    for method in ((args.method,) if args.method else METHODS):
        record_arm_start(args.output_dir, registered, method)
        print("Running", method, flush=True)
        r = run_reviewed_walk_forward(args.history, args.cache,
                                      budget=registered["budget"], seeds=tuple(registered["seeds"]),
                                      search_method=method, fee_reward_stage=True,
                                      output_dir=args.output_dir / method)
        print(method, r["status"], flush=True)
    if args.method is None:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
