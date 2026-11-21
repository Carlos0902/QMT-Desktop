"""A bounded local collector process; checkpoints survive pause and restart."""
from __future__ import annotations
import portalocker

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

from qmt.data.fund_lake import DatasetLake
from qmt.data.tushare_client import TushareClient


def _publish(path, payload):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_worker(root, *, seconds=86400., interval=.4, batch_queries=60, client=None, resume=False):
    if seconds <= 0 or interval < .3 or batch_queries < 1:
        raise ValueError("positive budget, interval >=.3 and batch_queries >=1 required")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {"pid": os.getpid(), "status": "starting", "queries": 0, "rows_received": 0,
             "started_at": datetime.now(timezone.utc).isoformat(), "batches": 0}
    with (root / ".collector.lock").open("a") as lock:
        try:
            portalocker.lock(lock, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.AlreadyLocked as exc:
            raise RuntimeError("another managed collector is already running") from exc
        if resume:
            (root / ".collector.pause").unlink(missing_ok=True)
        lake = DatasetLake(root)
        client = client or TushareClient(timeout=25, max_retries=0)
        try:
            while time.monotonic() - started < seconds:
                if (root / ".collector.pause").exists():
                    state["status"] = "paused"
                    break
                state.update(status="running", updated_at=datetime.now(timezone.utc).isoformat())
                _publish(root / "collector.json", state)
                budget = min(30., seconds - (time.monotonic() - started))
                if budget <= 0:
                    break
                try:
                    result = lake.run_budget(client, max_queries=batch_queries, max_seconds=budget,
                                             min_interval=interval)
                except TimeoutError:
                    state.update(status="waiting_for_catalog", updated_at=datetime.now(timezone.utc).isoformat())
                    _publish(root / "collector.json", state)
                    time.sleep(min(5., max(0., seconds - (time.monotonic() - started))))
                    continue
                state["queries"] += result["queries"]
                state["rows_received"] += result["rows_received"]
                state["batches"] += 1
                state["last_batch"] = {k: v for k, v in result.items() if k != "inventory"}
                state["tasks"] = result["inventory"]["tasks"]
                _publish(root / "collector.json", state)
                tasks = state["tasks"]
                if not (tasks.get("pending", 0) or tasks.get("retry", 0)):
                    state["status"] = ("complete_with_gaps" if any(tasks.get(k, 0) for k in
                                       ("failed", "truncated", "permission_blocked")) else "complete")
                    break
                if result["stop_reason"] in {"rate_limit_cooldown", "no_ready_tasks"}:
                    delay = max(5., float(result.get("retry_after_seconds", 0)))
                    state.update(status="cooldown", retry_after_seconds=delay)
                    _publish(root / "collector.json", state)
                    deadline = min(started + seconds, time.monotonic() + delay)
                    while time.monotonic() < deadline and not (root / ".collector.pause").exists():
                        time.sleep(min(5., max(0., deadline - time.monotonic())))
            if state["status"] not in {"complete", "complete_with_gaps", "paused"}:
                state["status"] = "budget_reached"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error_type=type(exc).__name__)
            raise
        finally:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            state["elapsed_seconds"] = round(time.monotonic() - started, 2)
            _publish(root / "collector.json", state)
            portalocker.unlock(lock)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=24.)
    parser.add_argument("--interval", type=float, default=.4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_worker(args.lake, seconds=args.hours * 3600, interval=args.interval, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
