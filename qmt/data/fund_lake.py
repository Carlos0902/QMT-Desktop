"""Resumable, provenance-preserving Tushare archive, separate from research caches.

This is an acquisition archive, not a certificate of historical tradability. Raw
pages are immutable Parquet files; DuckDB records both response provenance and
row revisions. Current fund metadata becomes available at acquisition time and
must never be used to backdate a historical investment universe.
"""
from __future__ import annotations
import portalocker

import contextlib
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import pandas as pd

from qmt.data.tushare_client import TushareQuotaError, TushareRateLimitError

__all__ = ["DatasetLake", "DatasetSpec", "DATASETS", "LakeBusyError"]


class LakeBusyError(TimeoutError):
    """Another catalog reader/writer exceeded the bounded lock wait; safe to retry."""


@dataclass(frozen=True)
class DatasetSpec:
    keys: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    page_size: int = 2000
    metadata: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "fund_basic": DatasetSpec(("ts_code",), ("found_date",), 4000, True),
    "fund_nav": DatasetSpec(("ts_code", "nav_date"), ("nav_date",)),
    "fund_portfolio": DatasetSpec(("ts_code", "end_date", "symbol"), ("end_date",)),
    "fund_div": DatasetSpec(("ts_code", "ann_date", "div_proc", "base_date", "record_date", "ex_date", "base_year"), ("ex_date", "record_date", "ann_date")),
    "fund_share": DatasetSpec(("ts_code", "trade_date"), ("trade_date",)),
    "trade_cal": DatasetSpec(("exchange", "cal_date"), ("cal_date",), 2000),
    "index_daily": DatasetSpec(("ts_code", "trade_date"), ("trade_date",)),
    "sw_daily": DatasetSpec(("ts_code", "trade_date"), ("trade_date",)),
    "fund_company": DatasetSpec(("name",), (), 2000, True),
    "fund_manager": DatasetSpec(("ts_code", "name", "begin_date", "end_date"), ("ann_date",), 2000, True),
}

# total_netasset is documented but absent from the API's default NAV response.
# Request the full known schema so the acquisition archive does not silently lose it.
DEFAULT_FIELDS: dict[str, tuple[str, ...]] = {
    "fund_nav": ("ts_code", "ann_date", "nav_date", "unit_nav", "accum_nav", "accum_div",
                 "net_asset", "total_netasset", "adj_nav", "update_flag"),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public(value: Any) -> Any:
    """Never serialize credentials, including nested task parameters."""
    if isinstance(value, Mapping):
        return {str(k): _public(v) for k, v in value.items()
                if not any(x in str(k).lower() for x in ("token", "password", "authorization", "secret"))}
    if isinstance(value, (list, tuple)):
        return [_public(v) for v in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value: Any) -> str:
    return json.dumps(_public(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _day(value: Any) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return None
    try:
        parsed = pd.to_datetime(text, format="%Y%m%d") if len(text) == 8 and text.isdigit() else pd.Timestamp(text)
        return parsed.date().isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def _date_range(start: str, end: str) -> tuple[date, date]:
    a, b = _day(start), _day(end)
    if not a or not b or a > b:
        raise ValueError("expected a valid inclusive start_date <= end_date")
    return date.fromisoformat(a), date.fromisoformat(b)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class DatasetLake:
    """Local archive with idempotent task planning and bounded collection runs.

    The injected client must expose ``query(api_name, params=..., fields=...)``.
    Use ``TushareClient(max_retries=0, timeout=...)``: retry scheduling lives in
    this catalog, so rate limits never cause a hidden one-minute blocking wait.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.duckdb"
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS shards (
                shard_id VARCHAR PRIMARY KEY, dataset VARCHAR, api_name VARCHAR,
                params_json VARCHAR, fetched_at VARCHAR, origin VARCHAR, origin_mtime VARCHAR,
                row_count BIGINT, schema_json VARCHAR, fingerprint VARCHAR, raw_path VARCHAR,
                task_id VARCHAR, availability_note VARCHAR)""")
            db.execute("""CREATE TABLE IF NOT EXISTS records (
                dataset VARCHAR, row_key VARCHAR, payload_json VARCHAR, payload_hash VARCHAR,
                source_date VARCHAR, source_available_at VARCHAR, availability_basis VARCHAR,
                ingested_at VARCHAR, shard_id VARCHAR, PRIMARY KEY(dataset,row_key))""")
            db.execute("""CREATE TABLE IF NOT EXISTS revisions (
                dataset VARCHAR, row_key VARCHAR, payload_json VARCHAR, payload_hash VARCHAR,
                source_date VARCHAR, source_available_at VARCHAR, availability_basis VARCHAR,
                ingested_at VARCHAR, shard_id VARCHAR,
                PRIMARY KEY(dataset,row_key,payload_hash))""")
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                task_id VARCHAR PRIMARY KEY, dataset VARCHAR, api_name VARCHAR, params_json VARCHAR,
                fields_json VARCHAR, page_size BIGINT, offset_rows BIGINT DEFAULT 0,
                status VARCHAR DEFAULT 'pending', priority INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0, pages BIGINT DEFAULT 0, rows_received BIGINT DEFAULT 0,
                previous_page_hash VARCHAR, not_before DOUBLE DEFAULT 0,
                last_error VARCHAR, created_at VARCHAR, updated_at VARCHAR)""")
            db.execute("""CREATE TABLE IF NOT EXISTS interfaces (
                api_name VARCHAR PRIMARY KEY, status VARCHAR, last_error VARCHAR, updated_at VARCHAR)""")
            db.execute("CREATE INDEX IF NOT EXISTS tasks_queue ON tasks(status,priority,not_before)")
            db.execute("""CREATE TABLE IF NOT EXISTS archival_files (
                file_id VARCHAR PRIMARY KEY,dataset VARCHAR,raw_path VARCHAR,filename VARCHAR,
                byte_fingerprint VARCHAR,row_count BIGINT,code_count BIGINT,
                schema_json VARCHAR,fetched_at VARCHAR,ingested_at VARCHAR,seal VARCHAR)""")
            db.execute("ALTER TABLE shards ADD COLUMN IF NOT EXISTS ingested_at VARCHAR")
            db.execute("ALTER TABLE shards ADD COLUMN IF NOT EXISTS code_count BIGINT")
            db.execute("ALTER TABLE shards ADD COLUMN IF NOT EXISTS quality_json VARCHAR")
            db.execute("ALTER TABLE shards ADD COLUMN IF NOT EXISTS requested_fields_json VARCHAR")
            db.execute("ALTER TABLE interfaces ADD COLUMN IF NOT EXISTS blocked_until DOUBLE DEFAULT 0")
            db.execute("ALTER TABLE interfaces ADD COLUMN IF NOT EXISTS rate_strikes INTEGER DEFAULT 0")

    @contextlib.contextmanager
    def _db(self):
        # DuckDB does not admit a second process while a read/write connection is
        # open. Serialize EVERY connection, including research scans and inventories,
        # instead of assuming the collector's writer lock also protects readers.
        with (self.root / ".catalog.lock").open("a") as handle:
            deadline = time.monotonic() + 30
            while True:
                try:
                    portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    break
                except portalocker.exceptions.AlreadyLocked as exc:
                    if time.monotonic() >= deadline:
                        raise LakeBusyError("catalog busy after 30 seconds; retry later") from exc
                    time.sleep(0.1)
            db = None
            try:
                for attempt in range(5):
                    try:
                        db = duckdb.connect(str(self.catalog_path))
                        break
                    except duckdb.IOException as exc:
                        if not any(x in str(exc).lower() for x in ("could not set lock", "conflicting lock", "used by another")):
                            raise
                        if attempt == 4:
                            raise LakeBusyError("catalog is held by another DuckDB client; retry later") from exc
                        time.sleep(0.1 * (attempt + 1))
                yield db
            finally:
                if db is not None:
                    db.close()
                portalocker.unlock(handle)

    @contextlib.contextmanager
    def _writer(self):
        # Explicitly reject a second collector, rather than stealing its tasks.
        with (self.root / ".writer.lock").open("a") as handle:
            try:
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
            except portalocker.exceptions.AlreadyLocked as exc:
                raise RuntimeError("another acquisition writer is active for this lake") from exc
            try:
                yield
            finally:
                portalocker.unlock(handle)

    def enqueue(self, dataset: str, params: Mapping[str, Any], *, api_name: str | None = None,
                fields: Sequence[str] | None = None, page_size: int | None = None,
                priority: int = 0, generation: str = "") -> str:
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset {dataset!r}")
        clean = _public(dict(params))
        clean.pop("offset", None)
        clean.pop("limit", None)
        api = api_name or dataset
        limit = page_size or DATASETS[dataset].page_size
        if not 1 <= limit <= 100000:
            raise ValueError("page_size must be between 1 and 100000")
        identity = [dataset, api, clean, list(fields or ()), limit]
        task_id = _hash(identity + ([generation] if generation else []))
        stamp = _utcnow()
        with self._db() as db:
            db.execute("""INSERT INTO tasks(task_id,dataset,api_name,params_json,fields_json,
                page_size,priority,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO NOTHING""",
                [task_id, dataset, api, _json(clean), _json(fields or []), limit, priority, stamp, stamp])
        return task_id

    def enqueue_many(self, dataset: str, params_list: Iterable[Mapping[str, Any]], *,
                     api_name: str | None = None, fields: Sequence[str] | None = None,
                     page_size: int | None = None, priority: int = 0,
                     generation: str = "") -> dict:
        """Bulk plan supplemental interfaces without per-task database opens."""
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset {dataset!r}")
        limit = page_size or DATASETS[dataset].page_size
        if not 1 <= limit <= 100000:
            raise ValueError("page_size must be between 1 and 100000")
        api, stamp = api_name or dataset, _utcnow()
        before = self.task_count()
        with self._db() as db:
            batch = []
            def flush():
                if not batch:
                    return
                db.register("many_tasks", pd.DataFrame(batch, columns=["task_id", "dataset", "api_name", "params_json", "fields_json", "page_size", "priority", "created_at", "updated_at"]))
                db.execute("""INSERT INTO tasks(task_id,dataset,api_name,params_json,fields_json,
                    page_size,priority,created_at,updated_at) SELECT * FROM many_tasks
                    ON CONFLICT(task_id) DO NOTHING""")
                batch.clear()
            for params in params_list:
                clean = _public(dict(params))
                clean.pop("limit", None)
                clean.pop("offset", None)
                identity = [dataset, api, clean, list(fields or ()), limit]
                task_id = _hash(identity + ([generation] if generation else []))
                batch.append((task_id, dataset, api, _json(clean), _json(fields or []), limit,
                              priority, stamp, stamp))
                if len(batch) == 10000:
                    flush()
            flush()
        return {"tasks_added": self.task_count() - before, "tasks_total": self.task_count()}

    def plan_basic(self, *, markets: Sequence[str] = ("O", "E"),
                   statuses: Sequence[str] = ("L", "I", "D"), page_size: int | None = None,
                   generation: str = "") -> dict:
        before = self.task_count()
        for market in markets:
            for status in statuses:
                self.enqueue("fund_basic", {"market": market, "status": status}, page_size=page_size, priority=100, generation=generation)
        return {"tasks_added": self.task_count() - before, "tasks_total": self.task_count()}

    def plan_nav_dates(self, start_date: str, end_date: str, *,
                       markets: Sequence[str] = ("O", "E"), page_size: int | None = None,
                       recent_first: bool = True, generation: str = "") -> dict:
        start, end = _date_range(start_date, end_date)
        before = self.task_count()
        days = (end - start).days + 1
        # Date-wide retrieval includes all fund types and avoids survivor-only code lists.
        tasks = []
        stamp = _utcnow()
        for n in range(days):
            day = end - timedelta(days=n) if recent_first else start + timedelta(days=n)
            for market in markets:
                params = {"nav_date": day.strftime("%Y%m%d"), "market": market}
                limit = page_size or DATASETS["fund_nav"].page_size
                identity = ["fund_nav", "fund_nav", params, [], limit]
                task_id = _hash(identity + ([generation] if generation else []))
                ordered_stamp = (datetime.fromisoformat(stamp) + timedelta(microseconds=n)).isoformat()
                tasks.append((task_id, "fund_nav", "fund_nav", _json(params), "[]", limit,
                              60 if generation else 50, ordered_stamp, stamp))
        with self._db() as db:
            db.register("planned_tasks", pd.DataFrame(tasks, columns=["task_id", "dataset", "api_name", "params_json", "fields_json", "page_size", "priority", "created_at", "updated_at"]))
            db.execute("""INSERT INTO tasks(task_id,dataset,api_name,params_json,fields_json,
                page_size,priority,created_at,updated_at) SELECT * FROM planned_tasks
                ON CONFLICT(task_id) DO NOTHING""")
        return {"tasks_added": self.task_count() - before, "tasks_total": self.task_count()}

    def refresh_recent(self, *, days: int = 14, end_date: str | None = None,
                       markets: Sequence[str] = ("O", "E"), page_size: int | None = None,
                       generation: str | None = None) -> dict:
        """Append a daily refresh generation; old checkpoints and raw pages survive."""
        if days < 1:
            raise ValueError("days must be positive")
        end = date.fromisoformat(_day(end_date) or datetime.now(timezone.utc).date().isoformat())
        return self.plan_nav_dates((end - timedelta(days=days-1)).isoformat(), end.isoformat(),
            markets=markets, page_size=page_size,
            generation=generation or datetime.now(timezone.utc).strftime("%Y%m%d"))

    def plan_funds(self, start_date: str, end_date: str, *,
                   datasets: Sequence[str] = ("fund_nav", "fund_div", "fund_portfolio"),
                   codes: Sequence[str] | None = None, chunk_years: int = 5,
                   page_size: int | None = None) -> dict:
        start, end = _date_range(start_date, end_date)
        if chunk_years < 1:
            raise ValueError("chunk_years must be positive")
        if codes is None:
            basic = self.read_table("fund_basic")
            code_col = "ts_code" if "ts_code" in basic else "fund_code"
            codes = basic[code_col].dropna().astype(str).unique().tolist() if code_col in basic else []
        before = self.task_count()
        planned = []
        limit = page_size or DATASETS["fund_nav"].page_size
        if not 1 <= limit <= 100000:
            raise ValueError("page_size must be between 1 and 100000")
        stamp = _utcnow()
        for code in sorted(set(codes)):
            normalized = code if "." in code else f"{code}.OF"
            for dataset in datasets:
                if dataset not in ("fund_nav", "fund_div", "fund_portfolio", "fund_share"):
                    raise ValueError("plan_funds supports fund_nav/fund_div/fund_portfolio/fund_share")
                cursor = start
                while cursor <= end:
                    chunk_end = min(end, date(min(cursor.year + chunk_years, 9999), 1, 1) - timedelta(days=1))
                    # fund_div exposes date range inconsistently across accounts. One per-code
                    # query with pagination is safer than pretending ignored ranges are complete.
                    params = {"ts_code": normalized}
                    if dataset != "fund_div":
                        params.update(start_date=cursor.strftime("%Y%m%d"), end_date=chunk_end.strftime("%Y%m%d"))
                    task_limit = page_size or DATASETS[dataset].page_size
                    task_id = _hash([dataset, dataset, params, [], task_limit])
                    planned.append((task_id, dataset, dataset, _json(params), "[]", task_limit, 20, stamp, stamp))
                    if dataset == "fund_div":
                        break
                    cursor = chunk_end + timedelta(days=1)
        if planned:
            with self._db() as db:
                db.execute("BEGIN TRANSACTION")
                try:
                    # Native batched inserts avoid tens of thousands of catalog opens or
                    # per-task autocommits when covering the full fund universe.
                    for offset in range(0, len(planned), 10000):
                        db.register("planned_funds", pd.DataFrame(planned[offset:offset+10000], columns=["task_id", "dataset", "api_name", "params_json", "fields_json", "page_size", "priority", "created_at", "updated_at"]))
                        db.execute("""INSERT INTO tasks(task_id,dataset,api_name,params_json,fields_json,
                            page_size,priority,created_at,updated_at) SELECT * FROM planned_funds
                            ON CONFLICT(task_id) DO NOTHING""")
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
        return {"tasks_added": self.task_count() - before, "tasks_total": self.task_count(), "funds": len(set(codes))}

    def task_count(self) -> int:
        with self._db() as db:
            return db.execute("SELECT count(*) FROM tasks").fetchone()[0]

    @staticmethod
    def _normal_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result.columns = [str(c).lower() for c in result.columns]
        result = result.drop(columns=[c for c in result if any(x in c for x in ("token", "password", "authorization", "secret"))])
        if "ts_code" not in result and "fund_code" in result:
            result["ts_code"] = result["fund_code"].astype(str).map(lambda x: x if "." in x else f"{x}.OF")
        if dataset == "fund_nav" and "nav_date" not in result and "end_date" in result:
            result["nav_date"] = result["end_date"]
        if dataset == "fund_portfolio" and "end_date" not in result and "report_end_date" in result:
            result["end_date"] = result["report_end_date"]
        if dataset == "fund_share" and "trade_date" not in result and "period_end" in result:
            result["trade_date"] = result["period_end"]
        return result

    def _store_page(self, dataset: str, frame: pd.DataFrame, *, api_name: str, params: dict,
                    task_id: str | None = None, origin: str = "tushare_api",
                    fetched_at: str | None = None, origin_mtime: str | None = None,
                    source_copy: Path | None = None, task_update: dict | None = None,
                    fields: Sequence[str] | None = None) -> dict:
        frame = self._normal_frame(dataset, frame)
        ingested = _utcnow()
        fetched = fetched_at or ingested
        rows = [_public(row) for row in frame.to_dict("records")]
        # Order-insensitive fingerprints catch servers that ignore offset and reorder rows.
        page_hash = _hash(sorted(_json(r) for r in rows))
        # Content identity and acquisition identity are different. A later A->B->A
        # revision must retain the final A receipt instead of adopting the first A's
        # timestamp. Task generations + page offset identify resumable API receipts;
        # explicit fetched_at identifies imported bootstrap receipts.
        receipt_id = _hash([task_id, params.get("offset", 0)]) if task_id else fetched
        shard_id = _hash([dataset, api_name, params, page_hash, origin, receipt_id] + ([list(fields)] if fields else []))
        relative = Path("raw") / dataset / f"{shard_id}.parquet"
        raw_path = self.root / relative
        manifest_path = raw_path.with_suffix(".json")
        schema = {str(k): str(v) for k, v in frame.dtypes.items()}
        required = {"fund_nav": ("ts_code", "nav_date"),
                    "fund_portfolio": ("ts_code", "end_date", "symbol"),
                    "fund_share": ("ts_code", "trade_date")}.get(dataset, (DATASETS[dataset].keys[0],))
        if len(frame) and any(k not in frame or frame[k].isna().any() for k in required):
            raise ValueError(f"{dataset}: row has no stable source identity")
        present_keys = [x for x in DATASETS[dataset].keys if x in frame]
        quality = {"duplicate_key_rows_within_page": int(frame.duplicated(present_keys).sum()) if present_keys else 0,
                   "missing_announcement_rows": (int(frame["ann_date"].map(_day).isna().sum()) if "ann_date" in frame else len(frame)) if dataset in ("fund_nav", "fund_portfolio", "fund_div") else 0}
        if dataset == "fund_nav":
            quality["nonpositive_or_invalid_unit_nav_rows"] = int((pd.to_numeric(frame.get("unit_nav", pd.Series(index=frame.index, dtype=float)), errors="coerce").fillna(0) <= 0).sum())
        manifest = {"shard_id": shard_id, "dataset": dataset, "api_name": api_name,
                    "params": _public(params), "fetched_at": fetched, "ingested_at": ingested,
                    "requested_fields": list(fields or []),
                    "origin": origin, "origin_mtime": origin_mtime, "row_count": len(frame),
                    "schema": schema, "fingerprint": page_hash,
                    "receipt_id": receipt_id,
                    "quality_counts": quality,
                    "historical_execution_certified": False,
                    "availability_note": "Source dates and ingestion timestamps are separate; current metadata is never backdated."}
        if not raw_path.exists():
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temp = raw_path.with_name(f".{raw_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                if source_copy is not None:
                    shutil.copyfile(source_copy, temp)
                else:
                    frame.to_parquet(temp, index=False)
                with temp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp, raw_path)
            finally:
                temp.unlink(missing_ok=True)
        if not manifest_path.exists():
            _atomic_bytes(manifest_path, (_json(manifest) + "\n").encode())
        else:
            # A process can stop after raw/manifest rename but before catalog commit.
            # Reuse that immutable response's timestamps when adopting its orphan files.
            original = json.loads(manifest_path.read_text())
            if original.get("fingerprint") != page_hash:
                raise ValueError("immutable raw page manifest fingerprint mismatch")
            fetched = original["fetched_at"]
            ingested = original["ingested_at"]
        with self._db() as db:
            db.execute("BEGIN TRANSACTION")
            try:
                db.execute("""INSERT INTO shards(shard_id,dataset,api_name,params_json,fetched_at,
                    origin,origin_mtime,row_count,schema_json,fingerprint,raw_path,task_id,
                    availability_note,ingested_at,code_count,quality_json,requested_fields_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(shard_id) DO NOTHING""",
                           [shard_id, dataset, api_name, _json(params), fetched, origin, origin_mtime,
                            len(frame), _json(schema), page_hash, str(relative), task_id,
                            manifest["availability_note"], ingested,
                            int(frame["ts_code"].nunique()) if "ts_code" in frame else 0, _json(quality), _json(fields or [])])
                if task_update is not None:
                    db.execute("""UPDATE tasks SET offset_rows=?,status=?,pages=pages+1,
                        rows_received=rows_received+?,previous_page_hash=?,attempts=0,
                        not_before=0,last_error=?,updated_at=? WHERE task_id=?""",
                        [task_update["offset"], task_update["status"], len(frame), page_hash,
                         task_update.get("error"), ingested, task_id])
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return {"rows": len(frame), "shard_id": shard_id, "fingerprint": page_hash}

    def import_legacy(self, cache_dir: str | Path, *, seal: str | None = None) -> dict:
        """Copy immutable Parquet archives using bounded memory, never Python row loops.

        All values, including sealed periods, are archived; returned summaries contain
        counts only. DuckDB scans selected columns for inventory. Original byte hashes
        make re-imports idempotent, and research reads must specify their cutoff.
        """
        imported: dict[str, int] = {}
        skipped: list[str] = []
        with self._writer():
            for path in sorted(Path(cache_dir).glob("*.parquet")):
                dataset = path.stem
                if dataset not in DATASETS:
                    skipped.append(path.name)
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                fingerprint = digest.hexdigest()
                file_id = _hash([dataset, fingerprint])
                relative = Path("raw") / dataset / f"legacy-{file_id}.parquet"
                target = self.root / relative
                mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                stamp = _utcnow()
                with self._db() as db:
                    exists = db.execute("SELECT row_count FROM archival_files WHERE file_id=?", [file_id]).fetchone()
                if exists:
                    imported[dataset] = exists[0]
                    continue
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                    try:
                        shutil.copyfile(path, temp)
                        with temp.open("rb") as handle:
                            os.fsync(handle.fileno())
                        os.replace(temp, target)
                    finally:
                        temp.unlink(missing_ok=True)
                with self._db() as db:
                    schema = db.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(target)]).fetchall()
                    names = {r[0] for r in schema}
                    code = '"ts_code"' if "ts_code" in names else '"fund_code"' if "fund_code" in names else None
                    count, codes = db.execute(f"SELECT count(*),{f'count(DISTINCT {code})' if code else '0'} FROM read_parquet(?)", [str(target)]).fetchone()
                    db.execute("INSERT INTO archival_files VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(file_id) DO NOTHING",
                        [file_id, dataset, str(relative), path.name, fingerprint, count, codes,
                         _json({r[0]: r[1] for r in schema}), mtime, stamp, seal])
                manifest = {"file_id": file_id, "dataset": dataset, "origin": "provenance_imported",
                    "api_name": "legacy_parquet_import", "params": {"filename": path.name},
                    "fetched_at": mtime, "fetched_at_basis": "legacy_file_mtime_not_verified_api_time",
                    "ingested_at": stamp, "fingerprint": fingerprint, "schema": {r[0]: r[1] for r in schema},
                    "rows": count, "historical_execution_certified": False,
                    "availability_note": "Legacy available_at is unverified; current metadata is ingestion-available only."}
                _atomic_bytes(target.with_suffix(".json"), (_json(manifest) + "\n").encode())
                imported[dataset] = count
        self._write_inventory_snapshot()
        return {"imported_rows": imported, "skipped_files": skipped,
                "historical_execution_certified": False}

    def read_table(self, dataset: str, cutoff: str | date | datetime | None = None, *,
                   acquired_before: str | None = None, include_provenance: bool = True,
                   codes: Sequence[str] | None = None, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """DuckDB filters and deduplicates native Parquet before materializing rows.

        This reads every recorded revision from immutable files, filters availability
        before ranking, and retains the last visible source revision. Fund metadata is
        available only at ingestion and cannot establish historical eligibility.
        """
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset {dataset!r}")
        cutoff_utc = None
        if cutoff is not None:
            parsed = pd.Timestamp(cutoff)
            if len(str(cutoff)) in (8, 10):
                parsed = parsed.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize("Asia/Shanghai")
            cutoff_utc = parsed.tz_convert("UTC").isoformat()
        with self._db() as db:
            files = db.execute("SELECT raw_path,coalesce(ingested_at,fetched_at),fetched_at,shard_id,origin FROM shards WHERE dataset=? AND row_count>0", [dataset]).fetchall()
            files += db.execute("SELECT raw_path,ingested_at,fetched_at,file_id,'provenance_imported' FROM archival_files WHERE dataset=?", [dataset]).fetchall()
            if not files:
                return pd.DataFrame()
            provenance = pd.DataFrame([(str(self.root / p), ing, fet, sha, ori) for p, ing, fet, sha, ori in files],
                columns=["__file", "__ingested", "__fetched", "__shard", "__origin"])
            db.register("lake_provenance", provenance)
            paths = provenance["__file"].tolist()
            relation = db.read_parquet(paths, union_by_name=True, filename=True)
            db.register("lake_raw", relation)
            names = [r[0] for r in db.execute("DESCRIBE lake_raw").fetchall() if r[0] != "filename"]
            normalized = [f'"{x}"' for x in names]
            if "ts_code" in names and "fund_code" in names:
                normalized[names.index("ts_code")] = "coalesce(ts_code,CASE WHEN contains(fund_code,'.') THEN fund_code ELSE fund_code || '.OF' END) AS ts_code"
                normalized[names.index("fund_code")] = "coalesce(fund_code,split_part(ts_code,'.',1)) AS fund_code"
            if "ts_code" not in names and "fund_code" in names:
                normalized.append("CASE WHEN contains(fund_code,'.') THEN fund_code ELSE fund_code || '.OF' END AS ts_code")
                names.append("ts_code")
            if "fund_code" not in names and "ts_code" in names:
                normalized.append("split_part(ts_code,'.',1) AS fund_code")
                names.append("fund_code")
            aliases = {"fund_nav": ("nav_date", "end_date"), "fund_portfolio": ("end_date", "report_end_date"), "fund_share": ("trade_date", "period_end")}
            if dataset in aliases:
                new, old = aliases[dataset]
                if new not in names and old in names:
                    normalized.append(f'"{old}" AS "{new}"')
                    names.append(new)
                elif new in names and old in names:
                    normalized[names.index(new)] = f'coalesce(cast("{new}" AS VARCHAR),cast("{old}" AS VARCHAR)) AS "{new}"'
            def dt(name):
                return f"coalesce(try_strptime(cast(\"{name}\" AS VARCHAR),'%Y%m%d'),try_cast(\"{name}\" AS TIMESTAMP))"
            observed_col = next((x for x in DATASETS[dataset].date_columns if x in names), None)
            observed = dt(observed_col) if observed_col else "NULL::TIMESTAMP"
            announced = dt("ann_date") if "ann_date" in names else "NULL::TIMESTAMP"
            legacy_available = "try_cast(available_at AS TIMESTAMP) AT TIME ZONE 'Asia/Shanghai'" if "available_at" in names else "NULL::TIMESTAMPTZ"
            if DATASETS[dataset].metadata or dataset == "trade_cal":
                available = "try_cast(__ingested AS TIMESTAMPTZ)"
                basis = "'current_metadata_observed_at_ingestion'"
            else:
                # An absent NAV announcement is not evidence of publication on
                # the next day. Retain it only as observed-at-ingestion data;
                # historical reads must exclude it until evidence is recovered.
                fallback = ("try_cast(__ingested AS TIMESTAMPTZ)" if dataset == "fund_nav" else
                            f"(({observed}) + INTERVAL 1 DAY) AT TIME ZONE 'UTC'" if dataset in ("fund_share", "index_daily", "sw_daily") else "NULL::TIMESTAMPTZ")
                available = f"CASE WHEN ({announced}) IS NOT NULL THEN (({announced}) + INTERVAL 1 DAY) AT TIME ZONE 'UTC' WHEN __origin='provenance_imported' THEN {legacy_available} ELSE {fallback} END"
                unknown_basis = "missing_announcement_observed_at_ingestion" if dataset == "fund_nav" else "observation_date_plus_one_day_unverified"
                basis = f"CASE WHEN ({announced}) IS NOT NULL THEN 'announcement_date_plus_one_day' WHEN __origin='provenance_imported' THEN 'legacy_available_at_unverified' ELSE '{unknown_basis}' END"
            clauses, params = [], []
            if cutoff_utc:
                clauses.append("_source_available_at <= try_cast(? AS TIMESTAMPTZ)")
                params.append(cutoff_utc)
            if acquired_before is not None:
                clauses.append("_ingested_at <= try_cast(? AS TIMESTAMPTZ)")
                params.append(acquired_before)
            if codes is not None:
                if "ts_code" not in names:
                    return pd.DataFrame()
                clauses.append("(ts_code IN (SELECT unnest(?)) OR split_part(ts_code,'.',1) IN (SELECT unnest(?)))")
                params.extend([list(codes), list(codes)])
            key_parts = [f'cast(({dt(k)}) AS DATE)' if k.endswith("date") else f'"{k}"' for k in DATASETS[dataset].keys if k in names]
            if not key_parts:
                raise ValueError(f"{dataset}: no source identity columns")
            order = "try_cast(__fetched AS TIMESTAMPTZ) DESC,_ingested_at DESC,_row_content_hash DESC" if DATASETS[dataset].metadata else "_source_available_at DESC NULLS LAST,try_cast(__fetched AS TIMESTAMPTZ) DESC,_ingested_at DESC,_row_content_hash DESC"
            requested = names if columns is None else [x for x in columns if x in names]
            selected = [f'cast(({dt(x)}) AS DATE) AS "{x}"' if x.endswith("_date") else f'"{x}"' for x in requested]
            if include_provenance:
                selected.extend(["_source_date", "_source_available_at", "_availability_basis", "_ingested_at", "_shard_id", "_fetched_at", "_row_content_hash"])
                selected.append(f"(count(DISTINCT _row_content_hash) OVER(PARTITION BY {','.join(key_parts)},try_cast(__fetched AS TIMESTAMPTZ))>1) AS _same_vintage_conflict")
            if not selected:
                return pd.DataFrame()
            query = f"""WITH normalized AS (
                SELECT {','.join(normalized)},p.* EXCLUDE(__file)
                FROM lake_raw r JOIN lake_provenance p ON r.filename=p.__file
            ), timed AS (
                SELECT *,cast(({observed}) AS DATE) AS _source_date,
                       ({available}) AS _source_available_at,({basis}) AS _availability_basis,
                       try_cast(__ingested AS TIMESTAMPTZ) AS _ingested_at,__shard AS _shard_id,
                       try_cast(__fetched AS TIMESTAMPTZ) AS _fetched_at,
                       sha256(to_json(row({','.join(f'"{x}"' for x in names)}))) AS _row_content_hash
                FROM normalized
            ) SELECT {','.join(selected)} FROM timed
            {('WHERE ' + ' AND '.join(clauses)) if clauses else ''}
            QUALIFY row_number() OVER(PARTITION BY {','.join(key_parts)} ORDER BY {order})=1"""
            return db.execute(query, params).fetchdf()

    def record_frame(self, dataset: str, frame: pd.DataFrame, *, params: Mapping[str, Any] | None = None,
                     fetched_at: str | None = None, api_name: str | None = None) -> dict:
        """Archive an already acquired response (for bootstrap/import integrations)."""
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset {dataset!r}")
        with self._writer():
            result = self._store_page(dataset, frame, api_name=api_name or dataset,
                params=dict(params or {}), fetched_at=fetched_at)
        self._write_inventory_snapshot()
        return result

    @staticmethod
    def _response_mismatch(dataset: str, frame: pd.DataFrame, params: dict) -> str | None:
        if frame.empty:
            return None
        if params.get("ts_code") and "ts_code" in frame:
            if not frame["ts_code"].eq(params["ts_code"]).all():
                return "response contains codes outside requested ts_code"
        if dataset == "fund_basic":
            for key in ("market", "status"):
                if params.get(key) and key in frame and not frame[key].eq(params[key]).all():
                    return f"response ignored requested {key}"
        date_key = {"fund_nav": "nav_date", "fund_portfolio": "end_date", "trade_cal": "cal_date"}.get(dataset, "trade_date")
        if date_key in frame:
            dates = frame[date_key].map(_day)
            if dates.isna().any():
                return "response has invalid source dates"
            # end_date is a range boundary for portfolio, not an exact report date.
            if date_key in ("nav_date", "trade_date", "cal_date") and params.get(date_key) and not dates.eq(_day(params[date_key])).all():
                return f"response ignored requested {date_key}"
            if dataset == "fund_portfolio" and params.get("period") and not dates.eq(_day(params["period"])).all():
                return "response ignored requested period"
            if params.get("start_date") and (dates < _day(params["start_date"])).any():
                return "response ignored requested start_date"
            if params.get("end_date") and (dates > _day(params["end_date"])).any():
                return "response ignored requested end_date"
        return None

    def _write_inventory_snapshot(self) -> None:
        _atomic_bytes(self.root / "inventory.json", (_json(self.inventory()) + "\n").encode())

    def reset_interface(self, api_name: str) -> int:
        """Explicitly retry permission-blocked tasks after account access changes."""
        with self._db() as db:
            count = db.execute("SELECT count(*) FROM tasks WHERE api_name=? AND status='permission_blocked'", [api_name]).fetchone()[0]
            db.execute("DELETE FROM interfaces WHERE api_name=?", [api_name])
            db.execute("UPDATE tasks SET status='pending',last_error=NULL,not_before=0 WHERE api_name=? AND status='permission_blocked'", [api_name])
        return count

    def retry_tasks(self, api_name: str | None = None, *, include_truncated: bool = False) -> int:
        """Explicitly resume failed tasks while preserving raw pages and offsets.

        Truncation requires the caller's explicit opt-in because a pagination contract
        may need repairing first. Permission blocks remain an interface-level action.
        """
        statuses = ["failed", "truncated"] if include_truncated else ["failed"]
        where = "status IN (SELECT unnest(?))"
        params: list[Any] = [statuses]
        if api_name is not None:
            where += " AND api_name=?"
            params.append(api_name)
        with self._db() as db:
            count = db.execute(f"SELECT count(*) FROM tasks WHERE {where}", params).fetchone()[0]
            db.execute(f"""UPDATE tasks SET status='pending',attempts=0,not_before=0,
                last_error=NULL,updated_at=? WHERE {where}""", [_utcnow(), *params])
        self._write_inventory_snapshot()
        return count

    def run_budget(self, client: Any, *, max_queries: int = 100, max_seconds: float = 45,
                   min_interval: float = 1.0, max_attempts: int = 3,
                   rate_limit_cooldown: float = 61, max_pages_per_task: int = 1000) -> dict:
        """Run only the allowed query/time budget; persist checkpoints after every page.

        A request already in flight can outlive the wall budget by its socket timeout.
        Errors schedule a future retry; permission failures disable the entire interface.
        A repeated full page (ignored offset), oversized response, or page ceiling leaves
        an explicit ``truncated`` task, never a false completeness claim.
        """
        if max_queries < 0 or max_seconds <= 0 or not 0 <= min_interval <= 60 or max_attempts < 1 or rate_limit_cooldown < 0:
            raise ValueError("invalid collection budget/retry configuration")
        started = time.monotonic()
        deadline = started + max_seconds
        queries = pages = rows = errors = 0
        last_request = 0.0
        reason = "query_budget"
        with self._writer():
            while queries < max_queries:
                if time.monotonic() >= deadline:
                    reason = "time_budget"
                    break
                with self._db() as db:
                    cur = db.execute("""SELECT task_id,dataset,api_name,params_json,fields_json,page_size,
                        offset_rows,pages,previous_page_hash,attempts FROM tasks
                        WHERE status IN ('pending','retry') AND not_before<=?
                        AND api_name NOT IN (SELECT api_name FROM interfaces
                            WHERE status='permission_blocked' OR coalesce(blocked_until,0)>?)
                        ORDER BY priority DESC,created_at,task_id LIMIT 1""", [time.time(), time.time()])
                    row = cur.fetchone()
                if row is None:
                    reason = "no_ready_tasks"
                    break
                task_id, dataset, api, params_json, fields_json, limit, offset, task_pages, previous_hash, attempts = row
                gap = min_interval - (time.monotonic() - last_request)
                if gap > 0:
                    if time.monotonic() + gap >= deadline:
                        reason = "time_budget"
                        break
                    time.sleep(gap)
                params = json.loads(params_json)
                params.update(limit=limit, offset=offset)
                if time.monotonic() >= deadline:
                    reason = "time_budget"
                    break
                queries += 1
                last_request = time.monotonic()
                try:
                    effective_fields = json.loads(fields_json) or DEFAULT_FIELDS.get(api)
                    frame = client.query(api, params=params, fields=effective_fields)
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("client.query must return pandas.DataFrame")
                    normalized = self._normal_frame(dataset, frame)
                    current_hash = _hash(sorted(_json(_public(r)) for r in normalized.to_dict("records")))
                    error = self._response_mismatch(dataset, normalized, params)
                    if error:
                        status = "truncated"
                    elif len(frame) > limit:
                        status, error = "truncated", "response exceeded requested limit; pagination contract unverified"
                    elif len(frame) and previous_hash == current_hash:
                        status, error = "truncated", "repeated page: server may ignore offset"
                    elif len(frame) < limit:
                        status = "done"
                    elif task_pages + 1 >= max_pages_per_task:
                        status, error = "truncated", "page ceiling reached before an under-full terminal page"
                    else:
                        status = "pending"
                    self._store_page(dataset, frame, api_name=api, params=params, task_id=task_id,
                                     task_update={"offset": offset + len(frame), "status": status, "error": error}, fields=effective_fields)
                    with self._db() as db:
                        db.execute("""UPDATE interfaces SET status='ready',blocked_until=0,
                            rate_strikes=0,last_error=NULL,updated_at=?
                            WHERE api_name=? AND status='rate_limited'""", [_utcnow(), api])
                    pages += 1
                    rows += len(frame)
                    if pages % 10 == 0:
                        self._write_inventory_snapshot()
                except TushareQuotaError as exc:
                    errors += 1
                    message = f"{type(exc).__name__}: permission denied for {api}"
                    with self._db() as db:
                        db.execute("""INSERT INTO interfaces(api_name,status,last_error,updated_at)
                            VALUES (?,?,?,?) ON CONFLICT(api_name) DO UPDATE SET
                            status=excluded.status,last_error=excluded.last_error,
                            updated_at=excluded.updated_at""", [api, "permission_blocked", message, _utcnow()])
                        db.execute("UPDATE tasks SET status='permission_blocked',last_error=?,updated_at=? WHERE api_name=? AND status IN ('pending','retry')", [message, _utcnow(), api])
                except TushareRateLimitError as exc:
                    errors += 1
                    message = f"{type(exc).__name__}: rate limit on {api}"
                    with self._db() as db:
                        row = db.execute("SELECT coalesce(rate_strikes,0) FROM interfaces WHERE api_name=?", [api]).fetchone()
                        strikes = min(32, (row[0] if row else 0) + 1)
                        delay = min(3600, rate_limit_cooldown * (2 ** (strikes - 1)))
                        blocked_until = time.time() + delay
                        db.execute("""INSERT INTO interfaces(api_name,status,last_error,updated_at,blocked_until,rate_strikes)
                            VALUES (?,?,?,?,?,?) ON CONFLICT(api_name) DO UPDATE SET
                            status=excluded.status,last_error=excluded.last_error,
                            updated_at=excluded.updated_at,blocked_until=excluded.blocked_until,
                            rate_strikes=excluded.rate_strikes""",
                            [api, "rate_limited", message, _utcnow(), blocked_until, strikes])
                        # Quota-window waits do not consume the transport failure budget.
                        # Other pending tasks on this API are gated by the interface row.
                        db.execute("""UPDATE tasks SET status='retry',not_before=?,last_error=?,
                            updated_at=? WHERE task_id=?""", [blocked_until, message, _utcnow(), task_id])
                    reason = "rate_limit_cooldown"
                    break
                except (LakeBusyError, duckdb.Error):
                    # A catalog wait or storage/query failure is not a Tushare
                    # transport attempt. Preserve the checkpoint and let the worker
                    # retry busy errors or surface database faults explicitly.
                    raise
                except Exception as exc:
                    errors += 1
                    attempts += 1
                    delay = min(30, 2 ** attempts)
                    status = "retry" if attempts < max_attempts else "failed"
                    # Exception strings may contain request credentials. Persist only type,
                    # interface, and a bounded non-sensitive reason, never arbitrary reprs.
                    message = f"{type(exc).__name__}: collection error on {api}"
                    with self._db() as db:
                        db.execute("UPDATE tasks SET status=?,attempts=?,not_before=?,last_error=?,updated_at=? WHERE task_id=?", [status, attempts, time.time() + delay, message, _utcnow(), task_id])
        with self._db() as db:
            next_ready = db.execute("""SELECT min(greatest(t.not_before,coalesce(i.blocked_until,0)))
                FROM tasks t LEFT JOIN interfaces i USING(api_name)
                WHERE t.status IN ('pending','retry') AND coalesce(i.status,'')!='permission_blocked'""").fetchone()[0]
        retry_after = max(0, math.ceil(next_ready - time.time())) if next_ready is not None else 0
        self._write_inventory_snapshot()
        return {"queries": queries, "pages": pages, "rows_received": rows, "errors": errors,
                "elapsed_seconds": round(time.monotonic() - started, 3), "stop_reason": reason,
                "retry_after_seconds": retry_after,
                "inventory": self.inventory()}

    def inventory(self) -> dict:
        """Counts only: safe to print without exposing sealed research observations."""
        with self._db() as db:
            datasets = db.execute("""SELECT dataset,sum(n),max(c),sum(legacy_n),sum(acquired_n)
                FROM (SELECT dataset,row_count n,code_count c,row_count legacy_n,0 acquired_n
                      FROM archival_files UNION ALL
                      SELECT dataset,row_count,code_count,0,row_count FROM shards)
                GROUP BY dataset ORDER BY dataset""").fetchall()
            task_counts = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status ORDER BY status").fetchall())
            interfaces = [{"api_name": a, "status": s, "error": e,
                "blocked_until": b, "rate_strikes": r} for a, s, e, b, r in db.execute(
                "SELECT api_name,status,last_error,coalesce(blocked_until,0),coalesce(rate_strikes,0) FROM interfaces ORDER BY api_name").fetchall()]
            shards, raw_rows = db.execute("SELECT count(*),coalesce(sum(row_count),0) FROM shards").fetchone()
            quality = db.execute("SELECT dataset,quality_json FROM shards WHERE quality_json IS NOT NULL").fetchall()
            quality_counts: dict[str, dict[str, int]] = {}
            for dataset, counts_json in quality:
                counts = quality_counts.setdefault(dataset, {})
                for name, value in json.loads(counts_json).items():
                    counts[name] = counts.get(name, 0) + value
        return {"datasets": {name: {"rows": n, "funds_or_codes_lower_bound": codes,
                    "legacy_observations": legacy, "acquired_observations": acquired}
                    for name, n, codes, legacy, acquired in datasets},
                "tasks": task_counts, "interfaces": interfaces, "shards": shards,
                "acquired_quality_counts": quality_counts,
                "raw_rows_received": raw_rows,
                "row_count_basis": "archived observations including overlapping revisions; read_table deduplicates",
                "historical_execution_certified": False}
