"""Immutable, audited publication-evidence and duplicate curation.

Raw archives are never edited. Exact semantic duplicates collapse; different
disclosure/retrieval versions remain distinguishable. Ambiguous same-vintage
values and unresolved publication dates are quarantined, never averaged or
assigned a guessed date. Legacy available_at is not an original announcement.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import duckdb
import pandas as pd

from qmt.data.fund_lake import DatasetLake

POLICY_VERSION = "publication-evidence-dedup-v1"
FIELDS = {
    "fund_nav": ["ts_code", "nav_date", "ann_date", "unit_nav", "adj_nav", "accum_nav",
                 "accum_div", "net_asset", "total_netasset", "update_flag"],
    "fund_portfolio": ["ts_code", "end_date", "ann_date", "symbol", "mkv", "amount",
                       "stk_mkv_ratio", "stk_float_ratio"],
    "fund_div": ["ts_code", "ann_date", "imp_anndate", "base_date", "div_proc", "record_date",
                 "ex_date", "pay_date", "earpay_date", "net_ex_date", "div_cash", "base_unit",
                 "ear_distr", "ear_amount", "account_date", "base_year"],
}
STRINGS = {"ts_code", "symbol", "update_flag", "div_proc", "base_year"}
KEYS = {"fund_nav": ["ts_code", "nav_date", "ann_date"],
        "fund_portfolio": ["ts_code", "end_date", "symbol", "ann_date"],
        "fund_div": ["ts_code", "ann_date", "div_proc", "base_date", "record_date", "ex_date", "base_year"]}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str, allow_nan=False)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_sql(name):
    return f"coalesce(try_cast(try_strptime(cast(\"{name}\" AS VARCHAR),'%Y%m%d') AS DATE),try_cast(\"{name}\" AS DATE))"


def _copy(db, query, path):
    db.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])


def capture_sources(lake):
    with lake._db() as db:
        names = ["shard_id", "dataset", "raw_path", "fetched_at", "ingested_at", "row_count", "fingerprint"]
        rows = db.execute("SELECT " + ",".join(names) +
            " FROM shards WHERE row_count>0 AND dataset IN ('fund_nav','fund_div','fund_portfolio') ORDER BY dataset,shard_id").fetchall()
        references = db.execute("SELECT shard_id,dataset,fingerprint,ingested_at FROM shards WHERE dataset IN ('fund_basic','trade_cal') ORDER BY dataset,shard_id").fetchall()
        legacy = db.execute("SELECT dataset,row_count,byte_fingerprint FROM archival_files ORDER BY dataset,file_id").fetchall()
    return {"lake_root": str(lake.root), "sources": [dict(zip(names, r)) for r in rows],
            "references": references, "acquired_before": datetime.now(timezone.utc).isoformat(),
            "legacy_not_used_as_announcement_evidence": legacy}


def _register_raw(db, dataset, source):
    pages = [r for r in source["sources"] if r["dataset"] == dataset]
    provenance = pd.DataFrame([{"filename": str(Path(source["lake_root"]) / r["raw_path"]),
        "fetched_at": r["fetched_at"], "ingested_at": r["ingested_at"], "source_shard_id": r["shard_id"]}
        for r in pages])
    db.register("provenance", provenance)
    db.read_parquet(provenance.filename.tolist(), union_by_name=True, filename=True).create_view("raw", replace=True)
    present = {r[0] for r in db.execute("DESCRIBE raw").fetchall()}
    columns = FIELDS[dataset]
    expressions = []
    for name in columns:
        if name not in present:
            value = "NULL::DATE" if name.endswith("date") else "NULL::VARCHAR" if name in STRINGS else "NULL::DOUBLE"
        elif name.endswith("date"):
            value = _date_sql(name)
        elif name in STRINGS:
            value = f'nullif(trim(cast("{name}" AS VARCHAR)),\'\')'
        else:
            value = f'try_cast("{name}" AS DOUBLE)'
        expressions.append(f'{value} AS "{name}"')
    return ("SELECT " + ",".join(expressions) +
        ",try_cast(p.fetched_at AS TIMESTAMPTZ) AS fetched_at,try_cast(p.ingested_at AS TIMESTAMPTZ) AS ingested_at,"
        "p.source_shard_id,r.filename AS source_file FROM raw r JOIN provenance p USING(filename)")


def _curate_dataset(db, dataset, source, staging, *, prepared=None):
    columns=FIELDS[dataset]
    if not prepared and not any(r["dataset"]==dataset for r in source["sources"]):
        return {"status": "NO_ACQUIRED_ROWS", "raw_rows": 0}
    db.execute("DROP TABLE IF EXISTS stage")
    if prepared:
        fields = FIELDS[dataset]+["fetched_at","ingested_at","source_shard_id","source_file"]
        db.execute("CREATE TABLE stage AS SELECT " + ",".join('"'+c+'"' for c in fields) +
                   " FROM read_parquet(?,hive_partitioning=false)", [list(map(str,prepared))])
    else:
        db.execute("CREATE TABLE stage AS " + _register_raw(db,dataset,source))
    raw_rows = db.execute("SELECT count(*) FROM stage").fetchone()[0]
    missing_before = db.execute("SELECT count(*) FROM stage WHERE ann_date IS NULL").fetchone()[0]
    db.execute("ALTER TABLE stage ADD COLUMN publication_basis VARCHAR DEFAULT 'original_ann_date'")
    db.execute("ALTER TABLE stage ADD COLUMN announcement_evidence_shard VARCHAR")
    db.execute("ALTER TABLE stage ADD COLUMN original_ann_date DATE")
    db.execute("UPDATE stage SET original_ann_date=ann_date")
    repaired = 0
    if dataset == "fund_nav" and missing_before:
        # A date can transfer only between identical fund/date/unit+adjusted NAV
        # observations. Donors with impossible dates or invalid prices cannot help.
        db.execute("""CREATE OR REPLACE TABLE targets AS
            SELECT DISTINCT ts_code,nav_date,unit_nav,adj_nav FROM stage WHERE ann_date IS NULL""")
        db.execute("""CREATE OR REPLACE TABLE donors AS
            SELECT s.ts_code,s.nav_date,s.unit_nav,s.adj_nav,max(s.ann_date) AS ann_date,
                   arg_max(s.source_shard_id,row(s.ann_date,s.source_shard_id)) AS evidence_shard
            FROM stage s JOIN targets t USING(ts_code,nav_date,unit_nav,adj_nav)
            WHERE s.ann_date>=s.nav_date AND s.ann_date<=cast(s.fetched_at AT TIME ZONE 'Asia/Shanghai' AS DATE)
              AND isfinite(s.unit_nav) AND s.unit_nav>0 AND isfinite(s.adj_nav) AND s.adj_nav>0
            GROUP BY 1,2,3,4""")
        db.execute("""UPDATE stage AS s SET ann_date=d.ann_date,
            publication_basis='matching_same_quote_announcement',announcement_evidence_shard=d.evidence_shard
            FROM donors d WHERE s.ann_date IS NULL AND s.ts_code=d.ts_code AND s.nav_date=d.nav_date
              AND s.unit_nav=d.unit_nav AND s.adj_nav=d.adj_nav""")
        repaired = db.execute("SELECT count(*) FROM stage WHERE publication_basis='matching_same_quote_announcement'").fetchone()[0]
    db.execute("ALTER TABLE stage ADD COLUMN reason VARCHAR")
    if dataset=="fund_nav":
        db.execute("ALTER TABLE stage ADD COLUMN source_found_date DATE")
        db.execute("UPDATE stage s SET source_found_date=k.source_found_date FROM known_funds k WHERE s.ts_code=k.ts_code")
    observed = "nav_date" if dataset == "fund_nav" else "end_date" if dataset == "fund_portfolio" else None
    checks = ["WHEN ts_code IS NULL THEN 'missing_identity'", "WHEN ann_date IS NULL THEN 'missing_announcement'",
              "WHEN ann_date>cast(fetched_at AT TIME ZONE 'Asia/Shanghai' AS DATE) THEN 'announcement_after_retrieval'"]
    if observed:
        checks += [f"WHEN {observed} IS NULL THEN 'missing_observation_date'",
                   f"WHEN ann_date<{observed} THEN 'announcement_before_observation'"]
    if dataset == "fund_nav":
        checks += ["WHEN NOT coalesce(isfinite(unit_nav) AND unit_nav>0 AND isfinite(adj_nav) AND adj_nav>0,FALSE) THEN 'invalid_nav'"]
        checks += ["WHEN source_found_date IS NOT NULL AND nav_date<source_found_date THEN 'valuation_before_recorded_inception'"]
    if dataset == "fund_portfolio":
        checks += ["WHEN symbol IS NULL THEN 'missing_security_identity'"]
    db.execute("UPDATE stage SET reason=CASE " + " ".join(checks) + " ELSE NULL END")
    hash_values = ["unit_nav", "adj_nav"] if dataset == "fund_nav" else [c for c in columns if c not in KEYS[dataset]]
    full_hash = "sha256(to_json(row(" + ",".join('"'+c+'"' for c in columns) + ")))"
    value_hash = "sha256(to_json(row(" + ",".join('"'+c+'"' for c in hash_values) + ")))"
    db.execute("ALTER TABLE stage ADD COLUMN original_payload_hash VARCHAR")
    db.execute("UPDATE stage SET original_payload_hash=" + full_hash)
    merged_rows = 0
    if dataset == "fund_div":
        # Same-event rows often differ only by a missing net_ex_date. Fill only
        # complementary nulls when EVERY known field agrees. The base date/year
        # are identity fields: six planned payouts cannot become one event.
        key = KEYS[dataset] + ["fetched_at"]
        values = [c for c in columns if c not in KEYS[dataset]]
        db.execute("CREATE OR REPLACE TABLE complementary AS SELECT " + ",".join(key) + "," +
            ",".join(f'max("{c}") AS "fill_{c}"' for c in values) +
            " FROM stage WHERE reason IS NULL GROUP BY " + ",".join(key) +
            " HAVING " + " AND ".join(f'count(DISTINCT "{c}")<=1' for c in values))
        match = " AND ".join(f's."{c}" IS NOT DISTINCT FROM g."{c}"' for c in key)
        has_fill = " OR ".join(f'(s."{c}" IS NULL AND g."fill_{c}" IS NOT NULL)' for c in values)
        merged_rows = db.execute("SELECT count(*) FROM stage s JOIN complementary g ON " + match +
                                " WHERE s.reason IS NULL AND (" + has_fill + ")").fetchone()[0]
        db.execute("UPDATE stage s SET " + ",".join(f'"{c}"=coalesce(s."{c}",g."fill_{c}")' for c in values) +
                   " FROM complementary g WHERE " + match + " AND s.reason IS NULL")
    db.execute("ALTER TABLE stage ADD COLUMN payload_hash VARCHAR")
    db.execute("ALTER TABLE stage ADD COLUMN economic_hash VARCHAR")
    db.execute("UPDATE stage SET payload_hash=" + full_hash + ",economic_hash=" + value_hash)
    keys = KEYS[dataset]
    db.execute("CREATE OR REPLACE TABLE conflicts AS SELECT " + ",".join(keys) +
        ",fetched_at FROM stage WHERE reason IS NULL GROUP BY " + ",".join(keys) +
        ",fetched_at HAVING count(DISTINCT economic_hash)>1")
    same_key = " AND ".join(f's."{c}" IS NOT DISTINCT FROM c."{c}"' for c in keys + ["fetched_at"])
    db.execute("UPDATE stage s SET reason='same_vintage_conflicting_values' FROM conflicts c WHERE " + same_key + " AND s.reason IS NULL")
    conflict_groups = db.execute("SELECT count(*) FROM conflicts").fetchone()[0]
    before_dedup = db.execute("SELECT count(*) FROM stage WHERE reason IS NULL").fetchone()[0]
    original_distinct = (db.execute("SELECT count(DISTINCT original_payload_hash) FROM stage WHERE reason IS NULL").fetchone()[0]
                         if dataset=="fund_div" else None)
    _copy(db, "SELECT * FROM stage WHERE reason IS NOT NULL ORDER BY ts_code,ann_date,source_shard_id,payload_hash,original_payload_hash",
          staging / (dataset + "_quarantine.parquet"))
    # Distinct economic/disclosure versions survive. Only exactly equal canonical
    # payloads collapse. Their observation spans and counts preserve audit context;
    # the immutable raw receipts remain the authority for retrieval-time replay.
    _copy(db, """SELECT * EXCLUDE(economic_hash,reason),
         min(ingested_at) OVER(PARTITION BY payload_hash) AS first_observed_at,
         max(ingested_at) OVER(PARTITION BY payload_hash) AS last_observed_at,
         count(*) OVER(PARTITION BY payload_hash) AS observation_count
         FROM stage WHERE reason IS NULL
         QUALIFY row_number() OVER(PARTITION BY payload_hash ORDER BY fetched_at DESC,
             CASE WHEN publication_basis='original_ann_date' THEN 0 ELSE 1 END,
             ingested_at DESC,source_shard_id DESC,original_payload_hash DESC)=1
         ORDER BY ts_code,ann_date,payload_hash""", staging / (dataset + ".parquet"))
    clean = db.execute("SELECT count(*) FROM read_parquet(?)", [str(staging / (dataset + ".parquet"))]).fetchone()[0]
    if original_distinct is None:
        original_distinct=clean
    quarantine = {r: n for r, n in db.execute("SELECT reason,count(*) FROM stage WHERE reason IS NOT NULL GROUP BY reason ORDER BY reason").fetchall()}
    assert raw_rows == before_dedup + sum(quarantine.values())
    assert clean <= before_dedup
    return {"status": "CURATED", "raw_rows": raw_rows, "missing_announcements_raw": missing_before,
            "repaired_announcement_rows": repaired, "quarantined_rows": sum(quarantine.values()),
            "missing_announcement_distinct_keys": db.execute("SELECT count(*) FROM (SELECT " +
                ",".join(KEYS[dataset]) + " FROM stage WHERE reason='missing_announcement' GROUP BY " +
                ",".join(KEYS[dataset]) + ")").fetchone()[0],
            "quarantine_reasons": quarantine, "same_vintage_conflict_groups": conflict_groups,
            "exact_duplicates_removed": before_dedup - original_distinct,
            "complementary_duplicates_collapsed": original_distinct - clean,
            "complementary_rows_completed": merged_rows, "clean_distinct_versions": clean,
            "clean_unique_source_codes": db.execute("SELECT count(DISTINCT ts_code) FROM read_parquet(?)", [str(staging/(dataset+'.parquet'))]).fetchone()[0]}


def _curate_bounded(db,dataset,source,staging,*,partition_threshold):
    pages=[r for r in source["sources"] if r["dataset"]==dataset]
    if dataset=="fund_div" or sum(r["row_count"] for r in pages)<=partition_threshold:
        return _curate_dataset(db,dataset,source,staging)
    # One streaming pass partitions by valuation/report month and source-code
    # bucket. Every identity and all its duplicate/repair candidates share a
    # partition. Avoid multi-million-row UPDATE version chains in DuckDB memory.
    observed="nav_date" if dataset=="fund_nav" else "end_date"
    query=_register_raw(db,dataset,source)
    parts=staging/("_input_"+dataset)
    db.execute("CREATE OR REPLACE VIEW canonical AS "+query)
    db.execute("COPY (SELECT *,coalesce(strftime("+observed+",'%Y-%m'),'unknown')||'-'||"
               "cast(coalesce(hash(ts_code)%4,0) AS VARCHAR) AS _bucket FROM canonical) TO ? "
               "(FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(_bucket))",[str(parts)])
    outputs=staging/("_output_"+dataset)
    outputs.mkdir()
    reports=[]
    for folder in sorted(p for p in parts.iterdir() if p.is_dir()):
        output=outputs/folder.name
        output.mkdir()
        reports.append(_curate_dataset(db,dataset,source,output,prepared=sorted(folder.glob('*.parquet'))))
        db.execute("DROP TABLE stage")
        db.execute("CHECKPOINT")
    db.execute("SET threads=1")
    db.execute("SET preserve_insertion_order=true")
    for suffix in [".parquet","_quarantine.parquet"]:
        files=[str(p) for p in sorted(outputs.glob('*/*'+suffix)) if p.name==dataset+suffix]
        # Input files and rows have deterministic partition ordering. No global
        # wide-row sort is needed; evaluation later projects and sorts NAVs.
        db.read_parquet(files,union_by_name=True,hive_partitioning=False).create_view('merged_parts',replace=True)
        _copy(db,'SELECT * FROM merged_parts',staging/(dataset+suffix))
        db.execute('DROP VIEW merged_parts')
    combined={"status":"CURATED","bounded_partitions":len(reports)}
    for key in reports[0]:
        if key not in {"status","quarantine_reasons","clean_unique_source_codes"}:
            combined[key]=sum(r[key] for r in reports)
    combined["quarantine_reasons"]={}
    for r in reports:
        for reason,n in r["quarantine_reasons"].items():
            combined["quarantine_reasons"][reason]=combined["quarantine_reasons"].get(reason,0)+n
    combined["clean_unique_source_codes"]=db.execute("SELECT count(DISTINCT ts_code) FROM read_parquet(?)",
        [str(staging/(dataset+'.parquet'))]).fetchone()[0]
    db.execute("SET preserve_insertion_order=false")
    db.execute("SET threads=2")
    db.execute("DROP VIEW canonical")
    shutil.rmtree(parts)
    shutil.rmtree(outputs)
    return combined


def build_quality_bundle(lake, output_dir, *, sources=None, partition_threshold=500000):
    lake = lake if isinstance(lake, DatasetLake) else DatasetLake(lake)
    source = sources or capture_sources(lake)
    code_hash = _hash_file(__file__)
    source_hash = hashlib.sha256(_json(source).encode()).hexdigest()
    identifier = "quality-" + hashlib.sha256((source_hash+code_hash).encode()).hexdigest()[:20]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / identifier
    if target.exists():
        return verify_quality_bundle(target)
    staging = Path(tempfile.mkdtemp(prefix=".quality-", dir=output))
    db = duckdb.connect(str(staging / "work.duckdb"))
    db.execute("SET memory_limit='3GB'")
    db.execute("SET threads=2")
    db.execute("SET preserve_insertion_order=false")
    db.execute("SET partitioned_write_max_open_files=8")
    try:
        (staging / "sources.json").write_text(_json(source)+"\n")
        basic = lake.read_table("fund_basic", acquired_before=source["acquired_before"])
        calendar = lake.read_table("trade_cal", acquired_before=source["acquired_before"])
        found=pd.to_datetime(basic.get('found_date',pd.Series(pd.NaT,index=basic.index)),errors='coerce',format='mixed')
        known_funds=pd.DataFrame({'ts_code':basic.ts_code,'source_found_date':found})
        if '_same_vintage_conflict' in basic:
            known_funds.loc[basic._same_vintage_conflict.fillna(False),'source_found_date']=pd.NaT
        db.register('known_funds',known_funds)
        results = {}
        for dataset in FIELDS:
            results[dataset] = _curate_bounded(db, dataset, source, staging, partition_threshold=partition_threshold)
            print(_json({"dataset": dataset, **results[dataset]}), file=sys.stderr, flush=True)
        db.close()
        # Frozen references are separate from historical data eligibility.
        basic.to_parquet(staging / "fund_basic.parquet", index=False)
        calendar.to_parquet(staging / "trade_cal.parquet", index=False)
        for path in staging.glob("work.duckdb*"):
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
        shutil.copyfile(__file__, staging / "quality_source.py")
        files = {p.name: _hash_file(p) for p in sorted(staging.iterdir()) if p.is_file()}
        manifest = {"quality_id": identifier, "policy_version": POLICY_VERSION, "source_hash": source_hash,
                    "code_hash": code_hash, "created_at": datetime.now(timezone.utc).isoformat(),
                    "datasets": results, "files": files, "legacy_announcement_inference_used": False,
                    "raw_archives_changed": False, "evidence_tier": "vendor_publication_evidence_not_original_vintages",
                    "rules": ["announcement repair requires same source code, valuation date, unit and adjusted NAV",
                              "unknown announcement dates never become valuation+one day",
                              "exact semantic duplicates collapse; disclosed/retrieved revisions are retained",
                              "same-vintage economic conflicts and invalid records are quarantined",
                              "NAV predating the currently recorded exact-code inception is quarantined as a metadata mismatch",
                              "current metadata and provider historical revisions are not certified historical vintages"]}
        manifest["manifest_hash"] = hashlib.sha256(_json(manifest).encode()).hexdigest()
        (staging / "manifest.json").write_text(_json(manifest)+"\n")
        os.rename(staging, target)
    except BaseException:
        db.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**manifest, "quality_path": str(target)}


def verify_quality_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if hashlib.sha256(_json({k:v for k,v in manifest.items() if k!="manifest_hash"}).encode()).hexdigest()!=manifest["manifest_hash"]:
        raise ValueError("quality manifest hash mismatch")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or _hash_file(path / name) != digest:
            raise ValueError("quality artifact hash mismatch: " + name)
    return {**manifest, "quality_path": str(path)}


def quality_evaluation_inputs(path, *, seal="2024-12-10", fund_type="股票型"):
    """Physically select published pre-seal versions, then choose visible quotes."""
    from qmt.evaluation import DEFAULT_SEALED_FROM
    if pd.Timestamp(seal)>pd.Timestamp(DEFAULT_SEALED_FROM):
        raise ValueError("sealed historical region cannot be opened")
    manifest = verify_quality_bundle(path)
    path = Path(manifest["quality_path"])
    from qmt.factor.cost_benchmark import domestic_fund_mask
    basic = pd.read_parquet(path / "fund_basic.parquet")
    eligible = basic[basic.fund_type.eq(fund_type) & basic.market.eq("O") & domestic_fund_mask(basic["name"])]
    codes = eligible.ts_code.dropna().astype(str).unique().tolist()
    db = duckdb.connect()
    try:
        nav = db.execute("""SELECT n.ts_code,n.nav_date,n.ann_date,n.adj_nav,n.publication_basis,
            n.announcement_evidence_shard FROM read_parquet(?) n
          WHERE nav_date<cast(? AS DATE) AND ann_date<cast(? AS DATE)
            AND ts_code IN (SELECT unnest(?))
            AND NOT EXISTS (SELECT 1 FROM read_parquet(?) q
              WHERE q.reason='same_vintage_conflicting_values' AND q.ts_code=n.ts_code
                AND q.nav_date=n.nav_date AND q.ann_date<cast(? AS DATE) AND q.fetched_at>=n.fetched_at)
          QUALIFY row_number() OVER(PARTITION BY ts_code,nav_date
            ORDER BY ann_date DESC,fetched_at DESC,ingested_at DESC,payload_hash DESC)=1
          ORDER BY ts_code,nav_date""", [str(path/"fund_nav.parquet"),seal,seal,codes,
                                         str(path/"fund_nav_quarantine.parquet"),seal]).fetchdf()
    finally:
        db.close()
    return nav, basic, pd.read_parquet(path / "trade_cal.parquet"), manifest
