"""Reproducible broad-universe NAV research, separated from execution approval.

Calendar positions are real exchange sessions, never weekdays. No NAV or outcome
is forward-filled. Signals use only strictly complete, timely published backward
windows. Labels mature at the full registered horizon and never enter the seal.
Each arm sees one common ex-ante universe; missing outcomes cannot cause portfolio
reselection. Current vendor metadata and revised NAV history are explicitly not a
historical-vintage/PIT universe, even when conservative disclosure lags are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
import platform
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from qmt.factor.dsl import BIN_OPS, CS_OPS, EW_UNARY, FormulaDSL
from qmt.factor.types import BaseFactorPanel, GateStatus, SearchConfig
from qmt.stats import newey_west_tstat, spearman

DEFAULT_SEALED_FROM = "2024-12-10"
ENGINE_VERSION = "fund-nav-ranking-v2"
BASELINES = ("low_vol_60", "mom_60", "mom_20", "reversal_5", "trend_60", "equal_weight")
LEAVES = ("mom20", "mom60", "vol20", "vol60", "reversal5", "trend60", "dd60")
LIMITATIONS = [
    "Exploratory adjusted-NAV ranking research; no subscription/redemption fees, dealing-status, confirmation or dividend cash-flow simulation. Returns are gross NAV proxies, not executable fee-net alpha.",
    "Current fund_basic metadata is not historically versioned. Type/manager/cohort membership may differ from historical facts; current status is never used to remove liquidated funds.",
    "Vendor adjusted NAV history may contain revisions and has no complete original publication vintages; conservative publication lags do not prove historical PIT correctness.",
    "Only collected funds are represented. Liquidated, merged and never-collected funds can be missing; survivorship and collection coverage remain unresolved.",
    "Share classes are not reliably mapped to underlying portfolios. Correlated A/C classes can overstate cross-sectional breadth; manager counts do not resolve this duplication.",
    "Year folds and explicit regime intervals are frozen before this run; previously viewed historical development periods are not fresh untouched confirmation evidence.",
]


def _clean(value):
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _json(value) -> str:
    return json.dumps(_clean(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _engine_hash() -> str:
    from qmt.factor import dsl, types
    from qmt.factor import cost_benchmark
    import qmt.stats
    paths = [Path(__file__), Path(dsl.__file__), Path(types.__file__), Path(qmt.stats.__file__),
             Path(cost_benchmark.__file__)]
    return hashlib.sha256(_json({p.name: _hash_file(p) for p in paths}).encode()).hexdigest()


def _runtime_versions() -> dict:
    return {"python": platform.python_version(), **{name: version(name) for name in ("pandas", "numpy", "pyarrow")}}


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("artifact id must be a safe identifier of at most 96 characters")
    return value


def _dates(values) -> pd.Series:
    # Tushare YYYYMMDD integers must not become nanoseconds since epoch.
    series = pd.Series(values, copy=False)
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series)
    else:
        out = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return out.dt.normalize()


def _column(frame: pd.DataFrame, names: Sequence[str], *, required=True) -> str | None:
    for name in names:
        if name in frame:
            return name
    if required:
        raise ValueError(f"missing one of required columns: {', '.join(names)}")
    return None


def _identities(frame: pd.DataFrame) -> pd.Series:
    """Coalesce lake source/canonical codes without creating duplicate columns."""
    _column(frame, ("ts_code", "fund_code"))
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name in ("ts_code", "fund_code"):
        if name in frame:
            value = frame[name].astype("string").str.strip().replace("", pd.NA)
            numeric = value.str.fullmatch(r"\d{1,6}", na=False)
            value.loc[numeric] = value.loc[numeric].str.zfill(6) + ".OF"
            result = result.fillna(value)
    return result


def _seal(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value).normalize()
    if stamp > pd.Timestamp(DEFAULT_SEALED_FROM):
        raise ValueError("sealed historical region cannot be opened by this research platform")
    return stamp


def _open_sessions(calendar: pd.DataFrame | Sequence) -> pd.DatetimeIndex:
    if isinstance(calendar, pd.DataFrame):
        column = _column(calendar, ("cal_date", "trade_date", "date"))
        frame = calendar
        if "exchange" in frame and frame["exchange"].nunique() > 1:
            raise ValueError("provide exactly one exchange calendar")
        if "is_open" in frame:
            frame = frame[pd.to_numeric(frame["is_open"], errors="coerce").eq(1)]
        values = frame[column]
    else:
        values = calendar
    dates = _dates(values)
    if dates.isna().any() or len(dates) < 2:
        raise ValueError("an actual exchange-session calendar with valid dates is required")
    return pd.DatetimeIndex(dates.unique()).sort_values()


def prepare_snapshot(nav: pd.DataFrame, fund_basic: pd.DataFrame,
                     calendar: pd.DataFrame | Sequence, output_dir: str | Path, *,
                     snapshot_id: str | None = None, sealed_from=DEFAULT_SEALED_FROM,
                     metadata_basis="current_snapshot", source="tushare", value_col="adj_nav",
                     calendar_source="tushare.trade_cal", provenance: Mapping | None = None) -> dict:
    """Write a sealed immutable snapshot; never mutate the acquisition cache.

    Required NAV fields: ts_code/fund_code, nav_date/end_date, adjusted NAV and
    ann_date/available_at. Dates without publication evidence are rejected from
    the snapshot and counted. An announcement with only a day is usable on the
    *next exchange session*, including late announcements. Identical duplicates
    collapse conservatively; conflicting adjusted NAV revisions fail closed.
    """
    if metadata_basis != "current_snapshot":
        raise ValueError("only explicit current_snapshot metadata is supported; PIT metadata needs a versioned adapter")
    if not calendar_source or not source:
        raise ValueError("source and calendar_source must be recorded")
    seal = _seal(sealed_from)
    sessions = _open_sessions(calendar)
    sessions = sessions[sessions < seal]
    if len(sessions) < 2:
        raise ValueError("calendar has no usable pre-seal sessions")
    valuation = _column(nav, ("nav_date", "end_date"))
    publication = _column(nav, ("ann_date", "publication_date", "available_at", "_source_available_at"))
    if value_col not in nav:
        raise ValueError(f"adjusted NAV column {value_col!r} is required; unit NAV is not a substitute")
    basic = fund_basic.drop(columns=[c for c in ("ts_code", "fund_code") if c in fund_basic]).copy()
    basic["fund_code"] = _identities(fund_basic)
    if basic["fund_code"].isna().any():
        raise ValueError("fund_basic codes must be nonmissing")
    if basic["fund_code"].duplicated().any():
        # Identical metadata duplicates can collapse. Distinct revisions require
        # collection-time provenance, never an arbitrary input-row order choice.
        basic = basic.drop_duplicates()
        if basic["fund_code"].duplicated().any():
            if "_ingested_at" not in basic or basic["_ingested_at"].isna().any():
                raise ValueError("conflicting current fund_basic revisions need _ingested_at provenance")
            vintages = pd.to_datetime(basic["_ingested_at"], errors="coerce", utc=True)
            if vintages.isna().any():
                raise ValueError("invalid current metadata collection-time provenance")
            latest = vintages.groupby(basic.fund_code).transform("max")
            basic = basic.loc[vintages.eq(latest)]
            if basic.fund_code.duplicated().any():
                raise ValueError("conflicting metadata rows at the same latest collection vintage")
    # Stringified metadata avoids heterogeneous vendor object columns in parquet;
    # these fields are descriptive, never causal historical features.
    for column in basic.columns:
        basic[column] = basic[column].astype("string")
    basic = basic.sort_values("fund_code").reset_index(drop=True)
    frame = pd.DataFrame({"fund_code": _identities(nav),
                          "nav_date": _dates(nav[valuation]),
                          "publication_date": _dates(nav[publication]),
                          "adj_nav": pd.to_numeric(nav[value_col], errors="coerce")})
    if "available_at" in nav and publication != "available_at":
        observed_availability = _dates(nav["available_at"])
        # Respect any later provider availability in addition to announcement.
        frame["publication_date"] = pd.concat([frame.publication_date, observed_availability], axis=1).max(axis=1)
    # A schema-unioned lake can have ann_date only on newly collected shards.
    # Its synthesized _source_available_at is fallback evidence, not a reason to
    # double-lag rows that already carry the original announcement date.
    for fallback in ("publication_date", "available_at", "_source_available_at"):
        if fallback in nav:
            frame["publication_date"] = frame.publication_date.fillna(_dates(nav[fallback]))
    invalid = (frame.fund_code.isna() | frame.nav_date.isna() | frame.publication_date.isna()
               | ~np.isfinite(frame.adj_nav) | frame.adj_nav.le(0))
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    unknown = ~frame.fund_code.isin(basic.fund_code)
    unknown_count = int(unknown.sum())
    frame = frame.loc[~unknown].copy()
    # Exact valuation dates only: weekend NAVs must not shift trading horizons.
    in_calendar = frame.nav_date.isin(sessions)
    outside_count = int((~in_calendar).sum())
    frame = frame.loc[in_calendar].copy()
    publication_day = frame[["nav_date", "publication_date"]].max(axis=1)
    positions = sessions.searchsorted(publication_day.to_numpy("datetime64[ns]"), side="right")
    mature = positions < len(sessions)
    not_usable_count = int((~mature).sum())
    frame = frame.loc[mature].copy()
    frame["usable_date"] = sessions.take(positions[mature]).to_numpy()
    group = frame.groupby(["fund_code", "nav_date"], sort=False)
    if (group.adj_nav.nunique() > 1).any():
        raise ValueError("conflicting adjusted NAV values for one fund/date; resolve source vintages explicitly")
    before = len(frame)
    frame = group.agg(adj_nav=("adj_nav", "first"),
                      publication_date=("publication_date", "max"),
                      usable_date=("usable_date", "max")).reset_index()
    frame = frame.sort_values(["fund_code", "nav_date"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no valid pre-seal NAV records remain")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=output))
    try:
        frame.to_parquet(staging / "nav.parquet", index=False)
        basic.to_parquet(staging / "fund_basic.parquet", index=False)
        pd.DataFrame({"trade_date": sessions}).to_parquet(staging / "calendar.parquet", index=False)
        manifest = {"schema_version": 1, "kind": "exploratory_nav_snapshot", "source": source,
                    "sealed_from": str(seal.date()), "metadata_basis": metadata_basis,
                    "calendar_source": calendar_source, "provenance": dict(provenance or {}),
                    "availability_policy": "next exchange session after max(valuation day, publication/provider availability day); no intraday publication inference",
                    "rows": len(frame), "funds": int(frame.fund_code.nunique()),
                    "metadata_funds": len(basic), "first_nav_date": str(frame.nav_date.min().date()),
                    "last_nav_date": str(frame.nav_date.max().date()),
                    "quality": {"input_nav_rows": len(nav), "invalid_rows_rejected": invalid_count,
                                "unknown_metadata_rows_rejected": unknown_count,
                                "noncalendar_or_postseal_rows_rejected": outside_count,
                                "unavailable_before_seal_rows_rejected": not_usable_count,
                                "identical_duplicates_collapsed": before - len(frame)},
                    "files": {name: _hash_file(staging / name) for name in
                              ("nav.parquet", "fund_basic.parquet", "calendar.parquet")},
                    "limitations": LIMITATIONS}
        digest = hashlib.sha256(_json(manifest).encode()).hexdigest()
        identifier = _safe_id(snapshot_id or f"snapshot-{digest[:20]}")
        destination = output / identifier
        if destination.exists():
            raise FileExistsError(f"snapshot already exists: {destination}")
        manifest.update(snapshot_id=identifier, snapshot_hash=digest)
        (staging / "manifest.json").write_text(_json(manifest) + "\n")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**manifest, "snapshot_path": str(destination.resolve())}


def verify_snapshot(snapshot_path: str | Path) -> dict:
    path = Path(snapshot_path)
    manifest = json.loads((path / "manifest.json").read_text())
    content = {k: v for k, v in manifest.items() if k not in ("snapshot_id", "snapshot_hash")}
    if hashlib.sha256(_json(content).encode()).hexdigest() != manifest["snapshot_hash"]:
        raise ValueError("snapshot manifest hash mismatch")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or _hash_file(path / name) != expected:
            raise ValueError(f"snapshot artifact hash mismatch: {name}")
    _seal(manifest["sealed_from"])
    return manifest


@dataclass(frozen=True)
class EvaluationProtocol:
    evaluation_start: str = "2015-01-01"
    evaluation_end: str = "2024-12-09"
    horizon_sessions: int = 40
    decision_stride_sessions: int = 40
    signal_lag_sessions: int = 2
    warmup_sessions: int = 60
    minimum_training_sessions: int = 252
    embargo_sessions: int = 1
    fund_types: tuple[str, ...] = ("股票型",)
    markets: tuple[str, ...] = ("O",)
    domestic_cny_only: bool = True
    min_cross_section: int = 20
    min_outcome_coverage: float = 0.95
    top_k: int = 5
    min_scored_dates: int = 12
    min_scored_years: int = 3
    min_date_coverage: float = 0.8
    bootstrap_repetitions: int = 499
    bootstrap_block_decisions: int = 2
    seed: int = 20261015
    candidates: Mapping[str, str] = field(default_factory=dict)
    # Optional externally specified calendar regimes, never inferred from outcomes.
    regimes: tuple[tuple[str, str, str], ...] = ()
    evidence_tier: str = "exploratory_nav_ranking"

    def __post_init__(self):
        if not 20 <= self.horizon_sessions <= 60:
            raise ValueError("fund outcome horizon must be between 20 and 60 exchange sessions")
        if self.decision_stride_sessions != self.horizon_sessions:
            raise ValueError("decision stride must equal horizon: no overlapping outcomes or unobserved turnover gaps")
        if not 1 <= self.signal_lag_sessions <= 3:
            raise ValueError("signal lag must be one to three exchange sessions")
        if self.warmup_sessions < 60 or self.minimum_training_sessions < 60 or self.embargo_sessions < 1:
            raise ValueError("at least 60 warmup/training sessions and one embargo session required")
        if self.min_cross_section < 5 or self.top_k < 1 or self.min_scored_dates < 3 or self.min_scored_years < 1:
            raise ValueError("insufficient cross-section, date/year requirements or invalid top_k")
        if not 0 < self.min_outcome_coverage <= 1 or not 0 < self.min_date_coverage <= 1:
            raise ValueError("coverage thresholds must be in (0, 1]")
        if self.bootstrap_repetitions < 99 or self.bootstrap_block_decisions < 1:
            raise ValueError("at least 99 bootstrap replicates and positive block length required")
        if pd.Timestamp(self.evaluation_start) > pd.Timestamp(self.evaluation_end):
            raise ValueError("evaluation dates must be chronological")
        if self.evidence_tier != "exploratory_nav_ranking":
            raise ValueError("NAV-only evidence cannot be promoted to executable net-alpha evaluation")
        if len(self.candidates) > 16:
            raise ValueError("register at most 16 explicit candidates per protocol")
        for name in self.candidates:
            _safe_id(name)
            if name in BASELINES:
                raise ValueError("candidate name collides with a fixed baseline")
        names = set()
        for name, start, end in self.regimes:
            _safe_id(name)
            if name in names or pd.Timestamp(start) > pd.Timestamp(end):
                raise ValueError("regime names must be unique and intervals chronological")
            names.add(name)


def _protocol(value: EvaluationProtocol | Mapping | None) -> EvaluationProtocol:
    if value is None:
        return EvaluationProtocol()
    return value if isinstance(value, EvaluationProtocol) else EvaluationProtocol(**value)


def _candidate_exprs(config: EvaluationProtocol):
    dsl = FormulaDSL()
    allowed = frozenset((BIN_OPS | CS_OPS | EW_UNARY) - {"neutralize"})
    search = SearchConfig(config_id="frozen-evaluation", allowed_fields=list(LEAVES),
                          allowed_operators=allowed, allowed_windows=(), max_complexity=16, max_depth=5)
    expressions = {}
    for name, formula in config.candidates.items():
        expression = dsl.parse(formula)
        verdict = dsl.validate(expression, search)
        if verdict.status != GateStatus.PASS:
            raise ValueError(f"candidate {name}: {'; '.join(verdict.reasons)}")
        expressions[name] = expression
    return dsl, expressions


def _frozen_folds(sessions: pd.DatetimeIndex, config: EvaluationProtocol, seal) -> list[dict]:
    begin = pd.Timestamp(config.evaluation_start)
    end = min(pd.Timestamp(config.evaluation_end), pd.Timestamp(seal) - pd.Timedelta(days=1))
    folds = []
    for year in range(begin.year, end.year + 1):
        test_start = max(begin, pd.Timestamp(year=year, month=1, day=1))
        test_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        first = int(sessions.searchsorted(test_start))
        last = int(sessions.searchsorted(test_end, side="right")) - 1
        first = max(first, config.minimum_training_sessions + config.embargo_sessions)
        if first > last or first >= len(sessions):
            continue
        # The last training label endpoint is strictly before the embargo. Its
        # anchor therefore precedes test by horizon + embargo, not merely one day.
        train_endpoint = first - config.embargo_sessions - 1
        train_anchor = train_endpoint - config.horizon_sessions
        decisions = list(range(first, last - config.horizon_sessions + 1,
                               config.decision_stride_sessions))
        if not decisions or train_anchor < config.warmup_sessions:
            continue
        folds.append({"fold_id": str(year), "train_start": str(sessions[0].date()),
                      "last_train_anchor": str(sessions[train_anchor].date()),
                      "last_train_label_end": str(sessions[train_endpoint].date()),
                      "test_start": str(sessions[first].date()), "test_end": str(sessions[last].date()),
                      "purge_sessions": config.horizon_sessions, "embargo_sessions": config.embargo_sessions,
                      "decisions": [str(sessions[p].date()) for p in decisions],
                      "outcome_ends": [str(sessions[p + config.horizon_sessions].date()) for p in decisions]})
    return folds


def register_protocol(snapshot_path: str | Path, registry_dir: str | Path, *,
                      protocol: EvaluationProtocol | Mapping | None = None,
                      protocol_id: str | None = None) -> dict:
    """Freeze cohorts, formulas, folds and thresholds before outcome scoring."""
    snapshot = verify_snapshot(snapshot_path)
    config = _protocol(protocol)
    _candidate_exprs(config)
    if pd.Timestamp(config.evaluation_end) >= pd.Timestamp(snapshot["sealed_from"]):
        raise ValueError("protocol evaluation end enters the sealed region")
    sessions = pd.DatetimeIndex(pd.read_parquet(Path(snapshot_path) / "calendar.parquet").trade_date)
    folds = _frozen_folds(sessions, config, snapshot["sealed_from"])
    payload = {"schema_version": 1, "engine_version": ENGINE_VERSION, "engine_hash": _engine_hash(),
               "runtime_versions": _runtime_versions(),
               "snapshot_hash": snapshot["snapshot_hash"], "config": asdict(config), "folds": folds,
               "baseline_formulas": {"low_vol_60": "rank(-std(ret,60,full_window))",
                                     "mom_60": "rank(mean(ret,60,full_window))",
                                     "mom_20": "rank(mean(ret,20,full_window))",
                                     "reversal_5": "rank(-mean(ret,5,full_window))",
                                     "trend_60": "rank(mean(ret,60)/std(ret,60))",
                                     "equal_weight": "equal weights over common ex-ante eligible funds"},
               "selection_policy": "fixed arms; no learned baseline selection; fractional allocation to all cutoff ties",
               "outcome_policy": "decision valuation NAV to exactly horizon exchange sessions later; endpoints must be observed and published before seal; no ffill or truncated labels",
               "training_policy": "fixed formulas need no fit; all stated training anchors have labels ending before embargo; no training or test driven arm selection",
               "regime_policy": "explicit calendar intervals frozen before score; no outcome-derived regime classification"}
    digest = hashlib.sha256(_json(payload).encode()).hexdigest()
    identifier = _safe_id(protocol_id or f"protocol-{digest[:20]}")
    payload.update(protocol_id=identifier, protocol_hash=digest)
    directory = Path(registry_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identifier}.json"
    # Publish a fully flushed file with an exclusive hard link: readers can
    # never observe an incomplete protocol, and existing ids cannot be replaced.
    temporary = directory / f".protocol-{uuid4().hex}.tmp"
    try:
        with temporary.open("x") as handle:
            handle.write(_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {**payload, "protocol_path": str(target.resolve())}


def _load_protocol(path: str | Path, snapshot_hash: str) -> tuple[dict, EvaluationProtocol]:
    payload = json.loads(Path(path).read_text())
    content = {k: v for k, v in payload.items() if k not in ("protocol_id", "protocol_hash")}
    if hashlib.sha256(_json(content).encode()).hexdigest() != payload["protocol_hash"]:
        raise ValueError("frozen protocol hash mismatch")
    if payload["snapshot_hash"] != snapshot_hash:
        raise ValueError("protocol belongs to a different data snapshot")
    if payload["engine_hash"] != _engine_hash():
        raise ValueError("evaluation code changed since protocol registration; freeze a new protocol")
    if payload["runtime_versions"] != _runtime_versions():
        raise ValueError("evaluation runtime changed since protocol registration; freeze a new protocol")
    return payload, _protocol(payload["config"])


def _features_and_outcomes(path: Path, sessions: pd.DatetimeIndex, decisions: pd.DatetimeIndex,
                           codes: list[str], config: EvaluationProtocol):
    """O(one-fund daily history + decisions*funds), avoiding full daily panels."""
    n_dates, n_codes = len(decisions), len(codes)
    arrays = {key: np.full((n_dates, n_codes), np.nan) for key in LEAVES}
    outcomes = np.full((n_dates, n_codes), np.nan)
    positions = sessions.get_indexer(decisions)
    exits = positions + config.horizon_sessions
    import pyarrow.parquet as pq
    # Physical snapshot order is fund/date, so batches can be streamed without
    # materializing a multi-million-row pandas table or dense daily-wide frames.
    carry = pd.DataFrame()
    code_index = {code: p for p, code in enumerate(codes)}

    def consume(group):
        code = str(group.fund_code.iloc[0])
        j = code_index.get(code)
        if j is None:
            return
        indexed = group.set_index("nav_date").reindex(sessions)
        px = indexed.adj_nav
        usable = pd.to_datetime(indexed.usable_date)
        # Preserve valuation-time returns. Publication gates the WHOLE backward
        # window at the actual decision, rather than permanently discarding a
        # return merely because it was published late at its original date.
        # Sixty returns require 61 source NAVs, including the first denominator.
        ret = px.pct_change(fill_method=None)
        complete = ret.rolling(config.warmup_sessions, min_periods=config.warmup_sessions).count().eq(config.warmup_sessions)
        publication_positions = pd.Series(sessions.get_indexer(usable), index=sessions,
                                          dtype=float).replace(-1, np.nan)
        quotes = config.warmup_sessions + 1
        latest_publication = publication_positions.rolling(quotes, min_periods=quotes).max()
        # End the feature window at decision-minus-registered-lag. Every quote
        # used in that window must already be usable by the actual decision.
        # A late quote can therefore recover before it rolls out of the window.
        visible_at_decision = latest_publication.shift(config.signal_lag_sessions).iloc[positions].to_numpy() <= positions
        mom20, mom60 = ret.rolling(20, min_periods=20).mean(), ret.rolling(60, min_periods=60).mean()
        vol20, vol60 = ret.rolling(20, min_periods=20).std(), ret.rolling(60, min_periods=60).std()
        values = {"mom20": mom20, "mom60": mom60, "vol20": vol20, "vol60": vol60,
                  "reversal5": -ret.rolling(5, min_periods=5).mean(),
                  "trend60": mom60 / vol60.where(vol60 > 1e-12),
                  "dd60": px / px.rolling(60, min_periods=60).max() - 1}
        for key, value in values.items():
            projected = value.where(complete).shift(config.signal_lag_sessions).iloc[positions].to_numpy()
            arrays[key][:, j] = np.where(visible_at_decision, projected, np.nan)
        p0, p1 = px.iloc[positions].to_numpy(), px.iloc[exits].to_numpy()
        # Snapshot drops all rows unavailable before seal; missing entry/exit
        # remains NaN, including liquidated funds without settlement evidence.
        outcomes[:, j] = p1 / p0 - 1

    for batch in pq.ParquetFile(path / "nav.parquet").iter_batches(batch_size=131072):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        last = frame.fund_code.iloc[-1]
        carry = frame.loc[frame.fund_code.eq(last)]
        done = frame.loc[~frame.fund_code.eq(last)]
        for _, group in done.groupby("fund_code", sort=False):
            consume(group)
    if not carry.empty:
        consume(carry)
    features = {key: pd.DataFrame(value, index=decisions, columns=codes) for key, value in arrays.items()}
    return features, pd.DataFrame(outcomes, index=decisions, columns=codes)


def _weights(scores: pd.Series, top_k: int | None) -> pd.Series:
    if top_k is None or len(scores) <= top_k:
        return pd.Series(1 / len(scores), index=scores.index)
    threshold = scores.nlargest(top_k).iloc[-1]
    above, tied = scores.gt(threshold), scores.eq(threshold)
    weight = pd.Series(0.0, index=scores.index)
    weight.loc[above] = 1 / top_k
    weight.loc[tied] = (top_k - int(above.sum())) / (top_k * int(tied.sum()))
    return weight[weight.gt(0)]


def _interval(values, config, *, seed_offset=0) -> dict:
    vector = np.asarray(values, dtype=float)
    vector = vector[np.isfinite(vector)]
    result = {"n": len(vector), "mean": float(vector.mean()) if len(vector) else None,
              "hac_tstat": None, "block_bootstrap_ci95": None}
    if len(vector) < max(5, 2 * config.bootstrap_block_decisions):
        return result
    lag = min(config.bootstrap_block_decisions, len(vector) // 3)
    result["hac_tstat"] = newey_west_tstat(vector, lag=lag)
    rng = np.random.default_rng(config.seed + seed_offset)
    block = min(config.bootstrap_block_decisions, len(vector))
    # Moving blocks, no wrap across the final observation into the initial one.
    starts = rng.integers(0, len(vector) - block + 1,
                          size=(config.bootstrap_repetitions, int(np.ceil(len(vector) / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(config.bootstrap_repetitions, -1)[:, :len(vector)]
    result["block_bootstrap_ci95"] = np.quantile(vector[indices].mean(axis=1), [.025, .975]).tolist()
    return _clean(result)


def run_evaluation(snapshot_path: str | Path, protocol_path: str | Path,
                   output_dir: str | Path, *, run_id: str | None = None) -> dict:
    """Evaluate the frozen family and preserve every run under a fresh id.

    Outputs include all date-level failures, paired baseline differences and
    per-year/regime diagnostics. This runner never searches, chooses a winning
    baseline, invents fees, or grants strict execution approval.
    """
    path = Path(snapshot_path)
    snapshot = verify_snapshot(path)
    frozen, config = _load_protocol(protocol_path, snapshot["snapshot_hash"])
    dsl, expressions = _candidate_exprs(config)
    identifier = _safe_id(run_id or f"run-{uuid4().hex}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / identifier
    if target.exists():
        raise FileExistsError(f"evaluation run already exists: {target}")
    sessions = pd.DatetimeIndex(pd.read_parquet(path / "calendar.parquet").trade_date)
    basic = pd.read_parquet(path / "fund_basic.parquet")
    metadata_total = len(basic)
    if config.markets:
        if "market" not in basic:
            raise ValueError("market metadata required for registered off-exchange cohort")
        basic = basic[basic.market.isin(config.markets)].copy()
    if config.domestic_cny_only:
        from qmt.factor.cost_benchmark import domestic_fund_mask
        if "name" not in basic:
            raise ValueError("fund names required to screen known foreign-calendar/currency products")
        basic = basic[domestic_fund_mask(basic["name"])].copy()
    if config.fund_types:
        if "fund_type" not in basic:
            raise ValueError("fund_type metadata required for registered type cohort")
        basic = basic[basic.fund_type.isin(config.fund_types)].copy()
    codes = sorted(basic.fund_code.astype(str).tolist())
    metadata = basic.set_index("fund_code")
    fold_by_date = {pd.Timestamp(d): fold["fold_id"] for fold in frozen["folds"] for d in fold["decisions"]}
    decisions = pd.DatetimeIndex(sorted(fold_by_date))
    features, outcomes = _features_and_outcomes(path, sessions, decisions, codes, config)
    scores = {"low_vol_60": -features["vol60"], "mom_60": features["mom60"],
              "mom_20": features["mom20"], "reversal_5": features["reversal5"],
              "trend_60": features["trend60"], "equal_weight": features["mom60"] * 0}
    panel = BaseFactorPanel(dates=[d.date() for d in decisions], funds=codes, factors=features)
    for name, expression in expressions.items():
        scores[name] = dsl.evaluate_frame(expression, panel).replace([np.inf, -np.inf], np.nan)
    names = list(scores)
    coverage_rows, metric_rows, weight_rows = [], [], []
    previous: dict[str, pd.Series] = {}
    previous_fold = None
    for d in decisions:
        fold = fold_by_date[d]
        if fold != previous_fold:
            previous = {}
        previous_fold = fold
        common = pd.Series(True, index=codes)
        for frame in scores.values():
            common &= np.isfinite(frame.loc[d])
        eligible = common.index[common]
        observed = outcomes.loc[d].reindex(eligible).replace([np.inf, -np.inf], np.nan)
        known = observed.dropna()
        fraction = len(known) / len(eligible) if len(eligible) else 0
        ic_ok = len(known) >= config.min_cross_section and fraction >= config.min_outcome_coverage
        portfolio_ok = ic_ok and len(known) == len(eligible)
        manager = metadata.reindex(eligible).get("management", pd.Series(index=eligible, dtype="string"))
        manager = manager.fillna("UNKNOWN").replace("", "UNKNOWN")
        known_manager = manager[manager.ne("UNKNOWN")]
        max_share = float(known_manager.value_counts().max() / len(eligible)) if len(known_manager) else None
        coverage_rows.append({"decision_date": d, "fold_id": fold,
                              "outcome_end": sessions[sessions.get_loc(d) + config.horizon_sessions],
                              "eligible_funds": len(eligible), "observed_outcomes": len(known),
                              "outcome_coverage": fraction, "ic_scored": ic_ok, "portfolio_scored": portfolio_ok,
                              "status": "COMPLETE" if portfolio_ok else "INCOMPLETE_OUTCOMES" if ic_ok else "INSUFFICIENT_COVERAGE",
                              "known_managers": known_manager.nunique(), "unknown_manager_funds": int(manager.eq("UNKNOWN").sum()),
                              "largest_known_manager_share": max_share})
        for arm, frame in scores.items():
            section = frame.loc[d].reindex(eligible)
            weights = _weights(section, None if arm == "equal_weight" else config.top_k) if len(section) else pd.Series(dtype=float)
            value, turnover = None, None
            if portfolio_ok:
                value = float((weights * observed.reindex(weights.index)).sum())
                old = previous.get(arm)
                turnover = 1.0 if old is None else 0.5 * float(weights.subtract(old, fill_value=0).abs().sum())
                previous[arm] = weights * (1 + observed.reindex(weights.index)) / (1 + value)
            else:
                previous.pop(arm, None)
            ic = spearman(section.reindex(known.index).to_numpy(), known.to_numpy()) if ic_ok and arm != "equal_weight" else None
            regimes = [name for name, start, end in config.regimes if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
            metric_rows.append({"decision_date": d, "fold_id": fold, "arm": arm,
                                "rank_ic": ic, "gross_nav_return_proxy": value,
                                "one_way_turnover_proxy": turnover, "regimes": ",".join(regimes)})
            # Ex-ante selections survive even when their outcomes are missing.
            for code, weight in weights.items():
                weight_rows.append({"decision_date": d, "arm": arm, "fund_code": code,
                                    "score": float(section.loc[code]), "target_weight": float(weight),
                                    "outcome_observed": bool(pd.notna(observed.loc[code]))})
    coverage = pd.DataFrame(coverage_rows, columns=["decision_date", "fold_id", "outcome_end", "eligible_funds", "observed_outcomes", "outcome_coverage", "ic_scored", "portfolio_scored", "status", "known_managers", "unknown_manager_funds", "largest_known_manager_share"])
    metrics = pd.DataFrame(metric_rows, columns=["decision_date", "fold_id", "arm", "rank_ic", "gross_nav_return_proxy", "one_way_turnover_proxy", "regimes"])
    weights = pd.DataFrame(weight_rows, columns=["decision_date", "arm", "fund_code", "score", "target_weight", "outcome_observed"])

    def summarize(rows):
        answer = {}
        for i, arm in enumerate(names):
            block = rows[rows.arm.eq(arm)]
            answer[arm] = {"rank_ic": _interval(block.rank_ic, config, seed_offset=i),
                           "gross_nav_return_proxy": _interval(block.gross_nav_return_proxy, config, seed_offset=i),
                           "mean_turnover_proxy": float(block.one_way_turnover_proxy.mean()) if block.one_way_turnover_proxy.notna().any() else None}
        return answer

    aggregate = summarize(metrics)
    by_year = {fold["fold_id"]: summarize(metrics[metrics.fold_id.eq(fold["fold_id"])]) for fold in frozen["folds"]}
    by_regime = {name: summarize(metrics[metrics.decision_date.between(pd.Timestamp(start), pd.Timestamp(end))]) for name, start, end in config.regimes}
    returns = metrics.pivot(index="decision_date", columns="arm", values="gross_nav_return_proxy") if len(metrics) else pd.DataFrame(columns=names)
    paired = {}
    for name in names:
        if name == "low_vol_60":
            continue
        difference = returns[name] - returns["low_vol_60"] if len(returns) else []
        paired[name] = _interval(difference, config, seed_offset=100 + names.index(name))
    scored = int(coverage.ic_scored.sum()) if len(coverage) else 0
    scored_years = int(coverage.loc[coverage.ic_scored, "fold_id"].nunique()) if len(coverage) else 0
    date_coverage = scored / len(coverage) if len(coverage) else 0
    enough = scored >= config.min_scored_dates and scored_years >= config.min_scored_years and date_coverage >= config.min_date_coverage
    report = {"schema_version": 1, "run_id": identifier, "snapshot_id": snapshot["snapshot_id"],
              "snapshot_hash": snapshot["snapshot_hash"], "protocol_id": frozen["protocol_id"],
              "protocol_hash": frozen["protocol_hash"], "engine_version": ENGINE_VERSION,
              "runtime_versions": frozen["runtime_versions"],
              "engine_hash": frozen["engine_hash"], "evidence_tier": config.evidence_tier,
              "verdict": "EXPLORATORY" if enough else "BLOCKED_INSUFFICIENT_COVERAGE",
              "alpha_approved": False, "strict_execution_status": "NOT_EVALUATED_MISSING_REVIEWED_EXECUTION_INPUTS",
              "planned_dates": len(coverage), "scored_ic_dates": scored,
              "scored_portfolio_proxy_dates": int(coverage.portfolio_scored.sum()) if len(coverage) else 0,
              "scored_years": scored_years, "date_coverage": date_coverage,
              "cohort_funds": len(codes), "current_type_cohort": list(config.fund_types),
              "cohort_scope": {"markets": list(config.markets), "domestic_cny_only": config.domestic_cny_only,
                               "metadata_funds_before_scope": metadata_total,
                               "foreign_scope_screen": "current-name heuristic; unidentified foreign exposure may remain",
                               "signal_lag_sessions": config.signal_lag_sessions},
              "sealed_from": snapshot["sealed_from"], "aggregate": aggregate, "by_year": by_year,
              "by_frozen_regime": by_regime, "paired_gross_proxy_difference_vs_fixed_low_vol_60": paired,
              "uncertainty": "descriptive moving-block bootstrap and HAC over nonoverlapping decisions; exploratory family, no multiplicity-adjusted alpha approval; seeds are not independent market histories",
              "turnover_policy": "half L1 from preceding observed outcome-drifted weights; initial cash purchase=1; reset at each fold or missing interval; terminal liquidation not charged because no fee-net execution claim",
              "limitations": LIMITATIONS,
              "strict_execution_next_step": "Use qmt.factor.walk_forward with reviewed FeeHistory, StatusHistory, DividendHistory and exact unit-NAV execution data on a separately certified eligible cohort."}
    staging = Path(tempfile.mkdtemp(prefix=".run-", dir=output))
    try:
        coverage.to_parquet(staging / "coverage.parquet", index=False)
        metrics.to_parquet(staging / "metrics.parquet", index=False)
        weights.to_parquet(staging / "weights.parquet", index=False)
        report["artifacts"] = {name: _hash_file(staging / name) for name in
                               ("coverage.parquet", "metrics.parquet", "weights.parquet")}
        # Include the exact frozen inputs, not only paths into a mutable cache.
        (staging / "protocol.json").write_text(_json(frozen) + "\n")
        report["artifacts"]["protocol.json"] = _hash_file(staging / "protocol.json")
        report["report_hash"] = hashlib.sha256(_json(report).encode()).hexdigest()
        (staging / "report.json").write_text(_json(report) + "\n")
        if target.exists():
            raise FileExistsError(f"evaluation run already exists: {target}")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**_clean(report), "run_path": str(target.resolve()), "report_path": str((target / "report.json").resolve())}
