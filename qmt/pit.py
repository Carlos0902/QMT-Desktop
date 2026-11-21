"""Point-in-time (PIT) enforcement — the single most important cross-cutting
invariant (§6 sec:cc-pit / §3 sec:ds-timeline).

Two mechanisms:

1. ``DecisionTimeline`` construction on a trading-day calendar:
   ``decision_time`` (T-day 14:30) → ``order_time`` → ``confirm_time`` →
   ``horizon_start`` / ``horizon_end``. This fixes the "what could be known when"
   chain before any data is read.

2. ``PointInTimeStore`` — a Parquet/DuckDB-backed store where every row carries
   ``available_at``. ``get_asof(as_of)`` returns ONLY rows with
   ``available_at <= as_of``; ``put`` rejects any row lacking ``available_at``
   (fail-closed at ingestion, §3 risknote). This is the enforcement point for
   invariant (1): no downstream module can read future data without going around
   this store, and the property test in tests/test_pit.py is the leak detector.
"""

from __future__ import annotations

import os
from bisect import bisect_right
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import duckdb
import pandas as pd

from qmt.config import CONFIG, DecisionTimingConfig
from qmt.types import DecisionTimeline, Horizon

__all__ = [
    "TradingCalendar",
    "build_decision_timeline",
    "confirm_lag_modeled",
    "DecisionTimeline",
    "PointInTimeStore",
    "AVAILABLE_AT",
]

# Canonical name of the mandatory PIT column on every stored row.
AVAILABLE_AT = "available_at"


# =============================================================================
# Trading-day calendar
# =============================================================================
class TradingCalendar:
    """Minimal trading-day calendar.

    Slice-1 default: business days (Mon–Fri), no holiday awareness. A real calendar
    (SSE/SZSE holidays) is injected later by passing an explicit sorted list of
    trading dates. All horizon math in ``build_decision_timeline`` runs through here
    so swapping in the real calendar changes nothing downstream.
    """

    def __init__(self, trading_days: Optional[Sequence[date]] = None) -> None:
        self._days: Optional[list[date]] = (
            sorted({d for d in trading_days}) if trading_days is not None else None
        )

    # --- predicates ---------------------------------------------------------
    def is_trading_day(self, d: date) -> bool:
        if self._days is not None:
            return d in set(self._days)
        return d.weekday() < 5  # Mon..Fri

    def next_trading_day(self, d: date) -> date:
        if self._days is not None:
            # Explicit calendar: first listed trading day strictly after d.
            idx = bisect_right(self._days, d)
            if idx >= len(self._days):
                raise ValueError(
                    f"no trading day after {d!r} in the supplied calendar (extend the calendar)."
                )
            return self._days[idx]
        cur = d + timedelta(days=1)
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
        return cur

    def add_trading_days(self, d: date, n: int) -> date:
        """Return the date that is ``n`` trading days after ``d`` (n >= 0)."""
        if n < 0:
            raise ValueError("add_trading_days requires n >= 0")
        cur = d
        remaining = n
        while remaining > 0:
            cur = self.next_trading_day(cur)
            remaining -= 1
        return cur

    def count_trading_days(self, start: date, end: date) -> int:
        """Number of trading days in the half-open interval ``(start, end]``.

        How many trading sessions have elapsed *after* ``start`` up to and including
        ``end`` (0 when ``end <= start``). Used to age daily-cadence data in trading
        days: a Friday NAV read the following Monday is 1 trading day old, not 3
        calendar days.
        """
        if end <= start:
            return 0
        if self._days is not None:
            lo = bisect_right(self._days, start)
            hi = bisect_right(self._days, end)
            return max(hi - lo, 0)
        # Business-day default: count Mon–Fri days in (start, end] via a whole-week
        # shortcut (5 weekdays per 7-day block) plus a walk over the <7-day remainder.
        span = (end - start).days
        full_weeks, extra = divmod(span, 7)
        count = full_weeks * 5
        cur = start
        for _ in range(extra):
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                count += 1
        return count


# =============================================================================
# DecisionTimeline construction
# =============================================================================
def build_decision_timeline(
    as_of: date,
    horizon: Horizon = Horizon.MID,
    calendar: Optional[TradingCalendar] = None,
    timing: Optional[DecisionTimingConfig] = None,
) -> DecisionTimeline:
    """Build the decision → order → confirm → forecast-window chain for a T-day.

    - ``decision_time`` = T-day 14:30 (per-fund overrides not applied here).
    - ``order_time`` = same instant (user submits before the cutoff).
    - ``confirm_time`` = next trading day (T+1 confirmation, OTC open-end default).
    - ``horizon_start`` = next trading day (first realistically executable point;
      T-day NAV is unknown at decision time).
    - ``horizon_end`` = horizon_start + (window_max - 1) trading days.
    """
    cal = calendar or TradingCalendar()
    tcfg = timing or CONFIG.timing

    if not cal.is_trading_day(as_of):
        raise ValueError(f"as_of {as_of!r} is not a trading day on the supplied calendar.")

    decision_time = datetime.combine(
        as_of, time(hour=tcfg.decision_hour, minute=tcfg.decision_minute)
    )
    order_time = decision_time  # submitted before the cutoff, at the decision instant
    confirm_day = cal.next_trading_day(as_of)
    confirm_time = datetime.combine(confirm_day, time(hour=0, minute=0))

    horizon_start = cal.next_trading_day(as_of)
    _, window_max = CONFIG.horizon.windows[horizon.value]
    # horizon_start counts as day 1 of the window.
    horizon_end = cal.add_trading_days(horizon_start, max(window_max - 1, 0))

    return DecisionTimeline(
        decision_time=decision_time,
        order_time=order_time,
        confirm_time=confirm_time,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )


def confirm_lag_modeled(
    as_of: date,
    calendar: Optional[TradingCalendar] = None,
    timing: Optional[DecisionTimingConfig] = None,
) -> bool:
    """Whether a real execution/confirmation delay (T+1) is modeled for a decision on ``as_of``.

    Returns ``True`` iff a :class:`DecisionTimeline` can be built and its ``confirm_time``
    falls on a strictly later trading day than the decision — i.e. the OTC open-end T+1
    confirmation lag was actually applied. A non-trading ``as_of`` (no timeline) → ``False``.

    This is the honest input for compliance gate 5 (§3.5-5, «不忽略执行延迟»): the gate must
    not be told a delay was modeled when none was (previously hardcoded ``True`` at every
    call site). The confirmation lag is intrinsic to the fixed OTC decision timeline, so on a
    normal trading day this is ``True``, but the boolean is now *derived*, not asserted.
    """
    try:
        tl = build_decision_timeline(as_of, calendar=calendar, timing=timing)
    except ValueError:
        return False
    return tl.confirm_time.date() > tl.decision_time.date()


# =============================================================================
# PointInTimeStore — the invariant-(1) enforcement point
# =============================================================================
RowLike = Union[Mapping[str, object], "pd.Series"]


class PointInTimeStore:
    """Idempotent, as_of-parameterized store. Every row must carry ``available_at``.

    Backed by DuckDB over an in-memory table per logical table name; optionally
    persisted to Parquet under a base directory (the /data_cache/ store). The three
    load-bearing guarantees:

    * ``put`` rejects any batch with a missing/null ``available_at`` (fail-closed at
      ingestion — a row without a known observable time cannot be point-in-time
      filtered, so it must never enter the store).
    * ``put`` is idempotent: rows identical across all columns are de-duplicated, so
      re-priming the same (interface, fund) does not double the stored rows (a second
      ingest of the same NAV/index bars would otherwise distort MA/vol windows).
    * ``get_asof(as_of)`` returns ONLY rows with ``available_at <= as_of``. There is
      no read path that omits ``as_of`` — future data cannot leak through this store.

    Persistence round-trips across processes: a path-backed store constructed over an
    already-primed ``data_cache`` discovers the per-table Parquet files and hydrates
    each lazily on first access, so a fresh process reads back what an earlier one wrote.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self._con = duckdb.connect(database=":memory:")
        self._tables: dict[str, pd.DataFrame] = {}
        self._base_dir: Optional[Path] = Path(base_dir) if base_dir is not None else None
        # Tables persisted by an earlier process: {table_name -> parquet path},
        # discovered on construction and hydrated lazily on first access.
        self._persisted: dict[str, Path] = {}
        if self._base_dir is not None:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            for pq in sorted(self._base_dir.glob("*.parquet")):
                self._persisted[pq.stem] = pq

    # --- hydration ----------------------------------------------------------
    def _ensure_loaded(self, table: str) -> None:
        """Hydrate ``table`` from its on-disk Parquet the first time it is touched.

        A store opened over an already-primed ``data_cache`` in a fresh process has no
        in-memory frame yet; this loads it on demand so ``get_asof`` / ``put`` see the
        previously persisted rows instead of an empty table.
        """
        if table in self._tables:
            return
        path = self._persisted.get(table)
        if path is None or not path.exists():
            return
        df = pd.read_parquet(path)
        if AVAILABLE_AT in df.columns:
            df[AVAILABLE_AT] = pd.to_datetime(df[AVAILABLE_AT])
        self._tables[table] = df.reset_index(drop=True)

    # --- write --------------------------------------------------------------
    def put(self, table: str, rows: Union[pd.DataFrame, Iterable[RowLike]]) -> int:
        """Merge ``rows`` into ``table``. Returns the number of *net-new* rows stored.

        Fail-closed: raises ``ValueError`` if the batch lacks an ``available_at``
        column or contains any null ``available_at`` — such rows can never be
        point-in-time filtered and must not enter the store.

        Idempotent: rows equal across all columns to ones already stored are dropped,
        so re-ingesting the same (interface, fund) is a no-op rather than a row-doubling
        append. Distinct restatements (same natural key, different value/available_at)
        are preserved — only exact duplicates collapse.
        """
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if df.empty:
            return 0
        if AVAILABLE_AT not in df.columns:
            raise ValueError(
                f"put({table!r}) rejected: every row must carry '{AVAILABLE_AT}' (invariant 1)."
            )
        df = df.copy()
        df[AVAILABLE_AT] = pd.to_datetime(df[AVAILABLE_AT])
        if df[AVAILABLE_AT].isna().any():
            raise ValueError(
                f"put({table!r}) rejected: null '{AVAILABLE_AT}' in batch (fail-closed at ingestion)."
            )

        # Hydrate any previously persisted rows first so a re-ingest de-duplicates
        # against what is already on disk instead of overwriting it.
        self._ensure_loaded(table)
        existing = self._tables.get(table)
        before = 0 if existing is None else len(existing)
        combined = df if existing is None else pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        self._tables[table] = combined

        if self._base_dir is not None:
            # Atomic persist: write to a temp file then rename. A direct write lets a
            # concurrent reader (or a crash) observe a half-written parquet — the
            # "magic bytes not found" corruption seen when ingestion and analysis
            # overlap. os.replace is atomic on POSIX for same-filesystem paths.
            final = self._parquet_path(table)
            tmp = final.with_suffix(final.suffix + ".tmp")
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, final)
            self._persisted[table] = final
        return len(combined) - before

    # --- read (the ONLY read path — always as_of-parameterized) -------------
    def get_asof(
        self,
        table: str,
        as_of: Union[datetime, date],
        filters: Optional[Mapping[str, object]] = None,
    ) -> pd.DataFrame:
        """Return rows of ``table`` with ``available_at <= as_of`` (optionally filtered).

        This is the sole read entry point; there is deliberately no unfiltered read.
        """
        self._ensure_loaded(table)
        if table not in self._tables:
            return pd.DataFrame()
        as_of_ts = pd.Timestamp(as_of)
        rel = self._con.from_df(self._tables[table])
        query = f'SELECT * FROM rel WHERE "{AVAILABLE_AT}" <= ?'
        params: list[object] = [as_of_ts.to_pydatetime()]
        if filters:
            for col, val in filters.items():
                query += f' AND "{col}" = ?'
                params.append(val)
        result = self._con.execute(query, params).df()
        return result

    def latest_asof(
        self,
        table: str,
        as_of: Union[datetime, date],
        filters: Optional[Mapping[str, object]] = None,
    ) -> Optional[pd.Series]:
        """Return the single most-recently-observable row (max available_at) <= as_of."""
        df = self.get_asof(table, as_of, filters)
        if df.empty:
            return None
        return df.sort_values(AVAILABLE_AT).iloc[-1]

    # --- introspection ------------------------------------------------------
    def tables(self) -> list[str]:
        """All table names, including persisted-but-not-yet-hydrated ones."""
        return sorted(set(self._tables) | set(self._persisted))

    def row_count(self, table: str) -> int:
        self._ensure_loaded(table)
        return 0 if table not in self._tables else len(self._tables[table])

    def _parquet_path(self, table: str) -> Path:
        assert self._base_dir is not None
        return self._base_dir / f"{table}.parquet"

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "PointInTimeStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
