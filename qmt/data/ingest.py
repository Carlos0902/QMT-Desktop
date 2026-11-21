"""Data priming / ingestion (slice: data-priming + backtest, part 1).

This is the **WRITE** side of the point-in-time contract. Where :mod:`qmt.data.sources`
provides the per-interface typed fetchers, this module *orchestrates* them into a single
"prime everything a fund needs" call and records what was fetched, so a backtest (or the
live pipeline) can later **READ** the primed store via
:meth:`~qmt.pit.PointInTimeStore.get_asof` and nothing else.

Hard rules honoured here (see CLAUDE.md / 06-cross-cutting.tex):

* **Point-in-time.** Ingest only *writes* rows (each already carrying the correct
  ``available_at`` set by :mod:`qmt.data.sources`); it never reads via a non-PIT path and
  never back-dates a fact. The read side (``run`` / ``backtest``) filters
  ``available_at <= as_of``. ``ingest`` and ``run`` are therefore cleanly separated:
  ingest WRITES, run READS.
* **Fail-open on quota (never crash).** Every Tushare call is guarded: a
  :class:`~qmt.data.tushare_client.TushareQuotaError` / rate-limit / transport failure is
  captured as a *gap* in the returned :class:`IngestReport` and the ingest continues with
  whatever it could fetch. A best-effort priming that can only get NAV (not holdings,
  because the account lacks the points) is a degraded success, not a stack trace.
* **Token only from config.** Fetchers construct a default
  :class:`~qmt.data.tushare_client.TushareClient` whose token comes from ``.env`` via
  :func:`qmt.config.get_settings`; nothing here hard-codes a token.

Two extra responsibilities the task calls out:

* :func:`build_industry_map` — best-effort ``stock ts_code -> Shenwan L1`` map built from
  Tushare ``index_classify`` + ``index_member`` *within quota*, cached under
  ``data_cache/`` so it is built once and reused. On any quota failure it returns ``{}``
  (never crashes). :func:`load_cached_industry_map` reads the cache back; the analysis
  orchestrator loads it and passes it to ``exposure_estimator`` so holdings map to real
  industries (instead of the ``unclassified`` sentinel) whenever the quota allowed the map
  to be built.
* :func:`ingest_universe` — prime the whole off-exchange ``fund_basic`` universe (all
  statuses, survivorship-safe) into the store.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from qmt.config import get_settings
from qmt.data.sources import (
    TABLE_FUND_BASIC,
    TABLE_FUND_NAV,
    TABLE_FUND_PORTFOLIO,
    TABLE_INDEX_DAILY,
    TABLE_SW_DAILY,
    fund_nav,
    fund_portfolio,
    fund_universe,
    index_daily,
    sw_daily,
)
from qmt.data.tushare_client import (
    TushareClient,
    TushareError,
    fund_code_to_ts_code,
    ts_code_to_fund_code,
)
from qmt.pit import PointInTimeStore
from qmt.types import FundProfile

logger = logging.getLogger(__name__)

__all__ = [
    "IngestReport",
    "ingest_fund",
    "ingest_universe",
    "build_industry_map",
    "load_cached_industry_map",
    "DEFAULT_BENCHMARK_INDEX",
    "SHENWAN_L1_INDEXES",
    "INDUSTRY_MAP_CACHE_FILE",
]

# The broad A-share benchmark whose EOD bars back the market-state / return baselines
# (matches ``market_state.assess_market_state`` and ``run._DEFAULT_BENCHMARK``: CSI 300).
DEFAULT_BENCHMARK_INDEX = "000300.SH"

# Shenwan (申万) 2021 first-level (L1) industry index codes (``sw_daily`` ts_codes). These
# are the industry regressors for returns-based exposure and sector market-state inputs.
# Best-effort and configurable — a quota-limited account simply records them as gaps.
SHENWAN_L1_INDEXES: tuple[str, ...] = (
    "801010.SI",  # 农林牧渔
    "801030.SI",  # 基础化工
    "801040.SI",  # 钢铁
    "801050.SI",  # 有色金属
    "801080.SI",  # 电子
    "801110.SI",  # 家用电器
    "801120.SI",  # 食品饮料
    "801130.SI",  # 纺织服饰
    "801140.SI",  # 轻工制造
    "801150.SI",  # 医药生物
    "801160.SI",  # 公用事业
    "801170.SI",  # 交通运输
    "801180.SI",  # 房地产
    "801200.SI",  # 商贸零售
    "801210.SI",  # 社会服务
    "801230.SI",  # 综合
    "801710.SI",  # 建筑材料
    "801720.SI",  # 建筑装饰
    "801730.SI",  # 电力设备
    "801740.SI",  # 国防军工
    "801750.SI",  # 计算机
    "801760.SI",  # 传媒
    "801770.SI",  # 通信
    "801780.SI",  # 银行
    "801790.SI",  # 非银金融
    "801880.SI",  # 汽车
    "801890.SI",  # 机械设备
    "801950.SI",  # 煤炭
    "801960.SI",  # 石油石化
    "801970.SI",  # 环保
    "801980.SI",  # 美容护理
)

# Cache filename for the built industry map (under ``data_cache/``). JSON, not Parquet:
# it is a small reference taxonomy (symbol -> industry), not a point-in-time table.
INDUSTRY_MAP_CACHE_FILE = "industry_map.json"


# =============================================================================
# Ingest report
# =============================================================================
@dataclass(frozen=True)
class IngestReport:
    """What a single :func:`ingest_fund` call fetched and wrote (a priming receipt).

    ``gaps`` records every interface that could not be fetched (quota / permission /
    rate-limit / transport) so a caller sees a degraded-but-successful priming instead of
    a crash. ``tables_written`` maps each PIT table name to the number of rows appended by
    this call.
    """

    fund_code: str
    as_of: date
    profile: Optional[FundProfile]
    in_universe: bool
    nav_rows: int
    holdings_reports: int
    index_series: dict[str, int]
    industry_map_size: int
    tables_written: dict[str, int]
    gaps: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the essential fund NAV was primed (the minimum a backtest needs)."""
        return self.nav_rows > 0


# =============================================================================
# small guards / helpers
# =============================================================================
def _resolve_client(client: Optional[TushareClient]) -> TushareClient:
    return client if client is not None else TushareClient()


def _ymd(d: Optional[date]) -> Optional[str]:
    return d.strftime("%Y%m%d") if d is not None else None


def _guard(gaps: list[str], label: str, fn: Callable[[], object]) -> object:
    """Run a Tushare-backed fetch, capturing any failure as a ``gap`` (never crash).

    A quota / permission / rate-limit / transport failure (or any unexpected error) is
    recorded in ``gaps`` and ``None`` is returned so the caller can continue priming
    whatever else is reachable — fail-open on the *fetch* path (the point-in-time and
    fail-closed guarantees live on the read/analysis side, not here).
    """
    try:
        return fn()
    except TushareError as exc:  # quota / rate-limit / api / transport
        gaps.append(f"{label}: {exc}")
        logger.info("ingest: skipped %s — %s", label, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive: a fetch must never crash ingest
        gaps.append(f"{label}: unexpected {type(exc).__name__}: {exc}")
        logger.warning("ingest: unexpected error fetching %s: %s", label, exc)
        return None


def _find_profile(profiles: Sequence[FundProfile], fund_code: str) -> Optional[FundProfile]:
    for p in profiles:
        if p.fund_code == fund_code:
            return p
    return None


def _uses_sw_daily(ts_code: str) -> bool:
    """Route a Shenwan (申万) ``.SI`` industry index to ``sw_daily``; others to ``index_daily``."""
    return ts_code.upper().endswith(".SI")


# =============================================================================
# ingest_universe — prime the whole off-exchange fund_basic universe
# =============================================================================
def ingest_universe(
    as_of: date,
    store: PointInTimeStore,
    *,
    client: Optional[TushareClient] = None,
) -> list[FundProfile]:
    """Prime the off-exchange ``fund_basic`` universe (all statuses) into ``store``.

    Delegates to :func:`qmt.data.sources.fund_universe`, which pulls **every** status
    (L/I/D) so the survivorship-correct universe can be reconstructed later from
    ``found_date`` / ``delist_date`` (never the current ``status`` snapshot). Quota
    failures return ``[]`` rather than crashing. ``as_of`` is accepted for symmetry with
    :func:`ingest_fund` (it does not bound the universe pull — a fund's ``available_at`` is
    its establishment day, and the read side filters by it).
    """
    cli = _resolve_client(client)
    gaps: list[str] = []
    profiles = _guard(gaps, TABLE_FUND_BASIC, lambda: fund_universe(client=cli, store=store))
    if profiles is None:
        return []
    return list(profiles)


# =============================================================================
# ingest_fund — prime everything one fund's analysis needs
# =============================================================================
def ingest_fund(
    fund_code: str,
    as_of: date,
    store: PointInTimeStore,
    *,
    with_holdings: bool = True,
    with_indexes: bool = True,
    client: Optional[TushareClient] = None,
    benchmark: str = DEFAULT_BENCHMARK_INDEX,
    industries: Sequence[str] = SHENWAN_L1_INDEXES,
    build_map: bool = True,
    ensure_universe: bool = True,
    cache_dir: Optional[Path] = None,
) -> IngestReport:
    """Prime ``store`` with everything the pipeline needs to analyze ``fund_code`` at ``as_of``.

    Fetches (via :mod:`qmt.data.sources`, each row tagged with its correct ``available_at``)
    and WRITES into ``store``:

    1. the off-exchange universe (``fund_basic``, all statuses) so the fund's
       :class:`FundProfile` is resolvable and survivorship is preserved;
    2. the fund's NAV history (``fund_nav``, ``available_at = ann_date``);
    3. the fund's disclosed holdings (``fund_portfolio``) when ``with_holdings``;
    4. the benchmark index and the Shenwan L1 industry indices (``index_daily`` /
       ``sw_daily``, EOD bars ``available_at = trade_date @ close``) when ``with_indexes``;
    5. a best-effort ``symbol -> Shenwan L1`` industry map (cached) when ``build_map``.

    Every fetch is quota-guarded: anything the account cannot reach is recorded in
    :attr:`IngestReport.gaps` and priming continues. Nothing is read back through a
    non-PIT path. ``as_of`` bounds the NAV / index pulls (``end_date``) and is echoed in
    the report.
    """
    cli = _resolve_client(client)
    gaps: list[str] = []
    tables_before = {t: store.row_count(t) for t in store.tables()}

    # --- 1) universe → FundProfile ------------------------------------------
    profiles: list[FundProfile] = []
    if ensure_universe:
        got = _guard(gaps, TABLE_FUND_BASIC, lambda: fund_universe(client=cli, store=store))
        profiles = list(got) if got else []
    profile = _find_profile(profiles, fund_code)

    # --- 2) fund NAV history (available_at = ann_date) ----------------------
    navs = _guard(
        gaps,
        TABLE_FUND_NAV,
        lambda: fund_nav(fund_code, client=cli, store=store, end_date=_ymd(as_of)),
    )
    nav_rows = len(navs) if navs else 0

    # --- 3) disclosed holdings (Tier-1 input to exposure) -------------------
    holdings_reports = 0
    holding_symbols: list[str] = []
    if with_holdings:
        snaps = _guard(
            gaps, TABLE_FUND_PORTFOLIO, lambda: fund_portfolio(fund_code, client=cli, store=store)
        )
        if snaps:
            holdings_reports = len(snaps)
            holding_symbols = [h.symbol for snap in snaps for h in snap.holdings]

    # --- 4) benchmark + Shenwan L1 industry index daily bars ----------------
    index_series: dict[str, int] = {}
    if with_indexes:
        for ts_code in (benchmark, *industries):
            fetch = sw_daily if _uses_sw_daily(ts_code) else index_daily
            df = _guard(
                gaps,
                ts_code,
                lambda ts_code=ts_code, fetch=fetch: fetch(
                    ts_code, client=cli, store=store, end_date=_ymd(as_of)
                ),
            )
            if df is not None:
                index_series[ts_code] = int(len(df))

    # --- 5) industry map (best-effort, cached) ------------------------------
    industry_map_size = 0
    if build_map:
        imap = build_industry_map(
            as_of,
            store,
            client=cli,
            symbols=holding_symbols or None,
            cache=True,
            cache_dir=cache_dir,
        )
        industry_map_size = len(imap)

    tables_after = {t: store.row_count(t) for t in store.tables()}
    tables_written = {
        t: tables_after[t] - tables_before.get(t, 0)
        for t in tables_after
        if tables_after[t] - tables_before.get(t, 0) > 0
    }

    return IngestReport(
        fund_code=fund_code,
        as_of=as_of,
        profile=profile,
        in_universe=profile is not None,
        nav_rows=nav_rows,
        holdings_reports=holdings_reports,
        index_series=index_series,
        industry_map_size=industry_map_size,
        tables_written=tables_written,
        gaps=gaps,
    )


# =============================================================================
# build_industry_map — best-effort stock -> Shenwan L1, cached, quota-safe
# =============================================================================
def _cache_path(cache_dir: Optional[Path]) -> Path:
    base = Path(cache_dir) if cache_dir is not None else get_settings().data_cache_dir
    base.mkdir(parents=True, exist_ok=True)
    return base / INDUSTRY_MAP_CACHE_FILE


def load_cached_industry_map(cache_dir: Optional[Path] = None) -> dict[str, str]:
    """Load the cached ``symbol -> Shenwan L1`` map, or ``{}`` if none has been built.

    Reads ``data_cache/industry_map.json`` (or ``cache_dir/…``). Any read/parse error
    degrades to ``{}`` — a missing or corrupt cache is a data gap, never a crash. Keys are
    the suffix-stripped 6-digit stock codes (matching ``exposure_estimator``'s lookup).
    """
    path = Path(cache_dir) / INDUSTRY_MAP_CACHE_FILE if cache_dir is not None else (
        get_settings().data_cache_dir / INDUSTRY_MAP_CACHE_FILE
    )
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}
    except Exception:  # pragma: no cover - defensive: a bad cache must never crash a read
        return {}


def build_industry_map(
    as_of: date,
    store: Optional[PointInTimeStore] = None,
    *,
    client: Optional[TushareClient] = None,
    symbols: Optional[Sequence[str]] = None,
    cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Build a best-effort ``stock ts_code -> Shenwan L1 industry name`` map (quota-safe).

    Strategy (the E40 Shenwan taxonomy join, kept best-effort): pull the L1 industry list
    via Tushare ``index_classify`` (``level='L1'``, ``src='SW2021'``), then for each L1
    index pull its constituents via ``index_member`` and record ``con_code -> industry``.
    When ``symbols`` is given the result is restricted to those stocks (cheaper and enough
    for one fund's holdings); otherwise the full membership is returned.

    Fail-open: if the account lacks the points for these interfaces (or any call fails),
    the partial map built so far is returned — ``{}`` in the worst case — and **nothing
    crashes**. On success the map is merged into the existing cache (so repeated per-fund
    calls accumulate) and written to ``data_cache/industry_map.json`` when ``cache``.

    Keys are suffix-stripped 6-digit codes so :func:`qmt.pipeline.exposure_estimator` can
    look up a holding symbol (which it canonicalizes the same way). ``as_of`` / ``store``
    are accepted for interface symmetry and future point-in-time-correct classification
    revisions (D13); the baseline taxonomy is treated as static.
    """
    cli = _resolve_client(client)
    wanted: Optional[set[str]] = None
    if symbols is not None:
        wanted = {_norm_symbol(s) for s in symbols}
        if not wanted:
            return {}

    mapping: dict[str, str] = {}
    try:
        classify = cli.query(
            "index_classify", {"level": "L1", "src": "SW2021"}, "index_code,industry_name"
        )
    except TushareError as exc:
        logger.info("build_industry_map: index_classify unavailable — %s", exc)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("build_industry_map: unexpected index_classify error: %s", exc)
        return {}

    if classify is None or getattr(classify, "empty", True):
        return {}

    for row in classify.to_dict("records"):
        index_code = str(row.get("index_code") or "").strip()
        name = str(row.get("industry_name") or "").strip()
        if not index_code or not name:
            continue
        try:
            members = cli.query("index_member", {"index_code": index_code}, "index_code,con_code")
        except TushareError as exc:
            logger.info("build_industry_map: index_member(%s) unavailable — %s", index_code, exc)
            continue  # partial map: keep what we have, skip this industry
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("build_industry_map: unexpected index_member error: %s", exc)
            continue
        if members is None or getattr(members, "empty", True):
            continue
        for m in members.to_dict("records"):
            base = _norm_symbol(str(m.get("con_code") or ""))
            if not base:
                continue
            if wanted is not None and base not in wanted:
                continue
            mapping[base] = name

    if cache and mapping:
        _write_cache(mapping, cache_dir)
    return mapping


def _norm_symbol(symbol: str) -> str:
    """Canonicalize a stock symbol for map lookup (strip the exchange suffix)."""
    return symbol.strip().split(".")[0]


def _write_cache(mapping: dict[str, str], cache_dir: Optional[Path]) -> None:
    """Merge ``mapping`` into the on-disk industry-map cache (best-effort, never crash)."""
    try:
        path = _cache_path(cache_dir)
        merged = load_cached_industry_map(cache_dir)
        merged.update(mapping)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, sort_keys=True)
    except Exception as exc:  # pragma: no cover - defensive: caching is optional
        logger.warning("build_industry_map: could not write cache: %s", exc)
