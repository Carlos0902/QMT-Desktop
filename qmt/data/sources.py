"""Typed, point-in-time-correct fetchers over the Tushare client (slice 2).

Each fetcher: (1) pulls raw rows via :class:`~qmt.data.tushare_client.TushareClient`,
(2) maps them into the foundation domain records from :mod:`qmt.types`, tagging the
**correct** ``available_at`` (invariant 1), and (3) caches the rows into a
:class:`~qmt.pit.PointInTimeStore` (Parquet under ``data_cache/``) so every downstream
read goes through the ``available_at <= as_of`` filter.

The load-bearing point-in-time rules realised here (see plan / 06-cross-cutting.tex):

* ``fund_nav``   — ``available_at = ann_date`` (disclosure), **never** ``end_date``
  (the valuation date), and **floored at the day after the valuation date**. For OTC
  open-end funds the NAV for day *D* is struck after close and published on *D+1*, so a
  decision at *D* 14:30 must not see it. Tushare sometimes tags the most-recent rows with
  ``ann_date <= nav_date`` (disclosure lag not yet applied); the floor (see
  :func:`_nav_available_at`) neutralises that same-day look-ahead.
* ``fund_portfolio`` — ``available_at = ann_date``; ``report_end_date = end_date``.
  On decision day the newest observable holdings can be 1.5–4 months stale. This is
  the empirical justification for "estimated exposure ≠ disclosed holdings".
* ``fund_universe`` — includes **all** statuses (L/I/D), never filters to ``status='L'``
  (survivorship). The point-in-time universe is reconstructed from
  ``found_date``/``delist_date``, not the current ``status`` snapshot.
* ``index_daily`` / ``sw_daily`` — EOD bars: ``available_at`` = trade_date at the EOD
  close (15:00). A *D*-day 14:30 signal must use only closes through *D-1*.

Known OTC data gaps (documented, flagged for a secondary source) are in :data:`DATA_GAPS`.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Optional

import pandas as pd

from qmt.config import CONFIG, get_settings
from qmt.data.tushare_client import (
    TushareClient,
    fund_code_to_ts_code,
    ts_code_to_fund_code,
)
from qmt.pit import AVAILABLE_AT, PointInTimeStore
from qmt.types import (
    DisclosedHoldings,
    DisclosureDepth,
    FundMeta,
    FundProfile,
    FundStatus,
    HoldingLine,
    NavPoint,
)

__all__ = [
    "fund_universe",
    "fund_nav",
    "fund_portfolio",
    "index_daily",
    "sw_daily",
    "TABLE_FUND_BASIC",
    "TABLE_FUND_NAV",
    "TABLE_FUND_PORTFOLIO",
    "TABLE_INDEX_DAILY",
    "TABLE_SW_DAILY",
    "DATA_GAPS",
]

# Point-in-time store table names (one per Tushare interface).
TABLE_FUND_BASIC = "fund_basic"
TABLE_FUND_NAV = "fund_nav"
TABLE_FUND_PORTFOLIO = "fund_portfolio"
TABLE_INDEX_DAILY = "index_daily"
TABLE_SW_DAILY = "sw_daily"

# D15 / plan "KNOWN DATA GAPS": things Tushare does NOT cleanly expose for OTC funds.
# Downstream modules must treat these as data gaps (secondary source / do-not-build),
# never silently assume zero.
DATA_GAPS: dict[str, str] = {
    "industry_allocation": (
        "fund_portfolio returns STOCK positions only — no industry/sector allocation "
        "vector. Must be DERIVED by mapping each symbol to an industry taxonomy "
        "(index_classify SW), and only reflects the stale, top-10-limited snapshot."
    ),
    "asset_class_split": (
        "The stock/bond/cash asset-allocation split is reported in periodic-report "
        "asset-allocation tables, which fund_portfolio does not expose for OTC funds "
        "(data gap — flag for a secondary source)."
    ),
    "daily_otc_size": (
        "Daily OTC open-end fund size / inflow-outflow is not available (fund_share is "
        "场内 ETF/LOF/SF/CEF only). OTC size is observable only quarterly via "
        "fund_nav.net_asset / total_netasset, tagged with fund_nav.ann_date."
    ),
    "intraday_nav": (
        "No intraday/realtime NAV or index value at the T 14:30 decision point — EOD "
        "tables cannot supply it; build no feature that requires it."
    ),
}

# fund_basic status snapshots to pull so survivorship is preserved (never just 'L').
_ALL_FUND_STATUSES = ("L", "I", "D")

# Field lists (kept explicit so a schema drift on Tushare's side is visible here).
_FUND_BASIC_FIELDS = (
    "ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,"
    "issue_date,delist_date,issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,"
    "exp_return,benchmark,status,invest_type,type,trustee,purc_startdate,redm_startdate,"
    "market"
)
_FUND_NAV_FIELDS = (
    "ts_code,ann_date,nav_date,end_date,unit_nav,accum_nav,accum_div,net_asset,"
    "total_netasset,adj_nav,update_flag"
)
_FUND_PORTFOLIO_FIELDS = (
    "ts_code,ann_date,end_date,symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio"
)


# =============================================================================
# small parsing helpers
# =============================================================================
def _parse_date(value: Any) -> Optional[date]:
    """Parse a Tushare ``YYYYMMDD`` string (or empty/None) into a ``date``."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return datetime.strptime(s[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _disclosure_available_at(ann_date: date) -> datetime:
    """Disclosure data becomes visible on its announcement day.

    ``ann_date`` for NAV/holdings is already the *day after* the valuation/period end
    (the report is published that morning), so midnight of ``ann_date`` is the correct
    PIT marker: a decision on ``ann_date`` at 14:30 sees it; the day before does not.
    """
    return datetime.combine(ann_date, time(0, 0))


def _nav_available_at(ann_date: date, valuation_date: date) -> datetime:
    """PIT marker for a daily NAV: ``ann_date`` disclosure, floored at the day after valuation.

    A daily open-end NAV for valuation date *D* is struck after *D*'s close and cannot be
    known before the next day's disclosure (typically T+1). Almost every row carries
    ``ann_date = D + 1`` and this reduces to :func:`_disclosure_available_at`. But Tushare
    occasionally reports ``ann_date <= nav_date`` for the most-recent rows of some funds
    (the disclosure lag not yet applied at the source) — a same-day look-ahead: a
    midnight-cutoff read on *D* would then see *D*'s NAV, which is only published that
    evening. Flooring at ``_disclosure_available_at(D + 1 day)`` removes that leak while
    leaving every correctly-tagged row (``ann_date >= D + 1``, including the far-later
    quarterly ``net_asset`` restatements) exactly as the source tagged it.
    """
    disclosed = _disclosure_available_at(ann_date)
    nav_floor = _disclosure_available_at(valuation_date + timedelta(days=1))
    return max(disclosed, nav_floor)


def _eod_available_at(trade_date: date) -> datetime:
    """EOD market bars become visible only at the close of their own trade_date.

    ``available_at`` = trade_date at the configured EOD close (15:00). A *D*-day 14:30
    decision therefore excludes the *D*-day bar (15:00 > 14:30) and includes *D-1*'s —
    exactly the "no same-day EOD data for a pre-close signal" rule.
    """
    t = time(CONFIG.timing.eod_close_hour, CONFIG.timing.eod_close_minute)
    return datetime.combine(trade_date, t)


def _depth_for(report_end_date: date) -> DisclosureDepth:
    """Q1/Q3 reports (period-end 03/09) disclose TOP10; semi-annual/annual (06/12) FULL."""
    return DisclosureDepth.FULL if report_end_date.month in (6, 12) else DisclosureDepth.TOP10


def _resolve_client(client: Optional[TushareClient]) -> TushareClient:
    return client if client is not None else TushareClient()


def _resolve_store(store: Optional[PointInTimeStore], persist: bool) -> Optional[PointInTimeStore]:
    """Return the store to write to, or None when persistence is off.

    A caller-supplied store always wins (tests inject an in-memory or tmp store). When
    ``persist`` and no store is given, a Parquet-backed store under the configured
    ``data_cache/`` directory is created lazily.
    """
    if store is not None:
        return store
    if not persist:
        return None
    return PointInTimeStore(base_dir=get_settings().ensure_data_cache())


def _put(store: Optional[PointInTimeStore], table: str, records: list[dict[str, Any]]) -> None:
    """Write flat records to the PIT store (every record already carries available_at)."""
    if store is None or not records:
        return
    df = pd.DataFrame(records)
    # Normalise pure-date columns to datetime64 so both the DuckDB (in-memory) and the
    # Parquet backends store them uniformly; available_at is left for the store's own
    # to_datetime + fail-closed null check.
    for col in ("nav_date", "report_end_date", "trade_date", "found_date", "list_date", "delist_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    store.put(table, df)


# =============================================================================
# (1) fund universe — fund_basic, ALL statuses (survivorship-safe)
# =============================================================================
def fund_universe(
    *,
    client: Optional[TushareClient] = None,
    store: Optional[PointInTimeStore] = None,
    statuses: tuple[str, ...] = _ALL_FUND_STATUSES,
    market: str = CONFIG.tushare.otc_market_code,
    persist: bool = True,
) -> list[FundProfile]:
    """Fetch the off-exchange fund universe (Tushare ``fund_basic``).

    Includes **every** status in ``statuses`` (default L/I/D) so delisted/merged funds
    are retained — filtering to ``status='L'`` would drop funds that were live at a past
    decision time and silently introduce survivorship bias. The point-in-time universe
    is reconstructed downstream from ``found_date``/``delist_date``.

    Each :class:`FundProfile` is tagged ``available_at`` = the fund's establishment day
    (``found_date``, falling back to ``list_date`` then ``issue_date``) — a fund is only
    externally observable once it exists (D29).

    **Paginated (D44).** Tushare ``fund_basic`` caps each call at ~15k rows; the OTC universe
    is larger (~20k+ live). Each status is fetched page-by-page (``offset``/``limit``) until a
    short page, so no fund is silently truncated. Page size / max pages come from
    :class:`~qmt.config.TushareConfig`.
    """
    cli = _resolve_client(client)
    dst = _resolve_store(store, persist)

    profiles: list[FundProfile] = []
    storage: list[dict[str, Any]] = []
    seen: set[str] = set()
    page_size = CONFIG.tushare.fund_basic_page_size
    max_pages = CONFIG.tushare.fund_basic_max_pages

    for status in statuses:
        rows: list[dict[str, Any]] = []
        for page in range(max_pages):
            raw = cli.query(
                "fund_basic",
                {"market": market, "status": status, "offset": page * page_size, "limit": page_size},
                _FUND_BASIC_FIELDS,
            )
            batch = raw.to_dict("records")
            rows.extend(batch)
            if len(batch) < page_size:  # short page → last page for this status
                break
        for row in rows:
            ts_code = str(row.get("ts_code") or "").strip()
            if not ts_code or ts_code in seen:
                continue
            seen.add(ts_code)
            fund_code = ts_code_to_fund_code(ts_code)

            found = _parse_date(row.get("found_date"))
            born = found or _parse_date(row.get("list_date")) or _parse_date(row.get("issue_date"))
            if born is None:
                # No establishment date at all → cannot assign a PIT marker → fail-closed drop.
                continue
            available_at = datetime.combine(born, time(0, 0))

            storage.append(
                {
                    "fund_code": fund_code,
                    "ts_code": ts_code,
                    "name": row.get("name"),
                    "fund_type": row.get("fund_type"),
                    "invest_type": row.get("invest_type"),
                    "market": row.get("market") or market,
                    "status": status,
                    "found_date": found,
                    "list_date": _parse_date(row.get("list_date")),
                    "delist_date": _parse_date(row.get("delist_date")),
                    "benchmark": row.get("benchmark"),
                    AVAILABLE_AT: available_at,
                }
            )

            # A valid domain FundProfile needs a canonical 6-digit code.
            if len(fund_code) != 6 or not fund_code.isdigit():
                continue
            try:
                status_enum = FundStatus(status)
            except ValueError:
                status_enum = None
            meta = FundMeta(
                fund_code=fund_code,
                name=str(row.get("name") or ""),
                management=row.get("management") or None,
                custodian=row.get("custodian") or None,
                benchmark=row.get("benchmark") or None,
                m_fee=_to_float(row.get("m_fee")),
                c_fee=_to_float(row.get("c_fee")),
                min_amount=_to_float(row.get("min_amount")),
                issue_amount=_to_float(row.get("issue_amount")),
            )
            profiles.append(
                FundProfile(
                    fund_code=fund_code,
                    name=str(row.get("name") or ""),
                    fund_type=row.get("fund_type") or None,
                    invest_type=row.get("invest_type") or None,
                    market=str(row.get("market") or market),
                    status=status_enum,
                    found_date=found,
                    list_date=_parse_date(row.get("list_date")),
                    delist_date=_parse_date(row.get("delist_date")),
                    benchmark=row.get("benchmark") or None,
                    meta=meta,
                    available_at=available_at,
                )
            )

    _put(dst, TABLE_FUND_BASIC, storage)
    return profiles


# =============================================================================
# (2) fund NAV history — fund_nav, available_at = ann_date (NOT end_date)
# =============================================================================
def fund_nav(
    fund_or_ts_code: str,
    *,
    client: Optional[TushareClient] = None,
    store: Optional[PointInTimeStore] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    persist: bool = True,
) -> list[NavPoint]:
    """Fetch a fund's NAV history (Tushare ``fund_nav``) as point-in-time NavPoints.

    ``available_at`` is set from ``ann_date`` (the disclosure date), NEVER ``end_date``
    (the valuation date) — the canonical look-ahead trap — and floored at the day after the
    valuation date (:func:`_nav_available_at`) so a Tushare ``ann_date <= nav_date`` tag on
    the most-recent rows cannot leak day *D*'s NAV at a *D* midnight read. Rows lacking
    ``ann_date`` (or the valuation date / unit NAV) are dropped fail-closed: a NAV with no
    known disclosure time cannot be point-in-time filtered and must not enter the store.
    """
    cli = _resolve_client(client)
    dst = _resolve_store(store, persist)
    ts_code = fund_code_to_ts_code(fund_or_ts_code)
    fund_code = ts_code_to_fund_code(ts_code)

    params: dict[str, Any] = {"ts_code": ts_code}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    raw = cli.query("fund_nav", params, _FUND_NAV_FIELDS)

    navs: list[NavPoint] = []
    storage: list[dict[str, Any]] = []
    for row in raw.to_dict("records"):
        ann = _parse_date(row.get("ann_date"))
        # end_date is the valuation date; some schema versions call it nav_date.
        valuation = _parse_date(row.get("end_date")) or _parse_date(row.get("nav_date"))
        unit = _to_float(row.get("unit_nav"))
        if ann is None or valuation is None or unit is None:
            continue  # fail-closed: undated / unusable NAV never enters the store
        # Disclosure marker, floored at the day after the valuation date so a Tushare
        # ann_date <= nav_date (same-day look-ahead) cannot leak D's NAV at a D midnight read.
        available_at = _nav_available_at(ann, valuation)
        navs.append(
            NavPoint(
                fund_code=fund_code,
                nav_date=valuation,
                unit_nav=unit,
                accum_nav=_to_float(row.get("accum_nav")),
                adj_nav=_to_float(row.get("adj_nav")),
                net_asset=_to_float(row.get("net_asset")),
                available_at=available_at,
            )
        )
        storage.append(
            {
                "fund_code": fund_code,
                "ts_code": ts_code,
                "nav_date": valuation,
                "unit_nav": unit,
                "accum_nav": _to_float(row.get("accum_nav")),
                "adj_nav": _to_float(row.get("adj_nav")),
                "net_asset": _to_float(row.get("net_asset")),
                AVAILABLE_AT: available_at,
            }
        )

    _put(dst, TABLE_FUND_NAV, storage)
    return navs


# =============================================================================
# (3) disclosed holdings — fund_portfolio, disclosure lag captured
# =============================================================================
def fund_portfolio(
    fund_or_ts_code: str,
    *,
    client: Optional[TushareClient] = None,
    store: Optional[PointInTimeStore] = None,
    persist: bool = True,
) -> list[DisclosedHoldings]:
    """Fetch a fund's disclosed top-10 holdings (Tushare ``fund_portfolio``).

    Rows are grouped into one :class:`DisclosedHoldings` snapshot per report, tagged
    ``report_end_date = end_date`` and ``available_at = ann_date`` — capturing the
    disclosure lag (``ann_date`` − ``end_date``). ``depth`` is derived from the report
    period (Q1/Q3 → TOP10, semi-annual/annual → FULL).

    DATA GAP (see :data:`DATA_GAPS`): this interface returns stock positions only — no
    industry allocation and no stock/bond/cash split. A snapshot here is a Tier-1 *input*
    to exposure estimation, never emitted as "current holdings".
    """
    cli = _resolve_client(client)
    dst = _resolve_store(store, persist)
    ts_code = fund_code_to_ts_code(fund_or_ts_code)
    fund_code = ts_code_to_fund_code(ts_code)

    raw = cli.query("fund_portfolio", {"ts_code": ts_code}, _FUND_PORTFOLIO_FIELDS)

    # group rows by (report_end_date, ann_date)
    grouped: dict[tuple[date, date], list[HoldingLine]] = {}
    storage: list[dict[str, Any]] = []
    for row in raw.to_dict("records"):
        ann = _parse_date(row.get("ann_date"))
        end = _parse_date(row.get("end_date"))
        symbol = str(row.get("symbol") or "").strip()
        if ann is None or end is None or not symbol:
            continue  # fail-closed: undated holdings line dropped
        line = HoldingLine(
            symbol=symbol,
            mkv=_to_float(row.get("mkv")),
            amount=_to_float(row.get("amount")),
            stk_mkv_ratio=_to_float(row.get("stk_mkv_ratio")),
            stk_float_ratio=_to_float(row.get("stk_float_ratio")),
        )
        grouped.setdefault((end, ann), []).append(line)
        storage.append(
            {
                "fund_code": fund_code,
                "ts_code": ts_code,
                "report_end_date": end,
                "depth": _depth_for(end).value,
                "symbol": symbol,
                "mkv": line.mkv,
                "amount": line.amount,
                "stk_mkv_ratio": line.stk_mkv_ratio,
                "stk_float_ratio": line.stk_float_ratio,
                AVAILABLE_AT: _disclosure_available_at(ann),
            }
        )

    snapshots: list[DisclosedHoldings] = []
    for (end, ann), lines in grouped.items():
        snapshots.append(
            DisclosedHoldings(
                fund_code=fund_code,
                report_end_date=end,
                depth=_depth_for(end),
                holdings=lines,
                available_at=_disclosure_available_at(ann),
            )
        )
    snapshots.sort(key=lambda s: (s.report_end_date, s.available_at))

    _put(dst, TABLE_FUND_PORTFOLIO, storage)
    return snapshots


# =============================================================================
# (6)(7) EOD index / SW industry daily series (available_at = trade_date @ EOD)
# =============================================================================
def _eod_series(
    api_name: str,
    table: str,
    ts_code: str,
    client: Optional[TushareClient],
    store: Optional[PointInTimeStore],
    start_date: Optional[str],
    end_date: Optional[str],
    persist: bool,
    fields: Optional[str],
) -> pd.DataFrame:
    cli = _resolve_client(client)
    dst = _resolve_store(store, persist)
    params: dict[str, Any] = {"ts_code": ts_code}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    raw = cli.query(api_name, params, fields)
    if raw.empty:
        return raw

    df = raw.copy()
    parsed = df["trade_date"].map(_parse_date)
    keep = parsed.notna()
    df = df.loc[keep].copy()  # fail-closed: bars without a trade_date cannot be PIT-tagged
    parsed = parsed[keep]
    df["trade_date"] = list(parsed)
    df[AVAILABLE_AT] = [_eod_available_at(d) for d in parsed]

    if dst is not None and not df.empty:
        _put(dst, table, df.to_dict("records"))
    return df.reset_index(drop=True)


def index_daily(
    ts_code: str,
    *,
    client: Optional[TushareClient] = None,
    store: Optional[PointInTimeStore] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    persist: bool = True,
    fields: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a broad / sector / style index daily series (Tushare ``index_daily``).

    Returns the raw bars tagged with ``available_at`` = trade_date at the EOD close, and
    caches them so exposure estimation / market-state reads inherit the point-in-time
    filter. EOD look-ahead rule: a *D*-day 14:30 signal uses only closes through *D-1*.
    """
    return _eod_series(
        "index_daily", TABLE_INDEX_DAILY, ts_code, client, store, start_date, end_date, persist, fields
    )


def sw_daily(
    ts_code: str,
    *,
    client: Optional[TushareClient] = None,
    store: Optional[PointInTimeStore] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    persist: bool = True,
    fields: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a Shenwan (申万) industry index daily series (Tushare ``sw_daily``).

    Same EOD PIT tagging as :func:`index_daily`; used as an industry regressor for
    returns-based exposure estimation and as a sector market-state input.
    """
    return _eod_series(
        "sw_daily", TABLE_SW_DAILY, ts_code, client, store, start_date, end_date, persist, fields
    )
