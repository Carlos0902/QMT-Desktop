"""Holdings-based industry exposure + the holding-exposure-leading signal (§7.1 hyp-3).

Instruction §7.1 hypothesis 3: a fund's *holding exposure* and the recent performance
of its underlying assets may lead its NAV. This module turns disclosed top-10 holdings
into a cross-sectional signal the search can use — genuinely new information beyond the
NAV price series.

Point-in-time discipline (this is where look-ahead is easiest to introduce, §3.2):
  * Each holdings snapshot is usable only from its ``available_at`` (= ann_date, the
    disclosure date), NOT its ``report_end_date``. We forward-fill the *latest disclosed*
    snapshot onto each decision date — the "estimated exposure ≠ current holdings"
    constraint: exposure is an estimate carrying a disclosure age, never current truth.
  * Industry index returns use only bars usable at the decision date (visible_lag).

Signal ``hold_mom``: at each date, a fund's exposure-weighted trailing industry return
= Σ_industry (disclosed weight in industry) × (industry index return over ``window``).
High when the fund is tilted toward industries with strong recent momentum.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore, TradingCalendar
from qmt.data.ingest import load_cached_industry_map
from qmt.data.sources import TABLE_FUND_PORTFOLIO, TABLE_SW_DAILY
from qmt.factor.panel import usable_price_frame

__all__ = ["industry_momentum_frame", "build_holdings_momentum_frame", "disclosure_age_days"]

_SENTINEL = pd.Timestamp("2100-01-01")


def _symbol6(sym: str) -> str:
    """Strip exchange suffix: '300750.SZ' -> '300750'."""
    return str(sym).split(".")[0]


def industry_momentum_frame(
    store: PointInTimeStore,
    *,
    window: int = 60,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """(usable-date x industry-name) trailing-``window`` return of each Shenwan index."""
    px = usable_price_frame(
        store, TABLE_SW_DAILY, value_col="close", code_col="ts_code",
        date_col="trade_date", visible_lag_days=1, calendar=calendar,
    )
    if px.empty:
        return px
    # ts_code -> industry name (keep first name seen per code)
    sw = store.get_asof(TABLE_SW_DAILY, _SENTINEL)[["ts_code", "name"]].drop_duplicates("ts_code")
    code2name = dict(zip(sw["ts_code"], sw["name"]))
    px = px.rename(columns=code2name)
    px = px.loc[:, ~px.columns.duplicated()]           # one column per industry name
    return px.pct_change(window, fill_method=None)


def _fund_exposure_snapshots(
    holdings: pd.DataFrame, industry_of: dict[str, str]
) -> pd.DataFrame:
    """(available_at x industry) disclosed weight (fraction of NAV) for one fund."""
    h = holdings.copy()
    h["industry"] = h["symbol"].map(lambda s: industry_of.get(_symbol6(s)))
    h = h.dropna(subset=["industry"])
    if h.empty:
        return pd.DataFrame()
    h["w"] = pd.to_numeric(h["stk_mkv_ratio"], errors="coerce") / 100.0
    h["av"] = pd.to_datetime(h["available_at"]).dt.normalize()
    snap = h.groupby(["av", "industry"])["w"].sum().unstack("industry")
    return snap.sort_index()


def build_holdings_momentum_frame(
    store: PointInTimeStore,
    codes: list[str],
    ref_index: pd.Index,
    *,
    window: int = 60,
    calendar: TradingCalendar | None = None,
    industry_of: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the (date x fund) ``hold_mom`` signal aligned to ``ref_index``.

    For each fund and date: exposure vector from the latest snapshot with
    ``available_at <= date`` (forward-filled), dotted with the industry momentum at
    that date. Funds without usable holdings get all-NaN columns (dropped downstream).
    ``industry_of`` (stock6 -> industry name) defaults to the cached Shenwan map.
    """
    imap = industry_of if industry_of is not None else load_cached_industry_map()
    ind_mom = industry_momentum_frame(store, window=window, calendar=calendar)
    if ind_mom.empty:
        return pd.DataFrame(index=ref_index)
    ref = pd.DatetimeIndex(pd.to_datetime(list(ref_index)))
    ind_mom = ind_mom.reindex(ref).ffill()             # industry momentum on the decision grid

    fp = store.get_asof(TABLE_FUND_PORTFOLIO, _SENTINEL)
    fp = fp[fp["fund_code"].astype(str).isin(set(map(str, codes)))]
    out: dict[str, pd.Series] = {}
    for fc, grp in fp.groupby("fund_code"):
        snaps = _fund_exposure_snapshots(grp, imap)
        if snaps.empty:
            continue
        # align exposure columns to industry-momentum columns; ffill snapshots onto grid
        snaps = snaps.reindex(columns=ind_mom.columns)
        snaps.index = pd.DatetimeIndex(snaps.index)
        expo = snaps.reindex(ref, method="ffill")      # latest disclosed exposure as-of each date
        prod = (expo * ind_mom).sum(axis=1, min_count=1)
        out[str(fc)] = prod
    if not out:
        return pd.DataFrame(index=ref_index)
    frame = pd.DataFrame(out)
    frame.index = ref_index                            # restore caller's index labels (dates)
    return frame


def disclosure_age_days(
    store: PointInTimeStore, fund_code: str, as_of: date
) -> int | None:
    """Trading-day-agnostic age (calendar days) of the latest holdings disclosure
    usable at ``as_of`` — the 'how stale is this estimate' number (§3.2)."""
    fp = store.get_asof(TABLE_FUND_PORTFOLIO, pd.Timestamp(as_of))
    fp = fp[fp["fund_code"].astype(str) == str(fund_code)]
    if fp.empty:
        return None
    latest = pd.to_datetime(fp["available_at"]).max()
    return int((pd.Timestamp(as_of) - latest).days)
