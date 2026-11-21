"""Assembled MARKET-level feature panel for the directional-forecasting search (§5 / §7).

:mod:`qmt.pipeline.features` is deliberately price-only and single-series. This module assembles a
*wider* panel for the systematic feature-engineering search — pulling volume/liquidity, breadth
across the 31 Shenwan industries, valuation (PE/PB/mktcap), vol-managed transforms, and calendar
effects — all aligned to the CSI300 trading grid and all **causal**: the value at bar ``j`` uses
only bars ``<= j``, so a decision at ``i`` reading row ``i-1`` cannot peek at its target.

Every column is grouped into a *family* so the search can test families individually and in
combination, and so permutation importance is interpretable. Nothing here fits or predicts; it only
builds the panel. Modeling + the multiple-testing-aware search live in
:mod:`qmt.pipeline.model_bench`.

Point-in-time notes:
* Industry closes are aligned to the CSI300 dates by forward-fill (a missing industry day takes its
  last *known* value — causal), never back-fill.
* Valuation (PE/mktcap) and volume are read as-of and used only on trailing windows.
* Calendar features are deterministic functions of the date (no lookahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from qmt.pipeline.features import feature_frame

__all__ = ["MarketPanel", "build_market_panel", "FEATURE_GROUPS"]

_BENCH = "000300.SH"
_EPS = 1e-9


@dataclass(frozen=True)
class MarketPanel:
    """Aligned causal feature panel on the CSI300 grid."""

    dates: list[date]
    close: np.ndarray            # CSI300 close (for forward-return targets)
    names: list[str]
    X: np.ndarray                # (n_dates, n_features), row j uses bars <= j
    groups: dict[str, list[str]]  # family -> feature names


# Families are declared here so the search harness can enumerate them without hard-coding names.
FEATURE_GROUPS = [
    "price_momentum", "volatility", "vol_managed", "volume_liquidity",
    "breadth", "valuation", "calendar",
]


def _zscore(s: pd.Series, win: int) -> np.ndarray:
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return ((s - m) / sd.replace(0.0, np.nan)).to_numpy()


def build_market_panel(store, benchmark: str = _BENCH) -> MarketPanel:
    """Assemble the full causal market feature panel on the benchmark's trading grid."""
    from qmt.data.sources import TABLE_INDEX_DAILY, TABLE_SW_DAILY

    idx = store.get_asof(TABLE_INDEX_DAILY, datetime(2200, 1, 1), {"ts_code": benchmark})
    if idx is None or idx.empty:
        return MarketPanel([], np.empty(0), [], np.empty((0, 0)), {})
    idx = idx.copy()
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    idx = idx.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    idx = idx[pd.to_numeric(idx["close"], errors="coerce") > 0].reset_index(drop=True)
    dts = [d.date() for d in idx["trade_date"]]
    C = pd.to_numeric(idx["close"], errors="coerce").to_numpy(float)
    amount = pd.to_numeric(idx.get("amount"), errors="coerce").to_numpy(float) if "amount" in idx else np.full(len(C), np.nan)
    vol = pd.to_numeric(idx.get("vol"), errors="coerce").to_numpy(float) if "vol" in idx else np.full(len(C), np.nan)
    n = len(C)
    ret1 = np.empty(n); ret1[0] = 0.0; ret1[1:] = np.diff(np.log(C))

    cols: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}

    def add(group: str, name: str, arr: np.ndarray) -> None:
        cols.append(np.asarray(arr, float)); names.append(name); groups[group].append(name)

    # --- price momentum / trend / oscillator (reuse the price-only library) ---
    pnames, PM = feature_frame(C)
    for c, nm in enumerate(pnames):
        grp = "volatility" if nm.startswith("vol") else "price_momentum"
        add(grp, nm, PM[:, c])
    pidx = {nm: c for c, nm in enumerate(pnames)}

    # --- vol-managed momentum (Moreira-Muir): momentum scaled by its own recent vol ---
    v20 = PM[:, pidx["vol20"]]
    for L in (20, 60, 120):
        add("vol_managed", f"volmgd_mom{L}", PM[:, pidx[f"mom{L}"]] / (v20 * np.sqrt(L) + _EPS))
    # vol term-structure: short vs long realized vol
    add("vol_managed", "vol_term_20_60", PM[:, pidx["vol20"]] / (PM[:, pidx["vol60"]] + _EPS) - 1.0)

    # --- volume / liquidity ---
    la = np.log(np.where(amount > 0, amount, np.nan))
    add("volume_liquidity", "amount_z250", _zscore(pd.Series(la), 250))
    add("volume_liquidity", "amount_mom60", pd.Series(la).diff(60).to_numpy())
    amihud = pd.Series(np.abs(ret1) / (amount / 1e8 + _EPS)).rolling(20, min_periods=10).mean().to_numpy()
    add("volume_liquidity", "amihud20", amihud)
    add("volume_liquidity", "vol_z250", _zscore(pd.Series(np.log(np.where(vol > 0, vol, np.nan))), 250))

    # --- breadth / cross-section across the 31 industries ---
    sw = store.get_asof(TABLE_SW_DAILY, datetime(2200, 1, 1))
    if sw is not None and not sw.empty:
        sw = sw.copy(); sw["trade_date"] = pd.to_datetime(sw["trade_date"])
        cpiv = sw.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").sort_index()
        cpiv = cpiv.reindex(pd.to_datetime(dts)).ffill()      # causal align to CSI300 grid
        IC = cpiv.to_numpy(float)                              # (n_dates, 31)
        ma60 = cpiv.rolling(60, min_periods=30).mean().to_numpy()
        above = np.where(np.isfinite(IC) & np.isfinite(ma60), (IC > ma60).astype(float), np.nan)
        breadth = np.nansum(above, axis=1) / np.maximum(np.sum(np.isfinite(IC) & np.isfinite(ma60), axis=1), 1)
        add("breadth", "pct_above_ma60", breadth)
        iret = np.vstack([np.zeros(IC.shape[1]), np.diff(np.log(np.where(IC > 0, IC, np.nan)), axis=0)])
        # nanstd on a row with <2 finite industries warns "Degrees of freedom <= 0"; compute it only
        # on the valid rows (a cross-sectional dispersion needs >=2 industries to be defined).
        _disp = np.full(iret.shape[0], np.nan)
        _valid = np.isfinite(iret).sum(axis=1) >= 2
        if _valid.any():
            _disp[_valid] = np.nanstd(iret[_valid], axis=1)
        add("breadth", "xs_dispersion20", pd.Series(_disp).rolling(20, min_periods=10).mean().to_numpy())
        # valuation (aggregate earnings yield z + PB z + EY 60d change)
        pepiv = sw.pivot_table(index="trade_date", columns="ts_code", values="pe", aggfunc="last").sort_index().reindex(pd.to_datetime(dts)).ffill()
        pbpiv = sw.pivot_table(index="trade_date", columns="ts_code", values="pb", aggfunc="last").sort_index().reindex(pd.to_datetime(dts)).ffill()
        mvpiv = sw.pivot_table(index="trade_date", columns="ts_code", values="total_mv", aggfunc="last").sort_index().reindex(pd.to_datetime(dts)).ffill()
        w = mvpiv.div(mvpiv.sum(axis=1), axis=0)
        ey = (1.0 / pepiv.where(pepiv > 0))
        agg_ey = (w * ey).sum(axis=1)
        agg_pb = (w * pbpiv.where(pbpiv > 0)).sum(axis=1)
        add("valuation", "ey_z500", _zscore(agg_ey, 500))
        add("valuation", "ey_chg60", agg_ey.diff(60).to_numpy())
        add("valuation", "pb_z500", -_zscore(agg_pb, 500))     # high PB = expensive -> negate so higher=cheaper
    else:
        for nm, g in (("pct_above_ma60", "breadth"), ("xs_dispersion20", "breadth"),
                      ("ey_z500", "valuation"), ("ey_chg60", "valuation"), ("pb_z500", "valuation")):
            add(g, nm, np.full(n, np.nan))

    # --- volume-conditioned reversal (Campbell-Grossman-Wang): last return, signed by low volume ---
    # CGW: return autocorrelation is more negative when volume is low (uninformed flow reverses).
    if np.isfinite(amount).any():
        az = _zscore(pd.Series(la), 60)
        r5 = np.concatenate([np.zeros(5), np.log(C[5:]) - np.log(C[:-5])])   # trailing 5d return, causal
        add("volume_liquidity", "cgw_reversal", -r5 * (-np.nan_to_num(az)))  # low-vol (neg z) * recent up -> expect reversal

    # --- calendar / seasonality (deterministic; no lookahead) ---
    months = np.array([d.month for d in dts], float)
    add("calendar", "month_sin", np.sin(2 * np.pi * months / 12.0))
    add("calendar", "month_cos", np.cos(2 * np.pi * months / 12.0))
    doms = np.array([d.day for d in dts], float)
    add("calendar", "turn_of_month", ((doms <= 3) | (doms >= 26)).astype(float))
    # Chinese New Year window (documented China effect: pre-holiday drift + post-holiday optimism).
    # Approximate CNY (lunar new year) dates; a bump peaking on CNY, ±15 calendar days.
    _cny = {2008:(2,7),2009:(1,26),2010:(2,14),2011:(2,3),2012:(1,23),2013:(2,10),2014:(1,31),
            2015:(2,19),2016:(2,8),2017:(1,28),2018:(2,16),2019:(2,5),2020:(1,25),2021:(2,12),
            2022:(2,1),2023:(1,22),2024:(2,10),2025:(1,29),2026:(2,17),2027:(2,6)}
    import datetime as _dtm
    cny = np.zeros(len(dts))
    for k, d in enumerate(dts):
        cd = _cny.get(d.year)
        if cd:
            dd = (d - _dtm.date(d.year, cd[0], cd[1])).days
            if abs(dd) <= 15:
                cny[k] = 1.0 - abs(dd) / 15.0       # triangular bump, 1 at CNY -> 0 at ±15d
    add("calendar", "cny_window", cny)
    # Heston-Sadka same-month seasonality: the market's OWN average return in this calendar month
    # over all PRIOR years (causal — only past months). A weak but documented monthly-mood effect.
    ann_ret = np.concatenate([np.zeros(1), np.diff(np.log(C))])
    same_month = np.full(len(dts), np.nan)
    from collections import defaultdict
    hist = defaultdict(list)
    cur_m = None; cur_sum = 0.0
    for k, d in enumerate(dts):
        m = d.month
        prior = hist.get(m)
        same_month[k] = float(np.mean(prior)) if prior else 0.0
        # accumulate this month's return into history at month end (causal: only closed months feed future)
        if k > 0 and dts[k].month != dts[k-1].month:
            hist[dts[k-1].month].append(cur_sum); cur_sum = 0.0
        cur_sum += ann_ret[k]
    add("calendar", "same_month_seasonality", same_month)

    # Causal forward-fill each column: an interior NaN (e.g. a transient non-positive amount making
    # a log undefined) takes the LAST KNOWN value — PIT-safe (uses only bars <= j) — so the all-
    # finite region has no interior gaps and the walk-forward embargo stays a valid time gap.
    cols = [pd.Series(a).ffill().to_numpy() for a in cols]
    # Drop a degenerate feature (all-NaN, or zero variance where it is finite) — it carries no
    # information and, worse, an all-NaN column would make the panel's all-finite region empty.
    kept_c, kept_n = [], []
    for a, nm in zip(cols, names):
        fin = a[np.isfinite(a)]
        if fin.size >= 10 and float(np.std(fin)) > 0.0:
            kept_c.append(a); kept_n.append(nm)
    X = np.column_stack(kept_c) if kept_c else np.empty((n, 0))
    kept = set(kept_n)
    groups = {g: [f for f in v if f in kept] for g, v in groups.items()}
    groups = {g: v for g, v in groups.items() if v}
    return MarketPanel(dts, C, kept_n, X, groups)
