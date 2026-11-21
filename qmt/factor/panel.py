"""Cross-sectional panel + label construction (net-new infra; adapts the
``layer_diagnostics`` read-once tape pattern from industry axis to fund axis).

Everything here is point-in-time-safe by *construction*: a value observed on
``obs_date`` is placed at row ``obs_date + visible_lag`` (default one trading day),
so a decision made at ``as_of`` that slices ``index <= as_of`` can never see a value
that was not yet public. NAV from ``qmt.data.sources.fund_nav`` already carries
``available_at = valuation+1``; EOD index/industry bars are ``trade_date @ 15:00``
(after the 14:30 decision) so they too become usable next trading day.

The frame is ``date x asset`` (funds or industries); ``forward_relative_returns``
produces the *relative* (peer-demeaned, alpha-caliber) label the cross-sectional
ranker predicts (instruction §3.3).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore, TradingCalendar
from qmt.factor.types import Label

__all__ = [
    "usable_price_frame",
    "returns_frame",
    "rolling_beta_frame",
    "forward_relative_returns",
    "forward_beta_adjusted_returns",
    "forward_drawdown",
    "build_labels",
    "cross_section_at",
]

_SENTINEL = datetime(2100, 1, 1)


def usable_price_frame(
    store: PointInTimeStore,
    table: str,
    *,
    value_col: str = "close",
    code_col: str = "ts_code",
    date_col: str = "trade_date",
    visible_lag_days: int = 1,
    calendar: TradingCalendar | None = None,
    codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a PIT table once and pivot to a ``(usable_date x asset)`` price frame.

    ``visible_lag_days`` shifts each observation forward so a same-day close cannot
    be consumed by a same-day pre-close decision (no look-ahead).
    """
    cal = calendar or TradingCalendar()
    raw = store.get_asof(table, _SENTINEL)
    if raw.empty:
        return pd.DataFrame()
    if codes is not None:
        raw = raw[raw[code_col].isin(list(codes))]
    keep = [code_col, date_col, value_col] + (["available_at"] if "available_at" in raw.columns else [])
    df = raw[keep].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    # shift each observation to the first date it may inform a decision:
    # usable = max(obs_date + lag trading days, ACTUAL disclosure available_at).
    # Ignoring available_at silently leaks late-published NAVs (skeptic-confirmed:
    # 3.6% of real fund_nav rows publish >3 calendar days late, p99 = 21 days).
    if visible_lag_days:
        shift_map = {d: cal.add_trading_days(d, visible_lag_days)
                     for d in pd.unique(df[date_col])}
        usable = df[date_col].map(shift_map)
    else:
        usable = df[date_col]
    if "available_at" in df.columns:
        avail = pd.to_datetime(df["available_at"]).dt.date
        df["usable"] = np.maximum(pd.to_datetime(usable), pd.to_datetime(avail)).dt.date
    else:
        df["usable"] = usable
    wide = (
        df.pivot_table(index="usable", columns=code_col, values=value_col, aggfunc="last")
        .sort_index()
    )
    return wide


def returns_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Simple one-period returns from a usable-price frame."""
    return price_frame.pct_change()


def _positions(index: pd.Index, as_of: date) -> int:
    """Index of the last usable row at or before ``as_of`` (-1 if none)."""
    arr = np.asarray(index)
    pos = int(np.searchsorted(arr, np.datetime64(as_of) if arr.dtype.kind == "M" else as_of, side="right")) - 1
    return pos


def rolling_beta_frame(
    price_frame: pd.DataFrame,
    market_px: pd.Series,
    *,
    window: int = 250,
    min_periods: int = 120,
) -> pd.DataFrame:
    """Point-in-time rolling beta of each asset vs the market (date x asset).

    Causal by construction (backward-looking rolling moments): the value at row
    ``d`` uses only returns ``<= d``. Used to strip the beta x market-direction
    component out of labels (instruction §3.3 "扣除市场影响", §3.6 fund-type
    control) — a mixed universe (beta 0.02..1.6) otherwise turns the relative-
    return label into a directional bet, drowning alpha in noise.
    """
    ret = price_frame.pct_change(fill_method=None)
    # ffill limit: a market series that ends early must go NaN (fail-closed), not be
    # extrapolated as a constant price fabricating zero market returns at the tail.
    mkt = market_px.reindex(price_frame.index).ffill(limit=5).pct_change(fill_method=None)
    r_roll = ret.rolling(window, min_periods=min_periods)
    m_roll = mkt.rolling(window, min_periods=min_periods)
    # population moments, consistently
    cov = ret.mul(mkt, axis=0).rolling(window, min_periods=min_periods).mean() \
        - r_roll.mean().mul(m_roll.mean(), axis=0)
    var = (mkt * mkt).rolling(window, min_periods=min_periods).mean() - m_roll.mean() ** 2
    var = var.where(var.abs() > 1e-12)
    return cov.div(var, axis=0)


def forward_beta_adjusted_returns(
    price_frame: pd.DataFrame,
    market_px: pd.Series,
    as_of_dates: Sequence[date],
    horizon_days: int,
    *,
    beta_window: int = 250,
) -> dict[date, pd.Series]:
    """Beta-adjusted alpha-caliber label: ``fwd_i − beta_i(as_of)·fwd_mkt``, then
    cross-sectionally demeaned (instruction §3.3).

    ``beta_i(as_of)`` is point-in-time (rolling, uses only data ``<= as_of``);
    the forward returns are the label side (future by definition).
    """
    beta = rolling_beta_frame(price_frame, market_px, window=beta_window)
    mkt = market_px.reindex(price_frame.index).ffill(limit=5)
    out: dict[date, pd.Series] = {}
    idx = price_frame.index
    n = len(idx)
    for as_of in as_of_dates:
        p0 = _positions(idx, as_of)
        if p0 < 0:
            continue
        p1 = min(p0 + horizon_days, n - 1)
        if p1 <= p0:
            continue
        # exit at the LAST OBSERVED price within the window — funds liquidated
        # mid-window settle at their final NAV instead of silently dropping out
        # (keeping dead funds in the label is the survivorship guard, §3.5)
        exit_px = price_frame.iloc[p0:p1 + 1].ffill().iloc[-1]
        fwd = (exit_px / price_frame.iloc[p0]) - 1.0
        fwd = fwd.replace([np.inf, -np.inf], np.nan)
        m0, m1 = mkt.iloc[p0], mkt.iloc[p1]
        fwd_mkt = (m1 / m0 - 1.0) if (pd.notna(m0) and pd.notna(m1) and m0 != 0) else np.nan
        b = beta.iloc[p0]
        if pd.isna(fwd_mkt):
            adj = fwd
        else:
            adj = fwd - b * fwd_mkt          # strip the directional component
        adj = adj.dropna()
        if adj.empty:
            continue
        out[as_of] = adj - adj.mean()        # peer demean removes residual common drift
    return out


def forward_relative_returns(
    price_frame: pd.DataFrame,
    as_of_dates: Sequence[date],
    horizon_days: int,
) -> dict[date, pd.Series]:
    """Per-as_of cross-section of forward *relative* return (peer-demeaned).

    Entry = price usable at ``as_of``; exit = price ~``horizon_days`` rows later.
    ``rel_return[i] = fwd_ret[i] - mean(fwd_ret)`` — the alpha-caliber target
    (instruction §3.3). Uses future prices (allowed: this is the label/outcome).
    """
    out: dict[date, pd.Series] = {}
    idx = price_frame.index
    n = len(idx)
    for as_of in as_of_dates:
        p0 = _positions(idx, as_of)
        if p0 < 0:
            continue
        p1 = min(p0 + horizon_days, n - 1)
        if p1 <= p0:
            continue
        entry = price_frame.iloc[p0]
        # settle liquidated funds at their last observed NAV within the window (§3.5)
        exit_ = price_frame.iloc[p0:p1 + 1].ffill().iloc[-1]
        fwd = (exit_ / entry) - 1.0
        fwd = fwd.replace([np.inf, -np.inf], np.nan).dropna()
        if fwd.empty:
            continue
        out[as_of] = fwd - fwd.mean()
    return out


def forward_drawdown(
    price_frame: pd.DataFrame,
    as_of_dates: Sequence[date],
    horizon_days: int,
) -> dict[date, pd.Series]:
    """Per-as_of max drawdown (<= 0) over the forward window for each asset."""
    out: dict[date, pd.Series] = {}
    idx = price_frame.index
    n = len(idx)
    for as_of in as_of_dates:
        p0 = _positions(idx, as_of)
        if p0 < 0:
            continue
        p1 = min(p0 + horizon_days, n - 1)
        if p1 <= p0:
            continue
        window = price_frame.iloc[p0:p1 + 1]
        running_max = window.cummax()
        dd = (window / running_max - 1.0).min()
        out[as_of] = dd.clip(upper=0.0)
    return out


def build_labels(
    price_frame: pd.DataFrame,
    as_of_dates: Sequence[date],
    *,
    horizon_days: int,
    rebalance: str = "weekly",
    market_px: pd.Series | None = None,
    beta_window: int = 250,
) -> list[Label]:
    """Materialise :class:`Label` rows for a cross-section over decision dates.

    With ``market_px`` (e.g. CSI300 usable closes), labels are **beta-adjusted**
    (§3.3 alpha caliber): fwd − PIT-beta·fwd_mkt, then demeaned. Without it, plain
    peer-demeaned relative returns (suitable for homogeneous universes).
    """
    if market_px is not None:
        rels = forward_beta_adjusted_returns(
            price_frame, market_px, as_of_dates, horizon_days, beta_window=beta_window
        )
    else:
        rels = forward_relative_returns(price_frame, as_of_dates, horizon_days)
    dds = forward_drawdown(price_frame, as_of_dates, horizon_days)
    labels: list[Label] = []
    for as_of in as_of_dates:
        rel = rels.get(as_of)
        if rel is None:
            continue
        dd = dds.get(as_of, pd.Series(dtype=float))
        for code, r in rel.items():
            labels.append(
                Label(
                    fund_code=str(code),
                    as_of=as_of,
                    horizon_days=horizon_days,
                    rel_return=float(r),
                    drawdown=float(dd.get(code, 0.0)),
                    rebalance=rebalance,
                )
            )
    return labels


def cross_section_at(labels: Sequence[Label], as_of: date) -> pd.Series:
    """Collapse a list of labels into the (fund -> rel_return) cross-section at ``as_of``."""
    rows = {lab.fund_code: lab.rel_return for lab in labels if lab.as_of == as_of}
    return pd.Series(rows, dtype=float)


def labels_by_date(labels: Sequence[Label]) -> dict[date, pd.Series]:
    """Group labels into ``{as_of: Series(fund -> rel_return)}`` for metrics."""
    grouped: dict[date, dict[str, float]] = {}
    for lab in labels:
        grouped.setdefault(lab.as_of, {})[lab.fund_code] = lab.rel_return
    return {d: pd.Series(v, dtype=float) for d, v in grouped.items()}
