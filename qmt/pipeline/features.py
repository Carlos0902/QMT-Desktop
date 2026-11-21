"""Point-in-time feature engineering from a price series (§5 modeling philosophy).

Every feature here is **causal by construction**: the value at bar ``j`` uses only bars ``<= j``.
Combined with the decision convention (a decision at bar ``i`` may read bars up to ``i-1``, because
the 14:30 decision cannot see day ``i``'s close), a model that reads ``feature_frame(P)[i-1]`` and
predicts the forward window ``[i+1, i+1+h]`` has a clean two-bar gap and cannot peek at its own
target. This is the same invariant the layer diagnostics enforce, expressed as reusable features.

The library is deliberately price/return-only (no fundamentals, no flows — the v1 data scope), and
deliberately transparent: momentum at several lookbacks, realized volatility, moving-average
distance, trend slope, drawdown, and a bounded oscillator. The point is not a large feature zoo but
a small, interpretable set whose out-of-sample value can be *measured* per feature
(:mod:`qmt.pipeline.model_bench` runs permutation importance under purged walk-forward CV).

Nothing here fits or predicts; it only turns a price path into a causal feature matrix. Modeling
lives in :mod:`qmt.pipeline.model_bench`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "FeatureSpec",
    "MOMENTUM_LOOKBACKS",
    "VOL_WINDOWS",
    "feature_names",
    "feature_frame",
    "relative_feature_frame",
    "forward_realized_vol",
    "market_valuation_zscore",
]

# Lookbacks (trading days). Kept modest and spread across scales; every one is justified by a
# distinct, documented effect (short reversal, 1-3 month momentum, 6-12 month trend).
MOMENTUM_LOOKBACKS = (5, 10, 20, 60, 120, 250)
VOL_WINDOWS = (10, 20, 60)
_MA_WINDOWS = (20, 60)
_SLOPE_WINDOW = 60
_DD_WINDOW = 120
_RSI_WINDOW = 14


@dataclass(frozen=True)
class FeatureSpec:
    """One named causal feature: ``fn(P, logp, ret1) -> np.ndarray`` of length ``len(P)``."""

    name: str
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    kind: str        # "momentum" | "vol" | "trend" | "risk" | "oscillator"


def _log_prices(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logp = np.log(np.asarray(P, dtype=float))
    ret1 = np.empty_like(logp)
    ret1[0] = 0.0
    ret1[1:] = np.diff(logp)
    return logp, ret1


# --- causal primitives (value at j uses only [.. j]) ---------------------------------------
def _momentum(L: int):
    def f(P, logp, ret1):
        out = np.full(logp.size, np.nan)
        out[L:] = logp[L:] - logp[:-L]
        return out
    return f


def _vol(L: int):
    def f(P, logp, ret1):
        return pd.Series(ret1).rolling(L, min_periods=L).std(ddof=0).to_numpy()
    return f


def _ma_dist(L: int):
    def f(P, logp, ret1):
        ma = pd.Series(P).rolling(L, min_periods=L).mean().to_numpy()
        return P / ma - 1.0
    return f


def _ma_ratio(La: int, Lb: int):
    def f(P, logp, ret1):
        a = pd.Series(P).rolling(La, min_periods=La).mean().to_numpy()
        b = pd.Series(P).rolling(Lb, min_periods=Lb).mean().to_numpy()
        return a / b - 1.0
    return f


def _slope(L: int):
    x = np.arange(L, dtype=float)
    x = x - x.mean()
    denom = float(x @ x)

    def f(P, logp, ret1):
        out = np.full(logp.size, np.nan)
        for j in range(L - 1, logp.size):
            y = logp[j - L + 1: j + 1]
            out[j] = float(x @ (y - y.mean())) / denom
        return out
    return f


def _drawdown(L: int):
    def f(P, logp, ret1):
        peak = pd.Series(P).rolling(L, min_periods=L).max().to_numpy()
        return P / peak - 1.0
    return f


def _rsi(L: int):
    def f(P, logp, ret1):
        gain = np.where(ret1 > 0, ret1, 0.0)
        loss = np.where(ret1 < 0, -ret1, 0.0)
        ag = pd.Series(gain).rolling(L, min_periods=L).mean().to_numpy()
        al = pd.Series(loss).rolling(L, min_periods=L).mean().to_numpy()
        rs = ag / np.where(al > 0, al, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return (rsi - 50.0) / 50.0        # centred to [-1, 1], 0 = neutral
    return f


def _specs() -> list[FeatureSpec]:
    s: list[FeatureSpec] = []
    for L in MOMENTUM_LOOKBACKS:
        s.append(FeatureSpec(f"mom{L}", _momentum(L), "momentum"))
    for L in VOL_WINDOWS:
        s.append(FeatureSpec(f"vol{L}", _vol(L), "vol"))
    for L in _MA_WINDOWS:
        s.append(FeatureSpec(f"px_ma{L}", _ma_dist(L), "trend"))
    s.append(FeatureSpec("ma20_ma60", _ma_ratio(20, 60), "trend"))
    s.append(FeatureSpec(f"slope{_SLOPE_WINDOW}", _slope(_SLOPE_WINDOW), "trend"))
    s.append(FeatureSpec(f"dd{_DD_WINDOW}", _drawdown(_DD_WINDOW), "risk"))
    s.append(FeatureSpec(f"rsi{_RSI_WINDOW}", _rsi(_RSI_WINDOW), "oscillator"))
    return s


_SPECS = _specs()


def feature_names() -> list[str]:
    return [sp.name for sp in _SPECS]


def feature_frame(P: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Causal feature matrix for one price series.

    Returns ``(names, M)`` where ``M[j]`` is the vector of features computed from bars ``<= j``.
    A decision at bar ``i`` uses ``M[i-1]``. Rows before enough history contain ``NaN`` (the caller
    drops them). ``M`` has shape ``(len(P), n_features)``.
    """
    P = np.asarray(P, dtype=float)
    logp, ret1 = _log_prices(P)
    cols = [sp.fn(P, logp, ret1) for sp in _SPECS]
    return feature_names(), np.column_stack(cols)


def relative_feature_frame(P_asset: np.ndarray, P_market: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Causal features of an asset **relative to the market** — for the cross-sectional layer.

    Momentum/MA/slope/RSI become the asset's value minus the market's (the industry-deviation
    framing: what the industry did *beyond* the broad move); volatility stays the asset's own.
    Both series must be aligned on the same trading grid.
    """
    names_a, A = feature_frame(P_asset)
    names_m, M = feature_frame(P_market)
    assert names_a == names_m
    rel = A.copy()
    for c, sp in enumerate(_SPECS):
        if sp.kind in ("momentum", "trend", "oscillator"):
            rel[:, c] = A[:, c] - M[:, c]
        # vol / risk (drawdown) stay the asset's own level
    return names_a, rel


def forward_realized_vol(P: np.ndarray, horizon: int) -> np.ndarray:
    """Realized volatility of daily log returns over the FORWARD window ``(j, j+h]`` (an OUTCOME).

    ``out[j]`` uses returns strictly after ``j`` — this is a target, never a feature. ``NaN`` where
    the forward window runs off the end.
    """
    P = np.asarray(P, dtype=float)
    _logp, ret1 = _log_prices(P)
    h = max(1, int(horizon))
    out = np.full(P.size, np.nan)
    for j in range(P.size - h):
        out[j] = float(ret1[j + 1: j + 1 + h].std(ddof=0))
    return out


def market_valuation_zscore(store, dates: list[date], *, window: int = 500) -> np.ndarray:
    """PIT aggregate market VALUATION (earnings-yield z-score) aligned to ``dates``.

    Built from ``sw_daily`` PE + market cap across the 31 Shenwan L1 industries: cap-weighted mean
    earnings yield (1/PE), z-scored over a trailing ``window``. **High z = cheap** (high earnings
    yield → historically higher forward returns); low z = expensive. This is the one signal class
    price-history lacks; it is the strongest *directional* lead in the benchmark (it leans the right
    way at every major turn — expensive at the 2015 peak, cheap at the 2019/2022/2024 bottoms) but
    does NOT pass the out-of-sample tilt gate on this ~12-year sample (D-BENCH-4). Provided so the
    importance analysis can *include* it and show it is real-but-underpowered, not omitted.

    Aligned to ``dates`` by as-of merge (each date takes the latest valuation available on/before
    it). Returns ``NaN`` before enough history. Causal: uses only trailing valuations.
    """
    from datetime import datetime as _dt

    from qmt.data.sources import TABLE_SW_DAILY

    df = store.get_asof(TABLE_SW_DAILY, _dt(2200, 1, 1))
    if df is None or df.empty or "pe" not in df.columns:
        return np.full(len(dates), np.nan)
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    pe = df.pivot_table(index="trade_date", columns="ts_code", values="pe", aggfunc="last").sort_index()
    mv = df.pivot_table(index="trade_date", columns="ts_code", values="total_mv", aggfunc="last").sort_index()
    ey = 1.0 / pe.where(pe > 0)
    w = mv.div(mv.sum(axis=1), axis=0)
    agg = (w * ey).sum(axis=1)
    z = (agg - agg.rolling(window, min_periods=window // 2).mean()) / agg.rolling(window, min_periods=window // 2).std()
    zs = pd.Series(z.values, index=[d.date() for d in agg.index]).sort_index()
    # as-of align: for each target date, the latest z on/before it
    out = np.full(len(dates), np.nan)
    zdates = list(zs.index)
    from bisect import bisect_right as _br
    for i, d in enumerate(dates):
        k = _br(zdates, d) - 1
        if k >= 0 and np.isfinite(zs.iloc[k]):
            out[i] = float(zs.iloc[k])
    return out
