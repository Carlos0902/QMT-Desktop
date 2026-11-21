"""layer_diagnostics — score the §5.3 hierarchical model's THREE layers independently (源文档 §3.3).

The architecture decomposes a forecast into ``大盘 + 行业相对 + 个股自身相对``. A single blended
accuracy number cannot tell you *which* layer is broken, so this module measures each layer on the
task that layer actually performs, point-in-time:

* **market layer (大盘)** — a probabilistic *level* forecast of the broad index. Diagnosed by
  directional skill (rank IC, hit-rate) **and** by interval calibration (coverage of
  ``[ret_low, ret_high]`` against its nominal). Its own target is the market's forward return.
* **industry layer (行业相对)** — a *cross-sectional relative* forecast: which industries beat the
  market. Diagnosed across **all 31 Shenwan L1 indices** (not just a fund's exposures — the full
  cross-section is the larger, lower-variance sample) by the cross-sectional rank IC per date, its
  information ratio, and a top-minus-bottom spread.
* **idiosyncratic layer (个股偏差)** — has a **zero central estimate by design** (the doc forbids a
  fabricated idiosyncratic alpha). A directional metric is therefore *meaningless*, and computing
  one would be fabricated evidence. Its only honest diagnosis is about **uncertainty**: does the
  band it adds actually cover the residual, and is ``idio_residual_vol`` consistent with the
  realised residual dispersion?

Every skill claim is passed through :mod:`qmt.stats`, because the daily-sampled, ``h``-day-forward
skill series is ``h``-fold **overlapping**: the naive i.i.d. t-statistic fires on pure noise 66% of
the time. The verdict requires a non-overlapping-subsample t, a Newey–West t, and a block-bootstrap
CI to agree; the naive t is reported only to expose how much it overstates.

Point-in-time (invariant 1): the *signal* for a decision at ``as_of`` reads only bars with
``available_at <= decision_moment(as_of)`` (14:30, so the same day's 15:00 close is invisible); the
*outcome* is read from a separate all-history path and starts at the first bar strictly **after**
``as_of`` (the first executable point). The two read paths are deliberately separate functions so a
future edit cannot accidentally cross them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from bisect import bisect_right
from datetime import date, datetime, time
from typing import Optional

import math
import numpy as np
import pandas as pd

from qmt.config import CONFIG
from qmt.data.sources import TABLE_INDEX_DAILY, TABLE_SW_DAILY
from qmt.pit import AVAILABLE_AT, PointInTimeStore

logger = logging.getLogger(__name__)


def _market_vol_multiplier(prices_upto: np.ndarray, cfg) -> float:
    """The forecaster's PIT vol-regime multiplier, reused so the diagnostics & curves grade/show the
    band the model actually produces (D-CVOL-1). Delegates to the single implementation."""
    from qmt.pipeline.hierarchical_forecaster import _vol_regime_multiplier

    return _vol_regime_multiplier(prices_upto, cfg)

__all__ = [
    "SeriesTape",
    "decision_moment",
    "forward_returns",
    "load_market_tape",
    "load_industry_tapes",
]

# All-history sentinel for reading *outcomes* (a historical outcome is, by construction, known).
# Kept inside pandas' nanosecond Timestamp range so the PIT filter converts.
_ALL_HISTORY = datetime(2200, 1, 1)


def decision_moment(as_of: date) -> datetime:
    """The T-day decision instant (14:30) — a read here cannot see the same day's 15:00 close."""
    t = CONFIG.timing
    return datetime.combine(as_of, time(hour=t.decision_hour, minute=t.decision_minute))


# =============================================================================
# One point-in-time read of an index series, sliced per decision
# =============================================================================
class SeriesTape:
    """One PIT read of an index/industry close series; ``visible_at(moment)`` slices it.

    Reproduces :func:`qmt.pipeline.hierarchical_forecaster._index_closes` **exactly** — restrict to
    ``available_at <= moment``, keep the **latest disclosure per ``trade_date``**, then drop NaN /
    non-positive closes (de-duplicate *before* filtering, so a ``trade_date`` whose latest
    restatement is invalid disappears rather than resurrecting the stale earlier value). Reading
    once and slicing is equivalent because ``get_asof`` is monotone in ``as_of`` and every decision
    moment is ``<= _ALL_HISTORY``.

    A per-call ``get_asof`` costs ~11 ms; a full replay (≈2,200 dates × 32 series) would take ~13
    minutes. The tape makes it ~3 s.
    """

    __slots__ = ("ts_code", "_td", "_av", "_close", "_unique", "_monotone", "dates", "closes")

    def __init__(self, store: PointInTimeStore, ts_code: str, table: Optional[str] = None) -> None:
        self.ts_code = ts_code
        self._td = np.empty(0, dtype=np.int64)
        self._av = np.empty(0, dtype=np.int64)
        self._close = np.empty(0, dtype=float)
        self._unique = True
        self._monotone = True
        self.dates: list[date] = []
        self.closes: np.ndarray = np.empty(0, dtype=float)

        df = None
        tables = (table,) if table else (TABLE_INDEX_DAILY, TABLE_SW_DAILY)
        for t in tables:
            cand = store.get_asof(t, _ALL_HISTORY, {"ts_code": ts_code})
            if cand is not None and not cand.empty and "close" in cand.columns:
                df = cand
                break
        if df is None or "trade_date" not in df.columns:
            return

        df = df.sort_values(["trade_date", AVAILABLE_AT])
        td_ts = pd.to_datetime(df["trade_date"])
        self._td = td_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._av = pd.to_datetime(df[AVAILABLE_AT]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)

        last = np.ones(self._td.size, dtype=bool)
        last[:-1] = self._td[1:] != self._td[:-1]
        self._unique = bool(last.all())
        self._monotone = bool(self._unique and np.all(np.diff(self._av) >= 0))

        # the full observable history (outcome path): latest row per trade_date, then filter
        c = self._close[last]
        keep = ~np.isnan(c) & (c > 0.0)
        self.dates = [d.date() for d in td_ts[last][keep]]
        self.closes = c[keep]

    def __len__(self) -> int:
        return self.closes.size

    def visible_at(self, moment: datetime, lookback: int = 0) -> np.ndarray:
        """Closes visible at ``moment`` (``available_at <= moment``), oldest→newest.

        ``lookback > 0`` trims to the most recent ``lookback`` observations, matching
        ``_index_closes``.
        """
        if self._td.size == 0:
            return np.empty(0, dtype=float)
        m = np.int64(pd.Timestamp(moment).value)
        if self._monotone:
            k = int(np.searchsorted(self._av, m, side="right"))
            c = self._close[:k]
        else:
            mask = self._av <= m
            c, td = self._close[mask], self._td[mask]
            if not self._unique and td.size:
                last = np.ones(td.size, dtype=bool)
                last[:-1] = td[1:] != td[:-1]
                c = c[last]
        keep = ~np.isnan(c) & (c > 0.0)
        c = c[keep]
        if lookback > 0:
            c = c[-lookback:]
        return c

    def visible_dates_and_closes(self, moment: datetime, lookback: int = 0):
        """Same as :meth:`visible_at` but also returns the aligned ``trade_date`` list."""
        if self._td.size == 0:
            return [], np.empty(0, dtype=float)
        m = np.int64(pd.Timestamp(moment).value)
        if self._monotone:
            k = int(np.searchsorted(self._av, m, side="right"))
            td, c = self._td[:k], self._close[:k]
        else:
            mask = self._av <= m
            td, c = self._td[mask], self._close[mask]
            if not self._unique and td.size:
                last = np.ones(td.size, dtype=bool)
                last[:-1] = td[1:] != td[:-1]
                td, c = td[last], c[last]
        keep = ~np.isnan(c) & (c > 0.0)
        td, c = td[keep], c[keep]
        if lookback > 0:
            td, c = td[-lookback:], c[-lookback:]
        return [pd.Timestamp(x).date() for x in td], c


def forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """``f[j] = prices[j+h]/prices[j] - 1`` (NaN where the window runs past the end)."""
    h = max(1, int(horizon))
    f = np.full(prices.size, np.nan, dtype=float)
    if prices.size > h:
        f[: prices.size - h] = prices[h:] / prices[: prices.size - h] - 1.0
    return f


# =============================================================================
# loaders
# =============================================================================
def load_market_tape(store: PointInTimeStore, benchmark: Optional[str] = None) -> SeriesTape:
    """The broad-market index tape (the 大盘 layer's own series)."""
    code = benchmark or CONFIG.hierarchical.benchmark_index
    return SeriesTape(store, code, table=TABLE_INDEX_DAILY)


def load_industry_tapes(store: PointInTimeStore, min_bars: int = 500) -> dict[str, SeriesTape]:
    """Every Shenwan L1 industry tape that has enough history (the 行业 layer's cross-section)."""
    from qmt.pipeline.hierarchical_forecaster import SW_L1_INDEX_BY_NAME

    out: dict[str, SeriesTape] = {}
    for name, code in SW_L1_INDEX_BY_NAME.items():
        tape = SeriesTape(store, code, table=TABLE_SW_DAILY)
        if len(tape) >= min_bars:
            out[name] = tape
        else:
            logger.info("industry %s (%s): only %d bars, skipped", name, code, len(tape))
    return out


# =============================================================================
# Proper scoring rule for a probabilistic forecast
# =============================================================================
def crps_empirical(sample: np.ndarray, y: float) -> float:
    """CRPS of an empirical predictive distribution ``sample`` against the realised scalar ``y``.

    ``CRPS = E|X - y| - 0.5 * E|X - X'|`` — a **proper** score, in return units, and comparable
    across candidate distributions of different width. That comparability is the point: the pinball
    loss at quantile level ``tau`` is mechanically smaller for a tighter ``tau``, so ranking
    candidates that also *sweep* ``tau`` by pinball loss compares apples to oranges. CRPS integrates
    over all quantiles and does not.

    The pairwise term uses the sorted-array identity ``sum_ij |x_i-x_j| = 2*sum_i (2i-n+1)*x_i``
    (0-indexed), so this is O(n log n), not O(n^2).

    **The pairwise term is normalised by ``n*(n-1)``, not ``n*n``** — the *fair* (unbiased)
    estimator. The ``n*n`` form is biased upward by ``E|X-X'| / (2n)``, a penalty that shrinks as
    the sample grows. That matters here because the layer comparisons deliberately score samples of
    **different sizes** against the same outcome: the market layer's bounded lookback (≈6y, D-HF-8) versus the
    unconditional baseline's entire visible history (up to ~2969 windows). Under the biased form the
    smaller sample is charged a larger penalty *purely for being smaller*, independent of forecast
    quality — a same-distribution simulation (n=750 vs n=3000, zero true skill difference) shows the
    biased estimator manufacturing a spurious edge of ``-3.1e-5`` for the larger sample, about a
    third of the real measured gap. The fair estimator drives that artifact to ``+1.2e-6``.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0 or not np.isfinite(y):
        return float("nan")
    term1 = float(np.mean(np.abs(x - y)))
    if n == 1:
        return term1                                       # a one-point sample has no spread term
    xs = np.sort(x)
    i = np.arange(n, dtype=float)
    pair = 2.0 * float(np.sum((2.0 * i - n + 1.0) * xs))   # = sum_ij |x_i - x_j|
    term2 = pair / (n * (n - 1.0))                         # fair/unbiased, not pair/(n*n)
    return term1 - 0.5 * term2


# =============================================================================
# The common trading grid shared by the market and every industry
# =============================================================================
@dataclass(frozen=True)
class Grid:
    """The aligned trading grid: dates plus market/industry close matrices."""

    dates: list[date]
    market: np.ndarray             # (N,)
    industries: np.ndarray         # (K, N)
    industry_names: list[str]

    @property
    def n(self) -> int:
        return len(self.dates)


def build_grid(market: SeriesTape, industries: dict[str, SeriesTape]) -> Grid:
    """Intersect the market and every industry tape onto one date grid (no forward-filling)."""
    if len(market) == 0 or not industries:
        return Grid([], np.empty(0), np.empty((0, 0)), [])
    idx = pd.Index(market.dates)
    for tp in industries.values():
        idx = idx.intersection(pd.Index(tp.dates))
    idx = idx.sort_values()
    if len(idx) == 0:
        return Grid([], np.empty(0), np.empty((0, 0)), [])

    def reindex(tape: SeriesTape) -> np.ndarray:
        s = pd.Series(tape.closes, index=pd.Index(tape.dates))
        return s.reindex(idx).to_numpy(dtype=float)

    names = sorted(industries)
    return Grid(
        dates=list(idx),
        market=reindex(market),
        industries=np.vstack([reindex(industries[n]) for n in names]) if names else np.empty((0, len(idx))),
        industry_names=names,
    )


# =============================================================================
# Layer reports
# =============================================================================
@dataclass(frozen=True)
class MarketLayerReport:
    """大盘层: a probabilistic LEVEL forecast of the broad index — its own target."""

    horizon: int
    n_obs: int
    ic: float                 # rank correlation of predicted median vs realised (descriptive)
    hit_rate: float
    bias: float               # mean(pred_mid) - mean(realised)
    coverage: float           # fraction of realised inside [ret_low, ret_high]
    nominal_coverage: float   # qhi - qlo
    crps: float               # proper score of the predictive distribution
    direction: "object"       # qmt.stats.Significance on sign(pred)*realised

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon, "n_obs": self.n_obs,
            "ic": _r(self.ic), "hit_rate": _r(self.hit_rate), "bias": _r(self.bias),
            "coverage": _r(self.coverage), "nominal_coverage": _r(self.nominal_coverage),
            "crps": _r(self.crps), "direction": self.direction.as_dict(),
        }


@dataclass(frozen=True)
class IndustryLayerReport:
    """行业相对层: a CROSS-SECTIONAL relative forecast over all Shenwan L1 indices."""

    horizon: int
    n_dates: int
    n_industries: int
    mean_ic: float
    pct_ic_positive: float
    ic: "object"              # Significance on the per-date cross-sectional IC series
    long_short: "object"      # Significance on the top-minus-bottom realised spread
    per_industry: dict        # name -> {"ic": .., "hit_rate": ..}

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon, "n_dates": self.n_dates, "n_industries": self.n_industries,
            "mean_ic": _r(self.mean_ic), "pct_ic_positive": _r(self.pct_ic_positive),
            "ic": self.ic.as_dict(), "long_short": self.long_short.as_dict(),
            "per_industry": {k: {kk: _r(vv) for kk, vv in v.items()} for k, v in self.per_industry.items()},
        }


@dataclass(frozen=True)
class IdioLayerReport:
    """个股偏差层: ZERO central estimate by design → uncertainty-only diagnosis.

    The residual is built from the **realised** market and industry components. Using the
    *forecast* components instead would fold the market layer's forecast error into "idiosyncratic"
    noise, and widening the idio band would then hide a broken market layer.
    """

    horizon: int
    fund_code: str
    n_obs: int
    resid_std: float
    configured_vol: float
    vol_ratio: float          # resid_std / configured_vol (1.0 = calibrated)
    band_coverage: float      # fraction of |resid| <= configured band
    unresolved_weight: float
    bias: "object"            # Significance of mean(resid) != 0 -> exposure mis-scaling, not idio
    note: str

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon, "fund_code": self.fund_code, "n_obs": self.n_obs,
            "resid_std": _r(self.resid_std), "configured_vol": _r(self.configured_vol),
            "vol_ratio": _r(self.vol_ratio), "band_coverage": _r(self.band_coverage),
            "unresolved_weight": _r(self.unresolved_weight),
            "bias": self.bias.as_dict(), "note": self.note,
        }


def _r(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, 6)


# =============================================================================
# Market layer
# =============================================================================
def evaluate_market_layer(grid: Grid, horizon: int, cfg=None, *, start: Optional[date] = None,
                          end: Optional[date] = None) -> MarketLayerReport:
    """Score the market layer against its own target: the broad index's forward return."""
    from qmt.stats import assess_significance, spearman

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    # `lb - h`: the forecaster passes `lb` *prices* to `_empirical_layer`, giving `lb - h` windows.
    lb = max(1, c.market_lookback_trading_days - h)
    qlo, qhi = c.market_ret_low_quantile, c.market_ret_high_quantile
    P, N = grid.market, grid.n
    fm = forward_returns(P, h)

    preds, reals, lows, highs, crps_vals = [], [], [], [], []
    for i in range(1, N):
        d = grid.dates[i]
        if (start and d < start) or (end and d > end):
            continue
        hi = i - 1 - h                       # last window fully observed at the 14:30 decision
        e = i + 1                            # first executable bar (strictly after the decision)
        if hi < 0 or e + h >= N:
            continue
        hist = fm[max(0, hi - lb + 1): hi + 1]
        hist = hist[np.isfinite(hist)]
        if hist.size < c.market_min_windows:
            continue
        lo_, mid, hi_ = np.quantile(hist, [qlo, 0.5, qhi])
        # Grade the band the forecaster actually produces: (1) de-bias the median by shrinking it
        # toward 0 (D-CVOL-3) — the band shifts with it; (2) scale the half-widths by the current
        # vol regime (D-CVOL-1), PIT from prices up to i-1.
        shift = mid * (c.market_median_shrink - 1.0)      # move mid toward 0; band follows
        mid = mid + shift; lo_ = lo_ + shift; hi_ = hi_ + shift
        reff = _market_vol_multiplier(P[:i], c)
        lo_, hi_ = mid + (lo_ - mid) * reff, mid + (hi_ - mid) * reff
        y = P[e + h] / P[e] - 1.0
        preds.append(mid); reals.append(y); lows.append(lo_); highs.append(hi_)
        crps_vals.append(crps_empirical(mid + (hist - np.median(hist)) * reff, y))  # mid already shrunk

    if len(preds) < 10:
        from qmt.stats import Significance
        nan = float("nan")
        return MarketLayerReport(h, len(preds), nan, nan, nan, nan, qhi - qlo, nan,
                                 Significance(nan, 0, h, nan, nan, 0, nan, nan, nan, False,
                                              "样本不足，无法检验（fail-closed）。"))

    pm, rm = np.array(preds), np.array(reals)
    lo_a, hi_a = np.array(lows), np.array(highs)
    return MarketLayerReport(
        horizon=h, n_obs=pm.size,
        ic=spearman(pm, rm),
        hit_rate=float(np.mean(np.sign(pm) == np.sign(rm))),
        bias=float(pm.mean() - rm.mean()),
        coverage=float(np.mean((rm >= lo_a) & (rm <= hi_a))),
        nominal_coverage=float(qhi - qlo),
        crps=float(np.mean(crps_vals)),
        direction=assess_significance(np.sign(pm) * rm, overlap=h),
    )


# =============================================================================
# Industry layer (cross-sectional)
# =============================================================================
def evaluate_industry_layer(grid: Grid, horizon: int, cfg=None, *, start: Optional[date] = None,
                            end: Optional[date] = None, top_k: int = 6) -> IndustryLayerReport:
    """Score the industry layer on its own task: ranking industries relative to the market."""
    from qmt.stats import assess_significance, spearman

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    lb = max(1, c.industry_lookback_trading_days - h)   # `lb` prices -> `lb - h` deviation windows
    P, IND, N, K = grid.market, grid.industries, grid.n, len(grid.industry_names)
    if K == 0 or N == 0:
        from qmt.stats import Significance
        nan = float("nan")
        empty = Significance(nan, 0, h, nan, nan, 0, nan, nan, nan, False, "无行业数据（fail-closed）。")
        return IndustryLayerReport(h, 0, 0, nan, nan, empty, empty, {})

    fm = forward_returns(P, h)
    fdev = np.vstack([forward_returns(IND[k], h) for k in range(K)]) - fm[None, :]

    ic_series, ls_series = [], []
    hits = np.zeros(K); cnt = np.zeros(K)
    pred_all, real_all = [[] for _ in range(K)], [[] for _ in range(K)]
    for i in range(1, N):
        d = grid.dates[i]
        if (start and d < start) or (end and d > end):
            continue
        hi = i - 1 - h
        e = i + 1
        if hi < 0 or e + h >= N:
            continue
        w = fdev[:, max(0, hi - lb + 1): hi + 1]
        if w.shape[1] < c.industry_min_windows:
            continue
        pred = np.nanmedian(w, axis=1)
        rm = P[e + h] / P[e] - 1.0
        real = (IND[:, e + h] / IND[:, e] - 1.0) - rm
        ok = np.isfinite(pred) & np.isfinite(real)
        if ok.sum() < 5:
            continue
        ic = spearman(pred[ok], real[ok])
        if not np.isfinite(ic):
            continue
        ic_series.append(ic)
        order = np.argsort(pred[ok])
        idx = np.flatnonzero(ok)[order]
        k = min(top_k, idx.size // 2)
        if k >= 1:
            ls_series.append(float(real[idx[-k:]].mean() - real[idx[:k]].mean()))
        hits[ok] += (np.sign(pred[ok]) == np.sign(real[ok]))
        cnt[ok] += 1
        for j in np.flatnonzero(ok):
            pred_all[j].append(pred[j]); real_all[j].append(real[j])

    ics = np.array(ic_series, dtype=float)
    per = {}
    for j, name in enumerate(grid.industry_names):
        if cnt[j] >= 20:
            per[name] = {"ic": spearman(np.array(pred_all[j]), np.array(real_all[j])),
                         "hit_rate": float(hits[j] / cnt[j])}
    return IndustryLayerReport(
        horizon=h, n_dates=ics.size, n_industries=K,
        mean_ic=float(ics.mean()) if ics.size else float("nan"),
        pct_ic_positive=float(np.mean(ics > 0)) if ics.size else float("nan"),
        ic=assess_significance(ics, overlap=h),
        long_short=assess_significance(np.array(ls_series, dtype=float), overlap=h),
        per_industry=per,
    )


# =============================================================================
# Idiosyncratic layer — uncertainty only (a directional metric here is forbidden)
# =============================================================================
class DirectionalMetricForbidden(ValueError):
    """Raised if anything asks the idiosyncratic layer for a directional/skill metric.

    Its central estimate is ZERO by construction (§5.3: never a fabricated 个股 alpha), so an IC,
    hit-rate or Sharpe computed on it would be measuring noise around a constant and presenting it
    as evidence. A coded guard, not a comment.
    """


def _fund_returns(store: PointInTimeStore, fund_code: str) -> tuple[list[date], np.ndarray]:
    """The fund's realised adj_nav path (OUTCOME read — all history, never a signal read)."""
    from qmt.backtest import _PriceTape, _end_moment

    tape = _PriceTape(store, fund_code, _end_moment(date(2200, 1, 1)))
    return tape.outcome_path()


def _fund_window(f_dates: list, f_vals: np.ndarray, start_after: date, end_on: date):
    """Realised fund return over the **real-time** interval ``(start_after, end_on]``.

    The forecast horizon ``h`` is a count of *index* trading days, but a fund's NAV calendar has
    holes (suspensions, late disclosure). Advancing ``h`` positions along the NAV array therefore
    measures a **different, systematically longer** real-time window than the one the forecast
    covers: on this data a "20-day" fund window spans 22.7-27.2 index days on average and up to 35
    (38-53% of samples are off), inflating the realised dispersion the interval is graded against.

    So we align by DATE: enter at the first NAV strictly after the decision, exit at the last NAV on
    or before the horizon's index-calendar end date. Returns ``None`` when the window is degenerate.
    """
    n = len(f_dates)
    if n == 0:
        return None
    lo = bisect_right(f_dates, start_after)          # first NAV strictly after the decision date
    hi = bisect_right(f_dates, end_on) - 1           # last NAV on or before the horizon end
    # `lo == 0` means the fund had no NAV at or before the decision date: it did not exist yet, so
    # there was nothing to decide about. (The old `next(fd > d)` scan silently returned index 0 here
    # and measured a window starting at the fund's inception, years away from the decision.)
    if lo == 0 or lo >= n or hi <= lo:
        return None
    v0, v1 = float(f_vals[lo]), float(f_vals[hi])
    if not (math.isfinite(v0) and math.isfinite(v1)) or v0 <= 0.0:
        return None
    return v1 / v0 - 1.0


def _idio_residual_series(
    store: PointInTimeStore,
    fund_code: str,
    grid: Grid,
    horizon: int,
    *,
    industry_map: Optional[dict] = None,
    cfg=None,
    sample_every: int = 21,
    max_unresolved: float = 0.5,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> tuple[list[date], list[float], list[float]]:
    """Chronological ``(dates, residuals, unresolved_weights)`` for the idio layer.

    Each residual is ``r_fund − loading·(within-book industry mix, realised)`` over the horizon's
    index-calendar window (see :func:`evaluate_idiosyncratic_layer` for why realised components and
    why the index calendar). With ``sample_every > horizon`` the windows are pairwise disjoint, so
    a trailing slice ``resids[:k]`` is fully closed before ``dates[k]`` — which is what makes this
    series usable as a **point-in-time calibration input**, not just a report.
    """
    from qmt.pipeline.exposure_estimator import UNCLASSIFIED_LABEL, estimate_exposure, estimate_market_beta
    from qmt.types import ExposureDimension

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    f_dates, f_vals = _fund_returns(store, fund_code)
    if len(f_dates) < h + 5:
        return [], [], []
    name_index = {n: j for j, n in enumerate(grid.industry_names)}

    r_dates: list[date] = []
    resids, unresolved_w = [], []
    for i in range(1, grid.n, max(1, sample_every)):
        d = grid.dates[i]
        if (start and d < start) or (end and d > end):
            continue
        e = i + 1
        if e + h >= grid.n:
            break
        # --- point-in-time exposures (signal read) ---
        exps = estimate_exposure(fund_code, d, store, industry_map=industry_map)
        ind = [x for x in exps if x.dimension is ExposureDimension.INDUSTRY]
        if not ind:
            continue
        # Mirror the forecaster's industry selection EXACTLY: it models only the top
        # `max_industries` by weight and routes the rest (plus unclassified) into the market proxy
        # via unmodeled_weight (D-HF-1). Crediting every mapped industry here would grade a model the
        # forecaster never runs — the residual would differ for any fund with > max_industries.
        cap = c.max_industries
        resolved = sorted(
            ((name_index[x.label], float(x.point_estimate)) for x in ind
             if x.label != UNCLASSIFIED_LABEL and x.label in name_index),
            key=lambda kv: kv[1], reverse=True,
        )
        w_res: dict = {}
        for j, w in resolved[:cap]:
            w_res[j] = w_res.get(j, 0.0) + w
        # unclassified + industries beyond the cap → market proxy (same as the forecaster)
        w_unres = (
            sum(float(x.point_estimate) for x in ind
                if x.label == UNCLASSIFIED_LABEL or x.label not in name_index)
            + sum(w for _j, w in resolved[cap:])
        )
        total = sum(w_res.values()) + w_unres
        if total <= 0 or w_unres / total > max_unresolved:
            unresolved_w.append(w_unres / total if total > 0 else 1.0)
            continue

        # --- REALISED components (outcome read), scaled by the fund's PIT market loading ---
        # Same estimator + fallback as forecast_hierarchical: without the loading, `explained`
        # models the fund as a 100%-loaded clone of its top-10 industry mix and the residual is
        # polluted by cash drag — the band then gets graded against model misspecification.
        beta = estimate_market_beta(
            fund_code, d, store,
            benchmark=c.benchmark_index, window=c.equity_beta_window_days,
            min_obs=c.equity_beta_min_obs, clip_high=c.equity_beta_clip_high,
        )
        loading = beta if beta is not None else min(total, 1.0)
        rm = grid.market[e + h] / grid.market[e] - 1.0
        explained = w_unres * rm
        for j, w in w_res.items():
            explained += w * (grid.industries[j][e + h] / grid.industries[j][e] - 1.0)
        explained *= loading / total          # within-book shares -> fund-level loading

        # --- realised fund return over the SAME real-time window the components span ---
        r_fund = _fund_window(f_dates, f_vals, d, grid.dates[e + h])
        if r_fund is None:
            continue
        r_dates.append(d)
        resids.append(r_fund - explained)
        unresolved_w.append(w_unres / total)
    return r_dates, resids, unresolved_w


def evaluate_idiosyncratic_layer(
    store: PointInTimeStore,
    fund_code: str,
    grid: Grid,
    horizon: int,
    *,
    industry_map: Optional[dict] = None,
    cfg=None,
    sample_every: int = 21,
    max_unresolved: float = 0.5,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> IdioLayerReport:
    """Score the idio layer's ONLY honest job: does its band cover the residual it must cover?

    The residual is ``R_fund - explained``, where ``explained`` is built from the **realised**
    market and industry components under the fund's point-in-time exposures::

        explained = sum_j w_j * r_j(realised)  +  w_unresolved * r_market(realised)

    (identically ``E*r_m + sum_j w_j*(r_j - r_m)``). Using the *forecast* components would fold the
    market layer's forecast error into this residual — then a wider idio band would "fix" the
    number while the market layer stayed broken. Fails closed when too much of the book cannot be
    mapped to an industry index.
    """
    from qmt.pipeline.exposure_estimator import UNCLASSIFIED_LABEL, estimate_exposure, estimate_market_beta
    from qmt.stats import Significance, assess_significance
    from qmt.types import ExposureDimension

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    nan = float("nan")
    _ovl = max(1, int(np.ceil(h / max(1, sample_every))))   # same overlap the real path reports
    empty_sig = Significance(nan, 0, _ovl, nan, nan, 0, nan, nan, nan, False, "样本不足（fail-closed）。")

    f_dates, f_vals = _fund_returns(store, fund_code)
    if len(f_dates) < h + 5:
        return IdioLayerReport(h, fund_code, 0, nan, c.idio_residual_vol, nan, nan, nan, empty_sig,
                               "基金净值历史不足，无法评估（fail-closed）。")

    r_dates, resids, unresolved_w = _idio_residual_series(
        store, fund_code, grid, h, industry_map=industry_map, cfg=c,
        sample_every=sample_every, max_unresolved=max_unresolved, start=start, end=end,
    )

    if len(resids) < 10:
        return IdioLayerReport(h, fund_code, len(resids), nan, c.idio_residual_vol, nan, nan,
                               float(np.mean(unresolved_w)) if unresolved_w else nan, empty_sig,
                               "可用观测不足（多为无法映射到行业指数的持仓），fail-closed。")

    e_arr = np.array(resids, dtype=float)
    resid_std = float(e_arr.std(ddof=1))
    # The config band/vol are calibrated at idio_reference_horizon_days and the forecaster applies
    # them scaled by sqrt(h/ref) — so the residual must be graded against the SAME scaled values,
    # else this report re-creates the flat-band artifact it exists to catch.
    hscale = math.sqrt(h / max(1, c.idio_reference_horizon_days))
    band = c.idio_residual_band * hscale
    scaled_vol = c.idio_residual_vol * hscale
    return IdioLayerReport(
        horizon=h, fund_code=fund_code, n_obs=e_arr.size,
        resid_std=resid_std, configured_vol=scaled_vol,
        vol_ratio=resid_std / scaled_vol if scaled_vol > 0 else nan,
        band_coverage=float(np.mean(np.abs(e_arr) <= band)),
        unresolved_weight=float(np.mean(unresolved_w)),
        # sampled every `sample_every` days, so overlap in units of samples is ceil(h/sample_every)
        bias=assess_significance(e_arr, overlap=max(1, int(np.ceil(h / max(1, sample_every))))),
        note="残差由『已实现』的大盘/行业分量构成，因此不含大盘层的预测误差（不会掩盖坏掉的大盘层）。",
    )


# =============================================================================
# Aggregate report — the three layers, sliced so degradation is visible over time
# =============================================================================
@dataclass(frozen=True)
class LayerReport:
    """The three layers, per horizon, plus per-period slices and a plain-language diagnosis."""

    horizons: list[int]
    market: dict            # horizon -> MarketLayerReport.as_dict()
    industry: dict          # horizon -> IndustryLayerReport.as_dict()
    idiosyncratic: dict     # fund_code -> IdioLayerReport.as_dict()
    combined: dict          # fund_code -> CombinedIntervalReport.as_dict() — the user-facing band
    by_period: dict         # period label -> {"market": {...}, "industry": {...}}
    baselines: dict         # layer -> LayerVsBaseline.as_dict() — the incremental-value verdict
    findings: list[str]
    grid: dict

    def as_dict(self) -> dict:
        return {
            "horizons": self.horizons, "market": self.market, "industry": self.industry,
            "idiosyncratic": self.idiosyncratic, "combined": self.combined,
            "by_period": self.by_period, "baselines": self.baselines, "findings": self.findings, "grid": self.grid,
        }


def _diagnose(market: dict, industry: dict, idio: dict, baselines: dict | None = None,
              combined: dict | None = None) -> list[str]:
    """Turn the numbers into the honest, fail-closed sentences the user asked for."""
    out: list[str] = []
    # --- incremental value vs each layer's OWN naive baseline (CLAUDE.md) ---
    # This, not the absolute IC, is what "this layer is broken" means. A layer can have a
    # positive-looking IC and still add nothing over the dumbest alternative to itself.
    for _key, b in (baselines or {}).items():
        if b.get("model_score") is None:
            continue
        arrow = "更高更好" if b["higher_is_better"] else "更低更好"
        sg = b["significance"]
        name = {"market": "大盘层", "industry": "行业层"}.get(b["layer"], b["layer"])
        base = {"unconditional_distribution": "无条件分布（用全部可见历史，不做任何窗口选择）",
                "persistence": "持续性（上期跑赢的下期继续跑赢）"}.get(b["baseline"], b["baseline"])
        if b["beats_baseline"]:
            out.append(f"【{name}】✅ 显著优于自身朴素基线 {base}："
                       f"{b['metric']} {b['model_score']:+.4f} vs {b['baseline_score']:+.4f}（{arrow}），"
                       f"配对差值经三种重叠修正一致显著。")
        else:
            def _t(v):  # a None t means the correction was unmeasurable, not "zero"
                return "不可测" if v is None else f"{v:+.2f}"

            out.append(f"【{name}】❌ **未能击败自身的朴素基线** {base}："
                       f"{b['metric']} {b['model_score']:+.4f} vs 基线 {b['baseline_score']:+.4f}（{arrow}），"
                       f"配对差值均值 {sg['mean']:+.5f}，非重叠 t={_t(sg['t_nonoverlap'])}，"
                       f"Newey-West t={_t(sg['t_newey_west'])}，自助区间含 0。"
                       f"按 CLAUDE.md「跑不赢朴素基线就没有证明增量价值」，该层目前**没有证明增量价值**。")
    # --- market layer ---
    dirs = [(h, m) for h, m in market.items()]
    if dirs and not any(m["direction"]["significant"] for _h, m in dirs):
        ics = ", ".join(f"h={h}: IC {m['ic']:+.3f}" for h, m in dirs)
        out.append(
            f"【大盘层】方向性无有效技能：所有周期的方向性增益都未通过重叠修正显著性检验（{ics}）。"
            f"注意朴素 t 值看起来很大（如 h=90 的 t={dirs[-1][1]['direction']['t_naive']:+.1f}），"
            f"那是窗口重叠造成的假象，非重叠/Newey-West 修正后均不显著。"
        )
    for h, m in dirs:
        cov, nom = m["coverage"], m["nominal_coverage"]
        if cov is not None and nom and abs(cov - nom) > 0.05:
            out.append(
                f"【大盘层】h={h} 的区间覆盖率 {cov:.3f} 明显偏离名义 {nom:.2f}："
                + ("区间过窄、系统性高估把握（危险方向）。" if cov < nom else "区间过宽、过度保守。")
            )
    # --- industry layer ---
    for h, q in industry.items():
        sig = q["ic"]["significant"]
        out.append(
            f"【行业层】h={h} 截面 IC 均值 {q['mean_ic']:+.4f}，"
            + ("经三种重叠修正一致确认显著（有真实排序能力）。" if sig
               else f"但重叠修正后**不显著**（非重叠 t={q['ic']['t_nonoverlap']:+.2f}，"
                    f"Newey-West t={q['ic']['t_newey_west']:+.2f}，自助区间含 0）——朴素 t={q['ic']['t_naive']:+.2f} 是假象。")
        )
        per = q.get("per_industry") or {}
        if per:
            good = sorted(per.items(), key=lambda kv: -(kv[1]["ic"] or -9))[:2]
            bad = sorted(per.items(), key=lambda kv: (kv[1]["ic"] or 9))[:2]
            out.append(
                "【行业层】能力高度不均：最好 " + "、".join(f"{k} IC {v['ic']:+.3f}" for k, v in good)
                + "；最差 " + "、".join(f"{k} IC {v['ic']:+.3f}" for k, v in bad)
                + "。整体正 IC 是被少数行业拉起来的，另一些行业方向是反的。"
            )
            break
    # --- idio layer, per horizon ---
    # The band scales by sqrt(h) (D-HF-9). We STILL evaluate every horizon separately — never
    # averaged — because a REGRESSION to a flat band would show the ratio drifting with h, and
    # averaging would hide it (a 3x-too-wide short band cancels a 1.7x-too-narrow long one and
    # reports "well calibrated", the single most dangerous thing this could say).
    ratio_by_h = {}
    for hk, per_fund in sorted((idio or {}).items(), key=lambda kv: int(kv[0])):
        rs = [r["vol_ratio"] for r in per_fund.values() if r.get("vol_ratio")]
        if rs:
            ratio_by_h[int(hk)] = float(np.mean(rs))
    for hh, mr in ratio_by_h.items():
        if mr < 0.8:
            out.append(f"【个股层】h={hh}：残差实际波动仅为配置 idio_residual_vol 的 {mr:.2f} 倍，"
                       f"不确定性带**过宽/过度保守**，会把可用信号淹没。")
        elif mr > 1.25:
            out.append(f"【个股层】h={hh}：残差实际波动是配置的 {mr:.2f} 倍，不确定性带**过窄**，"
                       f"系统会**过度自信（危险方向）**。")
        else:
            out.append(f"【个股层】h={hh}：残差波动/配置 = {mr:.2f}，标定基本合理。")
    if len(ratio_by_h) >= 2:
        lo_h, hi_h = min(ratio_by_h), max(ratio_by_h)
        if ratio_by_h[hi_h] > ratio_by_h[lo_h] * 1.5:
            out.append(
                f"【个股层】**回归警告：不确定性带随周期缩放失效**。按 sqrt(h) 缩放后各周期比值应大致相等，"
                f"但实测从 h={lo_h} 的 {ratio_by_h[lo_h]:.2f} 漂移到 h={hi_h} 的 {ratio_by_h[hi_h]:.2f}"
                f"（>1.5 倍）——短周期把信号淹没、长周期系统性过度自信。请检查 "
                f"`idio_reference_horizon_days` 与 sqrt(h) 缩放路径（D-HF-9）。"
            )
    for hk, per_fund in sorted((idio or {}).items(), key=lambda kv: int(kv[0])):
        biased = [f for f, r in per_fund.items() if (r.get("bias") or {}).get("significant")]
        if biased:
            out.append(f"【个股层】h={hk}：{'、'.join(biased)} 的残差均值显著非零 —— "
                       f"这不是个股噪声，而是暴露口径/缩放有偏。")
    skipped = sorted({f for per in (idio or {}).values() for f, r in per.items()
                      if (r.get("n_obs") or 0) < 10})
    if skipped:
        out.append(f"【个股层】{'、'.join(skipped)} 观测不足或持仓无法映射，已 fail-closed 跳过（不编造指标）。")
    out.append(
        "【个股层】口径提醒：大盘/行业分量现按基金自身净值对基准的**滚动 beta**（点位回归，"
        "`estimate_market_beta`）缩放——现金拖累与仓位变化已从残差中剔除。仍然保留的近似是："
        "前十大重仓只覆盖股票市值的 33%–47%，行业**构成**假设未披露部分与前十大同构；该近似的"
        "平均误差已体现在按已实现残差标定的带宽里。"
    )
    # --- the combined, user-facing interval: the only coverage a user actually experiences ---
    for hk, per_fund in sorted((combined or {}).items(), key=lambda kv: int(kv[0])):
        covs = [(f, r) for f, r in per_fund.items() if r.get("coverage") is not None]
        if not covs:
            continue
        nom = covs[0][1]["nominal_coverage"]
        mean_cov = float(np.mean([r["coverage"] for _f, r in covs]))
        detail = "、".join(f"{f} {r['coverage']:.3f}" for f, r in covs)
        if mean_cov > nom + 0.05:
            out.append(f"【合成区间】h={hk}：对用户展示的 [ret_low, ret_high] 实际覆盖率 {mean_cov:.3f}，"
                       f"高于承诺的名义 {nom:.2f}（{detail}）——区间**过宽**，预测被稀释。")
        elif mean_cov < nom - 0.05:
            out.append(f"【合成区间】h={hk}：实际覆盖率仅 {mean_cov:.3f} < 名义 {nom:.2f}（{detail}）——"
                       f"区间**过窄**，对用户系统性高估把握（危险方向）。")
        else:
            out.append(f"【合成区间】h={hk}：覆盖率 {mean_cov:.3f} ≈ 名义 {nom:.2f}，标定合理。")
        lost = [f for f, r in covs
                if r.get("winkler") is not None and r.get("winkler_default_band") is not None
                and r["winkler"] > r["winkler_default_band"]]
        if lost:
            out.append(f"【合成区间】h={hk}：{'、'.join(lost)} 的 Winkler 区间分数**劣于固定参考带**"
                       f"（不随周期缩放、无分层结构的单一常数带）这一朴素基线——分层+缩放机制在这些基金上没有证明增量价值。")
        else:
            out.append(f"【合成区间】h={hk}：分层+缩放带宽在所有基金上的 Winkler 分数都**优于固定参考带**"
                       f"（单一常数带）这一朴素基线，该机制证明了增量价值。")
    out.append(
        "【合成区间】读数注意：市场状态叠加层（`_apply_market_overlay`）只会**加宽、从不收窄**区间"
        "（fail-closed 设计），在部分样本上把带宽放大至多约 1.5 倍。因此『过宽』的一部分是这一有意的"
        "安全边际，不能全部归咎于分层带宽机制。"
    )
    return out


def evaluate_all_layers(
    store: PointInTimeStore,
    *,
    horizons: tuple[int, ...] = (3, 20, 90),
    fund_codes: tuple[str, ...] = (),
    industry_map: Optional[dict] = None,
    idio_horizon: int = 20,
    period_years: tuple[int, ...] = (),
    cfg=None,
) -> LayerReport:
    """Score all three layers, plus per-year slices so a degrading layer is visible over time."""
    grid = build_grid(load_market_tape(store), load_industry_tapes(store))
    if grid.n == 0:
        return LayerReport(list(horizons), {}, {}, {}, {}, {}, {}, ["无可用指数数据（fail-closed）。"], {})

    market = {h: evaluate_market_layer(grid, h, cfg).as_dict() for h in horizons}
    industry = {h: evaluate_industry_layer(grid, h, cfg).as_dict() for h in horizons if h >= 20}
    # Idio + combined are scored at EVERY horizon, not just `idio_horizon`. The band now scales by
    # sqrt(h) (D-HF-9); scoring all three horizons is how we CONFIRM that scaling holds — a
    # regression to a flat band would show the vol ratio drifting with h (over-confident at h=90,
    # the dangerous direction), which a single-horizon view or an average would hide.
    idio = {
        str(h): {
            f: evaluate_idiosyncratic_layer(store, f, grid, h, industry_map=industry_map, cfg=cfg).as_dict()
            for f in fund_codes
        }
        for h in horizons
    }
    combined = {
        str(h): {
            f: evaluate_combined_interval(store, f, grid, h, industry_map=industry_map,
                                          cfg=cfg, sample_every=42).as_dict()
            for f in fund_codes
        }
        for h in horizons
    }

    # Each layer against its own naive baseline — the only evidence that can call a layer "broken".
    baselines: dict = {
        "market": market_vs_unconditional(grid, idio_horizon, cfg).as_dict(),
        "industry": industry_vs_persistence(grid, idio_horizon, cfg).as_dict(),
    }

    by_period: dict = {}
    years = period_years or tuple(sorted({d.year for d in grid.dates}))
    for y in years:
        s, e = date(y, 1, 1), date(y, 12, 31)
        m = evaluate_market_layer(grid, idio_horizon, cfg, start=s, end=e)
        q = evaluate_industry_layer(grid, idio_horizon, cfg, start=s, end=e)
        if m.n_obs >= 30:
            by_period[str(y)] = {
                "market": {"ic": _r(m.ic), "hit_rate": _r(m.hit_rate), "coverage": _r(m.coverage),
                           "crps": _r(m.crps), "n": m.n_obs},
                "industry": {"mean_ic": _r(q.mean_ic), "pct_ic_positive": _r(q.pct_ic_positive), "n": q.n_dates},
            }

    return LayerReport(
        horizons=list(horizons), market=market, industry=industry, idiosyncratic=idio,
        combined=combined, by_period=by_period, baselines=baselines,
        findings=_diagnose(market, industry, idio, baselines, combined),
        grid={"n_dates": grid.n, "start": str(grid.dates[0]), "end": str(grid.dates[-1]),
              "n_industries": len(grid.industry_names)},
    )


# =============================================================================
# Naive-baseline deltas — "broken" means "does not beat its own naive baseline"
# =============================================================================
# CLAUDE.md: "Always compare against naive baselines ... A complex model that can't reliably beat
# these has not proven incremental value." A layer's absolute IC/CRPS cannot show it is broken; only
# the paired delta against the layer's own naive alternative can. Each baseline below is the
# cheapest, most obvious thing the layer is supposed to improve upon.
@dataclass(frozen=True)
class LayerVsBaseline:
    """A layer's paired improvement over its own naive baseline, with an overlap-corrected verdict."""

    layer: str
    baseline: str
    metric: str
    model_score: float
    baseline_score: float
    higher_is_better: bool
    significance: "object"       # Significance of the paired (model - baseline) series

    @property
    def beats_baseline(self) -> bool:
        return bool(self.significance.significant and self.significance.mean > 0)

    def as_dict(self) -> dict:
        return {
            "layer": self.layer, "baseline": self.baseline, "metric": self.metric,
            "model_score": _r(self.model_score), "baseline_score": _r(self.baseline_score),
            "higher_is_better": self.higher_is_better,
            "beats_baseline": self.beats_baseline,
            "significance": self.significance.as_dict(),
        }


def market_vs_unconditional(grid: Grid, horizon: int, cfg=None, *, start=None, end=None) -> LayerVsBaseline:
    """Market layer vs its naive baseline: the **unconditional** (all-visible-history) distribution.

    If the ``market_lookback`` window adds nothing over simply using every forward return ever
    observed, the market layer has no incremental value. Scored by CRPS (a proper score, lower is
    better) paired per observation, so the two share the same realisations.
    """
    from qmt.stats import assess_significance

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    # The forecaster feeds `market_lookback_trading_days` **prices** to `_empirical_layer`, which
    # yields `lb - h` overlapping windows. Slicing `lb` forward *returns* here would hand the
    # diagnostic a longer history than the layer it is meant to score.
    lb = max(1, c.market_lookback_trading_days - h)
    P, N = grid.market, grid.n
    fm = forward_returns(P, h)

    d_model, d_base = [], []
    for i in range(1, N):
        dte = grid.dates[i]
        if (start and dte < start) or (end and dte > end):
            continue
        hi, e = i - 1 - h, i + 1
        if hi < 0 or e + h >= N:
            continue
        full = fm[: hi + 1]
        full = full[np.isfinite(full)]
        if full.size < c.market_min_windows:
            continue
        win = full[-lb:] if lb > 0 else full
        y = P[e + h] / P[e] - 1.0
        # Grade the SHIPPED model: the market band is vol-regime-conditioned (D-CVOL-1), so its
        # distribution is the window scaled around its median by reff — the same transform
        # evaluate_market_layer grades. The baseline stays the raw unconditional full history (the
        # naive "no window, no conditioning" alternative). Grading the pre-conditioning window here
        # under-credited the conditioning and mis-stated the incremental-value verdict.
        reff = _market_vol_multiplier(P[:i], c)
        m = float(np.median(win))
        d_model.append(crps_empirical(m + (win - m) * reff, y))
        d_base.append(crps_empirical(full, y))

    if len(d_model) < 10:
        from qmt.stats import Significance
        nan = float("nan")
        return LayerVsBaseline("market", "unconditional_distribution", "CRPS", nan, nan, False,
                               Significance(nan, 0, h, nan, nan, 0, nan, nan, nan, False, "样本不足。"))
    m, b = np.array(d_model), np.array(d_base)
    # CRPS is lower-is-better, so the *improvement* is (baseline - model)
    return LayerVsBaseline("market", "unconditional_distribution", "CRPS",
                           float(m.mean()), float(b.mean()), False,
                           assess_significance(b - m, overlap=h))


def industry_vs_persistence(grid: Grid, horizon: int, cfg=None, *, start=None, end=None) -> LayerVsBaseline:
    """Industry layer vs its naive baseline: **persistence** (last realised deviation predicts next).

    The layer's job is to rank industries; the cheapest rival ranking is "whatever outperformed last
    period will outperform next period". Both are scored by the per-date cross-sectional rank IC and
    compared as a paired series over the same dates.
    """
    from qmt.stats import assess_significance, spearman

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    # Same price-vs-window correction as the market layer: `_deviation_layer` receives
    # `industry_lookback_trading_days` prices and produces `lb - h` deviation windows.
    lb = max(1, c.industry_lookback_trading_days - h)
    P, IND, N, K = grid.market, grid.industries, grid.n, len(grid.industry_names)
    from qmt.stats import Significance
    nan = float("nan")
    if K == 0:
        return LayerVsBaseline("industry", "persistence", "cross-sectional IC", nan, nan, True,
                               Significance(nan, 0, h, nan, nan, 0, nan, nan, nan, False, "无行业数据。"))

    fm = forward_returns(P, h)
    fdev = np.vstack([forward_returns(IND[k], h) for k in range(K)]) - fm[None, :]

    ic_m, ic_b = [], []
    for i in range(1, N):
        dte = grid.dates[i]
        if (start and dte < start) or (end and dte > end):
            continue
        hi, e = i - 1 - h, i + 1
        if hi < 0 or e + h >= N:
            continue
        w = fdev[:, max(0, hi - lb + 1): hi + 1]
        if w.shape[1] < c.industry_min_windows:
            continue
        pred_model = np.nanmedian(w, axis=1)
        pred_persist = fdev[:, hi]                       # the most recent fully-observed deviation
        rm = P[e + h] / P[e] - 1.0
        real = (IND[:, e + h] / IND[:, e] - 1.0) - rm
        ok = np.isfinite(pred_model) & np.isfinite(pred_persist) & np.isfinite(real)
        if ok.sum() < 5:
            continue
        a = spearman(pred_model[ok], real[ok])
        b = spearman(pred_persist[ok], real[ok])
        if np.isfinite(a) and np.isfinite(b):
            ic_m.append(a); ic_b.append(b)

    if len(ic_m) < 10:
        return LayerVsBaseline("industry", "persistence", "cross-sectional IC", nan, nan, True,
                               Significance(nan, 0, h, nan, nan, 0, nan, nan, nan, False, "样本不足。"))
    m, b = np.array(ic_m), np.array(ic_b)
    return LayerVsBaseline("industry", "persistence", "cross-sectional IC",
                           float(m.mean()), float(b.mean()), True,
                           assess_significance(m - b, overlap=h))


# =============================================================================
# End-to-end: does the COMBINED interval keep the promise it makes to the user?
# =============================================================================
# The three per-layer bands are only intermediate. What the product shows is the combined
# ``[ret_low, ret_high]``. Checking ``|resid| <= idio_band`` in isolation cannot tell us whether
# that promise holds: the layers can each be miscalibrated in offsetting directions. This is the
# only coverage number a user actually experiences.
@dataclass(frozen=True)
class CombinedIntervalReport:
    horizon: int
    fund_code: str
    n_obs: int
    nominal_coverage: float
    coverage: float
    mean_width: float
    winkler: float
    winkler_default_band: float
    unresolved_weight: float
    note: str

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon, "fund_code": self.fund_code, "n_obs": self.n_obs,
            "nominal_coverage": _r(self.nominal_coverage), "coverage": _r(self.coverage),
            "mean_width": _r(self.mean_width), "winkler": _r(self.winkler),
            "winkler_default_band": _r(self.winkler_default_band),
            "unresolved_weight": _r(self.unresolved_weight), "note": self.note,
        }


def evaluate_combined_interval(
    store: PointInTimeStore,
    fund_code: str,
    grid: Grid,
    horizon: int,
    *,
    industry_map: Optional[dict] = None,
    cfg=None,
    sample_every: int = 21,
    max_unresolved: float = 0.5,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> CombinedIntervalReport:
    """Coverage + Winkler score of the *combined* ``[ret_low, ret_high]`` against the realised return.

    Fails closed when most of the book cannot be mapped to an industry index: ``forecast_hierarchical``
    still emits a band there (it routes unresolvable weight into idiosyncratic widening rather than
    refusing), so without this guard the harness would publish a coverage number for a fund whose
    exposure it has already admitted it cannot resolve.
    """
    from qmt.pipeline.exposure_estimator import UNCLASSIFIED_LABEL, estimate_exposure
    from qmt.pipeline.hierarchical_forecaster import SW_L1_INDEX_BY_NAME, forecast_hierarchical
    from qmt.pipeline.market_state import assess_market_state
    from qmt.stats import winkler_score
    from qmt.types import ExposureDimension

    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    nominal = c.market_ret_high_quantile - c.market_ret_low_quantile
    alpha = max(1e-9, 1.0 - nominal)
    nan = float("nan")

    # Pick the Horizon bucket whose configured window actually equals `h`, so the interval we score
    # is the one the product would have shown for this horizon — not a neighbouring bucket.
    from qmt.types import Horizon
    hz = next((z for z in (Horizon.SHORT, Horizon.MID, Horizon.LONG)
               if int(CONFIG.horizon.windows[z.value][1]) == h), None)
    if hz is None:
        return CombinedIntervalReport(h, fund_code, 0, nominal, nan, nan, nan, nan, nan,
                                      f"h={h} 不对应任何产品口径的预测周期（fail-closed）。")

    f_dates, f_vals = _fund_returns(store, fund_code)
    if len(f_dates) < h + 5:
        return CombinedIntervalReport(h, fund_code, 0, nominal, nan, nan, nan, nan, nan,
                                      "净值历史不足（fail-closed）。")

    known = set(SW_L1_INDEX_BY_NAME) | set(industry_map or {})
    los, his, ys, unresolved = [], [], [], []
    for i in range(1, grid.n, max(1, sample_every)):
        d = grid.dates[i]
        if (start and d < start) or (end and d > end):
            continue
        if i + 1 + h >= grid.n:
            break
        try:
            exps = estimate_exposure(fund_code, d, store, industry_map=industry_map)
            ms = assess_market_state(d, store)
            res = forecast_hierarchical(fund_code, d, exps, ms, store, cfg=c)
        except Exception:
            continue
        ind = [x for x in exps if x.dimension is ExposureDimension.INDUSTRY]
        tot = sum(float(x.point_estimate) for x in ind)
        unres = sum(float(x.point_estimate) for x in ind
                    if x.label == UNCLASSIFIED_LABEL or x.label not in known)
        frac = (unres / tot) if tot > 0 else 1.0
        unresolved.append(frac)
        if frac > max_unresolved:
            continue
        fc = res.forecasts.get(hz)
        if fc is None or not math.isfinite(fc.ret_low) or not math.isfinite(fc.ret_high):
            continue
        y = _fund_window(f_dates, f_vals, d, grid.dates[i + 1 + h])
        if y is None:
            continue
        los.append(fc.ret_low); his.append(fc.ret_high)
        ys.append(y)

    uw = float(np.mean(unresolved)) if unresolved else nan
    if len(ys) < 10:
        why = ("持仓多数无法映射到行业指数（分层预测在此基金上不可信），fail-closed。"
               if math.isfinite(uw) and uw > max_unresolved else "可用观测不足（fail-closed）。")
        return CombinedIntervalReport(h, fund_code, len(ys), nominal, nan, nan, nan, nan, uw, why)
    lo, hi, y = np.array(los), np.array(his), np.array(ys)
    cov = float(np.mean((y >= lo) & (y <= hi)))
    w = float(np.mean(winkler_score(lo, hi, y, alpha)))
    # Naive baseline for the band itself: a single FLAT band at the reference-horizon width, applied
    # unchanged at every horizon (no sqrt(h) scaling, no layer structure) around the same midpoint.
    # Beating it is exactly the incremental value of the horizon-scaling + layered machinery.
    mid = 0.5 * (lo + hi)
    b = c.idio_residual_band
    w_def = float(np.mean(winkler_score(mid - b, mid + b, y, alpha)))
    verdict = ("区间过窄，系统性高估把握（危险方向）。" if cov < nominal - 0.05 else
               "区间过宽、过度保守。" if cov > nominal + 0.05 else "覆盖率接近名义值，标定合理。")
    return CombinedIntervalReport(h, fund_code, y.size, nominal, cov, float(np.mean(hi - lo)),
                                  w, w_def, uw, verdict)
