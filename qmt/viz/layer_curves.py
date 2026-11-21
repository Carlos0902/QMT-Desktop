"""Prediction-vs-realized curves for the market and industry layers (§5.3 分层模型).

The point of these curves is to *see* what the per-layer diagnostics (:mod:`qmt.pipeline.
layer_diagnostics`) measure: at every point-in-time decision the layer emits a **range**
(a median plus an 80% band), and reality later lands somewhere. Plotting the band against the
realized outcome makes calibration (does reality fall in the band ~80% of the time?), directional
skill (does the median lead reality?), and horizon behaviour visible at a glance.

Both layers reduce to the same picture — *predicted central line + uncertainty band + realized
line over time* — so there is exactly one plotting primitive, :func:`plot_forecast_band`, and the
market and industry composers both call it. The series that feed it are computed by extractors that
mirror :func:`evaluate_market_layer` / :func:`evaluate_industry_layer` **exactly** (same lookback,
same 14:30 decision → first-executable-bar timing, same quantiles), so the curves show the model
that is actually graded, not a lookalike.

Point-in-time (invariant 1): at decision index ``i`` the prediction uses only windows fully
observed by the ``i``-th 14:30 decision (``hi = i-1-h``); the realized outcome starts at the first
strictly-later bar (``e = i+1``) and ends at ``e+h``. Nothing at or beyond ``e`` touches the
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np

from qmt.config import CONFIG
from qmt.pipeline.layer_diagnostics import Grid, forward_returns

# One palette so market and industry figures read as a single system.
_C_BAND = "#8ecae6"     # predicted 80% band (fill)
_C_PRED = "#0353a4"     # predicted median (model)
_C_REAL = "#e26d5c"     # realized (real data)
_C_ZERO = "#9aa0a6"     # zero reference


@dataclass(frozen=True)
class ForecastSeries:
    """A point-in-time forecast track: dates, predicted median + 80% band, and the realized value.

    ``kind`` is ``"return"`` (market: index forward return) or ``"deviation"`` (industry: forward
    return minus the market's). ``nominal`` is the band's nominal coverage (``qhi - qlo``).
    """

    dates: list[date]
    mid: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    realized: np.ndarray
    kind: str
    nominal: float
    label: str = ""

    @property
    def coverage(self) -> float:
        if self.realized.size == 0:
            return float("nan")
        return float(np.mean((self.realized >= self.lo) & (self.realized <= self.hi)))

    @property
    def correlation(self) -> float:
        """Rank correlation of median vs realized — does the prediction lead reality at all."""
        from qmt.stats import spearman

        return spearman(self.mid, self.realized)


# =============================================================================
# The one reusable plotting primitive
# =============================================================================
def plot_forecast_band(
    ax,
    series: ForecastSeries,
    *,
    title: str = "",
    ylabel: Optional[str] = None,
    show_legend: bool = True,
) -> None:
    """Draw one forecast track onto ``ax``: shaded 80% band, predicted median, realized line.

    This is the single primitive both the market and industry figures reuse. It annotates the
    subtitle with realized coverage vs nominal and the median↔realized rank correlation, so each
    panel is self-describing.
    """
    import matplotlib.dates as mdates

    x = mdates.date2num(series.dates)
    ax.axhline(0.0, color=_C_ZERO, lw=0.7, ls=":", zorder=0)
    ax.fill_between(
        x, series.lo, series.hi, color=_C_BAND, alpha=0.45, lw=0,
        label=f"预测区间 {series.nominal:.0%}",
    )
    ax.plot(x, series.mid, color=_C_PRED, lw=1.3, label="模型预测（中位）", zorder=3)
    ax.plot(x, series.realized, color=_C_REAL, lw=1.0, alpha=0.9, label="真实（已实现）", zorder=2)

    cov, ic = series.coverage, series.correlation
    ic_txt = "—" if not np.isfinite(ic) else f"{ic:+.2f}"
    sub = f"覆盖率 {cov:.0%} / 名义 {series.nominal:.0%}　·　中位 vs 真实 相关 {ic_txt}　·　n={series.realized.size}"
    ax.set_title(f"{title}\n{sub}" if title else sub, fontsize=10, loc="left")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=8)
    ax.margins(x=0.01)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if show_legend:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =============================================================================
# Series extractors — mirror the evaluators exactly (same PIT timing & quantiles)
# =============================================================================
def market_forecast_series(
    grid: Grid, horizon: int, cfg=None, *, start: Optional[date] = None, end: Optional[date] = None
) -> ForecastSeries:
    """The market layer's own forecast track: the broad index's forward-return distribution vs the
    index's realized forward return. Mirrors :func:`evaluate_market_layer` (no beta — beta is the
    fund-level aggregation, not part of the market model itself)."""
    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    lb = max(1, c.market_lookback_trading_days - h)
    qlo, qhi = c.market_ret_low_quantile, c.market_ret_high_quantile
    P, N = grid.market, grid.n
    fm = forward_returns(P, h)

    ds: list[date] = []
    mid, lo, hi, real = [], [], [], []
    for i in range(1, N):
        d = grid.dates[i]
        if (start and d < start) or (end and d > end):
            continue
        hj, e = i - 1 - h, i + 1
        if hj < 0 or e + h >= N:
            continue
        hist = fm[max(0, hj - lb + 1): hj + 1]
        hist = hist[np.isfinite(hist)]
        if hist.size < c.market_min_windows:
            continue
        q_lo, q_mid, q_hi = np.quantile(hist, [qlo, 0.5, qhi])
        # Show the band the forecaster actually produces: de-bias the median (shrink toward 0,
        # D-CVOL-3) then scale the half-widths by the current vol regime (D-CVOL-1), PIT.
        from qmt.pipeline.layer_diagnostics import _market_vol_multiplier

        shift = q_mid * (c.market_median_shrink - 1.0)
        q_mid = q_mid + shift; q_lo = q_lo + shift; q_hi = q_hi + shift
        reff = _market_vol_multiplier(P[:i], c)
        q_lo, q_hi = q_mid + (q_lo - q_mid) * reff, q_mid + (q_hi - q_mid) * reff
        ds.append(d)
        mid.append(q_mid); lo.append(q_lo); hi.append(q_hi)
        real.append(P[e + h] / P[e] - 1.0)
    return ForecastSeries(
        ds, np.array(mid), np.array(lo), np.array(hi), np.array(real),
        kind="return", nominal=float(qhi - qlo), label="大盘（沪深300）",
    )


def industry_forecast_series(
    grid: Grid, industry: str, horizon: int, cfg=None, *,
    start: Optional[date] = None, end: Optional[date] = None,
) -> ForecastSeries:
    """One industry's forecast track: its predicted forward-return **deviation from the market**
    (median + band over the lookback) vs the realized deviation. Mirrors the per-date logic inside
    :func:`evaluate_industry_layer`, isolated to a single industry so it fits one panel."""
    c = cfg or CONFIG.hierarchical
    h = max(1, int(horizon))
    lb = max(1, c.industry_lookback_trading_days - h)
    qlo, qhi = c.industry_ret_low_quantile, c.industry_ret_high_quantile
    P, N = grid.market, grid.n
    try:
        k = grid.industry_names.index(industry)
    except ValueError as exc:
        raise KeyError(f"未知行业 {industry!r}；可选：{grid.industry_names}") from exc

    fm = forward_returns(P, h)
    fdev = forward_returns(grid.industries[k], h) - fm

    ds: list[date] = []
    mid, lo, hi, real = [], [], [], []
    for i in range(1, N):
        d = grid.dates[i]
        if (start and d < start) or (end and d > end):
            continue
        hj, e = i - 1 - h, i + 1
        if hj < 0 or e + h >= N:
            continue
        w = fdev[max(0, hj - lb + 1): hj + 1]
        w = w[np.isfinite(w)]
        if w.size < c.industry_min_windows:
            continue
        q_lo, q_mid, q_hi = np.quantile(w, [qlo, 0.5, qhi])
        rm = P[e + h] / P[e] - 1.0
        y = (grid.industries[k][e + h] / grid.industries[k][e] - 1.0) - rm
        ds.append(d)
        mid.append(q_mid); lo.append(q_lo); hi.append(q_hi)
        real.append(y)
    return ForecastSeries(
        ds, np.array(mid), np.array(lo), np.array(hi), np.array(real),
        kind="deviation", nominal=float(qhi - qlo), label=industry,
    )


# =============================================================================
# Composers — arrange panels; both reuse plot_forecast_band
# =============================================================================
def _new_figure(n_panels: int, *, title: str):
    # OO construction (Figure + Agg canvas), NOT pyplot: pyplot's global figure manager is not
    # thread-safe, and two concurrent renders in FastAPI's threadpool (market + industry) deadlock
    # it. A standalone Figure has no global state, so concurrent renders are safe.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _set_cjk_font()
    fig = Figure(figsize=(11, 2.7 * n_panels + 0.6))
    FigureCanvasAgg(fig)                       # attach an Agg canvas so savefig works without pyplot
    axes = fig.subplots(n_panels, 1, squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, ha="left")
    return fig, [row[0] for row in axes]


def plot_market_curves(
    grid: Grid, cfg=None, *, horizons: Sequence[int] = (3, 20, 90),
    start: Optional[date] = None, end: Optional[date] = None, out_path: Optional[str] = None,
):
    """Market layer: one panel per horizon, predicted forward-return band vs realized."""
    fig, axes = _new_figure(len(horizons), title="大盘模型：预测 vs 真实（沪深300 前瞻收益分布）")
    for ax, h in zip(axes, horizons):
        s = market_forecast_series(grid, h, cfg, start=start, end=end)
        plot_forecast_band(ax, s, title=f"周期 h={h} 交易日", ylabel="前瞻收益")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def plot_industry_curves(
    grid: Grid, cfg=None, *, horizon: int = 20, industries: Optional[Sequence[str]] = None,
    auto_extremes: int = 2, start: Optional[date] = None, end: Optional[date] = None,
    out_path: Optional[str] = None,
):
    """Industry layer: one panel per industry, predicted deviation band vs realized deviation.

    When ``industries`` is not given, auto-selects the ``auto_extremes`` best- and worst-ranked
    industries by realized median↔deviation correlation — so the panels show both where the model
    ranks correctly and where it is inverted (the diagnosis found the mean IC is carried by a few
    industries while others point the wrong way).
    """
    if industries is None:
        scored = []
        for name in grid.industry_names:
            s = industry_forecast_series(grid, name, horizon, cfg, start=start, end=end)
            if s.realized.size >= 20 and np.isfinite(s.correlation):
                scored.append((s.correlation, name))
        scored.sort(reverse=True)
        n = min(auto_extremes, len(scored) // 2)
        chosen = [nm for _c, nm in scored[:n]] + [nm for _c, nm in scored[-n:]]
    else:
        chosen = list(industries)

    fig, axes = _new_figure(
        len(chosen), title=f"行业模型：预测 vs 真实（行业相对大盘的前瞻偏离，h={horizon}）")
    for ax, name in zip(axes, chosen):
        s = industry_forecast_series(grid, name, horizon, cfg, start=start, end=end)
        plot_forecast_band(ax, s, title=f"{name}", ylabel="相对大盘偏离")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig


def _set_cjk_font() -> None:
    """Best-effort CJK font so Chinese labels render instead of tofu; silent if none is present."""
    import matplotlib

    for fam in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS",
                "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        try:
            from matplotlib.font_manager import findfont, FontProperties

            if findfont(FontProperties(family=fam), fallback_to_default=False):
                matplotlib.rcParams["font.sans-serif"] = [fam]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
