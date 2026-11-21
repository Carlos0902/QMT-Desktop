"""exposure_estimator — infer a fund's *current* true exposure (§1.4 / §2.6 / §1.8-3).

Responsibility (04-modules.tex sec:mod-expo): for an active fund, estimate the exposure
at the decision point ``as_of`` and keep the three information tiers apart
(§6 sec:cc-exposure / §3 sec:expo-tiers):

  Tier 1 — disclosed facts + their age  (the last public holdings, and how old they are);
  Tier 2 — the system estimate           (a ``point_estimate`` inside a ``low``/``high`` range);
  Tier 3 — the system's confidence       (falls as the disclosure ages).

The output is ``list[ExposureEstimate]`` — never the raw disclosed holdings dressed up as
"current holdings". Disclosed holdings are only an *input* here (invariant 2).

Baseline algorithm (05-algorithms.tex, ExposureEstimator row): "直接采用最近一次公开持仓，
按持仓龄给 confidence 打折" — take the most recent disclosed top-10 holdings that were
observable at ``as_of``, aggregate the stocks to industry (Shenwan L1) / style labels, and
discount confidence by the disclosure age via the sanctioned ``config.decayed_confidence``.
The ``low``/``high`` band widens with the drift probability so a stale snapshot is honestly
less certain. This is the simple, transparent baseline — NOT the complex model.

Candidate (documented hook only, E49): a rolling constrained NAV-vs-index regression
(``cvxpy``, weights>=0, sum<=1) — see :func:`_candidate_regression_exposure`. It is not
wired in: per the doc's "先基线，后候选，严格验证" rule a candidate replaces the baseline only
after time-ordered backtest evidence.

Point-in-time correctness (invariant 1): the only data read is via
``PointInTimeStore.get_asof(as_of)``; nothing after ``as_of`` can be seen.

Data gap (D41 / sources.DATA_GAPS): ``fund_portfolio`` exposes stock positions only — there
is no industry vector and no stock/bond/cash split. Industry labels must be DERIVED by
mapping each stock symbol to a Shenwan L1 taxonomy. When that mapping is unavailable (no
map injected and Tushare cannot supply it within quota) the unmapped weight is surfaced
under the explicit :data:`UNCLASSIFIED_LABEL` sentinel — never silently assumed zero.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from qmt.config import CONFIG, ExposureConfig, decayed_confidence
from qmt.data.sources import DATA_GAPS, TABLE_FUND_NAV, TABLE_FUND_PORTFOLIO, TABLE_INDEX_DAILY
from qmt.pit import AVAILABLE_AT, PointInTimeStore
from qmt.types import (
    DisclosedHoldings,
    DisclosureDepth,
    ExposureDimension,
    ExposureEstimate,
    HoldingLine,
)

logger = logging.getLogger(__name__)

__all__ = [
    "estimate_exposure",
    "latest_visible_holdings",
    "UNCLASSIFIED_LABEL",
    "INDUSTRY_ALLOCATION_GAP",
]

# Sentinel label for stock weight whose industry (or style) could not be resolved. Its
# presence in the output is the data-gap flag — a downstream module sees the gap explicitly
# instead of an industry vector that silently omits the unclassified weight.
UNCLASSIFIED_LABEL = "unclassified"

# The named data gap this module has to work around (D41 / sources.DATA_GAPS).
INDUSTRY_ALLOCATION_GAP = DATA_GAPS["industry_allocation"]


# =============================================================================
# Tier 1 — the most-recent disclosed holdings observable at as_of
# =============================================================================
def latest_visible_holdings(
    fund_code: str,
    as_of: date,
    store: PointInTimeStore,
) -> Optional[DisclosedHoldings]:
    """Reconstruct the newest :class:`DisclosedHoldings` snapshot visible at ``as_of``.

    Reads ONLY through ``store.get_asof`` (``available_at <= as_of``), so a report whose
    ``ann_date`` is after ``as_of`` is invisible (invariant 1). Among the visible reports
    the one with the latest ``available_at`` (then latest ``report_end_date``) is returned.
    ``None`` when nothing has been disclosed yet at ``as_of`` — the fail-closed case, no
    fabricated holdings.
    """
    df = store.get_asof(TABLE_FUND_PORTFOLIO, as_of, filters={"fund_code": fund_code})
    if df is None or df.empty:
        return None

    df = df.copy()
    df[AVAILABLE_AT] = pd.to_datetime(df[AVAILABLE_AT])
    latest_ann = df[AVAILABLE_AT].max()
    grp = df[df[AVAILABLE_AT] == latest_ann]

    report_end = as_of  # fallback
    if "report_end_date" in grp.columns:
        ends = pd.to_datetime(grp["report_end_date"])
        latest_end = ends.max()
        grp = grp[ends == latest_end]
        if pd.notna(latest_end):
            report_end = pd.Timestamp(latest_end).date()

    depth = DisclosureDepth.FULL if report_end.month in (6, 12) else DisclosureDepth.TOP10
    if "depth" in grp.columns:
        raw_depth = grp["depth"].iloc[0]
        try:
            depth = DisclosureDepth(str(raw_depth))
        except ValueError:
            pass

    lines: list[HoldingLine] = []
    for row in grp.to_dict("records"):
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        lines.append(
            HoldingLine(
                symbol=symbol,
                mkv=_maybe_float(row.get("mkv")),
                amount=_maybe_float(row.get("amount")),
                stk_mkv_ratio=_maybe_float(row.get("stk_mkv_ratio")),
                stk_float_ratio=_maybe_float(row.get("stk_float_ratio")),
            )
        )
    if not lines:
        return None

    return DisclosedHoldings(
        fund_code=fund_code,
        report_end_date=report_end,
        depth=depth,
        holdings=lines,
        available_at=pd.Timestamp(latest_ann).to_pydatetime(),
    )


# =============================================================================
# The public entrypoint (04-modules.tex lst:expo)
# =============================================================================
def estimate_exposure(
    fund_code: str,
    as_of: date,
    store: PointInTimeStore | None = None,
    *,
    industry_map: Optional[Mapping[str, str]] = None,
    style_map: Optional[Mapping[str, str]] = None,
    client: object | None = None,
    cfg: Optional[ExposureConfig] = None,
) -> list[ExposureEstimate]:
    """Estimate ``fund_code``'s current exposure at ``as_of`` as ``list[ExposureEstimate]``.

    Baseline (E48): take the most-recent disclosed top-10 holdings observable at ``as_of``,
    map each stock to an industry (Shenwan L1) label, aggregate the within-book weights per
    label, and emit one :class:`ExposureEstimate` per label carrying all three tiers —
    ``disclosure_age_days`` (tier 1), ``point_estimate``/``low``/``high`` (tier 2) and
    ``confidence`` (tier 3, = ``decayed_confidence(disclosure_age_days)``).

    Only ``as_of``-visible data is read (invariant 1); estimated exposure is never the raw
    disclosed holdings (invariant 2). Returns ``[]`` when nothing is observable yet.

    Keyword-only extensions beyond the doc's ``(fund_code, as_of)`` signature (all optional,
    so the doc/task signature is a strict subset):

    * ``store`` — the point-in-time store to read from (the sole read path). ``None`` builds
      the default cache-backed store, which is empty until hydrated by ``qmt.data.sources``.
    * ``industry_map`` / ``style_map`` — ``{symbol -> label}`` taxonomies. When
      ``industry_map`` is absent, :func:`_resolve_industry_map` attempts Tushare within
      quota and otherwise leaves the weight unclassified (data gap). ``style_map`` is only
      aggregated when supplied (style has no baseline Tushare source — a documented gap).
    * ``client`` — optional Tushare client the industry hook may use.
    * ``cfg`` — override :class:`ExposureConfig` (band widths / disclosure lags).
    """
    ecfg = cfg or CONFIG.exposure
    st = store if store is not None else _default_store()

    holdings = latest_visible_holdings(fund_code, as_of, st)
    if holdings is None:
        logger.info(
            "exposure_estimator: no disclosed holdings visible for %s at %s → [] (fail-closed).",
            fund_code,
            as_of,
        )
        return []

    disclosure_age_days = max((as_of - holdings.available_at.date()).days, 0)
    # Tier 3: confidence discounted by disclosure age (the sanctioned monotone decay).
    confidence = decayed_confidence(disclosure_age_days)
    # Drift probability rises as the snapshot ages (monotone, since confidence falls).
    drift_prob = min(max(1.0 - confidence, 0.0), 1.0)
    # Tier 2: band half-width widens with drift (a stale snapshot is honestly less certain).
    rel_halfwidth = ecfg.band_min_rel + (ecfg.band_max_rel - ecfg.band_min_rel) * drift_prob

    weights = _normalized_weights(holdings.holdings)

    estimates: list[ExposureEstimate] = []

    ind_map = _resolve_industry_map(holdings.holdings, industry_map, client)
    industry_agg = _aggregate_by_label(holdings.holdings, weights, ind_map)
    if UNCLASSIFIED_LABEL in industry_agg:
        logger.info(
            "exposure_estimator: %s of %s book unclassified at %s — data gap '%s'.",
            f"{industry_agg[UNCLASSIFIED_LABEL]:.0%}",
            fund_code,
            as_of,
            "industry_allocation",
        )
    estimates.extend(
        _emit(
            fund_code,
            as_of,
            ExposureDimension.INDUSTRY,
            industry_agg,
            disclosure_age_days,
            drift_prob,
            confidence,
            rel_halfwidth,
        )
    )

    if style_map:
        style_agg = _aggregate_by_label(holdings.holdings, weights, dict(style_map))
        estimates.extend(
            _emit(
                fund_code,
                as_of,
                ExposureDimension.STYLE,
                style_agg,
                disclosure_age_days,
                drift_prob,
                confidence,
                rel_halfwidth,
            )
        )

    return estimates


# =============================================================================
# helpers
# =============================================================================
def estimate_market_beta(
    fund_code: str,
    as_of: date,
    store: Optional[PointInTimeStore] = None,
    *,
    benchmark: str = "000300.SH",
    window: int = 250,
    min_obs: int = 60,
    clip_high: float = 1.3,
) -> Optional[float]:
    """PIT rolling OLS beta of the fund's NAV returns to the broad-market index (真实暴露 tier-2).

    This is the fund's **market loading** — the number that scales the 大盘 layer. The disclosed
    route to it is broken in two independent ways on real data: (1) the top-10 holdings are
    normalised to sum to 1.0 (within-disclosed-book shares, not NAV shares), and (2) portfolio
    ``mkv`` is disclosed for the COMBINED fund while ``fund_nav.net_asset`` is per share class, so
    ``Σmkv / net_asset`` measures an impossible 1.5–2.3. The fund's own returns don't lie: a
    trailing regression measures the *current effective* sensitivity — which is exactly what
    true-exposure estimation is for (current exposure, not a stale quarterly snapshot).

    Point-in-time: both series are read via ``get_asof(available_at <= as_of)`` — same convention
    as the forecaster's index reads. Fail-closed: fewer than ``min_obs`` common return observations
    → ``None`` (caller falls back to its previous behaviour). Clipped to ``[0, clip_high]``: a
    negative beta on an equity fund is estimation noise, and 公募 equity funds cannot lever.
    """
    src = store if store is not None else _default_store()

    nav = src.get_asof(TABLE_FUND_NAV, as_of, filters={"fund_code": fund_code})
    if nav is None or nav.empty or "nav_date" not in nav.columns:
        return None
    nav = nav.sort_values(["nav_date", "available_at"]).drop_duplicates(subset="nav_date", keep="last")
    col = "adj_nav" if ("adj_nav" in nav.columns and nav["adj_nav"].notna().any()) else "unit_nav"
    fv = pd.to_numeric(nav[col], errors="coerce")
    nd = pd.to_datetime(nav["nav_date"]).dt.date
    # Strict PIT: use only closes dated BEFORE the decision date. get_asof already filters on
    # available_at, but a NAV dated `as_of` itself is published that evening — not known at the
    # 14:30 decision — and some rows are mis-tagged available_at = nav_date@00:00, so this guard
    # protects the beta from a same-day leak regardless of tagging quality.
    keep = fv.notna() & (fv > 0.0) & (nd < as_of)
    f_dates = nd[keep].tolist()
    f_vals = fv[keep].to_numpy(dtype=float)

    idx = src.get_asof(TABLE_INDEX_DAILY, as_of, filters={"ts_code": benchmark})
    if idx is None or idx.empty or "trade_date" not in idx.columns:
        return None
    idx = idx.sort_values(["trade_date", "available_at"]).drop_duplicates(subset="trade_date", keep="last")
    mv = pd.to_numeric(idx["close"], errors="coerce")
    td = pd.to_datetime(idx["trade_date"]).dt.date
    mkeep = mv.notna() & (mv > 0.0) & (td < as_of)      # same strict-date guard as the NAV side
    m_dates = td[mkeep].tolist()
    m_vals = mv[mkeep].to_numpy(dtype=float)

    fpos = {d: i for i, d in enumerate(f_dates)}
    common = [(fpos[d], j) for j, d in enumerate(m_dates) if d in fpos]
    if len(common) < min_obs + 1:
        return None
    fi = np.array([a for a, _b in common])
    mi = np.array([b for _a, b in common])
    rf = np.diff(f_vals[fi]) / f_vals[fi][:-1]
    rm = np.diff(m_vals[mi]) / m_vals[mi][:-1]
    rf, rm = rf[-window:], rm[-window:]
    ok = np.isfinite(rf) & np.isfinite(rm)
    if ok.sum() < min_obs:
        return None
    rf, rm = rf[ok], rm[ok]
    # Consistent normalisation: np.cov defaults to ddof=1 while np.var defaults to ddof=0, which
    # would inflate beta by N/(N-1). Centre once and use population moments for both.
    rf_c, rm_c = rf - rf.mean(), rm - rm.mean()
    var_m = float(np.mean(rm_c * rm_c))
    if var_m <= 0.0:
        return None
    beta = float(np.mean(rf_c * rm_c) / var_m)
    return min(max(beta, 0.0), clip_high)


def _default_store() -> PointInTimeStore:
    """Build the default cache-backed store (mirrors ``qmt.data.sources._resolve_store``).

    Reads back whatever an earlier process persisted under ``data_cache`` (the store
    hydrates its per-table Parquet lazily on first access). If nothing has been primed
    yet, get_asof yields no rows → ``[]``. Callers that want a real estimate ensure the
    cache is populated by ``qmt.data.sources`` first (or inject a store).
    """
    from qmt.config import get_settings

    return PointInTimeStore(base_dir=get_settings().ensure_data_cache())


def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalized_weights(lines: list[HoldingLine]) -> list[float]:
    """Within-book weights normalized to sum to 1 over the disclosed (top-10) holdings.

    Uses ``stk_mkv_ratio`` (share of the fund's stock market value) when present, else
    ``mkv`` (position market value); normalization makes the percent-vs-fraction scale
    irrelevant. When no usable magnitude exists, falls back to equal weights. The result is
    a *within-disclosed-book* share, in [0, 1], summing to 1 across labels — the transparent
    baseline interpretation (not an equity/NAV ratio, which is a separate data gap).
    """
    n = len(lines)
    if n == 0:
        return []
    raw: list[float] = []
    for line in lines:
        magnitude = line.stk_mkv_ratio
        if magnitude is None:
            magnitude = line.mkv
        raw.append(float(magnitude) if magnitude is not None and magnitude > 0 else 0.0)
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / n] * n
    return [r / total for r in raw]


def _norm_symbol(symbol: str) -> str:
    """Canonicalize a stock symbol for map lookup (strip the exchange suffix)."""
    return symbol.strip().split(".")[0]


def _lookup(label_map: Mapping[str, str], symbol: str) -> Optional[str]:
    """Look up a symbol's label trying the raw form then the suffix-stripped form."""
    if symbol in label_map:
        return label_map[symbol]
    base = _norm_symbol(symbol)
    return label_map.get(base)


def _aggregate_by_label(
    lines: list[HoldingLine],
    weights: list[float],
    label_map: Mapping[str, str],
) -> dict[str, float]:
    """Sum normalized weights per label; unmapped weight accumulates under the sentinel."""
    agg: dict[str, float] = {}
    for line, w in zip(lines, weights):
        label = _lookup(label_map, line.symbol) if label_map else None
        key = label if label else UNCLASSIFIED_LABEL
        agg[key] = agg.get(key, 0.0) + w
    return agg


def _emit(
    fund_code: str,
    as_of: date,
    dimension: ExposureDimension,
    agg: Mapping[str, float],
    disclosure_age_days: int,
    drift_prob: float,
    confidence: float,
    rel_halfwidth: float,
) -> list[ExposureEstimate]:
    """Turn a label→weight aggregate into ExposureEstimates with a low/high band."""
    out: list[ExposureEstimate] = []
    for label, point in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])):
        if point <= 0.0:
            continue
        pt = min(max(point, 0.0), 1.0)
        low = max(0.0, pt * (1.0 - rel_halfwidth))
        high = min(1.0, pt * (1.0 + rel_halfwidth))
        out.append(
            ExposureEstimate(
                fund_code=fund_code,
                as_of=as_of,
                dimension=dimension,
                label=label,
                point_estimate=pt,
                low=low,
                high=high,
                disclosure_age_days=disclosure_age_days,
                drift_prob=drift_prob,
                confidence=confidence,
            )
        )
    return out


def _resolve_industry_map(
    lines: list[HoldingLine],
    industry_map: Optional[Mapping[str, str]],
    client: object | None,
) -> Mapping[str, str]:
    """Return the ``{symbol -> Shenwan L1}`` map to aggregate by.

    An injected ``industry_map`` always wins (the offline / tested path). Otherwise a
    best-effort Tushare lookup is attempted *within quota*; any failure (no client, quota
    exhausted, empty result) yields an empty map so the weight surfaces as a data gap. The
    heavy ``index_classify`` + ``index_member`` Shenwan taxonomy join (E40) is deferred — the
    Tushare branch here is a documented hook, not a backtested classifier.
    """
    if industry_map is not None:
        return industry_map
    if client is None:
        return {}
    return _shenwan_l1_map_from_tushare([ln.symbol for ln in lines], client)


def _shenwan_l1_map_from_tushare(symbols: list[str], client: object) -> Mapping[str, str]:
    """Best-effort Shenwan L1 membership from Tushare (documented hook, E40 deferred).

    Attempts to build ``{symbol -> L1 name}`` from ``index_classify`` (level L1) +
    ``index_member``. Guards every Tushare call: on any error (quota / rate limit / schema /
    transport) it returns ``{}`` so exposure estimation degrades to a flagged data gap
    instead of crashing. This is the "map ... if within quota, else flag data-gap" branch.
    """
    try:  # pragma: no cover - live-only path (deferred SW taxonomy join, E40)
        from qmt.data.tushare_client import TushareError

        query = getattr(client, "query", None)
        if query is None:
            return {}
        classify = query("index_classify", {"level": "L1", "src": "SW2021"}, "index_code,industry_name")
        if classify is None or getattr(classify, "empty", True):
            return {}
        wanted = {_norm_symbol(s) for s in symbols}
        mapping: dict[str, str] = {}
        for row in classify.to_dict("records"):
            index_code = str(row.get("index_code") or "").strip()
            name = str(row.get("industry_name") or "").strip()
            if not index_code or not name:
                continue
            members = query("index_member", {"index_code": index_code}, "index_code,con_code")
            if members is None or getattr(members, "empty", True):
                continue
            for m in members.to_dict("records"):
                con = str(m.get("con_code") or "").strip()
                base = _norm_symbol(con)
                if base in wanted:
                    mapping[base] = name
        return mapping
    except TushareError:  # pragma: no cover - live-only
        return {}
    except Exception:  # pragma: no cover - defensive: never let the hook crash estimation
        return {}


def _candidate_regression_exposure(*args: object, **kwargs: object) -> list[ExposureEstimate]:
    """CANDIDATE (E49) — rolling constrained NAV-vs-index regression. Documented hook only.

    Direction (05-algorithms.tex, ExposureEstimator candidate column): regress the fund's
    recent ``adj_nav`` returns on a panel of industry/style index returns over a rolling
    window (``ExposureConfig.regression_window_trading_days``) with constraints
    ``weights >= 0`` (``weights_nonneg``) and ``sum(weights) <= 1`` (``weights_sum_max``,
    residual = cash/other), e.g. via ``cvxpy``, updating the estimate over time.

    NOT wired in: per the doc's baseline-first rule a candidate replaces the baseline only
    after time-ordered backtest shows its estimates track the later verifiable disclosures
    more closely and it stays stable across market regimes.
    """
    raise NotImplementedError(
        "candidate rolling constrained NAV-vs-index regression (E49) is a documented hook; "
        "the baseline holding-age discount is active — see 05-algorithms.tex ExposureEstimator."
    )
