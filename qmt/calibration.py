"""Confidence-calibration production path (§4 sec:mod-uncertainty, §5 row \\UncertaintyM, §7).

The doc is emphatic that ``confidence`` must be **historically calibrated**, not a free
score: «如果系统过去说"把握较高"，那类似判断后来是否真的更常正确». The uncertainty module
(:mod:`qmt.pipeline.uncertainty`) supplies the two pieces — :func:`assess_uncertainty`
(raw score) and :func:`build_calibration_table` / :func:`calibrate` (bucketed hit-rate
lookup) — but nothing in the production path ever *built* a table, so the default
``run_analysis`` reported the raw, uncalibrated score.

This module closes that gap. It is the **backtest-driven builder**: it replays the
point-in-time analysis stages (exposure → market_state → horizon_forecaster →
uncertainty → to_product_view) over a historical window, pairs each replayed raw
confidence with whether the product-view **direction** call was subsequently right (the
realized forward return over the primary horizon), buckets the pairs via
:func:`~qmt.pipeline.uncertainty.build_calibration_table`, and **persists** the resulting
:class:`~qmt.types.CalibrationTable` to ``data_cache/calibration.json``.
:func:`~qmt.pipeline.analysis.run_analysis` loads that cached table by default and applies
it; when it is absent the reported confidence stays raw and is clearly flagged
*uncalibrated*.

Point-in-time discipline (invariant 1): every **signal** read (exposure / market / forecast
/ uncertainty) goes through :meth:`PointInTimeStore.get_asof` at the T-day 14:30 decision
moment, exactly as production. Only the **outcome** (the realized forward NAV used to score
whether the call was right) reads later NAV — which is legitimate, since a calibration is
built from history whose outcomes are, by construction, already known.

Doc-silent decisions (DECISIONS.md): the calibration *target* is direction-correctness with
a RANGE tolerance band (D-CAL-1/D-CAL-2); the JSON persistence format + cache location
(D-CAL-3).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from qmt.config import CONFIG, CalibrationConfig, get_settings
from qmt.data.sources import TABLE_FUND_NAV
from qmt.pit import PointInTimeStore, AVAILABLE_AT
from qmt.pipeline.exposure_estimator import estimate_exposure
from qmt.pipeline.horizon_forecaster import forecast
from qmt.pipeline.market_state import InsufficientMarketHistory, assess_market_state
from qmt.pipeline.product_view import primary_forecast, to_product_view
from qmt.pipeline.uncertainty import assess_uncertainty, build_calibration_table
from qmt.types import CalibrationTable, Direction

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATION_FILENAME",
    "ConfidenceObservation",
    "calibration_cache_path",
    "load_cached_calibration_table",
    "persist_calibration_table",
    "replay_confidence_observations",
    "build_calibration_from_history",
]

# Where the persisted table lives under the data-cache dir (D-CAL-3).
CALIBRATION_FILENAME = "calibration.json"

# A single replayed observation: (raw_confidence in [0,1], was the direction call correct).
ConfidenceObservation = Tuple[float, bool]

# RANGE-call tolerance: a 'range/震荡' judgment is scored correct iff the realized forward
# return stayed within +/- this band (doc-silent, D-CAL-2). UP/DOWN are scored by sign.
_RANGE_CORRECT_BAND = 0.02

# Sentinel outcome cutoff: read ALL disclosed NAV when measuring realized outcomes (a
# historical calibration's outcomes are already known; this is not a signal-time read).
# Kept within pandas' nanosecond Timestamp range (< 2262) so the PIT store filter converts.
_ALL_NAV_MOMENT = datetime(2200, 1, 1)

_DEFAULT_BENCHMARK = "000300.SH"


# =============================================================================
# persistence
# =============================================================================
def calibration_cache_path() -> Path:
    """Default on-disk location of the persisted calibration table (``data_cache/calibration.json``)."""
    return get_settings().data_cache_dir / CALIBRATION_FILENAME


def persist_calibration_table(
    table: CalibrationTable, path: Optional[Union[str, Path]] = None
) -> Path:
    """Write ``table`` to ``path`` (default :func:`calibration_cache_path`) as JSON; return the path."""
    p = Path(path) if path is not None else calibration_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(table.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_cached_calibration_table(
    path: Optional[Union[str, Path]] = None
) -> Optional[CalibrationTable]:
    """Load the persisted calibration table, or ``None`` if absent/unreadable.

    Never raises — a missing or malformed calibration file must degrade to the raw
    (uncalibrated) path, not break analysis (fail-safe on this optional reference).
    """
    p = Path(path) if path is not None else calibration_cache_path()
    if not p.exists():
        return None
    try:
        return CalibrationTable.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive: a bad cache never breaks analysis
        logger.warning("ignoring unreadable calibration table at %s", p)
        return None


# =============================================================================
# point-in-time data access (outcome NAV path)
# =============================================================================
def _decision_moment(as_of: date) -> datetime:
    """T-day decision instant (14:30) — the point-in-time read as_of for every signal."""
    t = CONFIG.timing
    return datetime.combine(as_of, time(hour=t.decision_hour, minute=t.decision_minute))


def _fund_nav_series(
    store: PointInTimeStore, code: str, moment: datetime
) -> tuple[list[date], np.ndarray]:
    """The fund's ``adj_nav`` path (by ``nav_date``) observable at ``moment``, oldest→newest.

    Restatements collapse to the latest-disclosed value per ``nav_date``; ``adj_nav`` is
    preferred, falling back to ``unit_nav``. Non-positive / NaN prices are dropped.
    """
    df = store.get_asof(TABLE_FUND_NAV, moment, {"fund_code": code})
    if df is None or df.empty or "nav_date" not in df.columns:
        return [], np.empty(0, dtype=float)
    df = df.sort_values(["nav_date", AVAILABLE_AT]).drop_duplicates(subset="nav_date", keep="last")
    col = None
    if "adj_nav" in df.columns and df["adj_nav"].notna().any():
        col = "adj_nav"
    elif "unit_nav" in df.columns and df["unit_nav"].notna().any():
        col = "unit_nav"
    if col is None:
        return [], np.empty(0, dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce")
    keep = vals.notna() & (vals > 0.0)
    dates = [pd.Timestamp(d).date() for d in df.loc[keep, "nav_date"]]
    return dates, vals[keep].to_numpy(dtype=float)


def _forward_return(
    o_dates: list[date], o_vals: np.ndarray, as_of: date, window_max: int
) -> Optional[float]:
    """Realized forward return from the first NAV strictly after ``as_of`` over ``window_max`` steps.

    Entry = first executable NAV after the decision (T+1, no same-day fill); exit = that
    index + ``window_max`` NAV points (≈ trading days). ``None`` when there is not enough
    forward NAV to measure the outcome.
    """
    entry_idx = next((i for i, d in enumerate(o_dates) if d > as_of), None)
    if entry_idx is None:
        return None
    exit_idx = entry_idx + window_max
    if exit_idx >= len(o_vals):
        return None
    entry = float(o_vals[entry_idx])
    if entry <= 0.0:
        return None
    return float(o_vals[exit_idx]) / entry - 1.0


def _direction_correct(direction: Direction, realized_return: float) -> bool:
    """Whether the product-view direction call matched the realized forward return (D-CAL-1)."""
    if direction is Direction.UP:
        return realized_return > 0.0
    if direction is Direction.DOWN:
        return realized_return < 0.0
    return abs(realized_return) <= _RANGE_CORRECT_BAND  # RANGE / 震荡


# =============================================================================
# replay + build
# =============================================================================
def replay_confidence_observations(
    code: str,
    start: date,
    end: date,
    store: PointInTimeStore,
    *,
    benchmark: str = _DEFAULT_BENCHMARK,
    outcome_asof: Optional[datetime] = None,
) -> list[ConfidenceObservation]:
    """Replay the analysis stages over ``[start, end]`` → ``(raw_confidence, was_correct)`` pairs.

    For each observable NAV date in the window, the signal (exposure / market_state /
    forecast / raw uncertainty / product-view direction) is computed point-in-time at the
    14:30 decision moment, and the direction call is scored against the realized forward
    return over the primary horizon. Dates whose market state cannot be assessed (thin index
    history) or whose forward outcome is not yet measurable are skipped (fail-closed skip).
    """
    outcome_moment = outcome_asof or _ALL_NAV_MOMENT
    o_dates, o_vals = _fund_nav_series(store, code, outcome_moment)
    if not o_dates:
        return []

    observations: list[ConfidenceObservation] = []
    for d in o_dates:
        if not (start <= d <= end):
            continue
        exposure = estimate_exposure(code, d, store)
        try:
            market = assess_market_state(d, store, benchmark)
        except InsufficientMarketHistory:
            continue
        forecasts = forecast(code, d, exposure, market, store)
        if not forecasts:
            continue
        primary = primary_forecast(forecasts)
        if primary is None:
            continue
        raw = assess_uncertainty(forecasts, exposure, market)
        pv = to_product_view(forecasts, raw, market, code, d)
        window_max = CONFIG.horizon.windows[primary.horizon.value][1]
        realized = _forward_return(o_dates, o_vals, d, window_max)
        if realized is None:
            continue
        observations.append((raw, _direction_correct(pv.direction, realized)))
    return observations


def build_calibration_from_history(
    codes: Union[str, Sequence[str]],
    start: date,
    end: date,
    store: Optional[PointInTimeStore] = None,
    *,
    benchmark: str = _DEFAULT_BENCHMARK,
    outcome_asof: Optional[datetime] = None,
    cfg: Optional[CalibrationConfig] = None,
    out_path: Optional[Union[str, Path]] = None,
    persist: bool = True,
) -> CalibrationTable:
    """Replay a historical window across ``codes``, build a calibration table, and persist it.

    This is the production calibration path (``qmt calibrate``): it replays forecasts vs.
    realized outcomes, buckets the ``(raw_confidence, was_correct)`` pairs into a
    :class:`~qmt.types.CalibrationTable`, and (by default) writes it to
    :func:`calibration_cache_path` so :func:`~qmt.pipeline.analysis.run_analysis` loads it.

    Args:
        codes: one 6-digit fund code or several (their observations are pooled).
        start, end: inclusive historical replay window.
        store: point-in-time store to read; ``None`` opens the persistent cache store.
        benchmark: broad index for market_state.
        outcome_asof: cutoff for reading realized-outcome NAV; ``None`` = all disclosed NAV.
        cfg: calibration bucketing config; defaults to ``CONFIG.calibration``.
        out_path: where to persist; ``None`` = :func:`calibration_cache_path`.
        persist: write the table to disk when ``True`` (default).
    """
    code_list = [codes] if isinstance(codes, str) else list(codes)
    st = store if store is not None else PointInTimeStore(get_settings().data_cache_dir)

    observations: list[ConfidenceObservation] = []
    for code in code_list:
        observations.extend(
            replay_confidence_observations(
                code, start, end, st, benchmark=benchmark, outcome_asof=outcome_asof
            )
        )

    table = build_calibration_table(observations, cfg=cfg)
    if persist:
        path = persist_calibration_table(table, out_path)
        logger.info(
            "built calibration table from %d observation(s) over %s..%s → %s",
            len(observations),
            start.isoformat(),
            end.isoformat(),
            path,
        )
    return table
