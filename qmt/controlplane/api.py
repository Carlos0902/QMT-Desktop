"""FastAPI control-plane backend (subsystem E89, Part-B addition beyond the doc).

A developer-facing HTTP control plane that lets you **CONTROL**, **DEBUG** and **REFINE**
the eleven-stage pipeline that lives in :mod:`qmt.pipeline`:

* **CONTROL** — search the primed fund universe (``/api/funds``), drive an analysis or an
  advice run for one fund at one point in time (``/api/analyze`` / ``/api/advice``), and
  run a point-in-time backtest (``/api/backtest``).
* **DEBUG** — ``/api/analyze`` returns the *full per-stage trace*, not just the final
  answer: fund scope, data visibility (fresh/stale/expired per source), the three-tier
  exposure estimate (disclosed-fact age → estimate range → confidence), the 4-dimension
  market state, the per-horizon forecast ranges, the calibrated confidence, the derived
  product-view labels, the explanation-card sections, and which compliance gates fired.
* **REFINE** — ``/api/config`` reads and **live-edits** the tunable calibration thresholds
  in :data:`qmt.config.CONFIG` (freshness cutoffs, confidence decay, market-state
  thresholds, action gates, product-view thresholds) so a change takes effect on the very
  next ``/api/analyze`` — validated and clamped to each field's declared bounds.

Hard rules honoured (see ``CLAUDE.md``):

* **Point-in-time.** Every request first *ingests* (WRITES, each row tagged with its true
  ``available_at`` by :mod:`qmt.data.sources`) into a :class:`~qmt.pit.PointInTimeStore`,
  then *runs* (READS ``get_asof(<= as_of)`` only). Ingest and run stay cleanly split.
* **Mode isolation.** ``/api/analyze`` calls :func:`qmt.pipeline.run.run_analysis`, whose
  signature has no :class:`~qmt.types.UserPortfolio`; only ``/api/advice`` ever builds a
  portfolio and calls :func:`~qmt.pipeline.run.run_advice`.
* **Fail-closed.** Out-of-scope / unconfirmable funds surface the compliance gate that
  fired (HTTP 422); an empty backtest window is a 400, never a silent empty result.
* **Token only from config.** Ingestion builds the default Tushare client whose token
  comes from ``.env`` via :func:`qmt.config.get_settings`; nothing here hard-codes it.
* **Compliance boundaries are NOT live-editable.** The :class:`ComplianceConfig` switches
  (fail-closed, suitability-required, …) are deliberately excluded from ``/api/config`` —
  they defer to formally effective regulation (§3.5-9), not a slider.

``create_app()`` builds the app; ``app`` at module scope is uvicorn-runnable:
``uvicorn qmt.controlplane.api:app --reload``.
"""

from __future__ import annotations

import html as _html
import json as _json
import logging
import queue as _queue
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Optional

import annotated_types as _at
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import qmt
from qmt.compliance import ComplianceGuard, ComplianceViolation, GateResult
from qmt.config import CONFIG, get_settings
from qmt.data.ingest import ingest_fund, ingest_universe
from qmt.data.sources import TABLE_FUND_BASIC, TABLE_FUND_NAV, TABLE_INDEX_DAILY
from qmt.pit import AVAILABLE_AT, PointInTimeStore, confirm_lag_modeled
from qmt.pipeline.run import AdviceResult, AnalysisResult, run_advice, run_analysis
from qmt.pipeline.portfolio_lookthrough import look_through
from qmt.pipeline.risk_budget import assess_risk_budget
from qmt.backtest import BacktestResult, run_backtest
from qmt.fees import default_fee_schedule
from qmt.types import (
    ExposureEstimate,
    FeeSchedule,
    Horizon,
    LookthroughResult,
    PortfolioHolding,
    RiskTolerance,
    SuitabilityProfile,
    UserPortfolio,
    VisibilitySet,
)

__all__ = [
    "create_app",
    "app",
    "EDITABLE_GROUPS",
    "list_editable_fields",
    "apply_config_override",
    "snapshot_config",
    "restore_config",
]

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DECISIONS_MD = Path(__file__).with_name("USER_GUIDE.md")
_TRACEABILITY_MD = Path(__file__).with_name("DATA_GUIDE.md")

# Ordered horizons for stable forecast serialization.
_HORIZON_ORDER = (Horizon.SHORT, Horizon.MID, Horizon.LONG)

# Config groups whose scalar (int/float/bool) fields may be live-edited. Deliberately
# excludes ``compliance`` (fail-closed switches defer to regulation, §3.5-9), ``timing``
# (holds a per-fund override dict), ``scope`` / ``tushare`` (tuples / codes), and
# ``validation`` (CI gate constants). See DECISIONS.md D-API-3.
EDITABLE_GROUPS: tuple[str, ...] = (
    "freshness",
    "confidence_decay",
    "market_state",
    "action",
    "product_view",
    "exposure",
    "horizon",
    "forecaster",
    "hierarchical",
    "risk_budget",
    "calibration",
)

# A parameter search temporarily swaps calibration constants on the shared CONFIG singleton
# while scoring each candidate (restoring in a finally). This lock serialises searches so two
# concurrent runs can never interleave their CONFIG mutations (D-TUNE-6). A dev control plane is
# effectively single-user, so a busy search returns 409 rather than queueing.
_TUNING_LOCK = threading.Lock()
# Serializes layer-curve PNG rendering: matplotlib's font manager / rcParams are process-global,
# so concurrent renders share mutable global state; one at a time is simplest and each is a few
# seconds. The lock also collapses a concurrent burst (market + industry) onto one grid build.
_CURVE_RENDER_LOCK = threading.Lock()
# Set for the DURATION of a search (while CONFIG holds transient candidate values). Config-reading
# endpoints check it and fail-closed with 409 rather than silently returning a forecast/analysis
# computed against a candidate config the user never set (read-isolation, D-TUNE-6).
_TUNING_ACTIVE = threading.Event()
# Cooperative-cancel flag: the frontend "Stop" button (POST /api/tuning/cancel) sets it; the
# search checks it at each candidate boundary and stops early (still restoring CONFIG). So "Stop"
# actually stops the server-side worker, not just the client stream.
_TUNING_CANCEL = threading.Event()


def _reject_if_tuning() -> None:
    """Fail-closed 409 while a parameter search holds transient candidate values on CONFIG.

    The tuning worker swaps calibration constants on the shared ``CONFIG`` singleton for the
    duration of a search; any endpoint that READS those constants (analyze / advice / backtest /
    portfolio suggestions / config) must not run meanwhile, or it would return a result computed
    against a candidate config the user never chose. Guarded endpoints reject with 409 for the
    few seconds a search runs (self-clears when it finishes) — honest over silently-wrong.
    """
    if _TUNING_ACTIVE.is_set():
        raise HTTPException(
            status_code=409,
            detail="训练进行中：为避免读到搜索期间的临时参数，请等本次训练结束后重试（通常几秒）。",
        )


# =============================================================================
# Request models
# =============================================================================
class AnalyzeRequest(BaseModel):
    """POST /api/analyze body."""

    fund_code: str = Field(..., description="6-digit off-exchange fund code, e.g. '010001'.")
    as_of: date = Field(..., description="Decision date (ISO 'YYYY-MM-DD'); data after it is hidden.")
    ingest: bool = Field(True, description="Prime the store first (set False to reuse primed data).")


class AdviceRequest(BaseModel):
    """POST /api/advice body — the SOLE portfolio-carrying request."""

    fund_code: str
    as_of: date
    portfolio: UserPortfolio
    suitability: SuitabilityProfile
    fees: FeeSchedule
    position: float = Field(0.0, ge=0.0, le=1.0, description="Current single-fund position [0,1].")
    ingest: bool = True


class HoldingInput(BaseModel):
    """One position row from the manage-position page (amount + when it was bought).

    The user enters ``buy_date``; the server derives the holding period (``as_of − buy_date``)
    that sizes the redemption-fee tier. ``hold_days`` is accepted for back-compat but a
    ``buy_date`` wins when both are present."""

    fund_code: str
    market_value: float = Field(..., gt=0.0, description="Amount currently held (¥); weights derived from it.")
    buy_date: Optional[date] = Field(None, description="When the fund was bought; hold days = as_of − buy_date.")
    hold_days: Optional[int] = Field(None, ge=0, description="Deprecated: explicit hold days (buy_date wins).")


def _effective_hold_days(h: HoldingInput, as_of: date) -> int:
    """Holding period (days) at ``as_of``: derived from ``buy_date`` if given, else ``hold_days``."""
    if h.buy_date is not None:
        return max((as_of - h.buy_date).days, 0)
    return max(h.hold_days or 0, 0)


class PortfolioSuggestionsRequest(BaseModel):
    """POST /api/portfolio/suggestions body — the whole portfolio + suitability, one 'generate' click.

    ``as_of`` is optional and defaults to **today** on the server: the manage-position page is a
    daily tool, so suggestions are always the newest available — the user does not pick a
    decision date. Point-in-time correctness still holds (only data observable ≤ the decision
    moment is used), so a today ``as_of`` naturally yields the freshest observable data."""

    holdings: list[HoldingInput] = Field(default_factory=list)
    available_cash: float = Field(
        0.0, ge=0.0, description="Investable cash (¥) not yet in any fund; allowed with no holdings."
    )
    must_fully_invest: bool = Field(
        False, description="Whether to try to deploy all usable cash (still bounded by the protection rule)."
    )
    min_cash_reserve: float = Field(0.0, ge=0.0, description="Cash to keep uninvested (¥).")
    as_of: Optional[date] = Field(None, description="Decision date; omitted → today (newest data).")
    risk_tolerance: RiskTolerance = Field(RiskTolerance.C3)
    investment_horizon_days: int = Field(365, ge=0)
    ingest: bool = True


class ConfigUpdateRequest(BaseModel):
    """PUT /api/config body — one dotted ``group.field`` path and its new value."""

    path: str = Field(..., description="Dotted path, e.g. 'market_state.vol_low_percentile'.")
    value: Any = Field(..., description="New value; validated + clamped to the field's bounds.")


class BacktestRequest(BaseModel):
    """POST /api/backtest body."""

    fund_code: str
    start: date
    end: date
    strategy: str = Field("baseline", description="baseline/ma_cross/buy_and_hold/trend/vol_control/fixed_sip.")
    ingest: bool = True


class TuningSearchRequest(BaseModel):
    """POST /api/tuning/search body — one walk-forward parameter search ('start training').

    The **method and the target are auto-selected** (D-TUNE-7): the system always trains the
    whole-market market-timing model (the §5.3 大盘 layer) on the broad index — the user picks
    neither a method nor a fund. The user gives only an overall **month** range; the system
    divides it into expanding walk-forward folds plus an untouched trailing holdout (D-TUNE-8/9)
    and selects on the *out-of-sample* fold score, never on the in-sample one.
    """

    start_month: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="Overall range start, 'YYYY-MM'.")
    end_month: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="Overall range end (inclusive), 'YYYY-MM'.")
    ingest: bool = Field(True, description="Prime the broad market index over the window first (set False to reuse).")


def _parse_month(value: str) -> date:
    """``'YYYY-MM'`` → the first day of that month."""
    year, month = value.split("-")
    return date(int(year), int(month), 1)


def _month_range_error_zh(exc: Exception) -> str:
    """User-facing (zh) message for an invalid training month range (D-TUNE-8/9)."""
    msg = str(exc)
    if "precedes" in msg:
        return "结束月份不能早于开始月份。"
    return "月份区间至少需要 4 个月：系统要划分出训练折、样本外验证折，以及从不参与择优的留出确认窗口。"


class TuningApplyRequest(BaseModel):
    """POST /api/tuning/apply body — write the winning settings onto the live CONFIG.

    When ``run_id`` names a persisted run that is **not promotable** (it failed the doc's
    "stably beats baseline" gate), the write is refused 422 unless ``force`` is set — a tuned
    setting that lost on the untouched holdout must not silently replace the baseline (D-TUNE-9).
    """

    params: dict[str, Any] = Field(
        ..., description="Winning ``{'group.field': value}`` overrides (from a search's best candidate)."
    )
    run_id: Optional[str] = Field(None, description="Persisted run these params came from (enables the promotion gate).")
    force: bool = Field(False, description="Apply even when the run is not promotable (explicit override).")


# =============================================================================
# Serialization helpers (enums→values, dates→ISO, ndarray→nested list)
# =============================================================================
def _dump(model: Optional[BaseModel]) -> Optional[dict]:
    """JSON-safe dump of a pydantic model (enum values, ISO dates), or ``None``."""
    return None if model is None else model.model_dump(mode="json")


def _dump_gate(g: GateResult) -> dict:
    """Serialize a compliance :class:`GateResult` (frozen dataclass)."""
    return {
        "gate": g.gate,
        "passed": g.passed,
        "reason": g.reason,
        "degraded_action": g.degraded_action.value if g.degraded_action is not None else None,
    }


def _dump_forecasts(forecasts: dict[Horizon, Any]) -> dict:
    """``{horizon.value: HorizonForecast}`` in a stable short/mid/long order."""
    out: dict[str, Any] = {}
    for h in _HORIZON_ORDER:
        fc = forecasts.get(h)
        if fc is not None:
            out[h.value] = fc.model_dump(mode="json")
    # any non-standard horizons (defensive)
    for h, fc in forecasts.items():
        if h.value not in out:
            out[h.value] = fc.model_dump(mode="json")
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _dump_exposure(exposure: list[ExposureEstimate]) -> list[dict]:
    """Serialize each exposure estimate with an explicit three-tier breakdown.

    Tier 1 = disclosed fact + its age; Tier 2 = the system estimate as a range; Tier 3 =
    the system's confidence in its own estimate (invariant 2 — estimate ≠ disclosure).
    """
    out: list[dict] = []
    for e in exposure:
        out.append(
            {
                "dimension": e.dimension.value,
                "label": e.label,
                "point_estimate": e.point_estimate,
                "low": e.low,
                "high": e.high,
                "range": f"{_pct(e.low)}–{_pct(e.high)}",
                "disclosure_age_days": e.disclosure_age_days,
                "drift_prob": e.drift_prob,
                "confidence": e.confidence,
                "tiers": {
                    "tier1_disclosed_fact": {
                        "disclosure_age_days": e.disclosure_age_days,
                        "drift_prob": e.drift_prob,
                    },
                    "tier2_system_estimate": {
                        "point_estimate": e.point_estimate,
                        "low": e.low,
                        "high": e.high,
                        "range": f"{_pct(e.low)}–{_pct(e.high)}",
                    },
                    "tier3_confidence": {"confidence": e.confidence},
                },
            }
        )
    return out


def _dump_visibility(vis: VisibilitySet) -> dict:
    """VisibilitySet + a per-source fresh/stale/expired + visible/hidden roll-up."""
    base = vis.model_dump(mode="json")
    summary = {"fresh": 0, "stale": 0, "expired": 0, "visible": 0, "hidden": 0}
    for s in vis.snapshots:
        summary[s.freshness.value] = summary.get(s.freshness.value, 0) + 1
        summary[s.visible.value] = summary.get(s.visible.value, 0) + 1
    base["summary"] = summary
    return base


def _dump_lookthrough(lr: Optional[LookthroughResult]) -> Optional[dict]:
    """Serialize a look-through result, converting the ndarray corr matrix to nested lists."""
    if lr is None:
        return None
    corr: Optional[list] = None
    if lr.corr_matrix is not None:
        corr = np.asarray(lr.corr_matrix, dtype=float).tolist()
    return {
        "industry_conc": lr.industry_conc,
        "style_conc": lr.style_conc,
        "theme_overlap": lr.theme_overlap,
        "fund_risk_contrib": lr.fund_risk_contrib,
        "budget_usage": lr.budget_usage,
        "fund_labels": lr.fund_labels,
        "corr_matrix": corr,
    }


def _df_records(df: pd.DataFrame) -> list[dict]:
    """A ``date, equity`` (or similar) frame → JSON records with ISO dates and float equity."""
    if df is None or df.empty:
        return []
    out: list[dict] = []
    for row in df.to_dict("records"):
        rec: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp,)):
                rec[k] = v.date().isoformat()
            elif isinstance(v, date):
                rec[k] = v.isoformat()
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            else:
                rec[k] = v
        out.append(rec)
    return out


# =============================================================================
# Compliance-gate reconstruction (post-hoc, from the returned result)
# =============================================================================
def _analysis_gates(result: AnalysisResult) -> list[dict]:
    """Re-evaluate the analysis-path compliance gates against the produced result.

    Gate 2 (support-tier) and gate 3 (advice-strength-cap) both report *fired* for a
    non-FULL (LIMITED/DEFERRED) degraded result — analysis_mode degrades rather than raising.
    Freshness/confidence (gate 4) is recomputed with the same ``data_expired`` heuristic
    ``run`` uses (empty exposure or empty forecasts); the regulation-deference marker (gate 9)
    is always surfaced. Honest debug reconstruction of "which gates fired", not a decision.
    """
    guard = ComplianceGuard()
    data_expired = (not result.exposure) or (not result.forecasts)
    return [
        _dump_gate(guard.gate_scope_required(result.scope)),
        _dump_gate(guard.gate_advice_strength_cap(result.scope)),
        _dump_gate(guard.gate_confidence_and_freshness(result.confidence, data_expired=data_expired)),
        _dump_gate(guard.gate_regulation_deference()),
    ]


def _advice_gates(
    result: AdviceResult, suitability: SuitabilityProfile, fees: FeeSchedule, as_of: date
) -> list[dict]:
    """Re-evaluate the advice-path gates; the action's own strategy gates are returned
    separately via ``ActionRecommendation.gated_by``.

    ``execution_delay_modeled`` is derived truthfully from the T+1 decision timeline for
    ``as_of`` (:func:`qmt.pit.confirm_lag_modeled`) — exactly what ``run_advice`` used — not
    a hardcoded ``True`` (§3.5-5)."""
    guard = ComplianceGuard()
    delay_modeled = confirm_lag_modeled(as_of)
    return [
        _dump_gate(guard.gate_suitability_required(suitability)),
        _dump_gate(guard.gate_net_of_cost(fees_provided=fees is not None, execution_delay_modeled=delay_modeled)),
        _dump_gate(guard.gate_risk_budget(result.risk_budget, is_add_position=False)),
        _dump_gate(guard.gate_regulation_deference()),
    ]


# =============================================================================
# Config live-edit (REFINE)
# =============================================================================
def _field_bounds(field_info: Any) -> tuple[Optional[float], Optional[float], bool, bool]:
    """Return ``(lo, hi, lo_exclusive, hi_exclusive)`` from a pydantic FieldInfo's metadata."""
    lo = hi = None
    lo_excl = hi_excl = False
    for m in field_info.metadata:
        if isinstance(m, _at.Ge):
            lo = m.ge
        elif isinstance(m, _at.Gt):
            lo, lo_excl = m.gt, True
        elif isinstance(m, _at.Le):
            hi = m.le
        elif isinstance(m, _at.Lt):
            hi, hi_excl = m.lt, True
    return lo, hi, lo_excl, hi_excl


def list_editable_fields() -> list[dict]:
    """Every live-editable scalar threshold: dotted path, current value, type and bounds."""
    out: list[dict] = []
    for group in EDITABLE_GROUPS:
        sub = getattr(CONFIG, group)
        for fname, fi in type(sub).model_fields.items():
            ann = fi.annotation
            if ann not in (int, float, bool):
                continue
            lo, hi, lo_excl, hi_excl = _field_bounds(fi)
            out.append(
                {
                    "path": f"{group}.{fname}",
                    "value": getattr(sub, fname),
                    "type": ann.__name__,
                    "min": lo,
                    "max": hi,
                    "min_exclusive": lo_excl,
                    "max_exclusive": hi_excl,
                    "description": fi.description or "",
                }
            )
    return out


def apply_config_override(path: str, value: Any) -> dict:
    """Validate + clamp ``value`` and live-apply it to ``CONFIG`` at ``group.field``.

    Rebuilds the target sub-config (which re-runs pydantic validation) and swaps it onto the
    shared frozen ``CONFIG`` singleton in place, so every pipeline module — which reads
    ``CONFIG.<group>.<field>`` at call time — picks up the change on the next run. Returns
    ``{path, value, clamped, previous}``. Raises :class:`ValueError` on an unknown/uneditable
    path or an uncoercible/invalid value.
    """
    if "." not in path:
        raise ValueError(f"path {path!r} must be 'group.field'.")
    group, field = path.split(".", 1)
    if group not in EDITABLE_GROUPS:
        raise ValueError(
            f"group {group!r} is not live-editable (editable: {', '.join(EDITABLE_GROUPS)})."
        )
    sub = getattr(CONFIG, group)
    fields = type(sub).model_fields
    if field not in fields:
        raise ValueError(f"unknown field {field!r} on config group {group!r}.")
    fi = fields[field]
    ann = fi.annotation
    if ann not in (int, float, bool):
        raise ValueError(f"field {path!r} (type {ann}) is not a scalar and cannot be edited.")

    previous = getattr(sub, field)
    clamped = False
    if ann is bool:
        coerced: Any = bool(value)
    else:
        try:
            coerced = ann(value)  # int(...) or float(...)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value {value!r} is not a valid {ann.__name__} for {path!r}.") from exc
        lo, hi, lo_excl, hi_excl = _field_bounds(fi)
        if lo is not None and coerced < lo:
            coerced = ann(lo + 1) if (lo_excl and ann is int) else (
                ann(lo + 1e-9) if lo_excl else ann(lo)
            )
            clamped = True
        if hi is not None and coerced > hi:
            coerced = ann(hi - 1) if (hi_excl and ann is int) else (
                ann(hi - 1e-9) if hi_excl else ann(hi)
            )
            clamped = True

    # Rebuild the sub-config (re-runs field validators) and swap it in place. CONFIG is a
    # frozen pydantic model, so bypass the freeze via object.__setattr__ — the identity of
    # the CONFIG object is preserved (every module holds the same reference).
    try:
        new_sub = type(sub)(**{**sub.model_dump(), field: coerced})
    except Exception as exc:  # pydantic ValidationError (e.g. exclusive-bound edge)
        raise ValueError(f"rejected {path}={coerced!r}: {exc}") from exc
    object.__setattr__(CONFIG, group, new_sub)
    return {"path": path, "value": coerced, "clamped": clamped, "previous": previous}


def snapshot_config() -> dict[str, BaseModel]:
    """Capture the current editable sub-config objects (for tests / rollback)."""
    return {g: getattr(CONFIG, g) for g in EDITABLE_GROUPS}


def restore_config(snap: dict[str, BaseModel]) -> None:
    """Restore sub-configs captured by :func:`snapshot_config` onto the shared CONFIG."""
    for g, sub in snap.items():
        object.__setattr__(CONFIG, g, sub)


def _effective_config() -> dict:
    """The full effective calibration config as JSON (all groups, editable + read-only)."""
    return CONFIG.model_dump(mode="json")


# =============================================================================
# Fund-universe search
# =============================================================================
def _full_universe_candidates(
    store: PointInTimeStore, as_of: date, exclude: set, cap: int
) -> list[str]:
    """FULL-tier fund codes to consider for cash allocation — the *analyzable* universe.

    Restricted to funds that have observable NAV data (``fund_nav``) so every candidate can
    actually be analyzed (no wasted forecast on metadata-only funds), then classified from
    ``fund_basic`` and kept only if FULL-tier. Returns up to ``cap`` codes not in ``exclude``
    (the held funds). Bounded by ``cap`` for latency — v1 screens all analyzable funds rather
    than a user watchlist (D-CASH-1)."""
    from datetime import datetime as _dt

    from qmt.pipeline.fund_scope import classify_scope
    from qmt.types import FundProfile, SupportTier

    if cap <= 0:
        return []
    nav = store.get_asof(TABLE_FUND_NAV, as_of)
    if nav is None or nav.empty or "fund_code" not in nav.columns:
        return []
    analyzable = [str(c) for c in nav["fund_code"].dropna().unique()]
    basic = store.get_asof(TABLE_FUND_BASIC, as_of)
    meta = {}
    if basic is not None and not basic.empty and "fund_code" in basic.columns:
        for _, row in basic.iterrows():
            meta.setdefault(str(row.get("fund_code") or ""), row)
    seen = set(exclude)
    out: list[str] = []
    avail = _dt.combine(as_of, _dt.min.time())
    for code in analyzable:
        if not code or code in seen:
            continue
        seen.add(code)
        row = meta.get(code)
        try:
            prof = FundProfile(
                fund_code=code,
                name=str((row.get("name") if row is not None else None) or code),
                fund_type=(str(row["fund_type"]) if row is not None and row.get("fund_type") is not None else None),
                invest_type=(str(row["invest_type"]) if row is not None and row.get("invest_type") is not None else None),
                market=str((row.get("market") if row is not None else None) or CONFIG.tushare.otc_market_code),
                available_at=avail,
            )
        except Exception:  # pragma: no cover - a malformed row never breaks the screen
            continue
        if classify_scope(prof).support_tier is SupportTier.FULL:
            out.append(code)
            if len(out) >= cap:
                break
    return out


def _search_universe(store: PointInTimeStore, as_of: date, query: str, limit: int) -> list[dict]:
    """Search the primed fund universe by code/name substring (case-insensitive).

    Reads ``fund_basic`` point-in-time (``get_asof(as_of)``); if nothing is primed yet it
    attempts a best-effort :func:`ingest_universe` (quota-guarded — an empty result on a
    points-limited account is fine). Returns the most-recently-observable profile per code.
    """
    df = store.get_asof(TABLE_FUND_BASIC, as_of)
    if df is None or df.empty:
        try:
            ingest_universe(as_of, store)
        except Exception:  # pragma: no cover - network/quota; never crash the endpoint
            pass
        df = store.get_asof(TABLE_FUND_BASIC, as_of)
    if df is None or df.empty:
        return []

    df = df.sort_values(AVAILABLE_AT).drop_duplicates(subset="fund_code", keep="last")
    q = (query or "").strip().lower()
    rows: list[dict] = []
    for row in df.to_dict("records"):
        code = str(row.get("fund_code") or "")
        name = str(row.get("name") or "")
        if q and q not in code.lower() and q not in name.lower():
            continue
        rows.append(
            {
                "fund_code": code,
                "name": name,
                "fund_type": row.get("fund_type") or None,
                "invest_type": row.get("invest_type") or None,
                "market": row.get("market") or None,
                "status": row.get("status") or None,
                "found_date": _iso_or_none(row.get("found_date")),
                "delist_date": _iso_or_none(row.get("delist_date")),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _iso_or_none(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


# =============================================================================
# Markdown rendering (dependency-free) for /api/decisions & /api/traceability
# =============================================================================
def _md_inline(text: str) -> str:
    """Escape HTML then apply inline `code` and **bold** spans."""
    text = _html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def _render_markdown(md: str) -> str:
    """A compact, dependency-free Markdown→HTML renderer.

    Handles headings, fenced code blocks, GitHub-style tables, horizontal rules, list
    items and paragraphs — enough to render DECISIONS.md / TRACEABILITY.md legibly. Not a
    full CommonMark implementation (documented D-API-4).
    """
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    in_code = False
    while i < n:
        line = lines[i]
        if line.strip().startswith("```"):
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(_html.escape(line, quote=False))
            i += 1
            continue

        # table: a row with '|' followed by a separator row of dashes
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            header = _split_row(line)
            i += 2  # skip header + separator
            body: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip():
                body.append(_split_row(lines[i]))
                i += 1
            out.append("<table><thead><tr>")
            out.extend(f"<th>{_md_inline(c)}</th>" for c in header)
            out.append("</tr></thead><tbody>")
            for r in body:
                out.append("<tr>")
                out.extend(f"<td>{_md_inline(c)}</td>" for c in r)
                out.append("</tr>")
            out.append("</tbody></table>")
            continue

        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_md_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if re.match(r"^([-*_])\1{2,}$", stripped):
            out.append("<hr/>")
            i += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            out.append(f"<li>{_md_inline(stripped[2:])}</li>")
            i += 1
            continue
        out.append(f"<p>{_md_inline(stripped)}</p>")
        i += 1
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into trimmed cells."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _serve_markdown(path: Path, title: str) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{title} not found at {path}.")
    md = path.read_text(encoding="utf-8")
    return {"title": title, "path": str(path), "markdown": md, "html": _render_markdown(md)}


# =============================================================================
# App factory
# =============================================================================
def _layer_cache_signature(store: "PointInTimeStore") -> tuple:
    """A cheap fingerprint of the store state the layer report depends on.

    Row counts of the four tables the report reads (NAV, holdings, market, industry). An ingest
    that appends PIT-tagged rows bumps a count, so the cache self-invalidates without every ingest
    endpoint having to remember to clear it. Config writes null the cache directly (D-API-1).
    """
    tables = ("fund_nav", "fund_portfolio", "index_daily", "sw_daily")
    counts = []
    for t in tables:
        try:
            counts.append(store.row_count(t))
        except Exception:
            counts.append(-1)
    return tuple(counts)


def _layer_fund_codes(store: "PointInTimeStore", cap: int = 4) -> list[str]:
    """Funds usable for the idiosyncratic layer: they need BOTH a NAV series and disclosed holdings.

    Capped: the idio + combined-interval passes replay the full forecaster per sampled date, so the
    report is scored on a handful of representative funds rather than the whole universe.
    """
    from datetime import datetime as _dt

    try:
        pf = store.get_asof("fund_portfolio", _dt(2200, 1, 1))
        nav = store.get_asof("fund_nav", _dt(2200, 1, 1))
    except Exception:
        return []
    if pf is None or nav is None or pf.empty or nav.empty:
        return []
    # `fund_code` (six digits) is the join key the NAV tape and exposure estimator use;
    # `ts_code` carries the ".OF" market suffix and would match nothing downstream.
    have_nav = set(nav["fund_code"].astype(str))
    codes = [c for c in sorted(set(pf["fund_code"].astype(str))) if c in have_nav]
    return codes[:cap]


def create_app() -> FastAPI:
    """Build and return the control-plane :class:`FastAPI` app.

    A single shared :class:`~qmt.pit.PointInTimeStore` (over the configured
    ``data_cache/``) is kept on ``app.state.store``; every ingest WRITES into it and every
    run READS from it, so a config-refine → re-analyze loop reuses already-primed data.
    """
    app = FastAPI(
        title="QMT control plane",
        version=getattr(qmt, "__version__", "0.1.0"),
        description="Control, debug and refine the QMT off-exchange equity-fund pipeline.",
    )

    # CORS — open for localhost dashboards (no credentials needed).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = get_settings()
    app.state.settings = settings
    app.state.store = PointInTimeStore(settings.ensure_data_cache())

    # -----------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "qmt-controlplane",
            "version": app.version,
            "data_cache_dir": str(settings.data_cache_dir),
            "config_loaded": True,
            "editable_config_groups": list(EDITABLE_GROUPS),
        }

    # -----------------------------------------------------------------
    @app.get("/api/funds")
    def funds(
        request: Request,
        query: str = Query("", description="Substring matched against fund_code / name."),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict:
        store: PointInTimeStore = request.app.state.store
        results = _search_universe(store, date.today(), query, limit)
        return {"query": query, "limit": limit, "count": len(results), "funds": results}

    # -----------------------------------------------------------------
    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest, request: Request) -> dict:
        """Ingest then run_analysis; return the FULL per-stage debug trace."""
        _reject_if_tuning()
        store: PointInTimeStore = request.app.state.store
        ingest_summary: Optional[dict] = None
        if req.ingest:
            report = ingest_fund(req.fund_code, req.as_of, store)
            ingest_summary = {
                "in_universe": report.in_universe,
                "nav_rows": report.nav_rows,
                "holdings_reports": report.holdings_reports,
                "index_series": report.index_series,
                "industry_map_size": report.industry_map_size,
                "tables_written": report.tables_written,
                "gaps": report.gaps,
                "ok": report.ok,
            }

        try:
            result: AnalysisResult = run_analysis(req.fund_code, req.as_of, store)
        except ComplianceViolation as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "refused": True,
                    "gate": exc.result.gate,
                    "reason": exc.result.reason,
                    "ingest": ingest_summary,
                },
            ) from exc

        return {
            "mode": "analysis",
            "fund_code": req.fund_code,
            "as_of": req.as_of.isoformat(),
            "ingest": ingest_summary,
            # Doc §4 lst:scope / §3.5-2/3: every 场外 fund is classified (HTTP 200). A non-FULL
            # (LIMITED/DEFERRED) type is DEGRADED — no strong judgment — not a 422 refusal;
            # ``support_tier`` in the scope payload tells FULL vs LIMITED vs DEFERRED. advice_mode
            # still 422s on non-FULL. ``degraded`` mirrors the old ``soft_declined`` flag.
            "degraded": (not result.scope.in_scope),
            "support_tier": result.scope.support_tier.value,
            "scope": _dump(result.scope),
            "visibility": _dump_visibility(result.visibility),
            "exposure": _dump_exposure(result.exposure),
            "market_state": _dump(result.market_state),
            "forecasts": _dump_forecasts(result.forecasts),
            "uncertainty": {
                "confidence": result.confidence,
                "calibrated": result.calibrated,
                "note": (
                    "historically-calibrated reliability (calibration table applied)."
                    if result.calibrated
                    else "uncalibrated (no calibration table) — raw reliability score; run `qmt calibrate`."
                ),
            },
            "product_view": _dump(result.product_view),
            "explanation": _dump(result.explanation),
            "compliance_gates": _analysis_gates(result),
        }

    # -----------------------------------------------------------------
    @app.post("/api/advice")
    def advice(req: AdviceRequest, request: Request) -> dict:
        """Ingest then run_advice (the SOLE portfolio-reading path); return advice + look-through."""
        _reject_if_tuning()
        store: PointInTimeStore = request.app.state.store
        ingest_summary: Optional[dict] = None
        if req.ingest:
            report = ingest_fund(req.fund_code, req.as_of, store)
            ingest_summary = {
                "nav_rows": report.nav_rows,
                "holdings_reports": report.holdings_reports,
                "gaps": report.gaps,
                "ok": report.ok,
            }
        try:
            result: AdviceResult = run_advice(
                req.fund_code,
                req.as_of,
                req.portfolio,
                req.suitability,
                req.fees,
                position=req.position,
                store=store,
            )
        except ComplianceViolation as exc:
            raise HTTPException(
                status_code=422,
                detail={"refused": True, "gate": exc.result.gate, "reason": exc.result.reason},
            ) from exc
        except ValueError as exc:  # fail-closed (e.g. empty portfolio)
            raise HTTPException(status_code=422, detail={"refused": True, "reason": str(exc)}) from exc

        return {
            "mode": "advice",
            "fund_code": req.fund_code,
            "as_of": req.as_of.isoformat(),
            "ingest": ingest_summary,
            "product_view": _dump(result.product_view),
            "action": {
                **_dump(result.action),
                "action_label": result.action.action.value,
            },
            "action_gates": list(result.action.gated_by),
            "lookthrough": _dump_lookthrough(result.lookthrough),
            "risk_budget": _dump(result.risk_budget),
            "explanation": _dump(result.explanation),
            "compliance_gates": _advice_gates(result, req.suitability, req.fees, req.as_of),
        }

    # -----------------------------------------------------------------
    @app.post("/api/portfolio/suggestions")
    def portfolio_suggestions(req: PortfolioSuggestionsRequest, request: Request) -> dict:
        """Advice across the held portfolio + available cash — the 'generate suggestions' click.

        Reads the user's positions (advice_mode), penetrates them to real industry/style
        concentration + risk budget, and returns a per-fund adjustment suggestion. FULL
        (equity-family) funds get a gated action; non-FULL types are surfaced as
        '该类型暂不支持个性化建议'. A user may hold NO funds (only cash): the response then
        carries the market environment + cash status (the system does not recommend specific
        funds to buy — out of scope). Only a totally-empty input (no cash, no holdings) → 422."""
        from qmt.pipeline.analysis import _resolve_profile
        from qmt.pipeline.fund_scope import classify_scope
        from qmt.pipeline.market_state import InsufficientMarketHistory, assess_market_state
        from qmt.types import SupportTier

        _reject_if_tuning()
        store: PointInTimeStore = request.app.state.store
        cash = max(req.available_cash, 0.0)
        if not req.holdings and cash <= 0:
            raise HTTPException(
                status_code=422,
                detail={"refused": True, "reason": "请输入可用资金，或至少录入一只持仓。"},
            )
        # Daily tool: default to today so suggestions always reflect the newest observable data
        # (point-in-time correctness still hides anything after the decision moment).
        as_of = req.as_of or date.today()
        # Holding period per fund, derived from the buy date (server-side, PIT-correct at as_of).
        hold_days_of = {h.fund_code: _effective_hold_days(h, as_of) for h in req.holdings}
        fund_total = sum(h.market_value for h in req.holdings)
        total_investable = cash + fund_total
        # Real allocation shares (incl. cash) for display; concentration uses the equity sleeve.
        book_weight = {h.fund_code: h.market_value / total_investable for h in req.holdings} if total_investable > 0 else {}
        cash_weight = (cash / total_investable) if total_investable > 0 else 1.0
        suitability = SuitabilityProfile(
            risk_tolerance=req.risk_tolerance, investment_horizon_days=req.investment_horizon_days
        )

        ingest_summary: Optional[dict] = None
        if req.ingest:
            gaps: list = []
            for h in req.holdings:  # each fund ingest also pulls the benchmark index (market state)
                rep = ingest_fund(h.fund_code, as_of, store)
                gaps.extend(rep.gaps or [])
            ingest_summary = {"funds": len(req.holdings), "gaps": gaps}

        # Market environment (broad, fund-independent) — best-effort from observable index bars.
        market_state = None
        try:
            market_state = assess_market_state(as_of, store)
        except (InsufficientMarketHistory, ValueError):
            market_state = None

        # Classify each holding + pre-check exposure. The industry/style look-through only
        # covers FULL funds whose exposure can actually be estimated. A FULL fund with NO
        # estimable exposure (e.g. an index / ETF-联接 whose disclosed holdings don't map to
        # stock-level industries) is reported honestly and EXCLUDED from the look-through
        # portfolio — so one such fund can no longer fail-close the whole book's suggestions.
        from qmt.pipeline.portfolio_lookthrough import estimate_exposure as _estimate_exposure

        tier: dict[str, Optional[SupportTier]] = {}
        has_exposure: dict[str, bool] = {}
        for h in req.holdings:
            prof = _resolve_profile(h.fund_code, as_of, store)
            t = classify_scope(prof).support_tier if prof is not None else None
            tier[h.fund_code] = t
            if t is SupportTier.FULL:
                try:
                    has_exposure[h.fund_code] = bool(_estimate_exposure(h.fund_code, as_of, store=store))
                except Exception:
                    has_exposure[h.fund_code] = False
        equity = [h for h in req.holdings
                  if tier[h.fund_code] is SupportTier.FULL and has_exposure.get(h.fund_code)]
        equity_mv = sum(h.market_value for h in equity)
        eq_weight = {h.fund_code: h.market_value / equity_mv for h in equity} if equity_mv > 0 else {}
        equity_portfolio = UserPortfolio(holdings=[
            PortfolioHolding(fund_code=h.fund_code, weight=eq_weight[h.fund_code], hold_days=hold_days_of[h.fund_code])
            for h in equity
        ])

        # Portfolio look-through + risk budget over the equity sleeve (degrade, don't 500, on
        # thin data — the per-fund suggestions still run).
        lookthrough = None
        risk_budget = None
        summary_note = ""
        if equity:
            try:
                lookthrough = look_through(equity_portfolio, as_of, store)
                risk_budget = assess_risk_budget(lookthrough, suitability)
            except ValueError as exc:
                summary_note = f"组合穿透暂不可用（数据不足）：{exc}"
        elif not req.holdings:
            summary_note = (
                f"当前为空仓（可用资金 ¥{cash:,.0f}）。系统会在全部可分析基金中筛选合适标的给出现金分配建议（见下方）；"
                "也可结合『市场环境』判断是否分批建仓，并按自身适当性谨慎决定。"
            )
        else:
            summary_note = "当前无权益类（FULL）持仓，暂无行业/风格集中度分析。"

        # Broad market environment note (顺风 / 逆风 / 乱流) — describes the market, not a fund.
        environment_note = ""
        if market_state is not None:
            calm = market_state.volatility.value == "low" or market_state.volatility.value == "normal"
            stressed = market_state.stress.value != "normal" or market_state.event.value == "shock"
            if stressed or market_state.volatility.value == "high":
                environment_note = "市场偏『乱流 / 逆风』（波动高或有压力/事件），若建仓建议谨慎、分批、控制节奏。"
            elif market_state.direction.value == "up" and calm:
                environment_note = "市场相对『顺风』（方向偏上、波动可控），但仍需结合估值与自身适当性，不宜追高。"
            else:
                environment_note = "市场方向不明朗，宜观望或以定投方式平滑成本。"

        # Per-fund adjustment suggestion (and capture per-fund signals for cash_allocation).
        # Every FULL fund gets a forecast-based action via run_analysis + decide_action; the
        # portfolio concentration (shared `lookthrough`/`risk_budget`, over the exposure-having
        # sleeve) is passed as context. A fund whose OWN exposure can't be estimated is still
        # advised (from its forecast) with a note — it just isn't folded into concentration.
        from qmt.pipeline.action_strategy import decide_action
        from qmt.pipeline.analysis import run_analysis as _run_analysis

        delay_modeled = confirm_lag_modeled(as_of)
        suggestions: list[dict] = []
        sig_forecasts: dict = {}
        sig_conf: dict = {}
        for h in req.holdings:
            fc = h.fund_code
            entry: dict = {
                "fund_code": fc, "weight": book_weight[fc],
                "hold_days": hold_days_of[fc], "support_tier": (tier[fc].value if tier[fc] else "unknown"),
            }
            if tier[fc] is not SupportTier.FULL:
                entry.update({
                    "supported": False,
                    "reason": "该类型暂不支持个性化加减仓建议（仅权益家族可给强建议）。"
                    if tier[fc] is not None else "无法确认基金类型（数据不足）。",
                })
                suggestions.append(entry)
                continue
            try:
                ares = _run_analysis(fc, as_of, store)
                if not ares.forecasts:
                    entry.update({"supported": False, "reason": "预测数据不足（净值历史过短），暂无法给出建议。"})
                    suggestions.append(entry)
                    continue
                action = decide_action(
                    ares.forecasts, ares.product_view.confidence, ares.market_state,
                    default_fee_schedule(fc), book_weight[fc], lookthrough, risk_budget,
                    suitability, hold_days_of[fc], execution_delay_modeled=delay_modeled,
                )
                entry.update({
                    "supported": True,
                    "action": action.action.value,
                    "rationale": action.rationale,
                    "gated_by": list(action.gated_by),
                    "direction": ares.product_view.direction.value,
                    "confidence": ares.product_view.confidence,
                })
                if not has_exposure.get(fc):
                    entry["note"] = "该基金暴露暂不可估计（如指数 / ETF 联接），未计入组合集中度。"
                sig_forecasts[fc] = ares.forecasts
                sig_conf[fc] = ares.product_view.confidence
            except ComplianceViolation as exc:
                entry.update({"supported": False, "gate": exc.result.gate, "reason": exc.result.reason})
            except ValueError as exc:  # fail-closed (e.g. thin data / unresolvable)
                entry.update({"supported": False, "reason": str(exc)})
            suggestions.append(entry)

        # --- cash_allocation: distribute usable cash across candidate FULL funds -----------
        # Candidates = held FULL funds (signals reused above) + FULL funds screened from the
        # primed universe (fund_basic), analyzed here from observable data. Non-FULL never enter
        # and the screen is capped for latency (§4 sec:mod-cashalloc, D-CASH-1).
        cash_plan_dump = None
        if cash > 0:
            from qmt.pipeline.analysis import run_analysis as _run_analysis
            from qmt.pipeline.cash_allocation import allocate_cash
            from qmt.types import UsableCashInput
            extra = _full_universe_candidates(
                store, as_of, exclude=set(sig_forecasts), cap=CONFIG.cash_allocation.max_candidates
            )
            for uc in extra:
                try:
                    ares = _run_analysis(uc, as_of, store)  # reads observable data; no forced ingest
                    if not ares.forecasts:
                        continue
                    sig_forecasts[uc] = ares.forecasts
                    sig_conf[uc] = ares.product_view.confidence
                except Exception:  # best-effort screen: a bad candidate is skipped, never fatal
                    continue
            candidates = list(sig_forecasts.keys())
            if candidates:
                contrib = (lookthrough.fund_risk_contrib if lookthrough is not None else {}) or {}
                mx = max(contrib.values()) if contrib else 0.0
                overlap = {c: (contrib.get(c, 0.0) / mx if mx > 0 else 0.0) for c in candidates}
                fees = {c: default_fee_schedule(c).purchase_fee_rate for c in candidates}
                usable = UsableCashInput(
                    amount=cash, must_fully_invest=req.must_fully_invest, min_cash_reserve=req.min_cash_reserve
                )
                plan = allocate_cash(usable, candidates, sig_forecasts, sig_conf, risk_budget,
                                     overlap=overlap, fees=fees)
                cash_plan_dump = _dump(plan)

        return {
            "as_of": as_of.isoformat(),
            "n_funds": len(req.holdings),
            "n_equity": len(equity),
            "available_cash": cash,
            "total_investable": total_investable,
            "cash_weight": cash_weight,
            "ingest": ingest_summary,
            "summary_note": summary_note,
            "market_state": _dump(market_state),
            "environment_note": environment_note,
            "lookthrough": _dump_lookthrough(lookthrough),
            "risk_budget": _dump(risk_budget),
            "cash_allocation": cash_plan_dump,
            "suggestions": suggestions,
        }

    # -----------------------------------------------------------------
    @app.get("/api/config")
    def get_config() -> dict:
        _reject_if_tuning()  # never show the transient candidate values a live search applies
        return {"config": _effective_config(), "editable": list_editable_fields()}

    @app.put("/api/config")
    def put_config(req: ConfigUpdateRequest) -> dict:
        _reject_if_tuning()  # a live search owns CONFIG; a concurrent edit would race its restore
        try:
            applied = apply_config_override(req.path, req.value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        app.state.layer_report = None      # the cached layer report was computed under the old config
        return {"applied": applied, "config": _effective_config(), "editable": list_editable_fields()}

    # -----------------------------------------------------------------
    @app.post("/api/backtest")
    def backtest(req: BacktestRequest, request: Request) -> dict:
        _reject_if_tuning()  # backtest reads the swept market_state config; don't run mid-search
        store: PointInTimeStore = request.app.state.store
        ingest_summary: Optional[dict] = None
        if req.ingest:
            report = ingest_fund(req.fund_code, req.end, store, with_holdings=False)
            ingest_summary = {"nav_rows": report.nav_rows, "gaps": report.gaps, "ok": report.ok}
        try:
            result: BacktestResult = run_backtest(
                req.fund_code,
                req.start,
                req.end,
                strategy=req.strategy,
                # Realistic, non-zero fees (§7 rules 6/7 — net-of-cost is never zero).
                fees=default_fee_schedule(req.fund_code),
                store=store,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        decisions = [
            {
                "as_of": d.as_of.isoformat(),
                "fund_code": d.fund_code,
                "fund_type": d.fund_type,
                "signal": d.signal,
                "executable_from": d.executable_from.isoformat() if d.executable_from else None,
                "regime": d.regime,
                "alive": d.alive,
            }
            for d in result.decision_log
        ]
        baselines = {
            name: {
                "net_return": b.net_return,
                "max_drawdown": b.max_drawdown,
                "equity_curve": _df_records(b.equity_curve),
            }
            for name, b in result.baselines.items()
        }
        return {
            "mode": "backtest",
            "ingest": ingest_summary,
            "summary": result.summary(),
            "strategy": result.strategy_name,
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "net_return": result.net_return,
            "max_drawdown": result.max_drawdown,
            "total_cost": result.total_cost,
            "n_decisions": result.n_decisions,
            "equity_curve": _df_records(result.equity_curve),
            "vs_baseline": result.vs_baseline,
            # `ma` (etc.) may BE the strategy: its delta is identically 0 and is not evidence.
            "self_baseline": result.self_baseline,
            "baselines": baselines,
            "net_return_by_regime": result.net_return_by_regime,
            "net_return_by_fund_type": result.net_return_by_fund_type,
            "decision_log": decisions,
        }

    # -----------------------------------------------------------------
    @app.get("/api/decisions")
    def decisions_doc() -> dict:
        return _serve_markdown(_DECISIONS_MD, "基金分析")

    @app.get("/api/traceability")
    def traceability_doc() -> dict:
        return _serve_markdown(_TRACEABILITY_MD, "数据与本地存储")

    # -----------------------------------------------------------------
    @app.get("/api/models")
    def models() -> dict:
        """Catalog of the pipeline's analytical models (kind / method / status / live params)."""
        from qmt.controlplane.models_registry import models_catalog

        return models_catalog()

    # -----------------------------------------------------------------
    # Training / tuning — walk-forward parameter search over the tunable models.
    # "Training" here is honestly a time-ordered hyper-parameter search (§5): a candidate
    # setting is scored point-in-time, net of cost, on a training window and confirmed
    # out-of-sample on a strictly-later validation window. See qmt.tuning / DECISIONS D-TUNE-*.
    # -----------------------------------------------------------------
    @app.get("/api/tuning/models")
    def tuning_models(request: Request) -> dict:
        """The auto-selected whole-market training the page shows (no method/fund picker, D-TUNE-7).

        Includes the benchmark's full observable month range so the page can default the training
        window to ALL available history.
        """
        from qmt.tuning import whole_market_catalog

        return whole_market_catalog(request.app.state.store)

    @app.post("/api/tuning/search")
    def tuning_search(req: TuningSearchRequest, request: Request) -> StreamingResponse:
        """Run the auto-selected whole-market training, streaming progress as Server-Sent Events.

        The method (market-timing) and the target (the broad market index, not a fund) are fixed
        by the system (D-TUNE-7). Emits ``start`` → ``baseline`` → one ``trial`` per candidate
        (training metrics, live) → ``validation`` for the top-k (out-of-sample) → ``done``. A busy
        search yields a single ``error`` event (409-style) instead of interleaving CONFIG mutations.
        """
        store: PointInTimeStore = request.app.state.store
        from qmt.tuning import (
            WHOLE_MARKET_BENCHMARK,
            PSEUDO_MARKET_CODE,
            auto_selected_spec,
            build_market_store,
            persist_run,
            plan_walk_forward,
            stream_walk_forward_search,
        )

        spec = auto_selected_spec()
        try:  # the system divides the chosen month range into walk-forward folds + a holdout
            plan = plan_walk_forward(_parse_month(req.start_month), _parse_month(req.end_month))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_month_range_error_zh(exc)) from exc

        def _sse(payload: dict) -> str:
            return f"data: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

        # Run the search in a worker THREAD that feeds a queue, and stream from the queue. The
        # thread owns the CONFIG mutation + lock and releases them in its own finally, which runs
        # to completion deterministically even if the CLIENT disconnects mid-stream (a generator
        # tied to the response lifetime would leak the lock on disconnect / page reload, bricking
        # the feature until restart — D-TUNE-6).
        q: "_queue.Queue" = _queue.Queue(maxsize=256)
        _SENTINEL = object()

        def _worker() -> None:
            try:
                if not _TUNING_LOCK.acquire(blocking=False):
                    q.put({"event": "error", "detail": "另一个训练任务正在进行，请稍候再试。"})
                    return
                try:
                    _TUNING_CANCEL.clear()
                    _TUNING_ACTIVE.set()  # read-isolation window opens (config-readers now 409)
                    # ensure the broad market index (the whole-market target, not a fund) is primed
                    as_of = plan.holdout_end
                    primed = store.get_asof(TABLE_INDEX_DAILY, as_of, {"ts_code": WHOLE_MARKET_BENCHMARK})
                    n_bars = 0 if primed is None else len(primed)
                    gaps: list = []
                    if req.ingest and n_bars == 0:  # only hit the network when the index is missing
                        try:
                            from qmt.data import sources as _sources

                            df = _sources.index_daily(
                                WHOLE_MARKET_BENCHMARK, store=store, end_date=as_of.strftime("%Y%m%d")
                            )
                            n_bars = int(len(df))
                        except Exception as exc:  # offline / no token → use whatever is already primed
                            gaps = [f"指数入库跳过：{exc}"]
                    q.put({"event": "ingest", "nav_rows": n_bars, "gaps": gaps, "ok": n_bars > 0})
                    # reshape the broad index into a fund-shaped store so the engine runs on it
                    market_store = build_market_store(store, WHOLE_MARKET_BENCHMARK)
                    sink: list = []
                    for ev in stream_walk_forward_search(
                        spec, PSEUDO_MARKET_CODE, plan, market_store,
                        metric=spec.metrics[0], benchmark=WHOLE_MARKET_BENCHMARK,
                        cancel=_TUNING_CANCEL.is_set, _result_sink=sink,
                    ):
                        # persist BEFORE the client sees 'done', so the run is already on disk
                        # when the page reloads its history (D-TUNE-10). Never fail the run.
                        if ev.get("event") == "done" and sink:
                            try:
                                persist_run(sink[0])
                            except Exception as exc:
                                logger.warning("could not persist tuning run: %s", exc)
                        q.put(ev)
                finally:
                    _TUNING_ACTIVE.clear()  # CONFIG restored by the search's own finally; window closes
                    _TUNING_LOCK.release()
            except Exception as exc:  # surface any failure as a stream event, never a 500 mid-stream
                logger.exception("tuning search failed")
                q.put({"event": "error", "detail": str(exc)})
            finally:
                q.put(_SENTINEL)

        threading.Thread(target=_worker, name="qmt-tuning-search", daemon=True).start()

        def _gen():
            while True:
                ev = q.get()
                if ev is _SENTINEL:
                    return
                yield _sse(ev)

        return StreamingResponse(_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/tuning/split")
    def tuning_split(
        start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
        end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    ) -> dict:
        """Preview the walk-forward division of a month range: folds + untouched holdout (D-TUNE-9)."""
        from qmt.tuning import HOLDOUT_FRACTION, plan_walk_forward

        try:
            plan = plan_walk_forward(_parse_month(start_month), _parse_month(end_month))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_month_range_error_zh(exc)) from exc
        d = plan.as_dict()
        d["holdout_fraction"] = HOLDOUT_FRACTION
        d["n_folds"] = len(plan.folds)
        return d

    @app.get("/api/tuning/layers")
    def tuning_layers(request: Request, refresh: bool = Query(False)) -> dict:
        """Score the three layers of the hierarchical model independently (源文档 §3.3 / techdesign §5.3).

        This answers "which layer is the problem", and it answers it the only way CLAUDE.md allows:
        each layer is compared against **its own naive baseline** and every skill claim is routed
        through the overlap-corrected, fail-closed :func:`qmt.stats.assess_significance`. A layer
        that cannot beat its own naive baseline has not proven incremental value, whatever its
        absolute IC looks like.

        Cached because a full replay is ~30 s; ``refresh=true`` recomputes. Rejected while a tuning
        run is in flight, since that run mutates the global CONFIG this report reads.
        """
        _reject_if_tuning()
        store: PointInTimeStore = request.app.state.store
        # The report is a function of (the data it reads, the config it grades under). Both can move
        # between calls — ingest appends NAV/holdings rows on other endpoints; PUT/apply changes the
        # config. Rather than hunt every mutation site, cache under a cheap signature and recompute
        # only when it changes: config writes already null the cache; row counts catch every ingest.
        sig = _layer_cache_signature(store)
        cached = getattr(app.state, "layer_report", None)
        if cached is not None and not refresh and getattr(app.state, "layer_report_sig", None) == sig:
            return cached
        from qmt.pipeline.layer_diagnostics import evaluate_all_layers

        settings = get_settings()
        im_path = settings.data_cache_dir / "industry_map.json"
        industry_map = _json.loads(im_path.read_text(encoding="utf-8")) if im_path.exists() else None
        funds = tuple(_layer_fund_codes(store))
        report = evaluate_all_layers(
            store, horizons=(3, 20, 90), fund_codes=funds, industry_map=industry_map, idio_horizon=20
        ).as_dict()
        app.state.layer_report = report
        app.state.layer_report_sig = sig
        return report

    @app.get("/api/tuning/curves/{layer}")
    def tuning_curves(
        layer: str, request: Request,
        horizon: int = Query(20, ge=1, le=250), refresh: bool = Query(False),
    ) -> StreamingResponse:
        """Render the layer's **预测 vs 真实** curves to a PNG (点位口径 / PIT).

        ``layer`` is ``market`` (one panel per horizon: the index forward-return distribution vs its
        realized return) or ``industry`` (one panel per auto-selected best/worst industry at
        ``horizon``: predicted deviation-from-market vs realized). Cheap relative to the full
        three-layer diagnosis (no per-fund forecaster replay), but still cached per store signature.
        Rejected mid-search, since the swept CONFIG would change the very curves being drawn.
        """
        import io

        _reject_if_tuning()
        if layer not in ("market", "industry"):
            raise HTTPException(status_code=404, detail="layer 只能是 market 或 industry。")
        store: PointInTimeStore = request.app.state.store
        key = (layer, int(horizon), _layer_cache_signature(store))
        cache = getattr(app.state, "curve_cache", None)
        if cache is None:
            cache = app.state.curve_cache = {}
        if not refresh and key in cache:
            return StreamingResponse(io.BytesIO(cache[key]), media_type="image/png")

        from qmt.pipeline.layer_diagnostics import build_grid, load_industry_tapes, load_market_tape
        from qmt.viz import plot_industry_curves, plot_market_curves

        # Serialize renders. The figures are built with matplotlib's OO API (thread-safe), but the
        # font manager and rcParams are process-global; serializing also means a second concurrent
        # request for the same key waits and then reads the freshly-cached bytes instead of redoing
        # the grid build. Endpoint runs in FastAPI's threadpool, so a plain Lock is correct here.
        with _CURVE_RENDER_LOCK:
            if not refresh and key in cache:            # filled while we waited for the lock
                return StreamingResponse(io.BytesIO(cache[key]), media_type="image/png")
            grid = build_grid(load_market_tape(store), load_industry_tapes(store))
            if grid.n == 0:
                raise HTTPException(status_code=422, detail="无可用指数数据（先入库大盘/行业指数）。")
            fig = (plot_market_curves(grid) if layer == "market"
                   else plot_industry_curves(grid, horizon=horizon))
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            cache[key] = buf.getvalue()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    @app.get("/api/tuning/benchmark")
    def tuning_benchmark(request: Request, horizon: int = Query(20, ge=1, le=250)) -> dict:
        """OOS model-ladder benchmark + permutation feature importance for the market layer (D-BENCH-1).

        Purged walk-forward, overlap-corrected. Shows, honestly, that features help VOLATILITY (OOS
        corr) but not DIRECTION (OOS IC, not significant) — so a fancier model is not adopted for the
        mean. Cached per store signature; ``_reject_if_tuning`` guarded.
        """
        _reject_if_tuning()
        store: PointInTimeStore = request.app.state.store
        key = ("benchmark", int(horizon), _layer_cache_signature(store))
        cache = getattr(app.state, "curve_cache", None)
        if cache is None:
            cache = app.state.curve_cache = {}
        if key in cache:
            return cache[key]

        import numpy as _np

        from qmt.pipeline.features import feature_frame, forward_realized_vol
        from qmt.pipeline.layer_diagnostics import forward_returns, load_market_tape
        from qmt.pipeline.model_bench import _hgb, permutation_importance, run_benchmark

        # Only the broad-market series is needed (not the full industry grid), so this works with
        # just the index ingested.
        P = np.asarray(load_market_tape(store).closes, dtype=float)
        n = P.size
        if n < 400:
            raise HTTPException(status_code=422, detail="市场历史不足（先入库大盘指数）。")
        names, M = feature_frame(P)
        h = int(horizon)
        fm = forward_returns(P, h)
        fv = forward_realized_vol(P, h)
        rows = [(M[i - 1], P[i + 1 + h] / P[i + 1] - 1.0, fv[i + 1])
                for i in range(1, n) if i + 1 + h < n]
        X = _np.array([r[0] for r in rows]); yret = _np.array([r[1] for r in rows])
        yvol = _np.array([r[2] for r in rows])
        with _CURVE_RENDER_LOCK:                       # serialize the heavy CV work
            if key in cache:
                return cache[key]
            dirres, _ = run_benchmark(X, yret, horizon=h, target=f"dir_h{h}", mom_col=names.index("mom60"))
            volres, _ = run_benchmark(X, yvol, horizon=h, target=f"vol_h{h}", mom_col=names.index("vol20"))
            imp_dir = permutation_importance(X, yret, names, _hgb, horizon=h, n_repeats=5, metric="ic")
            imp_vol = permutation_importance(X, yvol, names, _hgb, horizon=h, n_repeats=5, metric="ic")
            out = {
                "horizon": h,
                "direction": [r.as_dict() for r in dirres],
                "volatility": [{"model": r.model, "corr": r.extra.get("corr")} for r in volres],
                "importance_direction": imp_dir,
                "importance_volatility": imp_vol,
                "verdict": ("方向性：无模型通过重叠修正显著性——不采纳条件均值（保持中位近水平）。"
                            "波动率：特征显著提升样本外相关——已采纳条件波动率区间（D-CVOL-1）。"),
            }
            cache[key] = out
        return out

    @app.get("/api/tuning/tsbench")
    def tuning_tsbench() -> dict:
        """Serve the offline broad TS-model bake-off (D-BENCH-6), read from ``artifacts/``.

        This is the heavy 12-model ladder (classical + ML + seeded DL over 24y of CSI300), produced
        offline by ``scratchpad/run_ts_bench.py`` — not a live render, because a full refit takes
        ~14 min. The panel reads the last-computed artifact so the training page can display the
        honest finding: **nothing beats the random-walk / prevailing-mean benchmark at any horizon,
        and complexity monotonically increases OOS error.** A per-horizon ``winners`` list is derived
        from the same two-gate ``beats_rw`` rule the benchmark applies (DM t>2 AND directional
        significance), so the page never has to re-implement the verdict.
        """
        path = _REPO_ROOT / "artifacts" / "ts_model_bench.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="尚未生成（运行 scratchpad/run_ts_bench.py）。")
        import json as _json

        data = _json.loads(path.read_text(encoding="utf-8"))
        winners = {H: [r["name"] for r in rows if r.get("beats_rw")]
                   for H, rows in data.get("horizons", {}).items()}
        data["winners"] = winners
        data["verdict"] = ("在 h=3/20/90 上无任何模型跑赢随机游走/历史均值基准——随机游走即最优模型；"
                           "复杂度单调增大样本外误差（MLP 误差 2–8×，LSTM 1.3×，GBM 显著更差）。"
                           "h=90 上看似较高的方向 IC（如 ridge +0.32）是 ~39 个窗口的小样本假象，"
                           "其平方误差反而更大。未采纳任何模型（保持 D-BENCH-2 的近水平中位）。")
        return data

    @app.get("/api/tuning/runs")
    def tuning_runs(limit: int = Query(50, ge=1, le=200)) -> dict:
        """Newest-first summaries of the persisted training runs (D-TUNE-10)."""
        from qmt.tuning import list_runs

        runs = list_runs(limit)
        return {"count": len(runs), "runs": runs}

    @app.get("/api/tuning/runs/{run_id}")
    def tuning_run(run_id: str) -> dict:
        """One persisted training run, in full."""
        from qmt.tuning import load_run

        try:
            return load_run(run_id)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=404, detail=f"没有找到训练记录 {run_id}") from exc

    @app.post("/api/tuning/cancel")
    def tuning_cancel() -> dict:
        """Cooperatively cancel the in-flight parameter search (the "Stop" button)."""
        was_active = _TUNING_ACTIVE.is_set()
        _TUNING_CANCEL.set()
        return {"ok": True, "cancelling": was_active}

    @app.post("/api/tuning/apply")
    def tuning_apply(req: TuningApplyRequest) -> dict:
        """Write a search's winning settings onto the live CONFIG (takes effect on the next run).

        Gated by the promotion verdict: a candidate that did not stably beat the baseline is
        refused unless explicitly forced (D-TUNE-9).
        """
        # A live search owns CONFIG (it swaps candidate values in and restores from a snapshot);
        # a concurrent write here would be silently clobbered by that restore, and would corrupt
        # the candidate being scored. Same fail-closed 409 as PUT /api/config (D-TUNE-6).
        _reject_if_tuning()
        if not req.params:
            raise HTTPException(status_code=422, detail="没有要应用的参数。")
        if req.run_id and not req.force:
            from qmt.tuning import load_run

            try:
                run = load_run(req.run_id)
            except (FileNotFoundError, OSError) as exc:
                raise HTTPException(status_code=404, detail=f"没有找到训练记录 {req.run_id}") from exc
            promo = run.get("promotion", {})
            if not promo.get("promotable"):
                failed = [c["name"] for c in promo.get("criteria", []) if not c.get("passed")]
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "该候选未通过『稳定优于基线』验收，不能应用到模型："
                        + "；".join(failed)
                        + "。（如确需覆盖，请显式 force=true）"
                    ),
                )
        applied = []
        try:
            for path, value in req.params.items():
                applied.append(apply_config_override(path, value))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # The cached three-layer report was computed under the OLD config — serving it after an
        # apply would grade yesterday's model and label it with today's settings.
        app.state.layer_report = None
        return {"applied": applied, "config": _effective_config(), "editable": list_editable_fields()}

    # -----------------------------------------------------------------
    # Static dashboard (mounted LAST so /api/* routes take precedence). The dashboard
    # itself is added in a follow-up slice; the directory holds a placeholder for now.
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app


# uvicorn-runnable: ``uvicorn qmt.controlplane.api:app``
app = create_app()
