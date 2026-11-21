"""End-to-end factor-platform orchestration (ten layers, tech doc ch.2 / ch.8).

Wires the layers into one runnable, point-in-time-correct pass over a cross-section:

    safe_data -> base_factor_library -> (labels + validation) -> auto_search
    -> factor_screening (+ gates) -> factor_archive -> combination -> execution_sim

Honours the roadmap's **baseline-first** mandate (§5.6, §7.4): manual classic factors
are scored on the SAME out-of-sample split as the searched factors, and the result
reports honestly whether symbolic search beats them OOS. If it does not — the
expected outcome under the project's own falsifiable stance (§7.1/§7.4) — that is a
correct finding, not a bug: the platform's value is then governance/auditability.

A single full-history :class:`BaseFactorPanel` is built once at the last decision
date; every layer re-slices to ``<= as_of`` internally, so evaluating per decision
date stays leak-free (rolling windows are backward-looking).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore, TradingCalendar
from qmt.factor import baselines as _baselines
from qmt.factor import metrics as _metrics
from qmt.factor import panel as _panel
from qmt.factor import validation as _validation
from qmt.factor.archive import InMemoryFactorArchive
from qmt.factor.auto_search import SymbolicSearch
from qmt.factor.base_factors import BaseFactors
from qmt.factor.combination import FactorCombination
from qmt.factor.dsl import FormulaDSL
from qmt.factor.execution_sim import ExecSim
from qmt.factor.gates import GateConfig
from qmt.factor.safe_data import SafeData
from qmt.factor.screening import Screening
from qmt.factor.types import (
    CostTiers,
    FactorExpr,
    FactorRecord,
    FactorScore,
    FactorStatus,
    PortfolioState,
    SearchConfig,
    SearchLedger,
)

__all__ = ["FactorPipelineResult", "ScoredFactor", "run_factor_pipeline", "run_industry_demo"]


@dataclass
class ScoredFactor:
    name: str
    expr: FactorExpr
    score: FactorScore
    origin: str                              # "manual" | "searched"

    @property
    def gate_passed(self) -> bool:
        return bool(self.score.gate_report and self.score.gate_report.passed)


@dataclass
class FactorPipelineResult:
    """Structured, auditable outcome of one pipeline run."""

    as_of: date
    universe_size: int
    n_decision_dates: int
    horizon_days: int
    manual: list[ScoredFactor] = field(default_factory=list)
    searched: list[ScoredFactor] = field(default_factory=list)
    best_manual_rank_ic: float = 0.0          # raw OOS rank-IC (informative only)
    best_searched_rank_ic: float = 0.0
    best_manual_gated: float = 0.0            # rank-IC among gate-passing factors
    best_searched_gated: float = 0.0
    search_beats_baseline: bool = False       # gate-aware, OOS, honestly (§7.4)
    top_searched_text: str = ""               # for spotting level/artifact formulas
    n_trials: int = 0                         # search ledger size (overfit governance)
    n_gate_passed: int = 0
    portfolio_is_diagnostic: bool = False     # True => nothing passed gates; top-K used
    net_return_by_tier: dict[str, float] = field(default_factory=dict)
    archive_size: int = 0
    deflated_min_abs_t: float = 2.0           # searched-family bar (raised by trial count, §7.3)
    manual_min_abs_t: float = 2.0             # pre-registered-family bar (its own family size)
    locked_test_span: tuple | None = None     # held-out region, never scored (§7.3)
    diagnostics: list[dict] = field(default_factory=list)  # per-top-factor why-it-failed
    verdict: str = ""

    def summary(self) -> dict:
        return {
            "as_of": str(self.as_of),
            "universe_size": self.universe_size,
            "n_decision_dates": self.n_decision_dates,
            "horizon_days": self.horizon_days,
            "best_manual_rank_ic": round(self.best_manual_rank_ic, 4),
            "best_searched_rank_ic": round(self.best_searched_rank_ic, 4),
            "best_manual_gated": round(self.best_manual_gated, 4),
            "best_searched_gated": round(self.best_searched_gated, 4),
            "search_beats_baseline_oos_gated": self.search_beats_baseline,
            "top_searched_text": self.top_searched_text,
            "n_trials": self.n_trials,
            "n_gate_passed": self.n_gate_passed,
            "portfolio_is_diagnostic": self.portfolio_is_diagnostic,
            "net_return_by_tier": {k: round(v, 4) for k, v in self.net_return_by_tier.items()},
            "archive_size": self.archive_size,
            "deflated_min_abs_t": round(self.deflated_min_abs_t, 2),
            "manual_min_abs_t": round(self.manual_min_abs_t, 2),
            "locked_test_span": [str(d) for d in self.locked_test_span] if self.locked_test_span else None,
            "diagnostics": self.diagnostics,
            "verdict": self.verdict,
        }


def run_factor_pipeline(
    store: PointInTimeStore,
    table: str,
    *,
    value_col: str,
    code_col: str,
    date_col: str,
    horizon_days: int = 40,
    stride_days: int = 20,
    start_frac: float = 0.35,
    max_decision_dates: int | None = 120,
    industry_of: dict[str, str] | None = None,
    codes: list[str] | None = None,
    extra_factors: dict | None = None,
    market_px: pd.Series | None = None,
    min_date: date | None = None,
    locked_start: date | None = None,
    search_cfg: SearchConfig | None = None,
    costs: CostTiers | None = None,
    gate_config: GateConfig | None = None,
    calendar: TradingCalendar | None = None,
) -> FactorPipelineResult:
    """Run the full pipeline over a PIT cross-section and return an audit report."""
    cal = calendar or TradingCalendar()
    costs = costs or CostTiers()
    dsl = FormulaDSL(industry_of=industry_of)

    # --- safe_data: build & clean the point-in-time price panel --------------
    safe = SafeData(store, calendar=cal)
    tail_frame = _panel.usable_price_frame(
        store, table, value_col=value_col, code_col=code_col,
        date_col=date_col, visible_lag_days=1, calendar=cal, codes=codes,
    )
    if tail_frame.empty:
        raise ValueError(f"no usable data in table {table!r}")
    final_as_of = tail_frame.index.max()
    final_as_of = final_as_of.date() if hasattr(final_as_of, "date") else final_as_of
    sp = safe.price_panel(
        table, final_as_of, value_col=value_col, code_col=code_col, date_col=date_col,
        codes=codes,
    )
    # winsorize=False: NAV/close LEVELS must not be quantile-clipped (it flattens
    # legitimate new highs, and any full-history threshold would leak the future
    # into beta/vol/labels — skeptic-confirmed). clean() here only drops dead columns.
    sp = safe.clean(sp, winsorize=False)
    # optional history trim: dropping rows OLDER than min_date is causally safe (it
    # removes only the distant past; decision dates start after start_frac warmup
    # anyway) and keeps very wide universes (14k+ funds) within memory bounds.
    if min_date is not None:
        sp.frame = sp.frame.loc[sp.frame.index >= min_date]

    # --- base_factor_library: full-history base indicators -------------------
    # extra_factors (e.g. holdings-based hold_mom, §7.1 hyp-3) enter as new leaves;
    # with a market series, PIT rolling beta becomes a leaf too (usable by the DSL
    # to hedge/neutralize, and by the low_beta baseline).
    extras = dict(extra_factors or {})
    if market_px is not None:
        extras.setdefault("beta", _panel.rolling_beta_frame(sp.frame, market_px))
    bf = BaseFactors(industry_of=industry_of, extra=extras)
    bpanel = bf.compute_base_factors(sp, final_as_of)
    price_frame = sp.frame

    # --- labels + locked test + overlap-free validation split ----------------
    all_dates = _validation.decision_dates(price_frame, stride_days=stride_days, start_frac=start_frac)
    if max_decision_dates is not None and len(all_dates) > max_decision_dates:
        all_dates = all_dates[-max_decision_dates:]
    ledger = SearchLedger()
    # §7.3: carve off a LOCKED final test region (last ~15%) that nothing in this run
    # may touch — not the search, not screening, not the diagnostics. An EMBARGO of
    # ceil(horizon/stride) decision dates is dropped between as_ofs and the locked
    # region so no train/valid label window reaches into it (skeptic-confirmed leak:
    # without this, the last valid labels share up to 30/40 days with locked labels).
    embargo_steps = max(1, math.ceil(horizon_days / max(1, stride_days)))
    if locked_start is not None:
        # PINNED seal: the locked region is an anchor date, not a per-run fraction.
        # A fraction-derived carve silently SHIFTS when the universe/date grid
        # changes, letting a later run consume previously-sealed dates (observed:
        # extending the universe moved the carve from 2024-12 to 2025-05, spending
        # ~3 months of sealed data). Repeated-research runs must pin this.
        lock_idx = next((i for i, d in enumerate(all_dates) if d >= locked_start), len(all_dates))
        locked = all_dates[lock_idx:]
        as_ofs = all_dates[: max(0, lock_idx - embargo_steps)]
        if not as_ofs:
            raise ValueError(f"locked_start={locked_start} leaves no evaluable dates")
    else:
        n_locked = max(1, int(len(all_dates) * 0.15))
        if len(all_dates) > n_locked + embargo_steps + 6:
            locked = all_dates[-n_locked:]
            as_ofs = all_dates[: len(all_dates) - n_locked - embargo_steps]
        else:
            locked = []
            as_ofs = all_dates
    if locked:
        ledger.lock_test_span((locked[0], locked[-1]))
    labels = _panel.build_labels(
        price_frame, as_ofs, horizon_days=horizon_days, market_px=market_px,
    )
    # search trains on the early ~45%; screening evaluates on the later, multi-regime
    # remainder (both strictly before the locked test region).
    split = _validation.train_valid_split(
        as_ofs, horizon_days=horizon_days, train_frac=0.45,
        observation_dates=price_frame.index,
    )

    # --- auto_search: baseline-first symbolic search -------------------------
    # scale-free leaves only: raw price LEVEL `px` is excluded from cross-sectional
    # ranking (it is non-stationary and not comparable across assets — §7.3).
    scale_free_fields = [f for f in bpanel.factors.keys() if f != "px"]
    if search_cfg is not None:
        cfg = search_cfg
        if not cfg.allowed_fields:            # budget-only config → still forbid px
            cfg.allowed_fields = scale_free_fields
    else:
        cfg = SearchConfig(
            config_id=uuid.uuid4().hex[:8],
            label_horizon_days=horizon_days,
            allowed_fields=scale_free_fields,
            allowed_windows=(5, 20, 60),
            max_complexity=10,
            max_depth=4,
            max_trials=200,
            population=40,
            generations=5,
        )
    searcher = SymbolicSearch(dsl, industry_of=industry_of)
    searched_exprs = searcher.search(cfg, dsl, bpanel, labels, split, ledger)

    # --- deflated significance, attributed PER FAMILY (§7.3 / §4.3) ----------
    # The expected max of N noise t-stats grows like sqrt(2 ln N). The multiple-
    # testing burden differs by family: searched factors carry the full search
    # ledger (n_trials), while the pre-registered manual baselines carry only their
    # own family size. One shared bar would unfairly kill manual factors and could
    # falsely declare "search beats baseline" (skeptic-confirmed).
    manual_exprs = _baselines.manual_exprs(
        dsl, available_fields=set(bpanel.factors.keys()), price_only=False
    )
    deflated_t = max(2.0, float(math.sqrt(2.0 * math.log(max(ledger.n_trials, 2)))))
    manual_t = max(2.0, float(math.sqrt(2.0 * math.log(max(len(manual_exprs), 2)))))

    # --- factor_screening (+ gates) on the multi-regime OOS split ------------
    screen_manual = Screening(dsl, gate_config=gate_config, stride_days=stride_days, min_abs_t=manual_t)
    screen_searched = Screening(dsl, gate_config=gate_config, stride_days=stride_days, min_abs_t=deflated_t)
    archive = InMemoryFactorArchive()

    manual_scored = _score_and_archive(
        manual_exprs, "manual", screen_manual, archive, bpanel, labels, split, costs, cfg,
    )
    searched_named = {f"searched_{i}": e for i, e in enumerate(searched_exprs)}
    searched_scored = _score_and_archive(
        searched_named, "searched", screen_searched, archive, bpanel, labels, split, costs, cfg,
    )

    best_manual = max((s.score.rank_ic for s in manual_scored), default=0.0)
    best_searched = max((s.score.rank_ic for s in searched_scored), default=0.0)
    best_manual_gated = max((s.score.rank_ic for s in manual_scored if s.gate_passed), default=0.0)
    best_searched_gated = max((s.score.rank_ic for s in searched_scored if s.gate_passed), default=0.0)
    # honest, gate-aware: an edge only counts if a searched factor PASSES the gates
    # and beats the best gate-passing manual factor. Raw rank-IC alone is not enough
    # (e.g. neg(px) is a non-stationary level artifact — §7.3).
    beats = best_searched_gated > best_manual_gated + 1e-6
    top_searched = max(searched_scored, key=lambda s: s.score.rank_ic, default=None)
    top_searched_text = top_searched.expr.text if top_searched else ""

    # --- combination + execution_sim -----------------------------------------
    # Prefer gate-passing factors; if none clear the gates, still exercise the cost
    # path on the top-K by rank-IC, clearly flagged DIAGNOSTIC (not deployable).
    passing = [s for s in (manual_scored + searched_scored) if s.gate_passed]
    diagnostic = not passing
    combine_set = passing or sorted(
        manual_scored + searched_scored, key=lambda s: s.score.rank_ic, reverse=True
    )[:3]
    net_by_tier: dict[str, float] = {}
    if combine_set:
        combo = FactorCombination()
        fund_scores = combo.combine(
            [s.expr for s in combine_set],
            {s.expr.text: s.score for s in combine_set},
            bpanel,
            final_as_of,
        )
        execsim = ExecSim()
        empty = PortfolioState(as_of=final_as_of, holdings=pd.Series(dtype=float))
        exec_res = execsim.simulate(fund_scores, empty, final_as_of, costs)
        net_by_tier = exec_res.net_return_by_tier

    # --- transparency: for the top factors, show WHY they pass/fail --------
    lbd_full = _panel.labels_by_date(labels)
    diagnostics = _diagnose_top(
        manual_scored, searched_scored, dsl, bpanel, as_ofs, lbd_full, horizon_days, stride_days,
        manual_min_abs_t=manual_t, searched_min_abs_t=deflated_t,
    )

    verdict = _verdict(beats, best_searched_gated, best_manual_gated, len(passing),
                       best_searched, top_searched_text)

    return FactorPipelineResult(
        as_of=final_as_of,
        universe_size=len(bpanel.funds),
        n_decision_dates=len(as_ofs),
        horizon_days=horizon_days,
        manual=manual_scored,
        searched=searched_scored,
        best_manual_rank_ic=best_manual,
        best_searched_rank_ic=best_searched,
        best_manual_gated=best_manual_gated,
        best_searched_gated=best_searched_gated,
        search_beats_baseline=beats,
        top_searched_text=top_searched_text,
        n_trials=ledger.n_trials,
        n_gate_passed=len(passing),
        portfolio_is_diagnostic=diagnostic,
        net_return_by_tier=net_by_tier,
        archive_size=len(archive.query()),
        deflated_min_abs_t=deflated_t,
        manual_min_abs_t=manual_t,
        locked_test_span=(locked[0], locked[-1]) if locked else None,
        diagnostics=diagnostics,
        verdict=verdict,
    )


def _diagnose_top(manual_scored, searched_scored, dsl, bpanel, as_ofs, lbd_full,
                  horizon_days, stride_days, *, top: int = 4,
                  manual_min_abs_t: float = 2.0,
                  searched_min_abs_t: float = 2.0) -> list[dict]:
    """For the top factors by raw rank-IC, report full-history sub-period stability,
    overlap-corrected significance, and the first gate that failed — so 'why it
    failed' is visible without a separate script.

    Honesty notes (skeptic-confirmed): (a) significance uses each origin's OWN
    deflated bar — never the undeflated default, which would let a selected max
    look "significant" in the audit while failing its gate; (b) for SEARCHED
    factors the full span includes their own training region, so ``full_rank_ic``
    is upward-biased — flagged via ``train_contaminated``.
    """
    picks = (
        sorted(manual_scored, key=lambda s: -s.score.rank_ic)[:top]
        + sorted(searched_scored, key=lambda s: -s.score.rank_ic)[:top]
    )
    out: list[dict] = []
    for sf in picks:
        try:
            frame = dsl.evaluate_frame(sf.expr, bpanel)
            sbd = _metrics.scores_by_date_from_frame(frame, list(as_ofs))
        except Exception:
            continue
        _, rics = _metrics.per_date_ic(sbd, lbd_full)
        n = len(rics)
        thirds = np.array_split(np.asarray(rics, dtype=float), 3) if n >= 6 else [np.asarray(rics)]
        sub = [round(float(np.mean(b)), 4) if len(b) else 0.0 for b in thirds]
        bar = manual_min_abs_t if sf.origin == "manual" else searched_min_abs_t
        full = _metrics.ic_summary(sbd, lbd_full, horizon_days=horizon_days,
                                   stride_days=stride_days, min_abs_t=bar)
        gf = sf.score.gate_report.first_failure() if sf.score.gate_report else None
        out.append({
            "factor": sf.name if sf.origin == "manual" else sf.expr.text[:48],
            "origin": sf.origin,
            "valid_rank_ic": round(sf.score.rank_ic, 4),
            "full_rank_ic": round(full.mean_rank_ic, 4),
            "full_tstat": round(full.tstat, 2),
            "significant": bool(full.significant),
            "tstat_bar": round(bar, 2),
            "train_contaminated": sf.origin == "searched",
            "subperiod_rank_ic": sub,
            "stable_sign": len({v > 0 for v in sub if v != 0.0}) == 1 if sub else False,
            "first_gate_fail": gf.value if gf else None,
        })
    return out


def _score_and_archive(
    named: dict[str, FactorExpr],
    origin: str,
    screen: Screening,
    archive: InMemoryFactorArchive,
    bpanel,
    labels,
    split,
    costs: CostTiers,
    cfg: SearchConfig,
) -> list[ScoredFactor]:
    out: list[ScoredFactor] = []
    for name, expr in named.items():
        try:
            score = screen.score_standalone(expr, bpanel, labels, split, costs)
        except Exception as exc:  # a broken candidate must not crash the run; archive it
            _archive(archive, name, expr, None, cfg, origin, reject=str(exc), status=FactorStatus.RETIRED)
            continue
        passed = bool(score.gate_report and score.gate_report.passed)
        status = FactorStatus.SCREENED if passed else FactorStatus.ARCHIVED
        reject = None if passed else _first_gate_fail(score)
        _archive(archive, name, expr, score, cfg, origin, reject=reject, status=status)
        out.append(ScoredFactor(name=name, expr=expr, score=score, origin=origin))
    return out


def _archive(archive, name, expr, score, cfg, origin, *, reject, status) -> None:
    archive.put(
        FactorRecord(
            factor_id=f"{origin}:{name}",
            expr=expr,
            search_config_id=cfg.config_id,
            ic=getattr(score, "ic", 0.0) if score else 0.0,
            rank_ic=getattr(score, "rank_ic", 0.0) if score else 0.0,
            turnover=getattr(score, "turnover", 0.0) if score else 0.0,
            net_return_by_tier=getattr(score, "net_return_by_tier", {}) if score else {},
            status=status,
            score=(score.rank_ic if score else None),
            reject_reason=reject,
        )
    )


def _first_gate_fail(score: FactorScore) -> str | None:
    if score.gate_report is None:
        return "no gate report"
    g = score.gate_report.first_failure()
    return f"gate_fail:{g.value}" if g else None


_LEVEL_ARTIFACT_OPS = {"ts_mean", "ts_std", "ts_delta", "ts_drawdown", "ts_min",
                       "ts_max", "ts_rank", "ret", "vol20", "mom20"}


def _looks_like_level_artifact(text: str) -> bool:
    """A formula that ranks on raw price *level* (px) with no return/ts transform is
    a non-stationary cross-sectional artifact, not a real factor (§7.3)."""
    if "px" not in text:
        return False
    return not any(op in text for op in ("ts_", "ret", "mom", "vol", "dd"))


def _verdict(beats: bool, best_searched_gated: float, best_manual_gated: float,
             n_pass: int, best_searched_raw: float, top_text: str) -> str:
    if n_pass == 0:
        artifact = ""
        if top_text and _looks_like_level_artifact(top_text):
            artifact = (
                f" The highest raw rank-IC formula, `{top_text}`, ranks on price LEVEL "
                "— a non-stationary artifact the gates correctly reject (§7.3), not an edge."
            )
        return (
            f"NO factor (manual or searched) passed all four gates OOS.{artifact} "
            "Per §7.1/§7.4 this is a correct, honest outcome — on this cross-section the "
            "platform's value is governance/auditability, not returns. (Portfolio below "
            "is diagnostic-only.)"
        )
    if beats:
        return (
            f"A gate-passing searched factor (rank-IC {best_searched_gated:.3f}) beats the "
            f"best gate-passing manual factor ({best_manual_gated:.3f}) — provisional edge; "
            "must still survive purged multi-path validation and cost tiers before adoption."
        )
    return (
        f"Best gate-passing manual factor rank-IC {best_manual_gated:.3f} >= searched "
        f"{best_searched_gated:.3f}: symbolic search added no gate-passing edge OOS. "
        "Per §7.1/§7.4 the honest read is governance value, not returns."
    )


def run_industry_demo(
    store: PointInTimeStore | None = None,
    *,
    horizon_days: int = 40,
    stride_days: int = 20,
    max_decision_dates: int = 120,
) -> FactorPipelineResult:
    """Runnable end-to-end demo on the real 31 Shenwan-industry cross-section.

    The industry axis is a genuine cross-section with 2008–2026 depth (the fund NAV
    universe needs bulk ingestion first). Uses ``sw_daily`` closes; ``industry_of``
    maps each industry code to itself (neutralization is a no-op single-name group).
    """
    from qmt.config import get_settings
    from qmt.data.sources import TABLE_SW_DAILY

    close = store is None
    store = store or PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        codes = list(
            store.get_asof(TABLE_SW_DAILY, pd.Timestamp("2100-01-01"))["ts_code"].unique()
        )
        industry_of = {c: c for c in codes}
        return run_factor_pipeline(
            store, TABLE_SW_DAILY,
            value_col="close", code_col="ts_code", date_col="trade_date",
            horizon_days=horizon_days, stride_days=stride_days,
            max_decision_dates=max_decision_dates, industry_of=industry_of,
        )
    finally:
        if close:
            store.close()


def run_fund_demo(
    store: PointInTimeStore | None = None,
    *,
    horizon_days: int = 40,
    stride_days: int = 10,
    max_decision_dates: int = 260,
    cover_from: str = "2019-06-01",
    cover_to: str = "2024-01-01",
    max_funds: int | None = 400,
    use_holdings: bool = True,
    beta_adjust: bool = True,
) -> FactorPipelineResult:
    """Run the pipeline on the **real off-exchange fund cross-section** (``fund_nav``).

    Universe membership is POINT-IN-TIME: a fund qualifies by having NAV history
    starting at or before ``cover_from`` — a fact knowable on every decision date.
    Funds that later liquidate/merge REMAIN in the universe (their labels settle at
    the last observed NAV); filtering on "survived until ``cover_to``" would inject
    future information and a ~25% survivorship bias (skeptic-confirmed — it dropped
    127/513 funds that died in-window, flattering low-volatility factors).
    ``cover_to`` is retained for API compatibility but NO LONGER filters.
    """
    from qmt.config import get_settings
    from qmt.data.sources import TABLE_FUND_NAV

    close = store is None
    store = store or PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        nav = store.get_asof(TABLE_FUND_NAV, pd.Timestamp("2100-01-01"))
        nav["nav_date"] = pd.to_datetime(nav["nav_date"])
        per = nav.groupby("fund_code")["nav_date"].agg(["min", "max"])
        # PIT membership: inception (per["min"]) is knowable at decision time;
        # per["max"] (death date) is NOT — never filter on it.
        covered = per[per["min"] <= pd.Timestamp(cover_from)]
        codes = sorted(covered.index)
        if max_funds is not None and len(codes) > max_funds:
            codes = codes[:max_funds]
        if len(codes) < 10:
            raise ValueError(
                f"only {len(codes)} funds have history from {cover_from} — ingest more "
                "fund NAV (qmt.data.sources.fund_nav) before running the fund demo."
            )
        cal = TradingCalendar()
        # market series (CSI300 usable closes) for beta-adjusted labels + beta leaf
        market_px = None
        if beta_adjust:
            mkt = _panel.usable_price_frame(
                store, "index_daily", value_col="close", code_col="ts_code",
                date_col="trade_date", calendar=cal,
            )
            if "000300.SH" in mkt.columns:
                market_px = mkt["000300.SH"]
        # extra leaves: holdings signal (hyp-3) + flow/drift (hyp-2), where ingested
        extra: dict = {}
        pf = _panel.usable_price_frame(
            store, TABLE_FUND_NAV, value_col="adj_nav", code_col="fund_code",
            date_col="nav_date", codes=codes, calendar=cal,
        )
        if use_holdings:
            from qmt.factor.holdings import build_holdings_momentum_frame
            hm = build_holdings_momentum_frame(store, codes, pf.index, window=60, calendar=cal)
            if not hm.empty and hm.notna().any().any():
                extra["hold_mom"] = hm
        from qmt.factor.flows import build_flow_frame, build_drift_frame
        fl = build_flow_frame(store, codes, pf.index)
        if not fl.empty and fl.notna().any().any():
            extra["flow"] = fl
        if market_px is not None:
            dr = build_drift_frame(pf, market_px)
            if not dr.empty and dr.notna().any().any():
                extra["drift"] = dr
        extra = extra or None
        return run_factor_pipeline(
            store, TABLE_FUND_NAV,
            value_col="adj_nav", code_col="fund_code", date_col="nav_date",
            horizon_days=horizon_days, stride_days=stride_days,
            max_decision_dates=max_decision_dates, codes=codes, extra_factors=extra,
            market_px=market_px,
        )
    finally:
        if close:
            store.close()
