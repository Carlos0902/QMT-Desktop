"""portfolio_lookthrough — module 8, the first stage of the advice branch
(§2.8 / §1.8-8, sec:mod-lookthrough).

Responsibility (doc): penetrate all of a user's funds to reveal the *true* industry
and style concentration that individual fund names hide — "这可能是整个产品最有差异化
的功能之一". It reduces every held fund to its underlying industry/style exposure by
**reusing exposure_estimator** (:func:`estimate_exposure`), then weight-combines those
exposures to compute merged industry/style concentration, single-theme repeated
(hidden) exposure, an underlying-asset correlation proxy, per-fund risk contribution
and risk-budget usage.

Doc signature (04-modules.tex, lst:look)::

    def look_through(portfolio: UserPortfolio, as_of: date) -> LookthroughResult:
        \"\"\"把每只基金还原到底层行业/风格暴露（复用 exposure_estimator），
        再按权重合并，计算行业/风格集中度、单一主题重复暴露、隐性重复持仓、
        底层资产相关性、极端同跌风险、单基金风险贡献与风险预算使用情况。\"\"\"

The public entrypoint adds an optional ``store`` so the whole chain runs point-in-time
against one :class:`~qmt.pit.PointInTimeStore` (and fully offline in tests).

**ADVICE-MODE ONLY.** :class:`~qmt.types.UserPortfolio` is the user-assets entry type;
this module (with ``risk_budget`` and ``action_strategy``) is the *only* chain that may
read it (invariant 3 / §6 sec:cc-modes). ``analysis_mode`` must never appear in this
call graph. :meth:`ComplianceGuard.gate_no_user_assets_in_analysis` is invoked as the
runtime backstop.

The doc leaves every look-through *formula* unspecified (doc-silent gaps: the calc
"口径与阈值" for concentration / hidden overlap / correlation / co-drawdown / risk
contribution / budget usage). Baselines chosen here (all transparent, all logged in
DECISIONS.md D59–D63) — the candidate methods (returns-based correlation, factor-
covariance risk contribution) are left as documented hooks, never built first:

* **industry_conc / style_conc** — weight-combine each fund's per-label exposure:
  ``conc[label] = Σ_i weight_i · point_estimate_i(label)`` over the matching
  dimension. Weights are NOT renormalized (holding cash legitimately lowers
  concentration).
* **theme_overlap** — any label carried by ``>= _MIN_FUNDS_FOR_OVERLAP`` funds is a
  *repeated / hidden* theme; returned as labels sorted by merged concentration desc.
* **corr_matrix** — baseline proxy = cosine similarity of the funds' exposure
  vectors (Gram matrix of L2-normalized non-negative exposures ⇒ PSD, unit diagonal,
  entries in [0, 1]); it measures *exposure overlap*. Hook: replace with realized
  adj_nav-return correlation once the NAV history is wired.
* **fund_risk_contrib** — standard risk-contribution decomposition
  ``RC_i = w_i·(Σw)_i / (wᵀΣw)`` using that correlation as Σ (unit-vol baseline);
  normalized to sum to 1, keyed by fund_code.
* **budget_usage** — max merged concentration / ``single_industry_soft_cap`` (a real
  config value); > 1 means past the soft cap. The suitability-normalized authoritative
  figure is ``risk_budget``'s job.

Fail-closed (invariant 6): an empty portfolio, a non-positive total weight, or a fund
that resolves to *no* exposure all raise — a concentration tool must never silently
under-count risk by treating an unresolved fund as zero exposure.
"""

from __future__ import annotations

import inspect
from datetime import date
from typing import Optional

import numpy as np

from qmt.compliance import ComplianceGuard, ComplianceViolation
from qmt.config import CONFIG
from qmt.pit import PointInTimeStore
from qmt.types import (
    ExposureDimension,
    ExposureEstimate,
    FundCode,
    LookthroughResult,
    UserPortfolio,
)

__all__ = ["look_through", "estimate_exposure"]

# Doc-silent baseline threshold (D60): a theme carried by at least this many funds is
# treated as a *repeated / hidden* overlap. Module-local (config groups are frozen);
# logged in DECISIONS.md.
_MIN_FUNDS_FOR_OVERLAP = 2


def estimate_exposure(
    fund_code: FundCode,
    as_of: date,
    store: Optional[PointInTimeStore] = None,
) -> list[ExposureEstimate]:
    """Seam onto exposure_estimator (§4 lst:expo, module 3).

    ``look_through`` reuses the exposure estimator rather than re-deriving exposures.
    That module lands in its own slice, so the real implementation is imported lazily
    here — this file imports and is testable *now*, and offline tests monkeypatch this
    symbol to feed synthetic :class:`ExposureEstimate` rows (the doc worked example).

    Fails **closed** (invariant 6) with a clear error if the estimator is not yet
    available and no test double has been injected.
    """
    try:
        from qmt.pipeline.exposure_estimator import estimate_exposure as _impl
    except Exception as exc:  # pragma: no cover - exercised once the estimator lands
        raise RuntimeError(
            "portfolio_lookthrough.look_through requires "
            "qmt.pipeline.exposure_estimator.estimate_exposure (module 3). Until that "
            "slice lands, monkeypatch qmt.pipeline.portfolio_lookthrough.estimate_exposure "
            "to inject exposures."
        ) from exc
    # Thread the shared PIT store through when the real estimator accepts it, so the
    # whole advice chain reads one point-in-time snapshot.
    if "store" in inspect.signature(_impl).parameters:
        return _impl(fund_code, as_of, store=store)
    return _impl(fund_code, as_of)


def _fund_exposure_map(
    estimates: list[ExposureEstimate],
) -> dict[tuple[ExposureDimension, str], float]:
    """Collapse one fund's estimates to ``(dimension, label) -> summed point_estimate``.

    Defensive summation guards against an estimator emitting more than one row for the
    same dimension/label.
    """
    out: dict[tuple[ExposureDimension, str], float] = {}
    for est in estimates:
        key = (est.dimension, est.label)
        out[key] = out.get(key, 0.0) + est.point_estimate
    return out


def _cosine_correlation(vectors: list[np.ndarray]) -> np.ndarray:
    """Exposure-overlap correlation proxy = cosine similarity Gram matrix (D61).

    Non-negative exposure vectors ⇒ entries in [0, 1]; L2-normalized rows ⇒ exact unit
    diagonal and symmetry; a Gram matrix ⇒ PSD (needed so the risk-contribution
    denominator ``wᵀΣw`` is non-negative). A zero-exposure fund gets 0 similarity to
    everything and a self-correlation forced to 1.
    """
    n = len(vectors)
    norms = np.array([np.linalg.norm(v) for v in vectors])
    corr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            denom = norms[i] * norms[j]
            sim = float(np.dot(vectors[i], vectors[j]) / denom) if denom > 0 else 0.0
            corr[i, j] = corr[j, i] = sim
    return corr


def look_through(
    portfolio: UserPortfolio,
    as_of: date,
    store: Optional[PointInTimeStore] = None,
) -> LookthroughResult:
    """Penetrate every fund in ``portfolio`` to reveal true concentration (§2.8).

    ADVICE-MODE ONLY. Reduces each held fund to underlying industry/style exposure via
    :func:`estimate_exposure`, weight-combines them, and returns a
    :class:`LookthroughResult` (industry/style concentration, hidden theme overlap, an
    exposure-overlap correlation proxy, per-fund risk contribution, risk-budget usage).

    Reproduces the doc worked example: 5 differently-named funds (半导体 / AI / 科技成长 /
    创业板 / 新能源) → merged 科技成长 style exposure ≈ 63%, exposing that a portfolio that
    *looks* diversified is really one highly-correlated growth bet.

    Fails closed on an empty portfolio, non-positive total weight, or any fund that
    resolves to no exposure.
    """
    # --- advice-mode backstop (invariant 3 / §3.5-1) ------------------------------
    if portfolio is None:
        raise ComplianceViolation(
            ComplianceGuard().gate_no_user_assets_in_analysis(portfolio, in_analysis_mode=False)
        )
    gate = ComplianceGuard().gate_no_user_assets_in_analysis(portfolio, in_analysis_mode=False)
    if not gate.passed:
        raise ComplianceViolation(gate)

    holdings = list(portfolio.holdings)
    if not holdings:
        raise ValueError("look_through: empty portfolio — cannot assess concentration (fail-closed).")
    total_weight = sum(h.weight for h in holdings)
    if total_weight <= 0:
        raise ValueError("look_through: non-positive total portfolio weight (fail-closed).")

    fund_labels: list[str] = [h.fund_code for h in holdings]

    # --- 1. penetrate every fund to its underlying exposure ----------------------
    per_fund_maps: list[dict[tuple[ExposureDimension, str], float]] = []
    for h in holdings:
        estimates = estimate_exposure(h.fund_code, as_of, store=store)
        if not estimates:
            raise ValueError(
                f"look_through: fund {h.fund_code} resolved to no exposure — a "
                "concentration tool must not under-count risk (fail-closed)."
            )
        per_fund_maps.append(_fund_exposure_map(estimates))

    # --- 2. weight-combine into merged industry / style concentration ------------
    industry_conc: dict[str, float] = {}
    style_conc: dict[str, float] = {}
    # label -> number of funds carrying it (with positive exposure), for overlap.
    label_fund_count: dict[str, int] = {}
    merged_by_label: dict[str, float] = {}

    for h, fmap in zip(holdings, per_fund_maps):
        for (dim, label), point in fmap.items():
            if point <= 0:
                continue
            contrib = h.weight * point
            if dim is ExposureDimension.INDUSTRY:
                industry_conc[label] = industry_conc.get(label, 0.0) + contrib
            elif dim is ExposureDimension.STYLE:
                style_conc[label] = style_conc.get(label, 0.0) + contrib
            label_fund_count[label] = label_fund_count.get(label, 0) + 1
            merged_by_label[label] = merged_by_label.get(label, 0.0) + contrib

    # --- 3. single-theme repeated (hidden) exposure ------------------------------
    theme_overlap: list[str] = sorted(
        (lbl for lbl, cnt in label_fund_count.items() if cnt >= _MIN_FUNDS_FOR_OVERLAP),
        key=lambda lbl: merged_by_label.get(lbl, 0.0),
        reverse=True,
    )

    # --- 4. underlying-asset correlation proxy (exposure overlap) ----------------
    all_keys = sorted(
        {key for fmap in per_fund_maps for key in fmap},
        key=lambda k: (k[0].value, k[1]),
    )
    vectors = [
        np.array([fmap.get(key, 0.0) for key in all_keys], dtype=float)
        for fmap in per_fund_maps
    ]
    corr_matrix = _cosine_correlation(vectors)

    # --- 5. per-fund contribution to total portfolio risk ------------------------
    weights = np.array([h.weight for h in holdings], dtype=float)
    sigma_w = corr_matrix @ weights            # (Σw)_i
    portfolio_var = float(weights @ sigma_w)   # wᵀΣw
    if portfolio_var > 0:
        raw_contrib = weights * sigma_w / portfolio_var  # sums to 1
    else:
        raw_contrib = np.zeros_like(weights)
    fund_risk_contrib: dict[str, float] = {}
    for code, rc in zip(fund_labels, raw_contrib):
        fund_risk_contrib[code] = fund_risk_contrib.get(code, 0.0) + float(rc)

    # --- 6. risk-budget usage (preliminary; risk_budget refines w/ suitability) --
    all_conc = list(industry_conc.values()) + list(style_conc.values())
    max_conc = max(all_conc) if all_conc else 0.0
    soft_cap = CONFIG.risk_budget.single_industry_soft_cap
    budget_usage = max_conc / soft_cap if soft_cap > 0 else 0.0

    return LookthroughResult(
        industry_conc=industry_conc,
        style_conc=style_conc,
        theme_overlap=theme_overlap,
        corr_matrix=corr_matrix,
        fund_risk_contrib=fund_risk_contrib,
        budget_usage=budget_usage,
        fund_labels=fund_labels,
    )
