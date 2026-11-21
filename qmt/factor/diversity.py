"""Training-only semantic diversity for a formula pool.

Different ASTs can express exactly the same cross-sectional ordering. Compare
average absolute Spearman correlations within dates, rather than pooling raw
values across dates (which confounds temporal scale with cross-sectional signal).
Anti-correlated copies also carry the same information. This is a redundancy
filter inspired by pool-based alpha discovery, not an AlphaGen/AlphaPareto replica
and not evidence of incremental portfolio profit.
"""
from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from qmt.factor.types import FactorExpr


def rank_signature(scores: Mapping[date, pd.Series]) -> dict[date, pd.Series]:
    """Finite cross-sectional ranks on training decisions only."""
    out = {}
    for d, s in scores.items():
        clean = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) >= 5 and clean.nunique() > 1:
            out[d] = clean.rank(method="average", pct=True).astype("float32")
    return out


def mean_abs_rank_corr(
    left: Mapping[date, pd.Series], right: Mapping[date, pd.Series], *, min_dates: int = 3
) -> float | None:
    """Mean absolute within-date rank correlation; None means too little overlap.

    Re-rank on the common finite names: missingness must not alter the meaning of
    Spearman correlation. Each date has equal weight, regardless of universe size.
    """
    corrs = []
    for d in sorted(set(left) & set(right)):
        pair = pd.concat([left[d].rename("a"), right[d].rename("b")], axis=1).dropna()
        if len(pair) < 5:
            continue
        ranks = pair.rank(method="average").to_numpy(float)
        if np.std(ranks[:, 0]) == 0 or np.std(ranks[:, 1]) == 0:
            continue
        c = np.corrcoef(ranks.T)[0, 1]
        if np.isfinite(c):
            corrs.append(abs(float(c)))
    return float(np.mean(corrs)) if len(corrs) >= min_dates else None


def select_diverse(
    ranked: Sequence[FactorExpr],
    signatures: Mapping[str, Mapping[date, pd.Series]],
    *,
    limit: int,
    max_corr: float | None,
    min_dates: int = 3,
) -> list[FactorExpr]:
    """Greedily retain strongest train-ranked candidates with distinct semantics.

    Insufficient overlap cannot prove redundancy, so it never causes rejection.
    The caller supplies fitness ordering and training signatures; no labels or
    validation observations are consumed here. Thresholds are pre-registered.
    """
    if limit <= 0:
        return []
    if min_dates < 1:
        raise ValueError("diversity_min_dates must be positive")
    if max_corr is not None and not 0 < max_corr <= 1:
        raise ValueError("diversity_max_corr must lie in (0, 1] or be None")
    kept = []
    for expr in ranked:
        if any(expr.text == other.text for other in kept):
            continue
        own = signatures.get(expr.text, {})
        redundant = False
        if max_corr is not None:
            for other in kept:
                corr = mean_abs_rank_corr(own, signatures.get(other.text, {}), min_dates=min_dates)
                if corr is not None and corr >= max_corr:
                    redundant = True
                    break
        if not redundant:
            kept.append(expr)
            if len(kept) >= limit:
                break
    return kept
