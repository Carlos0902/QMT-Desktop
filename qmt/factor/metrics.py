"""Cross-sectional evaluation metrics (tech doc §6.1 / ch.7).

Every "does this factor have predictive power?" question is answered per-date on the
fund cross-section, then aggregated with overlap-corrected significance. We reuse
``qmt.stats`` verbatim — ``spearman`` is the per-date rank-IC primitive and
``assess_significance`` is the fail-closed verdict (bootstrap CI + two HAC-corrected
t-stats must agree). The overlap is expressed in *decision units*
(``ceil(horizon / stride)``), never day units, to avoid manufacturing significance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from qmt.stats import Significance, assess_significance, spearman

__all__ = [
    "ICResult", "aligned_pairs", "per_date_ic", "ic_summary",
    "group_monotonicity", "rank_turnover", "decay_profile",
    "scores_by_date_from_frame",
]


def scores_by_date_from_frame(
    frame: pd.DataFrame, dates: "list[date]"
) -> dict[date, pd.Series]:
    """Slice a full-history (date x fund) score frame into per-decision-date
    cross-sections, using as-of semantics (last row ``<= d``).

    Companion to ``FormulaDSL.evaluate_frame``: because every DSL operator is
    causal, ``frame`` row at the as-of position equals a per-date ``evaluate`` —
    this turns O(dates x history) evaluation into O(history) + O(dates) lookups.
    """
    out: dict[date, pd.Series] = {}
    if frame is None or frame.empty:
        return out
    idx = np.asarray(frame.index)
    datetime_index = idx.dtype.kind == "M"  # normalise date keys for datetime64 axes
    for d in dates:
        key = np.datetime64(pd.Timestamp(d)) if datetime_index else d
        pos = int(np.searchsorted(idx, key, side="right")) - 1
        if pos < 0:
            continue
        row = frame.iloc[pos].dropna()
        if not row.empty:
            out[d] = row
    return out


@dataclass(frozen=True)
class ICResult:
    """Aggregate IC verdict for one factor over a set of decision dates."""

    mean_ic: float
    mean_rank_ic: float
    n_dates: int
    overlap: int
    significance: Significance          # on the rank-IC series
    rank_ic_series: list[float]

    @property
    def significant(self) -> bool:
        return self.significance.significant

    @property
    def tstat(self) -> float:
        return self.significance.t_newey_west

    def as_dict(self) -> dict:
        return {
            "mean_ic": self.mean_ic,
            "mean_rank_ic": self.mean_rank_ic,
            "n_dates": self.n_dates,
            "overlap": self.overlap,
            "significant": self.significant,
            "tstat_nw": self.tstat,
        }


def aligned_pairs(scores: pd.Series, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join a factor cross-section with a label cross-section on fund code."""
    if scores is None or labels is None or scores.empty or labels.empty:
        return np.array([]), np.array([])
    joined = pd.concat([scores.rename("s"), labels.rename("y")], axis=1).dropna()
    return joined["s"].to_numpy(float), joined["y"].to_numpy(float)


def per_date_ic(
    scores_by_date: Mapping[date, pd.Series],
    labels_by_date: Mapping[date, pd.Series],
    *,
    min_names: int = 5,
) -> tuple[list[float], list[float]]:
    """Return (pearson-IC series, spearman-rankIC series) over shared dates."""
    ics: list[float] = []
    rank_ics: list[float] = []
    for d in sorted(set(scores_by_date) & set(labels_by_date)):
        s, y = aligned_pairs(scores_by_date[d], labels_by_date[d])
        if s.size < min_names or np.std(s) == 0 or np.std(y) == 0:
            continue
        ics.append(float(np.corrcoef(s, y)[0, 1]))
        rank_ics.append(spearman(s, y))
    ics = [v for v in ics if not math.isnan(v)]
    rank_ics = [v for v in rank_ics if not math.isnan(v)]
    return ics, rank_ics


def ic_summary(
    scores_by_date: Mapping[date, pd.Series],
    labels_by_date: Mapping[date, pd.Series],
    *,
    horizon_days: int,
    stride_days: int,
    min_abs_t: float = 2.0,
) -> ICResult:
    """Aggregate cross-sectional IC with overlap-corrected significance.

    ``overlap = ceil(horizon_days / stride_days)`` in decision units — the HAC lag
    the whole validation stack uses (see ``qmt.stats``/``ts_model_bench``).
    """
    ics, rank_ics = per_date_ic(scores_by_date, labels_by_date)
    overlap = max(1, math.ceil(horizon_days / max(1, stride_days)))
    arr = np.asarray(rank_ics, dtype=float)
    if arr.size == 0:
        sig = assess_significance(np.array([0.0]), overlap=overlap, min_abs_t=min_abs_t)
        return ICResult(0.0, 0.0, 0, overlap, sig, [])
    sig = assess_significance(arr, overlap=overlap, min_abs_t=min_abs_t)
    return ICResult(
        mean_ic=float(np.mean(ics)) if ics else 0.0,
        mean_rank_ic=float(np.mean(arr)),
        n_dates=int(arr.size),
        overlap=overlap,
        significance=sig,
        rank_ic_series=list(arr),
    )


def group_monotonicity(
    scores_by_date: Mapping[date, pd.Series],
    labels_by_date: Mapping[date, pd.Series],
    *,
    n_groups: int = 5,
    min_names: int = 10,
) -> tuple[list[float], bool]:
    """Quantile-bucket forward returns; return (bucket means, is_monotone).

    Monotone means the top bucket beats the bottom and the bucket-index vs
    bucket-mean rank correlation is strongly positive — the "分组收益单调" check.
    """
    sums = np.zeros(n_groups)
    counts = np.zeros(n_groups)
    for d in sorted(set(scores_by_date) & set(labels_by_date)):
        s, y = aligned_pairs(scores_by_date[d], labels_by_date[d])
        if s.size < min_names:
            continue
        order = np.argsort(s)
        buckets = np.array_split(order, n_groups)
        for g, idx in enumerate(buckets):
            if idx.size:
                sums[g] += y[idx].sum()
                counts[g] += idx.size
    means = [float(sums[g] / counts[g]) if counts[g] else float("nan")
             for g in range(n_groups)]
    valid = [m for m in means if not math.isnan(m)]
    if len(valid) < n_groups:
        return means, False
    idx_rank = np.arange(n_groups, dtype=float)
    mono_corr = spearman(idx_rank, np.array(means))
    is_monotone = (means[-1] > means[0]) and (mono_corr >= 0.8)
    return means, bool(is_monotone)


def rank_turnover(
    scores_by_date: Mapping[date, pd.Series],
    *,
    top_frac: float = 0.2,
) -> float:
    """Average name-level turnover of the top bucket between consecutive dates.

    High turnover means high cost; this feeds the tradability gate (§7.5).
    """
    dates = sorted(scores_by_date)
    if len(dates) < 2:
        return 0.0
    prev_top: set | None = None
    turns: list[float] = []
    for d in dates:
        s = scores_by_date[d].dropna()
        if s.empty:
            continue
        k = max(1, int(len(s) * top_frac))
        top = set(s.sort_values(ascending=False).head(k).index)
        if prev_top is not None and prev_top:
            churn = len(top ^ prev_top) / (2 * max(len(top), len(prev_top)))
            turns.append(churn)
        prev_top = top
    return float(np.mean(turns)) if turns else 0.0


def decay_profile(
    scores_by_date: Mapping[date, pd.Series],
    labels_by_horizon: Mapping[int, Mapping[date, pd.Series]],
    *,
    stride_days: int,
) -> dict[int, float]:
    """Mean rank-IC as a function of label horizon — the signal-decay curve (§6.1)."""
    out: dict[int, float] = {}
    for h, labels_by_date in labels_by_horizon.items():
        _, rank_ics = per_date_ic(scores_by_date, labels_by_date)
        out[h] = float(np.mean(rank_ics)) if rank_ics else 0.0
    return out
