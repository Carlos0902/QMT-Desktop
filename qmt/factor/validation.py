"""validation — decision-date series and purged/embargoed splits (instruction §4.2).

Random train/test splits leak because quant data is ordered and a 20–60-day label
spans many trading days (§4.1). We build :class:`ValidationSplit`s whose validation
region is strictly after training, separated by an ``embargo >= horizon`` so the two
regions' label windows never overlap. Fold boundaries reuse
``qmt.pipeline.model_bench.purged_blocks`` (the canonical López-de-Prado purge) over
the decision-date positions; the CPCV scheme yields multiple paths to see the
performance *distribution* rather than one lucky backtest.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from qmt.pipeline.model_bench import purged_blocks
from qmt.factor.types import ValidationScheme, ValidationSplit

__all__ = [
    "decision_dates",
    "train_valid_split",
    "purged_kfold_splits",
    "cpcv_splits",
    "embargo_rows_for_horizon",
]


def decision_dates(
    price_frame: pd.DataFrame,
    *,
    stride_days: int = 5,
    start_frac: float = 0.15,
) -> list[date]:
    """Every ``stride_days``-th usable row is a rebalance decision point.

    ``start_frac`` skips the earliest history so base factors (up to the longest
    window) have enough lookback before the first decision.
    """
    idx = list(price_frame.index)
    n = len(idx)
    if n == 0:
        return []
    start = int(n * start_frac)
    picks = idx[start::max(1, stride_days)]
    return [d.date() if hasattr(d, "date") else d for d in picks]


def embargo_rows_for_horizon(horizon_days: int, stride_days: int) -> int:
    """Embargo must cover the label window; expressed in *decision* steps."""
    return max(1, int(np.ceil(horizon_days / max(1, stride_days))))


def train_valid_split(
    dates: Sequence[date],
    *,
    horizon_days: int,
    train_frac: float = 0.6,
    embargo_days: int | None = None,
    split_id: str = "single",
    observation_dates: Sequence[date] | None = None,
) -> ValidationSplit:
    """One rolling train→(purged gap)→valid split.

    ``horizon_days`` and ``embargo_days`` count observed price rows, matching the
    forward-label builders. Supply the price-frame index as ``observation_dates``
    to purge by actual label endpoints, including irregular holidays. Training
    labels must end *strictly before* the first validation decision; validation
    dates without a complete forward horizon are excluded.

    Without an observation index, the legacy API uses weekday business days.
    That approximation cannot account for exchange holidays or missing rows;
    callers requiring an exact boundary must supply ``observation_dates``.
    Insufficient history raises instead of shortening the required purge.
    """
    dates = sorted({pd.Timestamp(d).date() for d in dates})
    n = len(dates)
    if n < 4:
        raise ValueError("need at least 4 decision dates to split")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie strictly between 0 and 1")
    if embargo_days is not None and embargo_days < 0:
        raise ValueError("embargo_days must be nonnegative")
    cut = int(n * train_frac)
    if cut < 1 or cut >= n:
        raise ValueError("train_frac leaves an empty train or validation region")
    emb = max(horizon_days, embargo_days if embargo_days is not None else 60)
    train = dates[:cut]
    if observation_dates is not None:
        observed = sorted({pd.Timestamp(d).date() for d in observation_dates})
        if not observed or train[0] < observed[0]:
            raise ValueError("observation_dates do not cover the training decisions")
        observed_arr = np.asarray(observed, dtype="datetime64[D]")

        def position(d: date) -> int:
            return int(np.searchsorted(observed_arr, np.datetime64(d), side="right")) - 1

        # Use the same as-of row lookup as panel.forward_*; never infer a trading
        # horizon from calendar spacing between sparse decision dates.
        train_end = position(train[-1]) + emb
        valid = [
            d for d in dates[cut:]
            if position(d) > train_end
            and position(d) + horizon_days < len(observed)
        ]
    else:
        train_label_end = (pd.Timestamp(train[-1]) + pd.offsets.BDay(emb)).date()
        valid = [d for d in dates[cut:] if d > train_label_end]
    if not valid:
        raise ValueError("insufficient dates for a complete purged train/validation split")
    return ValidationSplit(
        split_id=split_id,
        scheme=ValidationScheme.ROLLING,
        train_span=(train[0], train[-1]),
        valid_span=(valid[0], valid[-1]),
        embargo_days=emb,
        n_paths=1,
        as_of_series=list(valid),
    )


def purged_kfold_splits(
    dates: Sequence[date],
    *,
    horizon_days: int,
    stride_days: int,
    n_folds: int = 4,
    start_frac: float = 0.35,
) -> list[ValidationSplit]:
    """Expanding purged walk-forward folds via ``model_bench.purged_blocks``.

    Each fold trains on all decision dates before the (embargoed) test block and
    validates on the block — no label overlap between train and valid.
    """
    dates = sorted(dates)
    n = len(dates)
    if n < n_folds + 2:
        return [train_valid_split(dates, horizon_days=horizon_days, split_id="fold0")]
    emb = embargo_rows_for_horizon(horizon_days, stride_days)
    test_len = max(2, (n - int(n * start_frac)) // n_folds)
    splits: list[ValidationSplit] = []
    for i, (train_end, test_slice) in enumerate(
        purged_blocks(n, start_frac=start_frac, test=test_len, embargo=emb)
    ):
        if train_end < 2:
            continue
        train = dates[:train_end]
        valid = dates[test_slice]
        if not train or not valid or valid[0] <= train[-1]:
            continue
        splits.append(
            ValidationSplit(
                split_id=f"pkfold{i}",
                scheme=ValidationScheme.PURGED_KFOLD,
                train_span=(train[0], train[-1]),
                valid_span=(valid[0], valid[-1]),
                embargo_days=max(horizon_days, emb * stride_days),
                n_paths=1,
                as_of_series=list(valid),
            )
        )
    return splits or [train_valid_split(dates, horizon_days=horizon_days, split_id="fold0")]


def cpcv_splits(
    dates: Sequence[date],
    *,
    horizon_days: int,
    stride_days: int,
    n_folds: int = 5,
) -> list[ValidationSplit]:
    """Combinatorial-purged paths: reuse the k-fold blocks but tag ``n_paths`` so
    downstream aggregation reads a performance *distribution* (instruction §4.2)."""
    base = purged_kfold_splits(
        dates, horizon_days=horizon_days, stride_days=stride_days, n_folds=n_folds
    )
    for s in base:
        s.scheme = ValidationScheme.CPCV
        s.n_paths = len(base)
    return base


def _median_stride(dates: Sequence[date]) -> int:
    if len(dates) < 2:
        return 5
    deltas = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    deltas = [d for d in deltas if d > 0]
    return int(np.median(deltas)) if deltas else 5
