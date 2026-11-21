"""Purged walk-forward benchmarking for the forecasting layers (§5 / §7 modeling philosophy).

This is the honest test of whether *conditioning* on features (a richer model) buys real
out-of-sample skill over the transparent baselines the system already uses — the only evidence
CLAUDE.md accepts ("a complex model that can't reliably beat these has not proven incremental
value"). It is deliberately hostile to overfitting:

* **Purged, embargoed walk-forward** (López de Prado). Train on the past, leave an ``embargo = h``
  gap so the training targets never overlap the test window, then score the next block. Nothing in
  the test window's feature computation or target can have leaked into training.
* **Out-of-sample only.** Every reported number is on data the model never saw.
* **Overlap-corrected significance.** An ``h``-day forward metric sampled daily is autocorrelated;
  a naive t is inflated ~``sqrt(h)``. Verdicts route through :func:`qmt.stats.assess_significance`.
* **Permutation importance under the same CV**, so a feature is credited only for OOS signal.

The models are a *ladder* of increasing capacity so the marginal value of each step is visible:
unconditional (the current model) → a no-fit conditional rule → penalized linear → gradient-boosted
trees. One shared hyper-parameter set is used for every series (the whole-market discipline: the
same model faces the whole market, funds inherit it via aggregation — never per-fund tuning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from qmt.stats import Significance, assess_significance, spearman

__all__ = [
    "ModelSpec",
    "BenchResult",
    "MODEL_LADDER",
    "purged_blocks",
    "walk_forward_predict",
    "run_benchmark",
    "permutation_importance",
]


# =============================================================================
# Models — a fit/predict pair each; one shared hyper-parameter set (no per-series tuning)
# =============================================================================
@dataclass(frozen=True)
class ModelSpec:
    name: str
    make: Callable[[], "object"]     # returns an object with fit(X,y) / predict(X); or a rule
    capacity: str                    # "none" | "rule" | "linear" | "trees"


def _unconditional():
    """The current model: predict the training median target (no features)."""
    class _Uncond:
        def fit(self, X, y):
            self.m = float(np.median(y))
            return self
        def predict(self, X):
            return np.full(X.shape[0], self.m)
    return _Uncond()


def _feature_rule(col: int):
    """A no-fit conditional baseline: predict a single (standardized-sign) feature column."""
    class _Rule:
        def fit(self, X, y):
            self.mu = float(np.nanmean(X[:, col])); self.sd = float(np.nanstd(X[:, col])) or 1.0
            return self
        def predict(self, X):
            return (X[:, col] - self.mu) / self.sd
    return _Rule()


def _ridge(alpha: float = 10.0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Heavily regularized for a low-signal, few-thousand-sample regime: shallow trees, slow rate,
    # large leaves, L2. This is the SOTA-competitive choice here — deep nets overfit at this scale.
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=250, learning_rate=0.03,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0,
    )


def _ladder(mom_col: int) -> list[ModelSpec]:
    return [
        ModelSpec("unconditional", _unconditional, "none"),
        ModelSpec("feature_rule", lambda: _feature_rule(mom_col), "rule"),
        ModelSpec("ridge", _ridge, "linear"),
        ModelSpec("hgb", _hgb, "trees"),
    ]


# A default ladder assuming the momentum column is index 3 (mom60) in the standard feature frame.
MODEL_LADDER = _ladder


# =============================================================================
# Purged walk-forward
# =============================================================================
def _contiguous_finite(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite rows, but ONLY if they form a leading/trailing block — never interior.

    The walk-forward embargo is an ``h``-ROW gap that is only a valid ``h``-DAY gap while the rows
    stay contiguous in calendar time. Feature burn-in (leading NaNs) and forward-target truncation
    (trailing NaNs) are prefix/suffix drops that preserve contiguity. An *interior* NaN row would
    silently shorten the embargo below ``h`` days, so we refuse it loudly rather than leak.
    """
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if not ok.any():
        return X[:0], y[:0]
    lo = int(np.argmax(ok))
    hi = len(ok) - int(np.argmax(ok[::-1]))
    if not ok[lo:hi].all():
        raise ValueError(
            "non-finite rows are not a contiguous block; the row-embargo would no longer be a valid "
            "time gap — fix the feature/target NaNs before benchmarking (PIT safety)."
        )
    return X[lo:hi], y[lo:hi]


def purged_blocks(n: int, *, start_frac: float = 0.35, test: int = 126, embargo: int = 20):
    """Yield ``(train_end, test_slice)`` for an expanding, embargoed walk-forward.

    ``train`` is ``[0, train_end)`` with ``train_end = t0 - embargo`` (the embargo drops the most
    recent ``embargo`` training rows, whose forward targets would overlap the test features);
    ``test`` is ``[t0, t0+test)``.
    """
    t0 = int(n * start_frac)
    while t0 + test <= n:
        train_end = t0 - embargo
        if train_end >= 200:
            yield train_end, slice(t0, t0 + test)
        t0 += test


@dataclass
class BenchResult:
    target: str
    model: str
    capacity: str
    n_oos: int
    ic: float                        # spearman(pred, realized) — rank tracking of the outcome
    directional: Significance        # overlap-corrected on sign(pred - med(pred)) * realized
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "target": self.target, "model": self.model, "capacity": self.capacity,
            "n_oos": self.n_oos, "ic": None if not np.isfinite(self.ic) else round(self.ic, 5),
            "directional": self.directional.as_dict(), "extra": self.extra,
        }


def walk_forward_predict(
    X: np.ndarray, y: np.ndarray, model_make: Callable[[], object], *,
    embargo: int, start_frac: float = 0.35, test: int = 126,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated out-of-sample ``(test_positions, pred, realized)`` over the walk-forward.

    ``test_positions`` are row indices into ``X`` — so a caller can map predictions back onto dates
    to draw an **honest OOS curve** (never an in-sample fit). ``embargo`` is REQUIRED (pass the
    forecast horizon) so a caller cannot silently under-purge by relying on a wrong default.
    """
    pos, pr, re = [], [], []
    for train_end, te in purged_blocks(len(y), start_frac=start_frac, test=test, embargo=embargo):
        m = model_make().fit(X[:train_end], y[:train_end])
        pos.append(np.arange(te.start, te.stop))
        pr.append(np.asarray(m.predict(X[te]), dtype=float))
        re.append(y[te])
    if not pos:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(pos), np.concatenate(pr), np.concatenate(re)


def run_benchmark(
    X: np.ndarray, y: np.ndarray, *, horizon: int, target: str,
    models: Optional[Sequence[ModelSpec]] = None, mom_col: int = 3,
    start_frac: float = 0.35, test: int = 126,
) -> tuple[list[BenchResult], dict[str, np.ndarray]]:
    """Run the model ladder on one target; return per-model OOS results + concatenated predictions.

    ``target`` is a label (e.g. ``"market_ret_h20"`` or ``"market_vol_h20"``). For a *volatility*
    target read ``extra['corr']`` (Spearman of predicted vs realized vol) — the meaningful metric
    there is level tracking, not sign.
    """
    specs = list(models) if models is not None else MODEL_LADDER(mom_col)
    Xc, yc = _contiguous_finite(X, y)
    results: list[BenchResult] = []
    preds: dict[str, np.ndarray] = {}
    for sp in specs:
        _pos, p, r = walk_forward_predict(Xc, yc, sp.make, start_frac=start_frac, test=test,
                                          embargo=horizon)
        if p.size < 20:
            continue
        preds[sp.name] = p
        ic = spearman(p, r)
        dser = np.sign(p - np.median(p)) * r
        sig = assess_significance(dser, overlap=horizon)
        extra = {"corr": None if not np.isfinite(ic) else round(float(ic), 5)}
        results.append(BenchResult(target, sp.name, sp.capacity, int(r.size), ic, sig, extra))
    return results, preds


def permutation_importance(
    X: np.ndarray, y: np.ndarray, names: list[str], model_make: Callable[[], object], *,
    horizon: int, start_frac: float = 0.35, test: int = 126, n_repeats: int = 5,
    seed: int = 20240703, metric: str = "ic",
) -> list[dict]:
    """Permutation importance under the SAME purged walk-forward, so a feature earns credit only for
    genuine OOS signal.

    For each block: fit on train, score the test block, then for each feature shuffle *only that
    feature's test column* (breaking its relationship with the target while preserving its marginal
    distribution) and re-score. Importance = mean drop in the OOS metric. A feature that is pure
    noise scores ~0 (or negative); a feature the model actually uses scores positive.
    """
    Xc, yc = _contiguous_finite(X, y)
    rng = np.random.default_rng(seed)
    F = Xc.shape[1]
    drops: list[list[float]] = [[] for _ in range(F)]
    base_scores: list[float] = []

    def score(pred, real):
        if metric == "ic":
            return spearman(pred, real)
        return -float(np.mean((pred - real) ** 2))       # neg MSE (higher better)

    for train_end, te in purged_blocks(len(yc), start_frac=start_frac, test=test, embargo=horizon):
        m = model_make().fit(Xc[:train_end], yc[:train_end])
        Xte = Xc[te].copy(); yte = yc[te]
        base = score(np.asarray(m.predict(Xte), dtype=float), yte)
        if not np.isfinite(base):
            continue
        base_scores.append(base)
        for c in range(F):
            reps = []
            for _ in range(n_repeats):
                Xp = Xte.copy()
                Xp[:, c] = rng.permutation(Xp[:, c])
                s = score(np.asarray(m.predict(Xp), dtype=float), yte)
                if np.isfinite(s):
                    reps.append(base - s)      # drop = base - permuted (positive => feature helps)
            if reps:
                drops[c].append(float(np.mean(reps)))

    out = []
    for c, name in enumerate(names):
        d = np.array(drops[c], dtype=float)
        out.append({
            "feature": name,
            "importance": None if d.size == 0 else round(float(d.mean()), 6),
            "n_blocks": int(d.size),
        })
    out.sort(key=lambda r: (r["importance"] is None, -(r["importance"] or 0.0)))
    return out
