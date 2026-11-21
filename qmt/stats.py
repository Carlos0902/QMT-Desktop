"""Small, dependency-free statistics used by evaluation and model selection (§7 ch:testing).

Everything here exists because the naive statistic is *wrong* for this data and would manufacture
confidence the evidence does not support:

* **Overlapping windows.** A skill series sampled every trading day but measured over an ``h``-day
  forward window has ``h``-fold overlap, so its observations are heavily autocorrelated. The i.i.d.
  ``t = mean / (std/sqrt(n))`` is inflated by roughly ``sqrt(h)``. Measured on the industry layer's
  cross-sectional IC (h=20, n=2217): naive ``t = +7.94``, non-overlapping subsample ``t = +1.02``,
  Newey–West ``t = +2.16``. Reporting the naive number would have claimed overwhelming significance
  for what is at best a weak, marginal edge.
* **Ties.** Real objective values tie often (many candidates score exactly ``0.0``); a rank
  correlation that does not average tied ranks reports a spurious ``+/-1``.

So this module offers three *independent* corrections and a deliberately conservative verdict that
requires them to **agree** — a disagreement between corrections is itself the finding, and is
reported rather than resolved in the claim's favour (fail-closed, CLAUDE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "tied_ranks",
    "spearman",
    "newey_west_tstat",
    "nonoverlap_tstat",
    "block_bootstrap_ci",
    "Significance",
    "assess_significance",
    "effective_sample_size",
    "winkler_score",
]


# =============================================================================
# rank correlation (tie-aware)
# =============================================================================
def tied_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks with **average ranks for ties** (what Spearman is defined on)."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(a.size, dtype=float)
    s = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Spearman rank correlation; ``nan`` when either input is constant.

    A naive implementation ranks a constant array by ``argsort`` order and reports a spurious
    ``+/-1`` correlation, so a degenerate axis (e.g. a metric that does not vary with the swept
    parameter) would look perfectly predictive.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size != x.size:
        return float("nan")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    rx, ry = tied_ranks(x), tied_ranks(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# =============================================================================
# t-statistics under autocorrelation
# =============================================================================
def newey_west_tstat(x: np.ndarray, lag: int) -> float:
    """t-statistic for ``mean(x) == 0`` with a Newey–West HAC variance (Bartlett kernel).

    ``lag`` should be the overlap horizon minus one (``h - 1``): that is exactly how far the
    autocorrelation induced by overlapping ``h``-step windows reaches.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = float(x.mean())
    e = x - mu
    lag = max(0, min(int(lag), n - 1))
    s = float(e @ e) / n
    for l_ in range(1, lag + 1):
        gamma = float(e[l_:] @ e[:-l_]) / n
        s += 2.0 * (1.0 - l_ / (lag + 1.0)) * gamma
    if not math.isfinite(s) or s <= 0.0:
        return float("nan")
    return mu / math.sqrt(s / n)


def nonoverlap_tstat(x: np.ndarray, stride: int) -> tuple[float, int]:
    """i.i.d. t-statistic on a **non-overlapping** subsample (every ``stride``-th observation).

    The bluntest, most trustworthy correction: with ``stride = h`` the retained observations use
    disjoint forward windows, so the i.i.d. assumption actually holds. Returns ``(t, n_used)``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    stride = max(1, int(stride))
    sub = x[::stride]
    n = sub.size
    if n < 3:
        return float("nan"), n
    sd = float(sub.std(ddof=1))
    if sd == 0.0:
        return float("nan"), n
    return float(sub.mean() / (sd / math.sqrt(n))), n


def block_bootstrap_ci(
    x: np.ndarray,
    block: int,
    level: float = 0.95,
    *,
    resamples: int = 5000,
    seed: int = 20240703,
) -> tuple[float, float]:
    """Circular **block** bootstrap CI for ``mean(x)``, preserving within-block autocorrelation.

    An i.i.d. bootstrap resamples single observations and destroys the serial dependence that
    overlapping windows create, giving an interval far too narrow. Blocks of length ``h`` keep it.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return float("nan"), float("nan")
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    # index matrix of circular blocks: (resamples, n_blocks*block) -> trimmed to n
    starts = rng.integers(0, n, size=(resamples, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n      # (R, n_blocks, block)
    idx = idx.reshape(resamples, -1)[:, :n]
    means = x[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def effective_sample_size(n: int, overlap: int) -> int:
    """Rough independent-observation count for an ``overlap``-step overlapping series."""
    return max(1, int(n // max(1, overlap)))


# =============================================================================
# the conservative verdict
# =============================================================================
@dataclass(frozen=True)
class Significance:
    """Three independent corrections plus a fail-closed verdict.

    ``significant`` is True only when **all** available corrections agree on a non-zero effect with
    the same sign. If they disagree, the honest answer is "not established" and the disagreement is
    surfaced in ``note`` — never resolved in the claim's favour.
    """

    mean: float
    n: int
    overlap: int
    t_naive: float
    t_nonoverlap: float
    n_nonoverlap: int
    t_newey_west: float
    ci_low: float
    ci_high: float
    significant: bool
    note: str

    def as_dict(self) -> dict:
        def f(v: float) -> Optional[float]:
            return None if v is None or not math.isfinite(v) else round(float(v), 6)

        return {
            "mean": f(self.mean), "n": self.n, "overlap": self.overlap,
            "t_naive": f(self.t_naive), "t_nonoverlap": f(self.t_nonoverlap),
            "n_nonoverlap": self.n_nonoverlap, "t_newey_west": f(self.t_newey_west),
            "ci_low": f(self.ci_low), "ci_high": f(self.ci_high),
            "significant": self.significant, "note": self.note,
        }


def assess_significance(
    x: np.ndarray, overlap: int, *, level: float = 0.95, min_abs_t: float = 2.0
) -> Significance:
    """Assess whether ``mean(x)`` differs from zero, correcting for ``overlap``-step overlap.

    Computes the naive t (reported only to expose how much it overstates), a non-overlapping-
    subsample t, a Newey–West t, and a block-bootstrap CI. Declares significance only when the
    bootstrap CI excludes zero **and** both corrected t-statistics exceed ``min_abs_t`` with the
    same sign as the mean. The naive t never contributes to the verdict.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return Significance(float("nan"), n, overlap, float("nan"), float("nan"), 0,
                            float("nan"), float("nan"), float("nan"), False,
                            "样本不足，无法检验（fail-closed）。")

    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    t_naive = mu / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    t_sub, n_sub = nonoverlap_tstat(x, overlap)
    t_nw = newey_west_tstat(x, overlap - 1)
    lo, hi = block_bootstrap_ci(x, overlap, level)

    ci_excludes_zero = math.isfinite(lo) and math.isfinite(hi) and (lo > 0.0 or hi < 0.0)
    same_sign = lambda t: math.isfinite(t) and (t > 0) == (mu > 0)  # noqa: E731
    strong = lambda t: math.isfinite(t) and abs(t) >= min_abs_t     # noqa: E731
    sig = bool(ci_excludes_zero and strong(t_sub) and strong(t_nw) and same_sign(t_sub) and same_sign(t_nw))

    if sig:
        note = "三种重叠修正一致支持该效应显著。"
    else:
        bits = []
        if not ci_excludes_zero:
            bits.append("自助置信区间含 0")
        if not strong(t_sub):
            bits.append(f"非重叠子样本 t={t_sub:+.2f} 未达 {min_abs_t}")
        if not strong(t_nw):
            bits.append(f"Newey-West t={t_nw:+.2f} 未达 {min_abs_t}")
        note = (
            "证据不足（fail-closed）：" + "；".join(bits) +
            f"。朴素 t={t_naive:+.2f} 因窗口重叠被高估约 sqrt({overlap})≈{math.sqrt(max(overlap,1)):.1f} 倍，不作为依据。"
        )
    return Significance(mu, n, overlap, t_naive, t_sub, n_sub, t_nw, lo, hi, sig, note)


def winkler_score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Winkler / interval score for a central ``1 - alpha`` interval (lower is better).

    ``width + (2/alpha) * miss_distance``. It is a **proper** score at a *fixed* nominal level, so
    two candidate bands can be compared directly. Pinball loss cannot: it is mechanically smaller
    for a tighter tau, so ranking candidates that also *sweep* the quantile levels compares
    apples to oranges and reliably elects the narrowest levels regardless of skill. Quantile levels
    are a product choice (how much coverage we promise), not a hyper-parameter to be tuned away.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(y, dtype=float)
    width = hi - lo
    under = np.where(y < lo, lo - y, 0.0)
    over = np.where(y > hi, y - hi, 0.0)
    return width + (2.0 / alpha) * (under + over)
