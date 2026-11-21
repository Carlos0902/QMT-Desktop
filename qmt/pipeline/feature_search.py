"""Systematic, multiple-testing-aware feature-combination search (§5 / §7).

The user asked to "start basic and find which feature combination follows the trend better." Trying
many combinations and keeping the ones that look good is textbook p-hacking: with ~30-49 independent
90-day windows, *some* combination will track the trend by chance. This harness runs the search but
refuses to be fooled:

1. **Three-way TIME split** — TRAIN (fit) / SELECT (choose the combo & model) / HOLDOUT (opened once).
   The combination is chosen using only TRAIN+SELECT; the HOLDOUT is scored exactly once, for the
   chosen combos, and never influences selection.
2. **Purged walk-forward** inside each region (embargo = horizon), so a training target never
   overlaps a test feature.
3. **Multiple-testing deflation** — the best SELECT score is compared to a permutation null (shuffle
   the target, re-run the whole search); a combo "works" only if its HOLDOUT directional IC beats
   the unconditional baseline AND clears the overlap-corrected significance AND exceeds the
   deflated threshold from the null. The number of combinations tried is reported, always.

The verdict is deliberately conservative: it reports, per combination, the SELECT IC, the HOLDOUT
IC, and whether it survives — and it will say plainly "no combination follows the trend better
out-of-sample after deflation" when that is the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional, Sequence

import numpy as np

from qmt.pipeline.model_bench import _hgb, _ridge, _unconditional, purged_blocks
from qmt.stats import assess_significance, spearman

__all__ = ["SearchResult", "enumerate_feature_sets", "feature_search"]


# =============================================================================
# Models available to the search (one shared hyper-parameter set each)
# =============================================================================
def _elasticnet():
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))


def _forecast_combination():
    """Rapach-Strauss-Zhou equal-weight combination: average of UNIVARIATE OLS forecasts, one per
    feature. The literature's top method for low-signal return prediction — averaging cancels the
    idiosyncratic estimation error that sinks a single kitchen-sink regression."""
    class _Comb:
        def fit(self, X, y):
            X = np.asarray(X, float)
            self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd == 0] = 1.0
            Z = (X - self.mu) / self.sd
            ybar = float(np.mean(y))
            # univariate slope of y on each standardized feature (intercept = ybar)
            self.b = np.array([float(np.cov(Z[:, k], y)[0, 1] / (np.var(Z[:, k]) + 1e-12))
                               for k in range(Z.shape[1])])
            self.ybar = ybar
            return self
        def predict(self, X):
            Z = (np.asarray(X, float) - self.mu) / self.sd
            return self.ybar + np.mean(Z * self.b[None, :], axis=1)   # equal-weight avg of univariate fits
    return _Comb()


def _pls(n_components: int = 2):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class _PLS1D:
        def __init__(self):
            self.m = make_pipeline(StandardScaler(), PLSRegression(n_components=n_components))
        def fit(self, X, y):
            self.m.fit(np.asarray(X, float), np.asarray(y, float)); return self
        def predict(self, X):
            return np.asarray(self.m.predict(np.asarray(X, float))).ravel()
    return _PLS1D()


SEARCH_MODELS: dict[str, Callable[[], object]] = {
    "unconditional": _unconditional,
    "ridge": _ridge,
    "elasticnet": _elasticnet,
    "combination": _forecast_combination,
    "pls": _pls,
    "hgb": _hgb,
}


# =============================================================================
# Feature-set enumeration
# =============================================================================
def enumerate_feature_sets(groups: dict[str, list[str]], *, max_family_combo: int = 3) -> dict[str, list[str]]:
    """Feature sets to try: each single family, all families, and every family combo up to
    ``max_family_combo`` (so the search is exhaustive over small combinations without exploding)."""
    fams = list(groups)
    sets: dict[str, list[str]] = {}
    for f in fams:
        sets[f] = list(groups[f])
    for k in range(2, max_family_combo + 1):
        for combo in combinations(fams, k):
            name = "+".join(combo)
            feats: list[str] = []
            for f in combo:
                feats.extend(groups[f])
            sets[name] = feats
    sets["ALL"] = [x for f in fams for x in groups[f]]
    return sets


# =============================================================================
# Walk-forward OOS prediction scored on a target region only
# =============================================================================
def _oos_ic_on_region(
    X: np.ndarray, y: np.ndarray, model_make: Callable[[], object], *,
    region_lo: int, embargo: int, test: int = 189, region_hi: Optional[int] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Purged walk-forward; return (IC, preds, reals) for test blocks fully inside
    ``[region_lo, region_hi)``.

    The upper bound is REQUIRED for the SELECT scoring: without it the selection metric would sweep
    every block through the end of the series — including the HOLDOUT — so the holdout would help
    choose the winner and then "confirm" it (circular). Requiring the WHOLE block inside the region
    (``te.stop <= region_hi``) also prevents a block from straddling the select/holdout boundary.
    """
    preds, reals = [], []
    n = len(y)
    hi = n if region_hi is None else int(region_hi)
    for train_end, te in purged_blocks(n, start_frac=0.0, test=test, embargo=embargo):
        if te.start < region_lo or te.stop > hi:
            continue
        if train_end < 200:
            continue
        m = model_make().fit(X[:train_end], y[:train_end])
        preds.append(np.asarray(m.predict(X[te]), float)); reals.append(y[te])
    if not preds:
        return float("nan"), np.array([]), np.array([])
    p = np.concatenate(preds); r = np.concatenate(reals)
    return spearman(p, r), p, r


@dataclass
class SearchResult:
    horizon: int
    n_combos_tried: int
    baseline_holdout_ic: float
    best_select: dict          # {"combo","model","select_ic"}
    holdout: dict              # {"combo","model","holdout_ic","significant","beats_baseline","deflated_p"}
    leaderboard: list[dict]    # top combos by SELECT ic, with their HOLDOUT ic
    verdict: str
    n_holdout_blocks: int = 0   # ~independent h-day windows in the holdout (power gauge)

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon, "n_combos_tried": self.n_combos_tried,
            "baseline_holdout_ic": _r(self.baseline_holdout_ic),
            "best_select": self.best_select, "holdout": self.holdout,
            "leaderboard": self.leaderboard, "verdict": self.verdict,
            "n_holdout_blocks": self.n_holdout_blocks,
        }


def _r(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 5)


def feature_search(
    panel, horizon: int, *, models: Optional[Sequence[str]] = None,
    select_frac: float = 0.55, holdout_frac: float = 0.78, top_k: int = 5,
    deflation_runs: int = 40, seed: int = 20240703,
) -> SearchResult:
    """Run the multiple-testing-aware search on a :class:`MarketPanel` for one horizon.

    ``select_frac`` / ``holdout_frac`` split the all-finite region into TRAIN ``[0, s)`` used only
    for fitting, SELECT ``[s, h)`` used to CHOOSE the combo, HOLDOUT ``[h, n)`` opened once.
    """
    from qmt.pipeline.features import feature_frame  # noqa: F401 (keep import graph explicit)

    names = panel.names
    idx = {nm: c for c, nm in enumerate(names)}
    # target = forward return on the CSI300 close; align to the panel rows (feature at i-1 -> [i+1, i+1+h])
    C = panel.close
    n = len(C)
    h = max(1, int(horizon))
    Xrows, yv, keep_rows = [], [], []
    for i in range(1, n):
        e = i + 1
        if e + h >= n:
            break
        Xrows.append(panel.X[i - 1]); yv.append(C[e + h] / C[e] - 1.0); keep_rows.append(i)
    from qmt.pipeline.model_bench import _contiguous_finite

    Xall, yall = _contiguous_finite(np.asarray(Xrows), np.asarray(yv))   # refuse interior NaN gaps
    m = len(yall)
    s_lo = int(m * select_frac); h_lo = int(m * holdout_frac)
    # Effective INDEPENDENT holdout observations: an h-day forward window sampled daily overlaps
    # h-fold, so the holdout's real power is ~ (n_holdout / h) blocks. At h=90 this is tiny, so a
    # "no winner" there is absence-of-evidence, not evidence of absence — reported in the verdict.
    n_holdout_blocks = max(1, (m - h_lo) // h)

    sets = enumerate_feature_sets(panel.groups)
    model_names = list(models) if models else ["ridge", "elasticnet", "combination", "pls"]

    def cols(feats: list[str]) -> np.ndarray:
        return Xall[:, [idx[f] for f in feats]]

    # --- selection: score every (combo, model) by OOS IC on the SELECT region ---
    scored = []
    for cname, feats in sets.items():
        Xc = cols(feats)
        for mn in model_names:
            try:
                ic, _p, _r_ = _oos_ic_on_region(Xc, yall, SEARCH_MODELS[mn], region_lo=s_lo, region_hi=h_lo, embargo=h)
            except Exception:
                continue        # e.g. PLS on a 1-feature family; that (combo, model) is simply skipped
            if np.isfinite(ic):
                scored.append({"combo": cname, "model": mn, "select_ic": round(float(ic), 5)})
    scored.sort(key=lambda d: d["select_ic"], reverse=True)
    n_combos = len(scored)

    # Baseline for a rank IC is ZERO rank-skill — not the piecewise-constant unconditional
    # predictor's IC (which is arbitrary: it changes with the block boundaries and is meaningless as
    # a ranking). "Beats baseline" therefore means positive out-of-sample rank skill.
    base_ic = 0.0

    # --- confirm the top-K on the untouched HOLDOUT ---
    leaderboard = []
    for cand in scored[:top_k]:
        Xc = cols(sets[cand["combo"]])
        hic, hp, hr = _oos_ic_on_region(Xc, yall, SEARCH_MODELS[cand["model"]], region_lo=h_lo, embargo=h)
        sig = assess_significance(np.sign(hp - np.median(hp)) * hr, overlap=h) if hp.size else None
        leaderboard.append({
            **cand, "holdout_ic": _r(hic),
            "beats_baseline": bool(np.isfinite(hic) and hic > base_ic),
            "significant": bool(sig.significant) if sig else False,
        })

    # --- multiple-testing deflation: permute the target, re-run selection, record best SELECT IC ---
    # Capacity-matched permutation null: shuffle the target and re-run the SAME search (same set
    # space, same models) — so best-of-null reflects the real search's overfitting capacity, not a
    # smaller one. deflated_p = P(best null SELECT IC >= observed best). Linear models only (the
    # winners are linear; keeps it affordable). A circular BLOCK permutation (block = h) preserves
    # the target's autocorrelation so the null is not artificially easy to beat.
    rng = np.random.default_rng(seed)
    null_best = []
    null_models = [mn for mn in model_names if mn in ("ridge", "elasticnet", "combination", "pls")]
    nblk = int(np.ceil(len(yall) / h))
    for _ in range(deflation_runs):
        starts = rng.integers(0, len(yall), size=nblk)
        perm = (starts[:, None] + np.arange(h)[None, :]).ravel()[:len(yall)] % len(yall)
        yp = yall[perm]
        best = -np.inf
        for feats in sets.values():
            Xc = cols(feats)
            for mn in null_models:
                ic, _p, _rr = _oos_ic_on_region(Xc, yp, SEARCH_MODELS[mn], region_lo=s_lo, region_hi=h_lo, embargo=h)
                if np.isfinite(ic) and ic > best:
                    best = ic
        if np.isfinite(best):
            null_best.append(best)
    null_best = np.array(null_best)
    top_select = scored[0]["select_ic"] if scored else float("nan")
    deflated_p = float(np.mean(null_best >= top_select)) if null_best.size else float("nan")

    winner = leaderboard[0] if leaderboard else {}
    survives = bool(winner and winner["beats_baseline"] and winner["significant"]
                    and np.isfinite(deflated_p) and deflated_p < 0.05)
    holdout = {**winner, "deflated_p": _r(deflated_p)} if winner else {}
    if survives:
        verdict = (f"h={h}: 组合 {winner['combo']}/{winner['model']} 在留出集上跟随趋势更好且经多重检验去偏后仍显著"
                   f"（留出 IC {winner['holdout_ic']}, 去偏 p={_r(deflated_p)}）——采纳候选。")
    else:
        why = []
        if winner and not winner["beats_baseline"]:
            why.append("留出 IC 未超无条件基线")
        if winner and not winner["significant"]:
            why.append("方向性未通过重叠修正显著性")
        if np.isfinite(deflated_p) and deflated_p >= 0.05:
            why.append(f"最优 SELECT IC 在置换零分布下并不罕见（去偏 p={_r(deflated_p)}）")
        # Honest scope: this is a statement about THIS search (these 34 features, these models, this
        # 12-year sample), and the holdout has only ~n_holdout_blocks independent h-day windows, so
        # at long horizons it is absence-of-evidence, not proof no signal exists.
        power = (f"（本次搜索：34 特征 × 这些模型 × 本段样本；留出集仅约 {n_holdout_blocks} 个独立 {h} 日窗口，"
                 f"{'长周期功效很低，属证据不足而非证明无信号' if n_holdout_blocks < 12 else '功效有限'}）")
        verdict = (f"h={h}: 试了 {n_combos} 个(组合×模型)。没有组合在留出集上稳健地跟随趋势更好："
                   + "；".join(why) + "。" + power + "这正是多重检验会制造的假象。")
    return SearchResult(h, n_combos, base_ic,
                        {"combo": scored[0]["combo"], "model": scored[0]["model"], "select_ic": top_select} if scored else {},
                        holdout, leaderboard, verdict, n_holdout_blocks)
