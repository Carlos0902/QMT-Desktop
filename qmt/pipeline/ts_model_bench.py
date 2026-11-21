"""Broad time-series model benchmark for index-return direction (§5 / §7 modeling philosophy).

The user asked to "try different time-series prediction models to find the best-suited one." This
runs a wide ladder — classical Box–Jenkins / exponential-smoothing / Theta, shallow ML, and a
compact (single-config, seeded) deep MLP + LSTM sequence model — (a representative deep attempt at
this scale, not an exhaustive DL search) — against ONE honest question: at a point-in-time decision, does the
model forecast the sign/level of the forward ``h``-day return better than a **random walk**?

Discipline (identical for every model, so the comparison is fair):

* **Point-in-time, purged.** At decision date ``t`` a model sees data only up to ``t - embargo``
  (``embargo = h`` so a training target never overlaps the forecast window). Classical univariate
  models are refit at ``t``; ML/DL are refit on a coarser schedule and reused (still strictly-past).
* **Out-of-sample only**, decisions sampled every ``stride`` days across the OOS region.
* **Prevailing-mean benchmark.** The headline is the **Diebold–Mariano** test of equal squared-error
  accuracy vs the **historical-mean drift** (the Goyal–Welch standard — NOT a strawman ``pred=0``,
  which a model could beat just by predicting the equity premium), plus the directional rank IC via
  the overlap-corrected :func:`qmt.stats.assess_significance`. A model is "best-suited" only if it
  reduces error below the prevailing mean AND ranks direction significantly; the honest, expected
  outcome on a near-random-walk index is that **nothing does** — which is itself the answer.

Nothing here is wired into the forecaster. It is a measurement, reported via ``artifacts/``.

Result on 24y of CSI300 (2002–2026, 5945 bars; ``artifacts/ts_model_bench.json``): **no model beats
the prevailing-mean benchmark at h=3/20/90 — the random walk is best-suited.** Complexity
monotonically *hurts* (MLP squared error 2–8× the benchmark, LSTM 1.3×, GBM DM-t −2 to −3); the
seductive h=90 directional ICs (ridge +0.32) are a ~39-window mirage carrying *higher* squared error.
See DECISIONS D-BENCH-6 / D-BENCH-6a (the latter fixed a DM HAC-lag artifact — day-units on a
decision-sampled series — that had inflated theta h=90 to a spurious dm_t +2.49; corrected to +0.53).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from qmt.stats import assess_significance, spearman

__all__ = ["TSModelScore", "benchmark_timeseries", "CLASSICAL_MODELS", "diebold_mariano"]


# =============================================================================
# Forecasters — each maps (observable_logp, h, gap) -> forecast of the target-window cum log return
# =============================================================================
# ``logp`` is the FULL observable history at the decision (bars up to t-1 — the 14:30 cutoff). The
# target window is ``[t+1, t+1+h]``, i.e. steps ``[gap, gap+h]`` ahead of the last observed bar with
# ``gap = 2`` (the executable-delay: decision sees t-1, executes at t+1). A model forecasts ``gap+h``
# steps and returns the cumulative log return over that future window — NOT a past window (the fit is
# on all observable data; there is NO extra embargo here — the embargo belongs on the ML training
# rows whose *targets* would otherwise overlap, not on a univariate model's forecast origin).
_GAP = 2


def _fc_random_walk(logp: np.ndarray, h: int, gap: int = _GAP) -> float:
    return 0.0                                    # a random walk expects no change (the benchmark)


def _fc_drift(logp: np.ndarray, h: int, gap: int = _GAP) -> float:
    r = np.diff(logp)                             # historical-mean drift over the h-day window
    return float(np.mean(r) * h) if r.size else 0.0


def _window_from_path(path: np.ndarray, last: float, h: int, gap: int) -> float:
    """Cumulative log return over forecast steps ``[gap, gap+h]`` given a forecast path (of log
    prices) whose element ``k`` is the ``(k+1)``-step-ahead forecast, and ``last`` = last observed
    log price. Step ``gap`` price = ``path[gap-1]`` (or ``last`` if ``gap==0``)."""
    end = path[gap + h - 1]
    start = last if gap <= 0 else path[gap - 1]
    return float(end - start)


def _fc_ar(lags: int):
    def f(logp: np.ndarray, h: int, gap: int = _GAP) -> float:
        from statsmodels.tsa.ar_model import AutoReg

        r = np.diff(logp)
        if r.size < lags + 20:
            return 0.0
        try:
            fr = AutoReg(r, lags=lags, old_names=False).fit().forecast(gap + h)  # forecast returns
            return float(np.sum(fr[gap:gap + h]))                                # sum over the window
        except Exception:
            return 0.0
    return f


def _fc_arima(order: tuple[int, int, int]):
    def f(logp: np.ndarray, h: int, gap: int = _GAP) -> float:
        from statsmodels.tsa.arima.model import ARIMA

        if logp.size < 60:
            return 0.0
        try:
            path = np.asarray(ARIMA(logp, order=order).fit().forecast(gap + h))
            return _window_from_path(path, float(logp[-1]), h, gap)
        except Exception:
            return 0.0
    return f


def _fc_ets(damped: bool = True):
    def f(logp: np.ndarray, h: int, gap: int = _GAP) -> float:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if logp.size < 60:
            return 0.0
        try:
            path = np.asarray(ExponentialSmoothing(logp, trend="add", damped_trend=damped).fit().forecast(gap + h))
            return _window_from_path(path, float(logp[-1]), h, gap)
        except Exception:
            return 0.0
    return f


def _fc_theta():
    def f(logp: np.ndarray, h: int, gap: int = _GAP) -> float:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        if logp.size < 60:
            return 0.0
        try:
            P = np.exp(logp)
            path = np.log(np.asarray(ThetaModel(P, period=1).fit().forecast(gap + h)))
            return _window_from_path(path, float(logp[-1]), h, gap)
        except Exception:
            return 0.0
    return f


CLASSICAL_MODELS: dict[str, Callable[[np.ndarray, int], float]] = {
    "random_walk": _fc_random_walk,
    "drift_mean": _fc_drift,
    "ar5": _fc_ar(5),
    "arima_110": _fc_arima((1, 1, 0)),
    "arima_212": _fc_arima((2, 1, 2)),
    "ets_damped": _fc_ets(True),
    "theta": _fc_theta(),
}


# =============================================================================
# Diebold–Mariano test vs the random walk (squared-error loss)
# =============================================================================
def diebold_mariano(err_model: np.ndarray, err_bench: np.ndarray, overlap: int = 1) -> tuple[float, float]:
    """DM statistic (and its Newey–West-corrected value) for H0: equal squared-error accuracy.

    ``d = err_bench^2 - err_model^2`` — positive mean means the model beats the benchmark.

    ``overlap`` is the number of DECISIONS whose forecast windows overlap, i.e. ``ceil(h/stride)``.
    The loss-differential series ``d`` has one observation PER DECISION (decisions are spaced
    ``stride`` trading days apart), so its serial correlation runs to lag ``overlap-1`` in *decision*
    units — that is the correct Newey–West HAC lag. Passing a day-unit lag (``h-1``) onto a
    stride-sampled series grossly over-lags it and manufactures significance at long horizons
    (e.g. h=90, stride=90 → overlap 1, true lag 0, yet h-1=89 clamped to n-1 is a degenerate
    long-run variance on ~39 points). This mirrors the decision-unit overlap the directional gate
    already uses via ``assess_significance``.

    Returns ``(mean_loss_diff, nw_t)``; ``nw_t > 0`` and large ⇒ the model is significantly better.
    """
    d = err_bench ** 2 - err_model ** 2
    d = d[np.isfinite(d)]
    if d.size < 8:
        return float("nan"), float("nan")
    from qmt.stats import newey_west_tstat

    lag = min(max(0, overlap - 1), max(0, d.size - 2))
    return float(d.mean()), float(newey_west_tstat(d, lag))


@dataclass
class TSModelScore:
    name: str
    kind: str
    n: int
    dir_ic: float                # spearman(pred, realized)  — "does it follow the trend" (rank)
    hit_rate: float
    mse: float
    mse_ratio_vs_rw: float       # ratio vs the prevailing-mean benchmark; <1 = lower error
    dm_mean: float               # mean squared-error advantage over the prevailing-mean benchmark
    dm_t: float                  # Newey–West DM t (>~2 ⇒ significantly beats the prevailing mean)
    dir_significant: bool        # overlap-corrected directional significance
    beats_rw: bool

    def as_dict(self) -> dict:
        r = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 5)  # noqa: E731
        return {"name": self.name, "kind": self.kind, "n": self.n, "dir_ic": r(self.dir_ic),
                "hit_rate": r(self.hit_rate), "mse": r(self.mse),
                "mse_ratio_vs_rw": r(self.mse_ratio_vs_rw), "dm_mean": r(self.dm_mean),
                "dm_t": r(self.dm_t), "dir_significant": self.dir_significant,
                "beats_rw": self.beats_rw}


# =============================================================================
# ML / DL forecasters (feature-based). Fit on features up to t-embargo; refit on a schedule.
# =============================================================================
def _ml_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _ml_elasticnet():
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))


def _ml_gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.03,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def _make_mlp():
    import torch
    from torch import nn

    class _MLP:
        def __init__(self, n_in):
            torch.manual_seed(0)
            self.net = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))
            self.mu = None; self.sd = None
        def fit(self, X, y, epochs=120):
            X = np.asarray(X, float); self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            yt = torch.tensor(np.asarray(y, float).reshape(-1, 1), dtype=torch.float32)
            opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-3)
            lossf = nn.MSELoss()
            self.net.train()
            for _ in range(epochs):
                opt.zero_grad(); loss = lossf(self.net(Xt), yt); loss.backward(); opt.step()
            return self
        def predict(self, X):
            import torch as _t
            self.net.eval()
            Xt = _t.tensor((np.asarray(X, float) - self.mu) / self.sd, dtype=_t.float32)
            with _t.no_grad():
                return self.net(Xt).numpy().ravel()
    return _MLP


RETLAG_WIN = 60      # look-back window of daily returns fed to the sequence model


def _make_lstm():
    """Compact LSTM over the recent DAILY-RETURN sequence — the canonical deep *sequence* model, the
    fair representative of LSTM/GRU/TCN/transformer for this task. It consumes a ``(n, RETLAG_WIN)``
    matrix of the last ``RETLAG_WIN`` daily returns (built causally, so it flows through the same
    point-in-time loop as every other model) and reshapes each row into a length-``RETLAG_WIN``
    sequence internally. Kept tiny (1 layer, 16 units) because the honest expectation (the DLinear
    result + only a few thousand samples) is that it overfits and does not beat the random walk; it
    is included so the answer to "did you try deep sequence models" is yes, with evidence."""
    import torch
    from torch import nn

    class _LSTM:
        def __init__(self, _n_in=None):
            torch.manual_seed(0)
            self.net = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
            self.head = nn.Linear(16, 1)
            self.mu = None; self.sd = None
        def fit(self, X, y, epochs=40):
            X = np.asarray(X, float); self.mu = X.mean(); self.sd = X.std() + 1e-9
            Xt = torch.tensor(((X - self.mu) / self.sd)[:, :, None], dtype=torch.float32)
            yt = torch.tensor(np.asarray(y, float).reshape(-1, 1), dtype=torch.float32)
            opt = torch.optim.Adam(list(self.net.parameters()) + list(self.head.parameters()),
                                   lr=1e-3, weight_decay=1e-3)
            lossf = nn.MSELoss(); self.net.train()
            for _ in range(epochs):
                opt.zero_grad(); out, _ = self.net(Xt); pred = self.head(out[:, -1, :])
                lossf(pred, yt).backward(); opt.step()
            return self
        def predict(self, X):
            import torch as _t
            Xt = _t.tensor(((np.asarray(X, float) - self.mu) / self.sd)[:, :, None], dtype=_t.float32)
            self.net.eval()
            with _t.no_grad():
                out, _ = self.net(Xt)
                return self.head(out[:, -1, :]).numpy().ravel()
    return _LSTM


def _return_lag_matrix(logp: np.ndarray, win: int) -> np.ndarray:
    """Causal ``(n, win)`` matrix whose row ``i`` is the last ``win`` daily log returns through bar
    ``i`` (``ret[i-win+1..i]``); rows before enough history are NaN. Feeds the sequence model
    through the identical point-in-time loop as the engineered features."""
    ret = np.empty(logp.size); ret[0] = 0.0; ret[1:] = np.diff(logp)
    out = np.full((logp.size, win), np.nan)
    for i in range(win, logp.size):
        out[i] = ret[i - win + 1: i + 1]
    return out


def benchmark_timeseries(
    prices: np.ndarray, feats: np.ndarray, *, horizon: int, stride: int = 21,
    start_frac: float = 0.4, ml_refit_every: int = 3, include_ml: bool = True,
    include_dl: bool = True,
) -> list[TSModelScore]:
    """Benchmark the model ladder on one horizon. ``feats[i]`` are the causal features known at bar
    ``i`` (a decision at ``i`` uses ``feats[i-1]``); ``prices`` is the aligned close series."""
    P = np.asarray(prices, dtype=float)
    logp = np.log(P)
    n = P.size
    h = max(1, int(horizon))
    embargo = h

    decisions = [t for t in range(int(n * start_frac), n) if t + 1 + h < n and (t % stride == 0)]
    real = np.array([logp[t + 1 + h] - logp[t + 1] for t in decisions])

    scores: list[TSModelScore] = []

    # --- classical univariate: refit per decision on ALL observable data (bars up to t-1) ---
    # NOT logp[:t-h]: a univariate model forecasts FROM the end of its fit window, so an extra
    # h-bar embargo would make it forecast a stale window ending at the decision instead of the
    # future target. The forecaster itself skips the executable-gap (`gap`) so the window it scores
    # is exactly [t+1, t+1+h] — the same target as every other model.
    preds_by_model: dict[str, np.ndarray] = {}
    for name, fc in CLASSICAL_MODELS.items():
        preds = np.array([fc(logp[:t], h) for t in decisions])
        preds_by_model[name] = preds

    # --- ML / DL: refit on an expanding schedule, predict the feature row at t-1 ---
    # Each spec carries its OWN feature matrix, so the deep SEQUENCE model (LSTM) flows through the
    # identical point-in-time loop as the engineered-feature models — its "features" are just the
    # last RETLAG_WIN daily returns (a causal matrix), reshaped to a sequence inside the model.
    if include_ml or include_dl:
        ml_specs = []                                   # (name, kind, make, feature_matrix)
        if include_ml:
            ml_specs += [("ridge", "ml", _ml_ridge, feats),
                         ("elasticnet", "ml", _ml_elasticnet, feats),
                         ("gbm", "ml", _ml_gbm, feats)]
        if include_dl:
            try:
                MLP = _make_mlp()
                ml_specs.append(("mlp_torch", "dl", lambda: MLP(feats.shape[1]), feats))
            except Exception:
                pass
            try:
                LSTM = _make_lstm()
                retlag = _return_lag_matrix(logp, RETLAG_WIN)
                ml_specs.append(("lstm_torch", "dl", LSTM, retlag))   # deep SEQUENCE model
            except Exception:
                pass
        # target realized cum return per bar
        ytab = np.full(n, np.nan)
        for i in range(n):
            if i + 1 + h < n:
                ytab[i] = logp[i + 1 + h] - logp[i + 1]
        for name, kind, make, FM in ml_specs:
            preds = np.full(len(decisions), np.nan)
            model = None; last_fit = -10 ** 9
            for di, t in enumerate(decisions):
                tr_end = t - embargo
                rows = [i for i in range(1, tr_end) if np.all(np.isfinite(FM[i - 1])) and np.isfinite(ytab[i - 1])]
                if len(rows) < 200:
                    continue
                if model is None or (di - last_fit) >= ml_refit_every:
                    Xtr = FM[[r - 1 for r in rows]]; ytr = ytab[[r - 1 for r in rows]]
                    try:
                        model = make().fit(Xtr, ytr)
                        last_fit = di
                    except Exception:
                        model = None; continue
                xrow = FM[t - 1]
                if model is not None and np.all(np.isfinite(xrow)):
                    try:
                        preds[di] = float(np.asarray(model.predict(xrow.reshape(1, -1))).ravel()[0])
                    except Exception:
                        preds[di] = np.nan
            preds_by_model[name] = preds

    # --- score every model vs the PREVAILING-MEAN benchmark (the Goyal-Welch standard) ---
    # NOT vs pred=0: a model that merely predicts the sample's average return would "beat" a
    # zero forecast by capturing the equity premium — that is not following the trend. The honest
    # benchmark to beat is the historical-mean drift; a model only earns "beats_benchmark" if it
    # reduces squared error below the prevailing mean AND ranks direction significantly (both).
    bench = preds_by_model["drift_mean"]
    err_bench = bench - real
    kinds = {**{k: "classical" for k in CLASSICAL_MODELS}, "ridge": "ml", "elasticnet": "ml",
             "gbm": "ml", "mlp_torch": "dl", "lstm_torch": "dl"}
    overlap = max(1, int(math.ceil(h / stride)))
    for name, preds in preds_by_model.items():
        ok = np.isfinite(preds) & np.isfinite(real)
        if ok.sum() < 8:
            continue
        p, y = preds[ok], real[ok]
        err = p - y
        mse = float(np.mean(err ** 2))
        bench_mse = float(np.mean(err_bench[ok] ** 2))
        # DM HAC lag is the DECISION-unit overlap (ceil(h/stride)-1), not the day-unit h-1: `d` is
        # sampled once per decision, `stride` days apart — same overlap the directional gate uses.
        dm_mean, dm_t = diebold_mariano(err, err_bench[ok], overlap)
        ic = spearman(p, y)
        dser = np.sign(p - np.median(p)) * y
        sig = assess_significance(dser, overlap=overlap)
        # Both conditions, so a lower-error-by-luck model or a spuriously-significant IC cannot alone
        # crown a winner: beat the prevailing mean on squared error (DM t>2) AND rank-track direction.
        beats = bool(np.isfinite(dm_t) and dm_t > 2.0 and mse < bench_mse
                     and sig.significant and sig.mean > 0)
        scores.append(TSModelScore(
            name=name, kind=kinds.get(name, "classical"), n=int(ok.sum()),
            dir_ic=ic, hit_rate=float(np.mean(np.sign(p) == np.sign(y))),
            mse=mse, mse_ratio_vs_rw=(mse / bench_mse if bench_mse > 0 else float("nan")),
            dm_mean=dm_mean, dm_t=dm_t, dir_significant=bool(sig.significant),
            beats_rw=beats,
        ))
    scores.sort(key=lambda s: (s.beats_rw, -(s.mse_ratio_vs_rw if np.isfinite(s.mse_ratio_vs_rw) else 9)), reverse=True)
    return scores
