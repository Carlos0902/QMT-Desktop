"""Fund-flow and style-drift leaves — §7.1 hypothesis 2 (风格漂移 × 资金流交互).

Pre-registered operationalization (kept deliberately small; every variant counts
against the manual family's multiple-testing bar):

* ``flow``  — quarterly net-subscription rate: pct-change of disclosed shares
  outstanding between consecutive reports. A report dated ``period_end`` becomes
  usable only at ``available_at = period_end + 25 calendar days`` (fund quarterly
  reports disclose within ~15 working days; ``fund_share`` carries no ``ann_date``,
  and using the period end itself would peek weeks into the future — §3.2).
  The frame is a step function of the LATEST DISCLOSED value, forward-filled with a
  staleness limit so a fund that stops reporting fails closed (NaN), not stale-fresh.

* ``drift`` — style-drift proxy: |beta_fast − beta_slow| from causal rolling betas
  vs the market index. A fund whose short-horizon market exposure diverges from its
  long-horizon exposure is drifting from its own established style.

Hypothesis 2 predicts the *interaction* (drifting + sustained in/outflow) carries
information beyond either alone — wired as the pre-registered ``flow_x_drift``
candidate in :mod:`qmt.factor.baselines`.
"""

from __future__ import annotations

import pandas as pd

from qmt.pit import PointInTimeStore
from qmt.factor.panel import rolling_beta_frame

__all__ = ["build_flow_frame", "build_drift_frame", "FLOW_TABLE", "DISCLOSURE_LAG_DAYS"]

FLOW_TABLE = "fund_share"
DISCLOSURE_LAG_DAYS = 25          # conservative: quarter end -> actually knowable
_SENTINEL = pd.Timestamp("2100-01-01")
# a quarterly series older than ~1.5 quarters (in trading rows) is EXPIRED, not fresh
_STALE_LIMIT_ROWS = 95


def build_flow_frame(
    store: PointInTimeStore,
    codes: list[str],
    ref_index: pd.Index,
) -> pd.DataFrame:
    """(date x fund) latest-disclosed quarterly net-subscription rate.

    Point-in-time: each flow observation is placed at its ``available_at`` (already
    lagged at ingestion), then forward-filled onto the decision grid with a staleness
    cap — never at its period end.
    """
    try:
        raw = store.get_asof(FLOW_TABLE, _SENTINEL)
    except Exception:
        return pd.DataFrame(index=ref_index)
    if raw is None or raw.empty:
        return pd.DataFrame(index=ref_index)
    raw = raw[raw["fund_code"].astype(str).isin(set(map(str, codes)))]
    if raw.empty:
        return pd.DataFrame(index=ref_index)

    ref = pd.DatetimeIndex(pd.to_datetime(list(ref_index)))
    out: dict[str, pd.Series] = {}
    for fc, grp in raw.groupby("fund_code"):
        g = grp.sort_values("period_end")
        shares = pd.to_numeric(g["fd_share"], errors="coerce")
        flow = shares.pct_change(fill_method=None)          # quarter-over-quarter net flow
        avail = pd.to_datetime(g["available_at"]).dt.normalize()
        s = pd.Series(flow.values, index=avail).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            continue
        stepped = s.reindex(ref, method="ffill", limit=_STALE_LIMIT_ROWS)
        out[str(fc)] = stepped
    if not out:
        return pd.DataFrame(index=ref_index)
    frame = pd.DataFrame(out)
    frame.index = ref_index
    return frame


def build_drift_frame(
    price_frame: pd.DataFrame,
    market_px: pd.Series,
    *,
    fast: int = 60,
    slow: int = 250,
) -> pd.DataFrame:
    """|beta_fast − beta_slow| — causal style-drift proxy (date x fund).

    Both betas are backward-looking rolling moments (see
    :func:`qmt.factor.panel.rolling_beta_frame`), so the value at row d uses only
    data ``<= d``.
    """
    b_fast = rolling_beta_frame(price_frame, market_px, window=fast,
                                min_periods=max(20, fast // 2))
    b_slow = rolling_beta_frame(price_frame, market_px, window=slow,
                                min_periods=max(60, slow // 2))
    return (b_fast - b_slow).abs()
