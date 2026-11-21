"""S5 影子运行 — shadow run (instruction §6.2 stage 5; tech doc roadmap S5).

Before any capital, the system runs on real data under real trading rules and is
watched for: data timeliness, signal stability, cost conformity, execution
constraints, and backtest-vs-live gap. Deliverable: an auditable shadow ledger and
a go/no-go report.

Honesty contract (enforced in code, not prose):
  * The shadow portfolio is a **watchlist diagnostic** — no factor has passed the
    four gates, so nothing here is a deployable strategy or investment advice. The
    watchlist (marginal candidates + a control) is fixed in the ledger config at
    start; changing it mid-run is a new, logged run.
  * The LOCKED test region stays sealed: rehearsal (backfill on already-consumed
    research data, machinery validation only) must mature strictly BEFORE the
    locked span; live snapshots must sit strictly AFTER it. Both are hard checks.
  * Every snapshot and every maturation appends to a JSONL ledger — append-only,
    replayable, auditable (§5.2 spirit).

Lifecycle: ``step(as_of)`` builds the PIT panel at ``as_of``, scores the watchlist,
forms the top-decile watch portfolio, estimates 3-tier costs vs the previous
target, and logs a snapshot. Later, when ``as_of + horizon`` has passed,
``mature(now)`` fills in realized relative returns / rank-ICs for old snapshots and
runs :class:`LiveMonitoring` against expectations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore, TradingCalendar
from qmt.factor import panel as _panel
from qmt.factor.base_factors import BaseFactors
from qmt.factor.dsl import FormulaDSL
from qmt.factor.execution_sim import ExecSim
from qmt.factor.metrics import aligned_pairs
from qmt.factor.types import CostTiers, FundScores, PortfolioState
from qmt.stats import spearman

__all__ = ["ShadowConfig", "ShadowRunner", "DEFAULT_WATCHLIST", "explain_snapshot"]


def explain_snapshot(snap: dict) -> str:
    """把一份快照翻译成普通人能读懂的中文解读。

    合规定位:只做状态诊断与费用估算的解释,不给出买卖建议;明确说明当前
    没有因子通过验证门槛(NO-GO),分位低不等于预测下跌。
    """
    lines: list[str] = []
    lines.append("━━━ 影子快照解读(诊断信息,非投资建议)━━━")
    basis = "你的真实持仓" if snap.get("portfolio_basis") == "user_book" else "上一期的假想目标(未初始化持仓)"
    lines.append(f"快照日 {snap['as_of']} · 打分宇宙 {snap.get('n_scored', '?'):,} 只基金 · 基准:{basis}")

    hr = snap.get("held_review")
    total = None
    if hr:
        total = hr.get("total_value")
        cash_w = hr.get("cash_weight") or 0.0
        lines.append("")
        lines.append(f"你的组合:总资产 ¥{total:,.0f},其中现金约 ¥{total * cash_w:,.0f}({cash_w:.0%})")
        lines.append("")
        lines.append("持仓体检(观察名单综合分,当前主要衡量「近 60 日波动是否偏低」):")
        ranked = sorted(hr["positions"].items(),
                        key=lambda kv: -(kv[1].get("watchlist_percentile") or -1))
        for fc, r in ranked:
            w = r.get("weight")
            p = r.get("watchlist_percentile")
            wtxt = f"权重 {w:.1%}" if w is not None else "权重 ?"
            if p is None:
                lines.append(f"  {fc}  {wtxt}  未参与排名(成立太新或历史不足)")
            else:
                tone = "近期波动相对较低" if p >= 0.5 else "近期波动相对较高"
                lines.append(f"  {fc}  {wtxt}  综合分高于约 {p:.0%} 的基金 —— {tone}")
        lines.append("  (分位低 ≠ 预测会跌;打分只反映因子口径下的相对状态)")

    turn = snap.get("turnover")
    costs = snap.get("cost_by_tier") or {}
    if turn is not None:
        lines.append("")
        lines.append(f"与系统诊断组合的距离:{turn:.0%}(假如完全换成诊断组合,需要变动 {turn:.0%} 的仓位)")
        if costs and total:
            c = {k: costs.get(k, 0.0) * total for k in ("opt", "neu", "con")}
            lines.append(f"  估算换仓成本:乐观 ¥{c['opt']:,.0f} / 中性 ¥{c['neu']:,.0f} / 保守 ¥{c['con']:,.0f}"
                         " —— 大幅换仓的摩擦不可忽视")

    warns = snap.get("warnings") or []
    friendly: list[str] = []
    for w in warns:
        if "coverage collapse" in w:
            name = w.split(":", 1)[1].split(" scored")[0].strip() if ":" in w else "某因子"
            friendly.append(f"资金流数据源滞后约一年,「{name}」的观察暂停(不影响净值类打分)")
        elif "subscription/redemption" in w:
            friendly.append("申购/赎回状态数据缺失 —— 费用估算只包含赎回费档位")
        elif "stale" in w:
            friendly.append(f"数据时效提示:{w}")
    if friendly:
        lines.append("")
        lines.append("数据提示:")
        for f in dict.fromkeys(friendly):
            lines.append(f"  - {f}")

    lines.append("")
    primary = snap.get("primary")
    if primary:
        lines.append(f"结论:你已选择采用「{primary}」作为主因子(用户决定,已入账)。")
        lines.append("提示:该因子未通过平台的全部验证门槛(统计上不能排除无效应,且历史上")
        lines.append("属于 2021 年后的状态现象)—— 监控层会持续跟踪其实盘表现,衰减即报警。")
        lines.append("以上仍是状态诊断,不构成买卖建议。")
    else:
        lines.append("结论:目前没有任何因子通过平台的验证门槛(NO-GO)—— 系统不认为自己")
        lines.append("有可靠的选基能力;以上是状态诊断,不构成买卖建议。")
    lines.append("下一步:约 40 个交易日后运行 qmt factor shadow-mature,即可看到本次快照的实际结果。")
    return "\n".join(lines)

# Pre-registered watchlist: the two marginal hyp-2 candidates + one control.
# Fixed at run start; recorded in the ledger config (§7.3 — no silent mutation).
DEFAULT_WATCHLIST: dict[str, str] = {
    "flow_x_drift": "rank(mul(flow, drift))",
    "inflow": "rank(ts_mean(flow, 20))",
    "low_vol_60": "rank(neg(ts_std(ret, 60)))",   # control/reference
}


@dataclass
class ShadowConfig:
    ledger_dir: str
    watchlist: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_WATCHLIST))
    horizon_days: int = 40                     # maturation window (trading rows)
    top_frac: float = 0.10                     # watch portfolio = top decile...
    max_names: int = 50                        # ...capped at this many names — a top-decile
                                               # of a 12k universe would be 1,200+ funds,
                                               # meaningless as a human-readable diagnostic
    max_weight: float = 0.05                   # single-fund cap in the watch portfolio
    universe_min_history: str = "2024-12-31"   # funds with NAV history from before 2025
    locked_start: str = "2024-12-10"           # sealed final-test region (do not touch)
    locked_end: str = "2026-07-08"
    stale_nav_warn_days: int = 5               # data-timeliness threshold
    trailing_rows: int = 320                   # history kept for a snapshot: covers the
                                               # longest window (beta 250) + buffer; causally
                                               # identical for last-row evaluation, keeps a
                                               # 14k-fund universe in memory bounds
    primary: str | None = None                 # USER-OVERRIDE adopted factor: when set, the
                                               # watch portfolio is built from this factor
                                               # alone. Gates keep being computed and logged;
                                               # adoption despite gate-fail is an explicit,
                                               # audited user decision (ledger user_override).

    def locked(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.locked_start), pd.Timestamp(self.locked_end)

    @classmethod
    def load(cls, ledger_dir: str | Path) -> "ShadowConfig":
        """Construct from the persisted config.json when present (so overrides like
        ``primary`` and ``universe_min_history`` survive across CLI invocations)."""
        cfg = cls(ledger_dir=str(ledger_dir))
        p = Path(ledger_dir) / "config.json"
        if p.exists():
            try:
                stored = json.loads(p.read_text(encoding="utf-8"))
                for f in ("watchlist", "horizon_days", "top_frac", "max_names", "max_weight",
                          "universe_min_history", "locked_start", "locked_end",
                          "stale_nav_warn_days", "trailing_rows", "primary"):
                    if f in stored and stored[f] is not None:
                        setattr(cfg, f, stored[f])
            except Exception:
                pass  # fall back to defaults; config.json is a record, not a hard dep
        return cfg


class ShadowRunner:
    """Steps a shadow run over a PIT store; every action appends to the ledger."""

    def __init__(self, store: PointInTimeStore, config: ShadowConfig,
                 calendar: TradingCalendar | None = None) -> None:
        self.store = store
        self.cfg = config
        self.cal = calendar or TradingCalendar()
        self.dir = Path(config.ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._ledger = self.dir / "ledger.jsonl"
        self._config_file = self.dir / "config.json"
        if not self._config_file.exists():
            self._config_file.write_text(
                json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str)
            )
        self.dsl = FormulaDSL()

    # ------------------------------------------------------------------ #
    # data assembly (shared by step/mature)
    # ------------------------------------------------------------------ #
    def _universe(self) -> list[str]:
        nav = self.store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
        nav["nav_date"] = pd.to_datetime(nav["nav_date"])
        per = nav.groupby("fund_code")["nav_date"].agg(["min"])
        return sorted(per[per["min"] <= pd.Timestamp(self.cfg.universe_min_history)].index)

    def _price_frame(self, codes: list[str]) -> pd.DataFrame:
        return _panel.usable_price_frame(
            self.store, "fund_nav", value_col="adj_nav", code_col="fund_code",
            date_col="nav_date", codes=codes, calendar=self.cal,
        )

    def _leaves(self, codes: list[str], pf: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Optional extra leaves; each fails soft (a missing source just drops a leaf)."""
        extra: dict[str, pd.DataFrame] = {}
        try:
            mkt = _panel.usable_price_frame(
                self.store, "index_daily", value_col="close", code_col="ts_code",
                date_col="trade_date", calendar=self.cal,
            )
            if "000300.SH" in mkt.columns:
                market_px = mkt["000300.SH"]
                extra["beta"] = _panel.rolling_beta_frame(pf, market_px)
                from qmt.factor.flows import build_drift_frame
                extra["drift"] = build_drift_frame(pf, market_px)
        except Exception:
            pass
        try:
            from qmt.factor.flows import build_flow_frame
            fl = build_flow_frame(self.store, codes, pf.index)
            if not fl.empty and fl.notna().any().any():
                extra["flow"] = fl
        except Exception:
            pass
        return extra

    def _data_staleness(self, as_of: date) -> dict[str, int | None]:
        """Staleness per table at ``as_of``, in days.

        For fund-level tables the metric is the MEDIAN over funds of each fund's own
        latest ``available_at`` — a table-level max hides per-fund staleness (S5
        finding: fund_share's table max looked 4d fresh while 63% of funds had no
        report for ~a year).
        """
        out: dict[str, int | None] = {}
        ts = pd.Timestamp(as_of)
        for table in ("fund_nav", "index_daily", "sw_daily", "fund_share", "fund_portfolio"):
            try:
                df = self.store.get_asof(table, ts)
                if df is None or df.empty:
                    out[table] = None
                    continue
                avail = pd.to_datetime(df["available_at"])
                if "fund_code" in df.columns:
                    per_fund = avail.groupby(df["fund_code"]).max()
                    out[table] = int((ts - per_fund).dt.days.median())
                else:
                    out[table] = int((ts - avail.max()).days)
            except Exception:
                out[table] = None
        return out

    # ------------------------------------------------------------------ #
    # step — one shadow decision
    # ------------------------------------------------------------------ #
    def step(self, as_of: date, *, mode: str = "live") -> dict[str, Any]:
        """Take one shadow snapshot at ``as_of``. ``mode``: 'live' | 'rehearsal'.

        Live snapshots must be strictly AFTER the locked region; rehearsal
        snapshots must MATURE strictly BEFORE it (both enforced).
        """
        lo, hi = self.cfg.locked()
        ts = pd.Timestamp(as_of)
        if mode == "live":
            if ts <= hi:
                raise ValueError(
                    f"live shadow step at {as_of} would touch data at/before the locked "
                    f"region end {hi.date()} — locked region must stay sealed."
                )
        elif mode == "rehearsal":
            # maturation window (trading rows ≈ 1.6x calendar) must end before locked start
            approx_mature = ts + pd.Timedelta(days=int(self.cfg.horizon_days * 1.8))
            if approx_mature >= lo:
                raise ValueError(
                    f"rehearsal at {as_of} matures at ~{approx_mature.date()}, reaching the "
                    f"locked region starting {lo.date()} — pick an earlier date."
                )
        else:
            raise ValueError(f"unknown mode {mode!r}")

        codes = self._universe()
        pf = self._price_frame(codes)
        pf_cut = pf.loc[pf.index <= as_of]
        if pf_cut.empty:
            raise ValueError(f"no usable prices at {as_of}")
        if len(pf_cut) > self.cfg.trailing_rows:
            pf_cut = pf_cut.iloc[-self.cfg.trailing_rows:]
        from qmt.factor.types import SafePanel
        sp = SafePanel(as_of=as_of, frame=pf_cut, universe=list(pf_cut.columns))
        bpanel = BaseFactors(extra=self._leaves(codes, pf_cut)).compute_base_factors(sp, as_of)

        # score the (fixed) watchlist
        factor_scores: dict[str, dict[str, Any]] = {}
        zs: dict[str, pd.Series] = {}
        blend: pd.Series | None = None
        for name, text in self.cfg.watchlist.items():
            try:
                expr = self.dsl.parse(text)
                cs = self.dsl.evaluate(expr, bpanel, as_of)
            except Exception as exc:
                factor_scores[name] = {"n": 0, "error": str(exc)[:120]}
                continue
            cs = cs.dropna()
            factor_scores[name] = {
                "n": int(len(cs)),
                "mean": float(cs.mean()) if len(cs) else None,
                "std": float(cs.std()) if len(cs) else None,
            }
            z = (cs - cs.mean()) / (cs.std() if cs.std() else 1.0)
            zs[name] = z
            blend = z if blend is None else blend.add(z, fill_value=0.0)

        if blend is None or blend.dropna().empty:
            raise ValueError("watchlist produced no scores — check leaves/data")
        blend = blend.dropna()
        # user-override primary: the watch portfolio follows ONE adopted factor;
        # the full watchlist keeps being scored/matured for evidence accumulation.
        if self.cfg.primary and self.cfg.primary in zs and not zs[self.cfg.primary].dropna().empty:
            blend = zs[self.cfg.primary].dropna()

        # watch portfolio: top decile capped at max_names, equal weight, single-fund cap
        k = max(5, min(self.cfg.max_names, int(len(blend) * self.cfg.top_frac)))
        ranked_scores = blend.sort_values(ascending=False)
        top = ranked_scores.head(k)
        w = pd.Series(1.0 / len(top), index=top.index).clip(upper=self.cfg.max_weight)
        w = w / w.sum()

        # execution simulation basis (turnover-aware, 3 cost tiers):
        # user portfolio book if initialized (real weights + real hold_days), else
        # the previous snapshot's target (pure-shadow fallback).
        from qmt.factor.portfolio import load_book, mark_to_market, to_portfolio_state
        book = load_book(self.dir)
        held_review: dict[str, Any] | None = None
        if book is not None:
            pstate = to_portfolio_state(book, self.store, as_of)
            basis = "user_book"
            mtm = mark_to_market(book, self.store, as_of)
            # 持仓体检:每只真实持仓在观察名单综合分里的截面分位(诊断,非建议)
            pct = blend.rank(pct=True)
            held_review = {
                "total_value": mtm["total_value"],
                "cash_weight": mtm["cash_weight"],
                "positions": {
                    fc: {
                        "weight": r["weight"],
                        "hold_days": r["hold_days"],
                        "watchlist_percentile": (round(float(pct[fc]), 3) if fc in pct.index else None),
                    }
                    for fc, r in mtm["positions"].items()
                },
            }
        else:
            prev = self._last_snapshot()
            prev_w = pd.Series(prev["target_weights"]) if prev else pd.Series(dtype=float)
            pstate = PortfolioState(as_of=as_of, holdings=prev_w,
                                    hold_days={c: 30 for c in prev_w.index})  # 【推断】
            basis = "prev_target"
        exec_res = ExecSim().simulate(
            FundScores(as_of=as_of, scores=blend, weights=w), pstate, as_of, CostTiers(),
        )

        staleness = self._data_staleness(as_of)
        warns: list[str] = []
        nav_stale = staleness.get("fund_nav")
        if nav_stale is not None and nav_stale > self.cfg.stale_nav_warn_days:
            warns.append(f"fund_nav stale {nav_stale}d > {self.cfg.stale_nav_warn_days}d")
        # coverage-collapse guard: a watchlist factor scoring far fewer names than the
        # universe means its data source has gone stale live (fail-closed leaves) —
        # its accumulated evidence pauses rather than silently thinning.
        for name, st in factor_scores.items():
            n = st.get("n", 0) or 0
            if n < max(50, int(0.1 * len(blend))):
                warns.append(f"coverage collapse: {name} scored only {n} funds — "
                             "its data source is stale at the live edge; evidence "
                             "accumulation for this factor is effectively paused")
        warns.append("subscription/redemption status unavailable (DATA_GAPS) — "
                     "execution-restriction checks limited to redemption fees")

        snap = {
            "kind": "snapshot",
            "mode": mode,
            "as_of": str(as_of),
            "n_universe": len(codes),
            "n_scored": int(len(blend)),
            "primary": self.cfg.primary,       # user-override adopted factor (None=blend)
            "watch_top": [str(c) for c in ranked_scores.head(10).index],  # ordered, for advice
            "portfolio_basis": basis,          # user_book | prev_target
            "held_review": held_review,        # 真实持仓的截面体检(诊断,非建议)
            "data_staleness_days": staleness,
            "factor_scores": factor_scores,
            "target_weights": {str(k_): round(float(v), 6) for k_, v in w.items()},
            "turnover": round(float(exec_res.turnover), 4),
            "cost_by_tier": {k_: round(float(v), 6) for k_, v in exec_res.cost_by_tier.items()},
            "warnings": warns,
            "matured": False,
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._append(snap)
        return snap

    # ------------------------------------------------------------------ #
    # mature — settle old snapshots whose horizon has passed
    # ------------------------------------------------------------------ #
    def mature(self, now: date) -> list[dict[str, Any]]:
        """Realize outcomes for snapshots whose ``as_of + horizon`` <= now."""
        entries = self._read_ledger()
        snaps = [e for e in entries if e["kind"] == "snapshot" and not e.get("matured")]
        if not snaps:
            return []
        codes = self._universe()
        pf = self._price_frame(codes)
        pf_now = pf.loc[pf.index <= now]
        idx = list(pf_now.index)
        results: list[dict[str, Any]] = []
        matured_ids: list[str] = []
        for snap in snaps:
            s_date = pd.Timestamp(snap["as_of"]).date()
            pos0 = int(np.searchsorted(np.asarray(pf_now.index), s_date, side="right")) - 1
            pos1 = pos0 + self.cfg.horizon_days
            if pos0 < 0 or pos1 >= len(idx):
                continue                     # not yet matured
            entry = pf_now.iloc[pos0]
            exit_ = pf_now.iloc[pos0:pos1 + 1].ffill().iloc[-1]
            fwd = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
            rel = fwd - fwd.mean()
            w = pd.Series(snap["target_weights"])
            port_rel = float((rel.reindex(w.index).fillna(0.0) * w).sum())
            # realized rank-IC per watched factor needs the stored cross-section;
            # we re-evaluate PIT at the snapshot date (identical by causality)
            realized_ic: dict[str, float | None] = {}
            try:
                sp = None
                from qmt.factor.types import SafePanel
                pf_cut = pf.loc[pf.index <= s_date]
                sp = SafePanel(as_of=s_date, frame=pf_cut, universe=list(pf_cut.columns))
                bpanel = BaseFactors(extra=self._leaves(codes, pf_cut)).compute_base_factors(sp, s_date)
                for name, text in self.cfg.watchlist.items():
                    try:
                        cs = self.dsl.evaluate(self.dsl.parse(text), bpanel, s_date)
                        a, b = aligned_pairs(cs, rel)
                        realized_ic[name] = float(spearman(a, b)) if a.size >= 10 else None
                    except Exception:
                        realized_ic[name] = None
            except Exception:
                realized_ic = {name: None for name in self.cfg.watchlist}

            neu_cost = float(snap.get("cost_by_tier", {}).get("neu", 0.0))
            rec = {
                "kind": "maturation",
                "snapshot_as_of": snap["as_of"],
                "matured_at": str(now),
                "portfolio_rel_return_gross": round(port_rel, 6),
                "portfolio_rel_return_net_neu": round(port_rel - neu_cost, 6),
                "realized_rank_ic": {k_: (round(v, 4) if v is not None else None)
                                     for k_, v in realized_ic.items()},
                "written_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._append(rec)
            results.append(rec)
            matured_ids.append(snap["as_of"])
        if matured_ids:
            self._mark_matured(matured_ids)
        return results

    # ------------------------------------------------------------------ #
    # report — the S5 deliverable
    # ------------------------------------------------------------------ #
    def report(self) -> dict[str, Any]:
        entries = self._read_ledger()
        snaps = [e for e in entries if e["kind"] == "snapshot"]
        mats = [e for e in entries if e["kind"] == "maturation"]
        if not snaps:
            return {"status": "empty", "note": "no snapshots yet"}

        stale = [e["data_staleness_days"].get("fund_nav") for e in snaps
                 if e.get("data_staleness_days", {}).get("fund_nav") is not None]
        turnovers = [e["turnover"] for e in snaps]
        ic_series: dict[str, list[float]] = {}
        for m in mats:
            for k_, v in (m.get("realized_rank_ic") or {}).items():
                if v is not None:
                    ic_series.setdefault(k_, []).append(v)
        ic_summary = {
            k_: {"n": len(v), "mean": round(float(np.mean(v)), 4),
                 "positive_share": round(float(np.mean([x > 0 for x in v])), 2)}
            for k_, v in ic_series.items()
        }
        net_neu = [m["portfolio_rel_return_net_neu"] for m in mats]
        go_no_go = (
            "NO-GO for capital: no factor has passed the four gates; shadow run is "
            "watchlist evidence accumulation only. Re-assess when a watchlist factor's "
            "accumulated live rank-IC clears its deflated significance bar."
        )
        return {
            "n_snapshots": len(snaps),
            "n_matured": len(mats),
            "modes": sorted({e["mode"] for e in snaps}),
            "span": [snaps[0]["as_of"], snaps[-1]["as_of"]],
            "data_timeliness": {
                "nav_staleness_days_median": (float(np.median(stale)) if stale else None),
                "warnings": sorted({w for e in snaps for w in e.get("warnings", [])}),
            },
            "signal_stability": ic_summary,
            "cost_conformity": {
                "median_turnover": round(float(np.median(turnovers)), 4),
                "median_cost_neu": round(float(np.median(
                    [e["cost_by_tier"].get("neu", 0.0) for e in snaps])), 6),
            },
            "shadow_vs_backtest": {
                "matured_net_neu_mean": (round(float(np.mean(net_neu)), 6) if net_neu else None),
                "note": "backtest expectation ~0 (no gate-passing factor); any large gap is anomalous",
            },
            "go_no_go": go_no_go,
        }

    # ------------------------------------------------------------------ #
    # verdict — 自裁决:累计实盘 IC 有没有穿越显著性门槛
    # ------------------------------------------------------------------ #
    def verdict(self) -> dict[str, Any]:
        """把已结算(matured)的快照按因子累计实盘 rank-IC,做通胀显著性检验。

        这是「路 2」的自裁决核心:随时间积累的、真正样本外的实盘证据,
        自动判断每个观察因子(尤其用户采用的 primary)有没有开始赚它的钱。
        门槛按累计观测数通胀(sqrt(2·ln n))——观测越多、门槛越合理,不靠单点侥幸。
        """
        from qmt.stats import assess_significance
        entries = self._read_ledger()
        # 只认 LIVE 模式(封存区之后、真正样本外)的结算 —— rehearsal 是机器验证,
        # 不是实盘证据,绝不能计入裁决(否则用排练数据给因子背书,自欺)。
        live_dates = {e["as_of"] for e in entries
                      if e["kind"] == "snapshot" and e.get("mode") == "live"}
        mats = [e for e in entries if e["kind"] == "maturation"
                and e.get("snapshot_as_of") in live_dates]
        primary = self.cfg.primary
        # 按 snapshot 日期去重(同日多次快照只计一次,避免 demo 重复夸大 n)
        by_date: dict[str, dict] = {}
        for m in mats:
            by_date[m.get("snapshot_as_of", m.get("matured_at"))] = m
        uniq = list(by_date.values())

        factors: dict[str, Any] = {}
        for name in self.cfg.watchlist:
            ics = [m["realized_rank_ic"].get(name) for m in uniq
                   if m.get("realized_rank_ic", {}).get(name) is not None]
            if len(ics) < 2:
                factors[name] = {"n": len(ics), "status": "insufficient",
                                 "note": "结算样本不足,继续积累"}
                continue
            arr = np.asarray(ics, dtype=float)
            bar = max(2.0, float(np.sqrt(2.0 * np.log(max(len(arr), 2)))))
            sig = assess_significance(arr, overlap=1, min_abs_t=bar)
            factors[name] = {
                "n": len(arr), "mean_ic": round(float(arr.mean()), 4),
                "positive_share": round(float(np.mean(arr > 0)), 2),
                "tstat": round(sig.t_newey_west, 2), "bar": round(bar, 2),
                "significant": bool(sig.significant),
                "status": "earning" if sig.significant else "accumulating",
            }
        return {
            "n_matured_dates": len(uniq),
            "primary": primary,
            "factors": factors,
            "note": "实盘样本外证据;门槛按累计观测数通胀。四道门 NO-GO 在此因子过门前不变。",
        }

    def explain_verdict(self) -> str:
        v = self.verdict()
        lines = ["━━━ 影子证据自裁决(实盘样本外,非投资建议)━━━"]
        n = v["n_matured_dates"]
        lines.append(f"已结算的实盘快照日:{n} 个")
        if n < 2:
            lines.append("样本还太少,无法裁决 —— 第一批实盘快照约在 2026-09 中旬(下单后 40 交易日)成熟。")
            lines.append("每周跑 qmt factor shadow-monitor,系统会自动积累并在够格时通知你。")
            return "\n".join(lines)
        prim = v["primary"]
        for name, r in v["factors"].items():
            tag = "★采用" if name == prim else "  "
            if r.get("status") == "insufficient":
                lines.append(f"{tag} {name}:结算样本 {r['n']} 个,不足以裁决")
                continue
            verdict_word = ("已达显著(可考虑进入正式四道门复核)" if r["significant"]
                            else f"仍在积累(t={r['tstat']:+.2f} < 门槛 {r['bar']})")
            lines.append(f"{tag} {name}:实盘 IC 均值 {r['mean_ic']:+.3f}、正率 {r['positive_share']:.0%}、"
                         f"n={r['n']} → {verdict_word}")
        lines.append("")
        prim_r = v["factors"].get(prim, {}) if prim else {}
        if prim and prim_r.get("significant"):
            lines.append(f"结论:你采用的「{prim}」实盘 IC 已穿越通胀显著门槛 —— 建议做一次正式四道门复核,"
                         "并考虑是否动用封存区做最终确认。")
        elif prim:
            lines.append(f"结论:你采用的「{prim}」仍在积累证据,尚未达显著 —— 继续持有观察,时间会给答案;"
                         "封存区继续封存,不烧。")
        else:
            lines.append("结论:未设采用因子;各观察因子证据仍在积累。")
        return "\n".join(lines)

    def monitor(self, now: date) -> dict[str, Any]:
        """路 2 一键闭环:若数据推进则补拍快照 → 结算到期快照 → 输出自裁决。"""
        stepped = None
        last = self._last_snapshot()
        last_date = pd.Timestamp(last["as_of"]).date() if last else None
        lo, hi = self.cfg.locked()
        # 只在数据推进、且在封存区之后(live)时补拍;否则只结算+裁决
        if pd.Timestamp(now) > hi and (last_date is None or now > last_date):
            try:
                stepped = self.step(now, mode="live")
            except ValueError:
                stepped = None
        matured = self.mature(now)
        return {"stepped": bool(stepped), "n_matured_now": len(matured),
                "verdict": self.verdict()}

    # ------------------------------------------------------------------ #
    # plain-language interpretation (诊断解读,非投资建议)
    # ------------------------------------------------------------------ #
    def explain_last(self) -> str:
        snaps = [e for e in self._read_ledger() if e["kind"] == "snapshot"]
        if not snaps:
            return "账本里还没有快照 — 先运行 qmt factor shadow-step。"
        return explain_snapshot(snaps[-1])

    # ------------------------------------------------------------------ #
    # ledger plumbing (append-only)
    # ------------------------------------------------------------------ #
    def _append(self, rec: dict[str, Any]) -> None:
        with self._ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _read_ledger(self) -> list[dict[str, Any]]:
        if not self._ledger.exists():
            return []
        out = []
        for line in self._ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def _last_snapshot(self) -> dict[str, Any] | None:
        snaps = [e for e in self._read_ledger() if e["kind"] == "snapshot"]
        return snaps[-1] if snaps else None

    def _mark_matured(self, snapshot_dates: list[str]) -> None:
        """Rewrite ledger marking snapshots matured (the one permitted mutation —
        a flag flip; original rows retained verbatim otherwise)."""
        entries = self._read_ledger()
        for e in entries:
            if e["kind"] == "snapshot" and e["as_of"] in snapshot_dates:
                e["matured"] = True
        tmp = self._ledger.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        tmp.replace(self._ledger)
