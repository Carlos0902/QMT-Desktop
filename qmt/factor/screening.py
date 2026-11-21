"""factor_screening — 两步因子筛选 (instruction §5.4 / §7.5; tech doc §sec:mod-screening).

因子从 ``auto_search`` 产出后不是马上入库，而要过**两步**闸门：

  第一步 单因子是否有用 (:meth:`Screening.score_standalone`)
      在 ``split.as_of_series`` 的每个决策点上用 ``dsl.evaluate`` 求本因子横截面，
      与 ``build_labels`` 的前向相对收益对齐，交给 :func:`metrics.ic_summary`
      得到 rank-IC 与重叠修正后的 t 值；再用 :func:`metrics.group_monotonicity`
      看分组收益是否单调、:func:`metrics.rank_turnover` 估换手，并在三档成本假设下
      给出净收益 (成本随换手升高，保守档 > 中性档 > 乐观档)。regime_stable 把
      验证期切成前后两半，看 rank-IC 符号是否一致 (不是只在某一段有效)。

  第二步 组合增量 (:meth:`Screening.score_incremental`)
      逐点求每个已有因子的横截面，与本因子做横截面相关，取最大者为
      ``max_corr_existing``；相关过高即 ``redundant`` (现有因子的复杂重写)。
      ``increment`` 用简单口径：把本因子对已有因子做横截面残差化后，残差是否仍有
      正的 rank-IC —— 即在已有因子之外是否还带来独立方向。

诚实边界：这里只判「是否值得进组合」，不承诺预测力；黑箱只作基线，核心是公式因子。
点时点安全由 ``dsl.evaluate`` 保证 (在 ``as_of`` 只读 ``<= as_of`` 的行)。
"""

from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from qmt.factor import gates as _gates
from qmt.factor import metrics as _metrics
from qmt.factor.dsl import FormulaDSL
from qmt.factor.gates import GateConfig
from qmt.factor.panel import labels_by_date as _labels_by_date
from qmt.factor.types import (
    BaseFactorPanel,
    CostComponents,
    CostTier,
    CostTiers,
    FactorExpr,
    FactorScore,
    FactorScreening,
    Label,
    ValidationSplit,
)
from qmt.stats import spearman

__all__ = ["Screening"]

# 三档成本按此顺序计算，保证净收益 opt >= neu >= con (成本档单增)。
_TIER_ORDER = (CostTier.OPT, CostTier.NEU, CostTier.CON)


class Screening(FactorScreening):
    """具体实现 :class:`qmt.factor.types.FactorScreening` (instruction §5.4)。"""

    def __init__(
        self,
        dsl: FormulaDSL,
        gate_config: GateConfig | None = None,
        stride_days: int = 5,
        min_abs_t: float = 2.0,
    ) -> None:
        self.dsl = dsl
        self.gate_config = gate_config or GateConfig()
        self.stride_days = int(stride_days)
        # significance bar; the pipeline raises this above 2.0 as the search tries more
        # formulas (multiple-testing / deflated significance, §7.3/§4.3).
        self.min_abs_t = float(min_abs_t)

    # ------------------------------------------------------------------
    # 第一步：单因子是否有用
    # ------------------------------------------------------------------
    def score_standalone(
        self,
        expr: FactorExpr,
        panel: BaseFactorPanel,
        labels: Sequence[Label],
        split: ValidationSplit,
        costs: CostTiers,
    ) -> FactorScore:
        """单因子 OOS、成本净、验证期内的有用性评分 (instruction §5.4 第一步)。"""
        as_of_dates = list(split.as_of_series)
        horizon = self._horizon_of(labels)

        scores_by_date = self._scores_by_date(expr, panel, as_of_dates)
        lbd = _labels_by_date(labels)

        ic = _metrics.ic_summary(
            scores_by_date,
            lbd,
            horizon_days=horizon,
            stride_days=self.stride_days,
            min_abs_t=self.min_abs_t,
        )
        means, monotonic = _metrics.group_monotonicity(scores_by_date, lbd)
        turnover = _metrics.rank_turnover(scores_by_date)
        gross = self._long_short_gross(means)
        net_by_tier = self._net_return_by_tier(gross, turnover, costs)
        regime_stable = self._regime_stable(scores_by_date, lbd)

        score = FactorScore(
            factor_id=expr.text,
            ic=ic.mean_ic,
            rank_ic=ic.mean_rank_ic,
            ic_tstat=ic.tstat,
            ic_significant=ic.significant,
            monotonic=monotonic,
            turnover=turnover,
            net_return_by_tier=net_by_tier,
            regime_stable=regime_stable,
        )
        score.gate_report = _gates.evaluate_gates(score, self.gate_config)
        return score

    # ------------------------------------------------------------------
    # 第二步：组合增量
    # ------------------------------------------------------------------
    def score_incremental(
        self,
        expr: FactorExpr,
        existing: Sequence[FactorExpr],
        panel: BaseFactorPanel,
        labels: Sequence[Label],
        split: ValidationSplit,
    ) -> FactorScore:
        """本因子在已有因子之外是否仍带来独立方向 (instruction §5.4 第二步)。"""
        as_of_dates = list(split.as_of_series)
        horizon = self._horizon_of(labels)

        own = self._scores_by_date(expr, panel, as_of_dates)
        lbd = _labels_by_date(labels)

        # 自身 rank-IC (供 increment 无已有因子时的退化口径)
        own_ic = _metrics.ic_summary(
            own, lbd, horizon_days=horizon, stride_days=self.stride_days
        ).mean_rank_ic

        # 与每个已有因子的横截面相关，取最大者
        existing_scores = [
            self._scores_by_date(e, panel, as_of_dates) for e in existing
        ]
        max_corr = 0.0
        for es in existing_scores:
            c = abs(self._mean_cross_corr(own, es))
            if c > max_corr:
                max_corr = c

        redundant = max_corr >= self.gate_config.max_corr_existing

        # increment：把本因子对已有因子横截面残差化，残差是否仍有正 rank-IC
        if existing_scores:
            resid = self._residualize(own, existing_scores)
            increment = _metrics.ic_summary(
                resid, lbd, horizon_days=horizon, stride_days=self.stride_days
            ).mean_rank_ic
        else:
            increment = own_ic

        score = FactorScore(
            factor_id=expr.text,
            rank_ic=own_ic,
            max_corr_existing=float(max_corr),
            increment=float(increment),
            redundant=bool(redundant),
        )
        score.gate_report = _gates.evaluate_gates(score, self.gate_config)
        return score

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _scores_by_date(
        self, expr: FactorExpr, panel: BaseFactorPanel, dates: Sequence[date]
    ) -> dict[date, pd.Series]:
        """逐决策点取本因子横截面。

        快速路径：DSL 算子全部因果（ts 只看过去、截面算子只看当行），故全帧求值一次
        再按行读取，与逐日 ``evaluate`` 数学等价（点时点安全不变），把 O(dates×history)
        降为 O(history)。无 ``evaluate_frame`` 的 DSL 退回逐日求值。
        """
        if hasattr(self.dsl, "evaluate_frame"):
            try:
                frame = self.dsl.evaluate_frame(expr, panel)
                return _metrics.scores_by_date_from_frame(frame, list(dates))
            except Exception:
                pass  # fall back to the per-date path below
        out: dict[date, pd.Series] = {}
        for d in dates:
            try:
                s = self.dsl.evaluate(expr, panel, d)
            except Exception:
                continue  # same tolerance as the fast path — skip, never crash mid-screen
            if s is not None and not s.empty:
                out[d] = s
        return out

    @staticmethod
    def _horizon_of(labels: Sequence[Label]) -> int:
        for lab in labels:
            return int(lab.horizon_days)
        return 40  # 默认值 (§3.4)

    @staticmethod
    def _long_short_gross(means: Sequence[float]) -> float:
        """分组多空毛收益 = 最高组均值 - 最低组均值 (来自 group_monotonicity)。"""
        vals = [m for m in means if m is not None and not np.isnan(m)]
        if len(vals) < 2:
            return 0.0
        return float(vals[-1] - vals[0])

    @staticmethod
    def _rate_sum(cc: CostComponents) -> float:
        """一档成本的近似费率之和；按开关递增，保证 opt < neu < con。"""
        s = 0.0
        if cc.commission:
            s += cc.commission_rate
        if cc.spread == "partial":
            s += cc.spread_rate * 0.5
        elif cc.spread == "full":
            s += cc.spread_rate
        if cc.slippage:
            s += cc.slippage_rate
        if cc.market_impact == "simple":
            s += cc.impact_rate * 0.5
        elif cc.market_impact == "full":
            s += cc.impact_rate
        # 保守档专属项：做空/税费/不利成交/基金申赎费 → 固定增量，确保 con 最高
        if cc.short_cost:
            s += 0.0010
        if cc.taxes:
            s += 0.0005
        if cc.adverse_fill:
            s += 0.0005
        if cc.fund_fees:
            s += 0.0005
        return s

    def _net_return_by_tier(
        self, gross: float, turnover: float, costs: CostTiers
    ) -> dict[str, float]:
        """三档下净收益 = 多空毛收益 - 该档成本 (成本随 turnover 升高，两腿计)。"""
        net: dict[str, float] = {}
        for tier in _TIER_ORDER:
            rate = self._rate_sum(costs.get(tier))
            cost = 2.0 * turnover * rate  # 多空两腿换手
            net[tier.value] = float(gross - cost)
        return net

    def _regime_stable(
        self,
        scores_by_date: Mapping[date, pd.Series],
        labels_by_date: Mapping[date, pd.Series],
    ) -> bool:
        """切子区间检查「无严重反转」才算稳健 (非仅某段有效)。

        自适应：>=6 个决策日切三段，否则切两段。稳健的判据不是「每段都强正」，而是
        「没有哪一段与整体方向严重反转」——某段接近 0 可容忍，但某段明显反号 (如
        neg(px) 的级别漂移，前段 -0.07 / 后段 +0.09) 即判不稳健。
        """
        common = sorted(set(scores_by_date) & set(labels_by_date))
        if len(common) < 4:
            return False
        overall = self._mean_rank_ic_on(scores_by_date, labels_by_date, common)
        if overall == 0.0:
            return False
        n_blocks = 3 if len(common) >= 6 else 2
        blocks = np.array_split(np.array(common, dtype=object), n_blocks)
        ics = [
            self._mean_rank_ic_on(scores_by_date, labels_by_date, list(block))
            for block in blocks
        ]
        tol = 0.5 * abs(overall)  # a sub-period within this band of 0 is "no reversal"

        def agrees(v: float) -> bool:
            return (v > 0) == (overall > 0) or abs(v) < tol

        return all(agrees(v) for v in ics)

    @staticmethod
    def _mean_rank_ic_on(
        scores_by_date: Mapping[date, pd.Series],
        labels_by_date: Mapping[date, pd.Series],
        dates: Sequence[date],
    ) -> float:
        sub_s = {d: scores_by_date[d] for d in dates if d in scores_by_date}
        sub_y = {d: labels_by_date[d] for d in dates if d in labels_by_date}
        _, rank_ics = _metrics.per_date_ic(sub_s, sub_y)
        return float(np.mean(rank_ics)) if rank_ics else 0.0

    @staticmethod
    def _mean_cross_corr(
        a: Mapping[date, pd.Series], b: Mapping[date, pd.Series]
    ) -> float:
        """两因子逐点横截面 (Spearman) 相关的均值。"""
        corrs: list[float] = []
        for d in sorted(set(a) & set(b)):
            x, y = _metrics.aligned_pairs(a[d], b[d])
            if x.size < 5 or np.std(x) == 0 or np.std(y) == 0:
                continue
            c = spearman(x, y)
            if not np.isnan(c):
                corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    @staticmethod
    def _residualize(
        own: Mapping[date, pd.Series],
        existing_scores: Sequence[Mapping[date, pd.Series]],
    ) -> dict[date, pd.Series]:
        """逐点把本因子横截面对已有因子做 OLS 残差化，返回残差横截面。"""
        out: dict[date, pd.Series] = {}
        for d, s in own.items():
            cols = {"__own__": s}
            for i, es in enumerate(existing_scores):
                if d in es:
                    cols[f"e{i}"] = es[d]
            frame = pd.DataFrame(cols).dropna()
            if len(frame) < 5 or frame.shape[1] < 2:
                out[d] = s
                continue
            y = frame["__own__"].to_numpy(float)
            X = frame.drop(columns="__own__").to_numpy(float)
            X = np.column_stack([np.ones(len(X)), X])
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                out[d] = s
                continue
            out[d] = pd.Series(resid, index=frame.index)
        return out
