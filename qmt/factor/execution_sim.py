r"""execution_sim —— 执行仿真层（第 9 层，对应 ch.4 §sec:mod-execsim / 源文档 §4.4、§4.6、§7.6）。

在真正交易前，用\emph{更接近真实交易条件}模拟目标持仓的换手与成本：把上游
:class:`~qmt.factor.types.Combination` 的目标权重与当前 :class:`PortfolioState`
对齐，算出换手 ``turnover``，再在 :class:`CostTiers` 的三档（乐观 opt / 中性 neu /
保守 con）成本假设下分别扣成本，得到 :class:`ExecResult`。

三档成本假设（源文档 §4.6，原文照搬，见 ch.4 表）：
  * opt —— 只计佣金和\emph{部分}买卖价差；
  * neu —— 再加入滑点和\emph{简单}市场冲击（并计基金申赎费）；
  * con —— 进一步加入做空、税费、不利成交和容量折损（全额市场冲击）。

设计要点：
  * **成本单调不减**：opt <= neu <= con 是硬不变量——只在乐观档赚钱、中性/保守
    档都失效的策略不得上线（源文档 §4.6 风险框）。本实现按各档开关\emph{累加}
    非负成本，默认 :class:`CostTiers` 三档开关本身单调，故不变量成立。
  * **基金专属费用**：``fund_fees=True`` 时，对\emph{减仓（赎回）}部分按
    :meth:`FeeSchedule.redemption_rate` (依 ``hold_days``) 近似赎回费——短持有期
    （<7 日）赎回费更高（源文档 §7 rule 7）；对\emph{加仓（申购）}部分计申购费。
  * **点时点**：不读取任何晚于 ``as_of`` 的数据；毛收益 ``gross_return`` 默认为 0
    （测试可只验成本单调），亦可由 ``gross_return_fn`` 注入近似前瞻毛收益，用来
    识别“只在乐观档赚钱”的策略。
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Optional

import pandas as pd

from qmt.factor.types import (
    CostComponents,
    CostTier,
    CostTiers,
    ExecResult,
    FundScores,
    PortfolioState,
)

try:  # 复用既有费用基础设施；缺失时容错（不新增依赖）
    from qmt.fees import default_fee_schedule as _default_fee_schedule
except Exception:  # pragma: no cover - 防御性：qmt.fees 不可用时退回零赎回费
    _default_fee_schedule = None  # type: ignore[assignment]

__all__ = ["ExecSim"]


class ExecSim:
    """执行仿真层的具体实现（实现 :class:`~qmt.factor.types.ExecutionSim` Protocol）。

    :param fee_resolver: ``fund_code -> FeeSchedule`` 的解析器，默认用
        :func:`qmt.fees.default_fee_schedule`（场外权益基金的现实默认费率表）。
    :param gross_return_fn: 可选 ``(target, as_of) -> float`` 毛收益估计；默认返回
        0.0（测试可只验成本单调）。用于识别“只在乐观档赚钱”的策略。
    """

    #: 【推断】默认成本率——CostComponents 未给这些档的标量费率，取保守小值。
    SHORT_COST_RATE: float = 0.0010   # 做空/融券成本（近似）
    TAX_RATE: float = 0.0010          # 税费（近似）
    ADVERSE_FILL_RATE: float = 0.0010  # 不利成交 / 容量折损（近似）

    #: 买卖价差按档计入的比例：none 不计 / partial 半额 / full 全额。
    _SPREAD_FRAC = {"none": 0.0, "partial": 0.5, "full": 1.0}
    #: 市场冲击按档计入的比例：none / simple 简单半额 / full 全额。
    _IMPACT_FRAC = {"none": 0.0, "simple": 0.5, "full": 1.0}

    def __init__(
        self,
        fee_resolver: Optional[Callable[[str], object]] = None,
        gross_return_fn: Optional[Callable[[FundScores, date], float]] = None,
    ) -> None:
        if fee_resolver is not None:
            self._fee_resolver = fee_resolver
        elif _default_fee_schedule is not None:
            self._fee_resolver = lambda code: _default_fee_schedule(code)
        else:  # pragma: no cover - 无费用基础设施时的兜底
            self._fee_resolver = None  # type: ignore[assignment]
        self._gross_return_fn = gross_return_fn
        self._fee_cache: dict[str, object] = {}

    # -- 费用解析（带缓存）---------------------------------------------------
    def _fee(self, fund_code: str) -> Optional[object]:
        if self._fee_resolver is None:
            return None
        if fund_code not in self._fee_cache:
            try:
                self._fee_cache[fund_code] = self._fee_resolver(fund_code)
            except Exception:  # 费率表解析失败（如代码不合规）：跳过基金费，不中断仿真
                self._fee_cache[fund_code] = None
        return self._fee_cache[fund_code]

    def simulate_path(self, unit_nav, adjusted_nav, decision_dates, score_callback=None,
                      *, config, weight_callback=None, fees_resolver=None, **visibility):
        """Cash/unit timeline for actual portfolio increments and net backtests.

        ``simulate`` below remains a one-point target-cost diagnostic. Its
        instantaneous weights and three generic tiers are not executable NAV
        wealth. All timeline users share fund_execution's confirmation, FIFO,
        cash settlement, subscription amounts and NAV-embedded fee conventions.
        """
        from qmt.factor.fund_execution import run_fund_backtest
        return run_fund_backtest(unit_nav, adjusted_nav, decision_dates, score_callback,
                                 config=config, weight_callback=weight_callback,
                                 fees_resolver=(self._fee_resolver if fees_resolver is None else fees_resolver),
                                 **visibility)

    # -- 基金申赎费：对减仓部分计赎回费、加仓部分计申购费 --------------------
    def _fund_fee_cost(
        self, diff: pd.Series, hold_days: dict[str, int]
    ) -> float:
        """按持仓变动方向近似基金申赎费（源文档 §4.4 基金专属成本）。

        ``diff = target - holding``：负=减仓（赎回，按 ``hold_days`` 分档赎回费，
        短持有期更高）；正=加仓（申购费）。
        """
        total = 0.0
        for fund, dw in diff.items():
            if dw == 0.0:
                continue
            sched = self._fee(str(fund))
            if sched is None:
                continue
            if dw < 0.0:  # 减仓 -> 赎回费（短持有期更高）
                hd = int(hold_days.get(str(fund), 0))
                rate = sched.redemption_rate(hd)
                total += (-float(dw)) * rate
            else:  # 加仓 -> 申购费
                rate = float(getattr(sched, "purchase_fee_rate", 0.0))
                total += float(dw) * rate
        return total

    # -- 单档成本 ----------------------------------------------------------
    def _tier_cost(
        self,
        comp: CostComponents,
        turnover: float,
        diff: pd.Series,
        hold_days: dict[str, int],
    ) -> float:
        """一档成本 = 换手 * 逐笔比例成本之和 + （启用时）基金申赎费。所有项非负。"""
        rate = 0.0
        if comp.commission:
            rate += float(comp.commission_rate)
        rate += self._SPREAD_FRAC.get(comp.spread, 0.0) * float(comp.spread_rate)
        if comp.slippage:
            rate += float(comp.slippage_rate)
        rate += self._IMPACT_FRAC.get(comp.market_impact, 0.0) * float(comp.impact_rate)
        if comp.short_cost:
            rate += self.SHORT_COST_RATE
        if comp.taxes:
            rate += self.TAX_RATE
        if comp.adverse_fill:
            rate += self.ADVERSE_FILL_RATE
        cost = turnover * rate
        if comp.fund_fees:
            cost += self._fund_fee_cost(diff, hold_days)
        return cost

    # -- 主接口 ------------------------------------------------------------
    def simulate(
        self,
        target: FundScores,
        portfolio_state: PortfolioState,
        as_of: date,
        costs: CostTiers,
    ) -> ExecResult:
        """在三档成本下模拟从当前持仓换到目标持仓的净收益（ch.4 §sec:mod-execsim）。

        换手 ``turnover = 0.5 * sum(|target.weights - holdings|)``（索引对齐、缺失补 0）；
        每档 ``net_return_by_tier[tier] = gross_return - cost_by_tier[tier]``。
        不变量：``cost_opt <= cost_neu <= cost_con``；保守档净收益 <= 0 时给 notes 警示。
        """
        # 目标权重：优先 risk overlay 后的 weights，缺失则退回 scores。
        tw = target.weights if target.weights is not None else target.scores
        target_w = pd.Series(tw, dtype=float).dropna()
        holdings = pd.Series(portfolio_state.holdings, dtype=float).dropna()

        # 索引对齐，缺失按 0（未持有/清仓）。
        idx = target_w.index.union(holdings.index)
        tv = target_w.reindex(idx).fillna(0.0)
        hv = holdings.reindex(idx).fillna(0.0)
        diff = tv - hv  # target - holding

        turnover = 0.5 * float(diff.abs().sum())

        hold_days = portfolio_state.hold_days or {}

        cost_by_tier: dict[str, float] = {}
        for tier in (CostTier.OPT, CostTier.NEU, CostTier.CON):
            comp = costs.get(tier)
            cost_by_tier[tier.value] = self._tier_cost(comp, turnover, diff, hold_days)

        # 毛收益（默认 0；可由注入函数给近似前瞻毛收益）。
        gross = 0.0
        if self._gross_return_fn is not None:
            gross = float(self._gross_return_fn(target, as_of))

        net_return_by_tier = {
            k: gross - v for k, v in cost_by_tier.items()
        }

        notes: list[str] = []
        con_net = net_return_by_tier[CostTier.CON.value]
        neu_net = net_return_by_tier[CostTier.NEU.value]
        opt_net = net_return_by_tier[CostTier.OPT.value]
        # 仅在确有换手（发生交易）时评估“净成本后是否仍具增量”。
        if turnover > 0.0 and con_net <= 0.0 < opt_net:
            notes.append(
                "保守档净收益<=0 而乐观档为正：疑似只在乐观档赚钱的策略，不得贸然"
                "上线（源文档 §4.6）。"
            )
        elif turnover > 0.0 and con_net <= 0.0:
            notes.append("保守档净收益<=0：中性/保守成本下不具增量。")
        if turnover > 0.0 and neu_net <= 0.0 < opt_net:
            notes.append("中性档净收益<=0：交易性门槛（锚定中性档）未通过。")

        return ExecResult(
            as_of=as_of,
            target=tv,
            executed=tv,  # MVP：假设按目标全额成交（确认延迟/暂停留待扩展）
            gross_return=gross,
            net_return_by_tier=net_return_by_tier,
            cost_by_tier=cost_by_tier,
            turnover=turnover,
            notes=notes,
        )
