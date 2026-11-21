"""base_factor_library —— 基础指标层（tech doc §sec:mod-basefactor / instruction §5.1 L3, §5.3 L1）。

在数据可信之后（\\SafeData{} 之下、\\FactorDSL{} 之上），本层把点时点安全的宽价格帧
整理成一批\\emph{具名基础指标}——收益、波动率、动量、回撤，并可选并入上游给的规模/费率/资金流。
这些指标是 :mod:`qmt.factor.auto_search` 自动拼公式的\\textbf{原材料}（符号学习的叶子变量），
本层\\emph{不搜索、不打分}。

关键约定与不变量：
  * ``SafePanel.frame`` 约定为宽表：行 = usable-date（点时点可用日），列 = 资产代码
    （基金 / 行业），值 = 复权价格（NAV / 收盘）。
  * 点时点安全（§cc-pit）：在 ``as_of`` 求值只能用 ``index <= as_of`` 的行；本层显式
    ``.loc[index <= as_of]`` 截断，绝不泄露未来。
  * 先复现经典人工因子再扩展（§5.6）：产出的 ``ret`` / ``px`` / ``vol20`` 必须让
    :data:`qmt.factor.baselines.MANUAL_FACTORS` 里的价量类公式（mom_60 / low_vol_20 /
    trend_60 / drawdown_60 …）可被 :class:`~qmt.factor.dsl.FormulaDSL` 直接求值。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt.factor.types import BaseFactorPanel, SafePanel

__all__ = ["BaseFactors"]


class BaseFactors:
    """基础指标库，具体实现 :class:`qmt.factor.types.BaseFactorLibrary`（§sec:mod-basefactor）。

    参数
    ----
    industry_of:
        资产代码 -> 行业标签，供下游 ``neutralize`` 横截面中性化使用；透传到
        :class:`BaseFactorPanel`，本层不消费。
    extra:
        可选的 ``{因子名: (date x asset) 帧}``，用于并入上游已算好的
        ``size`` / ``fee`` / ``flow`` 等基本面 / 资金面指标。每帧同样按 ``as_of``
        截断后对齐到价格帧的日期 / 资产轴。
    """

    def __init__(
        self,
        industry_of: dict[str, str] | None = None,
        extra: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.industry_of: dict[str, str] = dict(industry_of or {})
        self.extra: dict[str, pd.DataFrame] = dict(extra or {})

    # -- 核心接口 -----------------------------------------------------------
    def compute_base_factors(self, panel: SafePanel, as_of: date) -> BaseFactorPanel:
        """从安全面板的宽价格帧派生基础指标（继承点时点安全，§5.1 L3）。

        产出至少 ``px`` / ``ret`` / ``vol20``，并附带 ``mom20`` / ``dd60``；若构造时
        传入 ``extra``（size/fee/flow），一并对齐产出。仅使用 ``index <= as_of`` 的行。
        """
        frame = panel.frame
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return BaseFactorPanel(
                dates=[], funds=[], factors={}, industry_of=dict(self.industry_of)
            )

        # 点时点截断：只保留决策日 as_of（含）之前可见的行，并按时间排序。
        # （DatetimeIndex 面板需把 date 归一为 Timestamp，否则 pandas>=2 抛 TypeError。）
        px = frame.sort_index()
        cutoff = pd.Timestamp(as_of) if isinstance(px.index, pd.DatetimeIndex) else as_of
        px = px.loc[px.index <= cutoff]
        if px.empty:
            return BaseFactorPanel(
                dates=[], funds=[], factors={}, industry_of=dict(self.industry_of)
            )
        px = px.astype(float)

        ret = px.pct_change(fill_method=None)

        factors: dict[str, pd.DataFrame] = {
            "px": px,                                  # 复权价格（仅供回撤等尺度相对算子，不作横截面排序叶子）
            "ret": ret,                                # 单期收益（尺度无关）
            "vol20": ret.rolling(20, min_periods=10).std(),   # 20 日已实现波动（尺度无关）
            "vol60": ret.rolling(60, min_periods=30).std(),   # 60 日已实现波动
            "mom20": px.pct_change(20, fill_method=None),                # 20 日动量＝收益（尺度无关）
            "mom60": px.pct_change(60, fill_method=None),                # 60 日动量＝收益（尺度无关）
            "dd60": self._drawdown(px, 60),            # 60 日回撤比率（<= 0，尺度无关）
        }

        # 并入上游提供的规模 / 费率 / 资金流：同样点时点截断并对齐到价格轴。
        for name, ext in self.extra.items():
            aligned = self._align_extra(ext, px, as_of)
            if aligned is not None:
                factors[name] = aligned

        dates = [d for d in px.index]
        funds = [str(c) for c in px.columns]
        return BaseFactorPanel(
            dates=dates,
            funds=funds,
            factors=factors,
            industry_of=dict(self.industry_of),
        )

    # -- 内部工具 -----------------------------------------------------------
    @staticmethod
    def _drawdown(px: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动窗口回撤 ``(px - rolling_max) / rolling_max``（<= 0），与 DSL 的
        ``ts_drawdown`` 口径一致，便于人工因子对照。"""
        rmax = px.rolling(window, min_periods=max(2, window // 2)).max()
        dd = (px - rmax) / rmax.where(rmax.abs() > 1e-12)
        return dd

    @staticmethod
    def _align_extra(
        ext: pd.DataFrame, px: pd.DataFrame, as_of: date
    ) -> pd.DataFrame | None:
        """把 extra 帧点时点截断（``index <= as_of``）并对齐到价格帧的日期 / 资产轴。"""
        if ext is None or not isinstance(ext, pd.DataFrame) or ext.empty:
            return None
        e = ext.sort_index()
        cutoff = pd.Timestamp(as_of) if isinstance(e.index, pd.DatetimeIndex) else as_of
        e = e.loc[e.index <= cutoff]
        if e.empty:
            return None
        # 对齐资产列；日期以前向填充映射到价格帧的可用日（extra 常为低频披露）。
        e = e.reindex(columns=px.columns)
        e = e.reindex(index=px.index, method="ffill")
        return e
