"""安全数据层 SafeData —— 点时点视图 + 清洗（tech doc ch.4 §sec:mod-safedata；
instruction §3.2 避免偷看未来 / §3.6 缺失分层处理）。

职责（对应 :class:`qmt.factor.types.SafeDataLayer` Protocol）：

* :meth:`SafeData.as_of_view` —— 给定一组 :class:`DataField`，只保留在 ``as_of``
  当刻「历史真正可见」的字段：``available_at <= as_of`` 且 ``visible_at_decision``。
  缺 ``available_at`` 的字段直接 ``raise ValueError``（fail-closed，§3.2；与
  :class:`qmt.pit.PointInTimeStore.put` 的入库拒收同构）。
* :meth:`SafeData.clean` —— winsorize 极端值（默认 1%/99% 分位裁剪）、缺失分层处理
  （§3.6：绝不简单填 0；整列缺失丢弃，个别缺失保留为 NaN 交由下游 MissingPolicy）。
* :meth:`SafeData.price_panel` —— 便捷方法，用 :func:`qmt.factor.panel.usable_price_frame`
  从 PIT store 建「行=usable-date（点时点可用日）、列=资产、值=复权价格」宽帧，切到
  ``<= as_of``，为每列构造 :class:`DataField` 并组装 :class:`SafePanel`。

点时点安全是物理保证：宽价格帧的行索引为 usable-date（观测日 + visible_lag），本层
再切 ``index <= as_of``，故 ``as_of`` 决策绝不会用到当刻尚不可见的数据。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore, TradingCalendar
from qmt.factor.panel import usable_price_frame
from qmt.factor.types import (
    DataField,
    DataSource,
    Freshness,
    MissingPolicy,
    SafeDataLayer,
    SafePanel,
    UpdateFreq,
)

__all__ = ["SafeData"]


def _to_dt(d: date | datetime) -> datetime:
    """把 ``date`` 归一为当日零点 ``datetime`` 以便与 ``available_at`` 比较。"""
    if isinstance(d, datetime):
        return d
    return datetime.combine(d, datetime.min.time())


class SafeData(SafeDataLayer):
    """点时点安全数据层的具体实现（§sec:mod-safedata）。"""

    def __init__(
        self,
        store: PointInTimeStore,
        calendar: TradingCalendar | None = None,
        *,
        winsor_lower: float = 0.01,
        winsor_upper: float = 0.99,
    ) -> None:
        self.store = store
        self.calendar = calendar or TradingCalendar()
        if not (0.0 <= winsor_lower < winsor_upper <= 1.0):
            raise ValueError("winsor 分位需满足 0<=lower<upper<=1")
        self.winsor_lower = winsor_lower
        self.winsor_upper = winsor_upper

    # ------------------------------------------------------------------ #
    # as_of_view —— 字段级点时点过滤（§3.2）
    # ------------------------------------------------------------------ #
    def as_of_view(self, fields: Sequence[DataField], as_of: datetime) -> SafePanel:
        """只保留 ``available_at <= as_of`` 且 ``visible_at_decision`` 的字段。

        缺 ``available_at`` 的字段直接抛错（不静默丢弃、不当作 0 时刻），因为一条无
        可见时刻的字段进入回测会偷看未来（instruction §3.2、§3.5）。
        """
        as_of_ts = _to_dt(as_of)
        as_of_date = as_of_ts.date()

        kept: dict[str, DataField] = {}
        for f in fields:
            if getattr(f, "available_at", None) is None:
                raise ValueError(
                    f"DataField {getattr(f, 'field_name', '?')!r} 缺 available_at —— "
                    "无法进入点时点视图（会偷看未来，instruction §3.2）。"
                )
            if not f.visible_at_decision:
                continue
            if _to_dt(f.available_at) <= as_of_ts:
                kept[f.field_name] = f

        # frame：每个存活字段的溯源元数据一行（field_name x provenance 列）。
        rows = [
            {
                "field_name": f.field_name,
                "source": f.source.value if isinstance(f.source, DataSource) else f.source,
                "available_at": _to_dt(f.available_at),
                "freshness": f.freshness.value
                if isinstance(f.freshness, Freshness)
                else f.freshness,
                "missing_policy": f.missing_policy.value
                if isinstance(f.missing_policy, MissingPolicy)
                else f.missing_policy,
            }
            for f in kept.values()
        ]
        frame = (
            pd.DataFrame(rows).set_index("field_name")
            if rows
            else pd.DataFrame(columns=["source", "available_at", "freshness", "missing_policy"])
        )
        return SafePanel(as_of=as_of_date, frame=frame, fields=kept, universe=[])

    # ------------------------------------------------------------------ #
    # clean —— winsorize + 缺失分层（§3.6）
    # ------------------------------------------------------------------ #
    def clean(self, panel: SafePanel, *, winsorize: bool = True) -> SafePanel:
        """裁剪极端值并做缺失分层处理，返回新的 :class:`SafePanel`（不改入参）。

        * winsorize（**因果**）：对每个数值列按 *expanding* 分位裁剪——第 t 行的
          ``[q_lower, q_upper]`` 阈值只用 ``<= t`` 的数据计算（min_periods 预热期内
          不裁剪）。绝不使用全历史分位，否则未来数据会通过阈值污染过去的行，
          破坏点时点安全（对抗审校确认过的真实泄漏路径）。
        * ``winsorize=False``：跳过裁剪，只做缺失分层。**价格水平帧应当用 False**——
          对趋势性 NAV 裁剪价格水平会把新高系统性压平；极端值防护属于收益/字段层。
        * 缺失分层：整列全缺 → 丢弃并把该字段标记为 EXPIRED；个别缺失 → 保留为 NaN，
          交由下游按各字段 :class:`MissingPolicy` 处理，**绝不简单填 0**（§3.6）。
        """
        frame = panel.frame.copy()
        numeric_cols = list(frame.select_dtypes(include=[np.number]).columns)

        dropped: list[str] = []
        for col in numeric_cols:
            s = frame[col]
            if s.notna().sum() == 0:
                # 整列缺失：丢弃，避免把「无数据」误当「0」。
                dropped.append(col)
                continue
            if winsorize:
                mp = min(60, max(10, int(s.notna().sum() // 4)))
                lo = s.expanding(min_periods=mp).quantile(self.winsor_lower)
                hi = s.expanding(min_periods=mp).quantile(self.winsor_upper)
                # 预热期（阈值 NaN）不裁剪；其后每行只被 <= 该行 的数据约束（因果）
                clipped = s.where(lo.isna() | (s >= lo), lo)
                clipped = clipped.where(hi.isna() | (clipped <= hi), hi)
                frame[col] = clipped

        if dropped:
            frame = frame.drop(columns=dropped)

        # 更新字段元数据 / universe。
        new_fields = dict(panel.fields)
        for col in dropped:
            if col in new_fields:
                fd = new_fields[col]
                fd.freshness = Freshness.EXPIRED  # 全缺 → 视为过期
        universe = [c for c in panel.universe if c not in dropped]

        return SafePanel(
            as_of=panel.as_of,
            frame=frame,
            fields=new_fields,
            universe=universe,
        )

    # ------------------------------------------------------------------ #
    # price_panel —— 从 PIT store 建点时点宽价格帧（便捷方法）
    # ------------------------------------------------------------------ #
    def price_panel(
        self,
        table: str,
        as_of: date,
        *,
        value_col: str = "adj_nav",
        code_col: str = "fund_code",
        date_col: str = "nav_date",
        codes: Sequence[str] | None = None,
        visible_lag_days: int = 1,
    ) -> SafePanel:
        """建「行=usable-date、列=资产、值=复权价格」宽帧并切到 ``<= as_of``。

        每列（资产）配一条 :class:`DataField`，其 ``available_at`` 取该列最后一个
        usable 观测日（即最新一次可见时刻）。行索引已被截断到 ``<= as_of``，故本帧
        在 ``as_of`` 求值不会偷看未来。
        """
        wide = usable_price_frame(
            self.store,
            table,
            value_col=value_col,
            code_col=code_col,
            date_col=date_col,
            visible_lag_days=visible_lag_days,
            calendar=self.calendar,
            codes=codes,
        )

        as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
        if not wide.empty:
            # 点时点切片：只保留 usable-date <= as_of 的行。
            wide = wide.loc[[d for d in wide.index if d <= as_of_date]]

        fields: dict[str, DataField] = {}
        universe: list[str] = []
        for col in wide.columns:
            col_series = wide[col].dropna()
            if col_series.empty:
                continue
            last_usable = col_series.index[-1]  # 该列最后可见日（<= as_of）
            fields[str(col)] = DataField(
                field_name=str(col),
                source=DataSource.MARKET,
                available_at=_to_dt(last_usable),
                update_freq=UpdateFreq.DAILY,
                freshness=Freshness.FRESH,
                visible_at_decision=True,
                missing_policy=MissingPolicy.AS_OF_LAG,
            )
            universe.append(str(col))

        return SafePanel(as_of=as_of_date, frame=wide, fields=fields, universe=universe)
