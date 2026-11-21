r"""monitoring —— 监控复盘层（tech doc §sec:mod-monitoring / instruction §5.1 第10层）。

策略进入影子运行 / 实盘后，持续观察四类退化并回写档案：
  - perf_drop   表现变差：实盘净收益明显低于回测预期（neu 档净收益）；
  - factor_decay 因子失效：实盘 IC/单调性衰减；失效因子转 ``retired`` 状态并回写；
  - risk_drift  风险漂移：实盘暴露偏离组合层设定；
  - cost_overrun 成本超预期：实盘净成本高于保守（con）档假设。

设计要点：把因子失效、风险漂移、成本超预期变成\emph{可观测、可留痕}的信号，
让"某因子何时、因何退役"可审计（tech doc §sec:mod-monitoring designnote）。
本层不忽略因子衰减与风险漂移（§04-modules L28）。
"""

from __future__ import annotations

from qmt.factor.types import (
    Anomaly,
    ExecResult,
    FactorArchive,
    FactorStatus,
    Monitoring,
    PerfSnapshot,
)

__all__ = ["LiveMonitoring"]


# 复盘时视为"仍在线、可被退役"的因子状态（已 retired/archived 的不再重复处理）。
_ACTIVE_STATUSES = frozenset(
    {FactorStatus.CANDIDATE, FactorStatus.SCREENED, FactorStatus.COMBINED}
)


class LiveMonitoring(Monitoring):
    """实盘 / 影子运行监控复盘（实现 :class:`~qmt.factor.types.Monitoring`）。

    阈值均为可调容忍度，缺省偏保守；触发即记录 :class:`Anomaly`，因子失效时
    把相关档案回写为 ``retired`` 并写入 ``reject_reason``（tech doc §sec:mod-monitoring）。
    """

    def __init__(
        self,
        perf_drop_tol: float = 0.5,
        min_live_ic: float = 0.0,
        drift_tol: float = 0.2,
    ) -> None:
        """
        参数：
          perf_drop_tol：实盘净收益低于 neu 档净收益多少（绝对差）判为表现变差。
          min_live_ic：实盘 IC 低于该值判为因子失效。
          drift_tol：任一暴露维度偏离预期超过该幅度判为风险漂移。
        """
        self.perf_drop_tol = float(perf_drop_tol)
        self.min_live_ic = float(min_live_ic)
        self.drift_tol = float(drift_tol)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def check(
        self,
        live_perf: PerfSnapshot,
        expected: ExecResult,
        archive: FactorArchive,
        as_of,
    ) -> list[Anomaly]:
        """比对实盘表现与回测预期，产出异常列表并回写失效因子（§5.1 第10层）。"""
        anomalies: list[Anomaly] = []

        anomalies.extend(self._check_perf_drop(live_perf, expected, as_of))
        anomalies.extend(self._check_factor_decay(live_perf, archive, as_of))
        anomalies.extend(self._check_risk_drift(live_perf, expected, as_of))
        anomalies.extend(self._check_cost_overrun(live_perf, expected, as_of))

        return anomalies

    # ------------------------------------------------------------------
    # 各类异常检测
    # ------------------------------------------------------------------
    def _check_perf_drop(
        self, live_perf: PerfSnapshot, expected: ExecResult, as_of
    ) -> list[Anomaly]:
        """表现变差：实盘净收益明显低于 neu 档预期净收益。"""
        neu = expected.net_return_by_tier.get("neu")
        if neu is None:
            # 无 neu 档基准无法比对；退化到 gross_return（诚实边界：不臆造基准）。
            neu = expected.gross_return
        gap = neu - live_perf.realized_return
        if gap > self.perf_drop_tol:
            return [
                Anomaly(
                    as_of=as_of,
                    kind="perf_drop",
                    factor_id=None,
                    detail=(
                        f"实盘净收益 {live_perf.realized_return:.4f} 低于 neu 档预期 "
                        f"{neu:.4f}，差幅 {gap:.4f} > 容忍 {self.perf_drop_tol:.4f}"
                    ),
                    severity="warn",
                )
            ]
        return []

    def _check_factor_decay(
        self, live_perf: PerfSnapshot, archive: FactorArchive, as_of
    ) -> list[Anomaly]:
        """因子失效：实盘 IC 衰减到阈值以下；把在线因子回写为 retired。"""
        ic = live_perf.realized_ic
        if ic is None or ic >= self.min_live_ic:
            return []

        reason = (
            f"实盘 IC {ic:.4f} 低于最低要求 {self.min_live_ic:.4f}（factor_decay，"
            f"as_of={as_of}）"
        )

        # 查档案里仍在线的因子并逐一退役（tech doc §sec:mod-monitoring：失效因子转 retired）。
        active = [
            rec for rec in archive.query() if rec.status in _ACTIVE_STATUSES
        ]

        anomalies: list[Anomaly] = []
        if not active:
            # 没有可退役的在线因子，仍留痕一条汇总异常。
            anomalies.append(
                Anomaly(
                    as_of=as_of,
                    kind="factor_decay",
                    factor_id=None,
                    detail=reason,
                    severity="critical",
                )
            )
            return anomalies

        for rec in active:
            rec.status = FactorStatus.RETIRED
            rec.reject_reason = reason
            archive.put(rec)  # 回写档案（审计单一事实来源）。
            anomalies.append(
                Anomaly(
                    as_of=as_of,
                    kind="factor_decay",
                    factor_id=rec.factor_id,
                    detail=reason,
                    severity="critical",
                )
            )
        return anomalies

    def _check_risk_drift(
        self, live_perf: PerfSnapshot, expected: ExecResult, as_of
    ) -> list[Anomaly]:
        """风险漂移：任一暴露维度偏离组合层预期超过 drift_tol。"""
        expected_exposure = self._expected_exposure(expected)
        worst_key: str | None = None
        worst_gap = 0.0
        for key, live_val in live_perf.exposure.items():
            exp_val = expected_exposure.get(key, 0.0)
            gap = abs(float(live_val) - float(exp_val))
            if gap > worst_gap:
                worst_gap = gap
                worst_key = key

        if worst_key is not None and worst_gap > self.drift_tol:
            exp_val = expected_exposure.get(worst_key, 0.0)
            return [
                Anomaly(
                    as_of=as_of,
                    kind="risk_drift",
                    factor_id=None,
                    detail=(
                        f"暴露 {worst_key!r} 实盘 {live_perf.exposure[worst_key]:.4f} 偏离预期 "
                        f"{exp_val:.4f}，差幅 {worst_gap:.4f} > 容忍 {self.drift_tol:.4f}"
                    ),
                    severity="warn",
                )
            ]
        return []

    def _check_cost_overrun(
        self, live_perf: PerfSnapshot, expected: ExecResult, as_of
    ) -> list[Anomaly]:
        """成本超预期：实盘净成本高于保守（con）档假设成本。"""
        con = expected.cost_by_tier.get("con")
        if con is None:
            return []
        if live_perf.net_cost > con:
            return [
                Anomaly(
                    as_of=as_of,
                    kind="cost_overrun",
                    factor_id=None,
                    detail=(
                        f"实盘净成本 {live_perf.net_cost:.4f} 高于 con 档假设 "
                        f"{con:.4f}"
                    ),
                    severity="warn",
                )
            ]
        return []

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _expected_exposure(expected: ExecResult) -> dict[str, float]:
        """组合层预期暴露：取实际成交目标权重（executed 优先，退化到 target）。"""
        series = expected.executed
        if series is None or len(series) == 0:
            series = expected.target
        if series is None:
            return {}
        return {str(k): float(v) for k, v in series.items()}
