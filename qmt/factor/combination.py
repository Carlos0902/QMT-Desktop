"""combination —— 组合层（非简单平均，instruction §5.5 / tech doc §sec:mod-combination）.

把多个通过筛选的因子合成为最终基金打分，分三步（源文档 §5.5）：

  1. **统一口径**：对每个因子在 ``as_of`` 求横截面，再截面 z-score，去量纲。
  2. **按类别组合**：把因子归入价量 / 财务 / 风险三类；类别内等权，类别间
     按类别质量（成员因子的 OOS rank-IC）加权——而非把所有好因子简单平均。
  3. **风险限制**：由合成分转 long-only 目标权重，施加单基金权重上限
     ``max_weight``、单基金类型（行业）集中度上限 ``max_category_share``，
     必要时截断并重归一。

设计意图：防止"研究上有效的因子"在组合里叠成"风险过于集中的仓位"。简单平均会
让高相关因子沿同一风险方向叠加；按类别加权 + 权重/集中度上限能把这类隐性集中拆开
（见 §5.5 risknote）。本类实现 :class:`qmt.factor.types.Combination`。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from qmt.factor.dsl import FormulaDSL
from qmt.factor.types import (
    BaseFactorPanel,
    FactorExpr,
    FactorScore,
    FundScores,
)

__all__ = ["FactorCombination", "CATEGORIES", "infer_category"]

# 因子类别（价量 / 财务 / 风险）——组合按类别分开处理，而非一锅端平均。
CATEGORIES = ("price_volume", "financial", "risk")

# 类别关键词：先按因子名 / factor_id 命中，再按其引用的 base-factor 字段兜底。
_NAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "risk": ("vol", "drawdown", "dd", "risk", "beta", "tail"),
    "financial": ("size", "fee", "flow", "value", "quality", "earn", "fund"),
    "price_volume": ("mom", "rev", "trend", "price", "px", "ret", "vol_switch"),
}
# base-factor 字段 -> 类别（名字未命中时兜底；vol20 归风险类）。
_FIELD_CATEGORY: dict[str, str] = {
    "ret": "price_volume",
    "px": "price_volume",
    "vol20": "risk",
    "size": "financial",
    "fee": "financial",
    "flow": "financial",
}

_EPS = 1e-12


def infer_category(factor: FactorExpr, factor_id: str | None = None) -> str:
    """推断因子类别（价量 / 财务 / 风险）。

    优先看因子标识 / 公式文本里的关键词，其次看引用的 base-factor 字段；
    都不命中则默认价量类（v1 以场外权益基金的价量信号为主）。
    """
    # 1) 名字 / id / 公式文本关键词（风险 > 财务 > 价量 优先级，避免 low_vol 落到价量）
    text = " ".join(filter(None, [factor_id or "", factor.text or ""])).lower()
    for cat in ("risk", "financial", "price_volume"):
        if any(kw in text for kw in _NAME_KEYWORDS[cat]):
            return cat
    # 2) 字段兜底：若引用到风险/财务字段则归对应类
    fields = set(factor.data_fields)
    for cat in ("risk", "financial"):
        if any(_FIELD_CATEGORY.get(f) == cat for f in fields):
            return cat
    return "price_volume"


def _zscore(s: pd.Series) -> pd.Series:
    """截面 z-score（统一量纲）；标准差为 0 时退化为去均值。"""
    s = s.astype(float).dropna()
    if s.empty:
        return s
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd < _EPS:
        return s - mu
    return (s - mu) / sd


class FactorCombination:
    """非简单平均的组合层（instruction §5.5）。

    ``max_weight``          单只基金目标权重上限。
    ``max_category_share``  单个基金类型/行业（``panel.industry_of``）的权重集中度上限。

    实现 :class:`qmt.factor.types.Combination` 的 ``combine`` 方法。
    """

    def __init__(self, max_weight: float = 0.15, max_category_share: float = 0.5) -> None:
        if not 0.0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0,1], got {max_weight}")
        if not 0.0 < max_category_share <= 1.0:
            raise ValueError(
                f"max_category_share must be in (0,1], got {max_category_share}"
            )
        self.max_weight = float(max_weight)
        self.max_category_share = float(max_category_share)
        self._dsl = FormulaDSL()

    # -- Combination protocol -------------------------------------------------
    def combine(
        self,
        factors: Sequence[FactorExpr],
        scores: Mapping[str, FactorScore],
        panel: BaseFactorPanel,
        as_of,
    ) -> FundScores:
        """三步合成最终基金打分（§5.5）。返回 :class:`FundScores`。"""
        notes: list[str] = []

        # === 步骤 1：逐因子求横截面并统一口径（z-score）===
        pairs = self._pair_factors_scores(factors, scores)
        # 类别 -> [(标准化得分 Series, 质量 q)]
        by_cat_members: dict[str, list[tuple[pd.Series, float]]] = {
            c: [] for c in CATEGORIES
        }
        for factor, fid, score in pairs:
            try:
                raw = self._dsl.evaluate(factor, panel, as_of)
            except (KeyError, TypeError, ValueError) as exc:  # 缺字段/无法求值 → 跳过
                notes.append(f"skip factor {fid or factor.text!r}: {exc}")
                continue
            z = _zscore(raw)
            if z.empty:
                notes.append(f"skip factor {fid or factor.text!r}: empty cross-section")
                continue
            cat = infer_category(factor, fid)
            quality = self._factor_quality(score)
            by_cat_members[cat].append((z, quality))

        # === 步骤 2：类别内等权、类别间按类别质量加权 ===
        by_category: dict[str, pd.Series] = {}
        cat_quality: dict[str, float] = {}
        for cat, members in by_cat_members.items():
            if not members:
                continue
            # 类别内等权（体现"不让单因子支配"）
            frame = pd.concat([m[0] for m in members], axis=1)
            by_category[cat] = frame.mean(axis=1)
            # 类别质量 = 成员质量均值（供类别间加权）
            cat_quality[cat] = float(np.mean([m[1] for m in members]))

        if not by_category:
            notes.append("no usable factor produced a cross-section")
            empty = pd.Series(dtype=float)
            return FundScores(as_of=as_of, scores=empty, weights=empty,
                              by_category={}, risk_notes=notes)

        composite = self._blend_categories(by_category, cat_quality, notes)

        # === 步骤 3：风险限制 —— 合成分转 long-only 目标权重并施加上限 ===
        weights = self._risk_overlay(composite, panel, notes)

        return FundScores(
            as_of=as_of,
            scores=composite,
            weights=weights,
            by_category=by_category,
            risk_notes=notes,
        )

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _pair_factors_scores(
        factors: Sequence[FactorExpr],
        scores: Mapping[str, FactorScore],
    ) -> list[tuple[FactorExpr, str | None, FactorScore | None]]:
        """把每个 FactorExpr 关联到它的 FactorScore。

        依次尝试：(1) 以公式文本为键；(2) 以 score.factor_id 恰好等于文本；
        (3) 按位置兜底（factors 与 scores 顺序一致的常见情形）。
        """
        score_items = list(scores.items())
        # 反查：factor_id -> score（用于文本恰为 id 的情形）
        pairs: list[tuple[FactorExpr, str | None, FactorScore | None]] = []
        for i, factor in enumerate(factors):
            sc: FactorScore | None = None
            fid: str | None = None
            if factor.text in scores:
                fid, sc = factor.text, scores[factor.text]
            elif i < len(score_items):
                fid, sc = score_items[i]
            if sc is not None and fid is None:
                fid = sc.factor_id
            pairs.append((factor, fid, sc))
        return pairs

    @staticmethod
    def _factor_quality(score: FactorScore | None) -> float:
        """单因子质量分（用于类别间加权）：OOS rank-IC 的正部，附小底以免全零。

        rank-IC 缺失时回退到 ic；再退回极小正数（保证类别至少等权参与）。
        """
        if score is None:
            return _EPS
        q = score.rank_ic if score.rank_ic else score.ic
        return max(float(q), 0.0) + _EPS

    @staticmethod
    def _blend_categories(
        by_category: Mapping[str, pd.Series],
        cat_quality: Mapping[str, float],
        notes: list[str],
    ) -> pd.Series:
        """类别间按质量加权求和；质量全零时退化为类别等权（仍非简单平均）。"""
        cats = list(by_category)
        qual = np.array([max(cat_quality.get(c, 0.0), 0.0) for c in cats], dtype=float)
        if qual.sum() < _EPS:
            weights = np.full(len(cats), 1.0 / len(cats))
            notes.append("category qualities all ~0 → equal category weights")
        else:
            weights = qual / qual.sum()
        frame = pd.concat([by_category[c] for c in cats], axis=1, keys=cats)
        blended = frame.mul(pd.Series(weights, index=cats), axis=1).sum(
            axis=1, min_count=1
        )
        return blended.dropna()

    def _risk_overlay(
        self,
        composite: pd.Series,
        panel: BaseFactorPanel,
        notes: list[str],
    ) -> pd.Series:
        """合成分 -> long-only 目标权重，施加单权重与行业集中度上限。"""
        # long-only：取正部作为持仓意愿；全非正则回退为等权（无可做多标的时保守持有）
        appetite = composite.clip(lower=0.0)
        if appetite.sum() < _EPS:
            notes.append("no positive composite score → fall back to equal weights")
            appetite = pd.Series(1.0, index=composite.index)
        w = appetite / appetite.sum()

        # 可行性检查：单权重上限至少要能容纳 1/N 的均摊
        n = len(w)
        if self.max_weight * n < 1.0 - 1e-9:
            notes.append(
                f"max_weight={self.max_weight} × N={n} < 1；权重上限不可行，"
                "退化为等权"
            )
            return pd.Series(1.0 / n, index=w.index)

        w = self._cap_single(w)
        w = self._cap_category(w, panel, notes)
        # 行业重分配可能破坏单权重上限；先归一再做最后一次注水式单权重截断
        # （_cap_single 会把总权重重新压回 1，故放在最后以保证 sum==1 且 max<=cap）
        total = w.sum()
        if total > _EPS:
            w = w / total
        w = self._cap_single(w)
        return w

    def _cap_single(self, w: pd.Series) -> pd.Series:
        """单基金权重上限：注水法。

        capped 的基金冻结在上限（不再随后续重分配被抬高），剩余质量只在未冻结的
        基金间按比例重分配，迭代至稳定——这样总权重保持 1 且没有权重超过上限。
        """
        cap = self.max_weight
        w = w.copy()
        frozen = pd.Series(False, index=w.index)
        for _ in range(len(w) + 2):
            over = (~frozen) & (w > cap + 1e-12)
            if not over.any():
                break
            w[over] = cap
            frozen |= over
            free_mass = 1.0 - cap * frozen.sum()
            active = ~frozen
            active_sum = w[active].sum()
            if not active.any() or active_sum < _EPS or free_mass <= 0:
                break
            w[active] = w[active] * (free_mass / active_sum)
        return w

    def _cap_category(
        self,
        w: pd.Series,
        panel: BaseFactorPanel,
        notes: list[str],
    ) -> pd.Series:
        """单基金类型/行业集中度上限（panel.industry_of）。

        某行业权重超限时按比例压缩到上限，释放的权重分给其它行业的基金。
        无行业映射时跳过（并记一条 note）。
        """
        industry_of = getattr(panel, "industry_of", None) or {}
        if not industry_of:
            notes.append("no industry_of on panel → skip category-share cap")
            return w
        cap = self.max_category_share
        w = w.copy()
        ind = pd.Series({c: industry_of.get(c, "_none") for c in w.index})
        for _ in range(len(set(ind)) + 2):
            shares = w.groupby(ind).sum()
            over_cats = shares[shares > cap + 1e-12].index.tolist()
            if not over_cats:
                break
            freed = 0.0
            for cat in over_cats:
                members = ind[ind == cat].index
                cur = w[members].sum()
                scale = cap / cur
                freed += cur - cap
                w[members] = w[members] * scale
            # 释放的权重按现权重比例分给非超限行业的基金
            others = ind[~ind.isin(over_cats)].index
            other_sum = w[others].sum()
            if other_sum < _EPS or freed <= 0:
                notes.append(
                    "category-share cap: no room to redistribute (few industries) "
                    "→ residual concentration remains"
                )
                break
            w[others] = w[others] + freed * (w[others] / other_sum)
        return w
