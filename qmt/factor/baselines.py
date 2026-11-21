"""Manual classic factors — the baseline-first anchor (instruction §5.6 / §7.4).

The roadmap (S1) mandates reproducing known manual factors *before* any search:
they validate the pipeline and are the ladder rung symbolic learning must beat. If
:class:`~qmt.factor.auto_search` cannot stably out-perform these out-of-sample, the
project's value is governance/explainability, not returns (§7.1/§7.4) — this module
makes that comparison concrete.

Formulas reference base factors produced by ``base_factor_library`` (canonical names
in :data:`BASE_FACTORS`). Price-only factors (momentum/reversal/low-vol) need just
``ret``/``px`` and so run on any cross-section (funds *or* the industry demo);
``size``/``fee``/``flow`` factors activate when those base factors are present.
"""

from __future__ import annotations

from qmt.factor.dsl import FormulaDSL
from qmt.factor.types import FactorExpr

__all__ = ["BASE_FACTORS", "MANUAL_FACTORS", "PRICE_ONLY_FACTORS", "manual_exprs"]

# Base-factor leaves the baselines reference. All formulas below are SCALE-FREE
# (returns / return-std / drawdown ratio). Raw price LEVEL `px` is never used for
# cross-sectional *ranking* — it is non-stationary and not comparable across assets
# (§7.3); the pipeline also excludes `px` from the auto_search leaf set. `px` appears
# only inside the scale-free `ts_drawdown` ratio.
BASE_FACTORS = ("ret", "px", "vol20", "vol60", "mom20", "mom60", "dd60", "size", "fee", "flow")

# name -> DSL formula text (cross-sectional rank of a classic, scale-free signal)
MANUAL_FACTORS: dict[str, str] = {
    # return / volatility only — always available, all scale-free (need only `ret`)
    "mom_60": "rank(ts_mean(ret, 60))",           # 动量：近 60 日平均收益
    "mom_20": "rank(ts_mean(ret, 20))",           # 中期动量
    "reversal_5": "rank(neg(ts_mean(ret, 5)))",   # 短期反转
    "low_vol_20": "rank(neg(ts_std(ret, 20)))",   # 低波动
    "low_vol_60": "rank(neg(ts_std(ret, 60)))",
    "trend_60": "rank(div_g(ts_mean(ret, 60), ts_std(ret, 60)))",  # 风险调整动量（收益/波动，尺度无关）
    "drawdown_60": "rank(neg(ts_drawdown(px, 60)))",  # 回撤越小越好（回撤为比率，尺度无关）
    # fund-attribute factors — activate when the base factor exists
    "small_size": "rank(neg(size))",              # 小规模
    "low_fee": "rank(neg(fee))",                  # 低费率
    "inflow": "rank(ts_mean(flow, 20))",          # 资金持续流入
    # holdings-exposure signal (§7.1 hyp-3): tilt toward strong-momentum industries
    "hold_mom": "rank(hold_mom)",                 # 持仓暴露 × 行业近期动量
    # PIT rolling beta leaf (activates when market series provided): classic low-beta
    "low_beta": "rank(neg(beta))",                # 低 beta 异象
    # §7.1 hyp-2 pre-registered candidates (activate when flow/drift leaves exist)
    "style_drift": "rank(neg(drift))",            # 风格纪律：漂移越小越好
    "flow_x_drift": "rank(mul(flow, drift))",     # 假设2本体：漂移 × 资金流交互
}

PRICE_ONLY_FACTORS = (
    "mom_60", "mom_20", "reversal_5", "low_vol_20", "low_vol_60",
    "trend_60", "drawdown_60",
)


def manual_exprs(
    dsl: FormulaDSL,
    available_fields: set[str] | None = None,
    *,
    price_only: bool = False,
) -> dict[str, FactorExpr]:
    """Parse the manual factors into :class:`FactorExpr`, keeping only those whose
    referenced base factors are available (guards fund-attribute factors on the
    price-only industry demo)."""
    names = PRICE_ONLY_FACTORS if price_only else tuple(MANUAL_FACTORS)
    out: dict[str, FactorExpr] = {}
    for name in names:
        expr = dsl.parse(MANUAL_FACTORS[name])
        if available_fields is not None and not set(expr.data_fields) <= available_fields:
            continue
        out[name] = expr
    return out
