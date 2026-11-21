"""models_registry — a catalog of the analytical *models* the QMT pipeline runs.

The control plane needs a page answering "what models do we have, of what kind, with what
parameters and status". In this system a *model* is the concrete method a pipeline stage uses
to estimate / predict / decide — and, per the doc's "simple transparent baseline first, then
candidate methods" philosophy (§5), every stage ships a **baseline** with optional deferred
**candidate** hooks. This module curates that catalog once and merges the *live* parameter
values from :data:`qmt.config.CONFIG` at read time, so the page never drifts from the running
constants.

Kinds (``category``): 准入 / 数据 / 估计 / 市场状态 / 预测 / 不确定性 / 校准 / 组合 / 风险 /
行动 / 派生 / 回测基线.  Method (``method``): ``baseline`` (transparent implemented baseline) ·
``rule`` (config-driven gate) · ``derivation`` (labels derived from numbers, not predictive) ·
``artifact`` (data-driven table, e.g. calibration) · ``strategy`` (a backtest baseline).
Mode (``mode``): ``analysis`` · ``advice`` · ``both`` · ``evaluation``.

This is metadata only — it imports no pipeline module and runs no analysis.
"""

from __future__ import annotations

from typing import Optional

from qmt.config import CONFIG


# ---------------------------------------------------------------------------
# Curated catalog. Editorial fields (category/method/status/doc_ref/summary/
# candidates) live here; ``config_group`` links to a live CONFIG group whose
# current values are merged in by :func:`models_catalog`.
# ---------------------------------------------------------------------------
_MODELS: list[dict] = [
    {
        "id": "fund_scope",
        "name_zh": "基金分类 / 口径路由", "name_en": "Fund Scope",
        "category": "准入", "method": "rule", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.fund_scope", "config_group": "scope",
        "doc_ref": "§4 sec:mod-scope / §1 sec:scope / §3 sec:ds-scope / §6 gate 2-3",
        "summary": "按 fund_type / invest_type / name 将每只场外基金判入 13 类 ScopeCategory，"
                   "并据类型选择分析口径与支持档位：权益家族 FULL（完整链路）；债券/货币/QDII/FOF/"
                   "商品/养老/主动混合等 LIMITED（披露事实+风险来源+免责声明『该类型深度分析暂未支持』，"
                   "无强判断）；场内 ETF DEFERRED（仅识别）。不硬拒绝任何基金；全配置化、无网络、失败保守。",
        "candidates": ["按披露持仓校验类型（候选，未启用）", "对基金合同/名称做 NLP 归类（候选）"],
    },
    {
        "id": "data_visibility",
        "name_zh": "数据可见性", "name_en": "Data Visibility",
        "category": "数据", "method": "derivation", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.data_visibility", "config_group": "freshness",
        "doc_ref": "§4 sec:mod-visibility / §2.1 §2.3",
        "summary": "给每个数据字段打 可见/隐藏 + 新鲜度（fresh/stale/expired），"
                   "是 point-in-time 的第一道闸门。",
        "candidates": [],
    },
    {
        "id": "exposure_estimator",
        "name_zh": "真实暴露估计", "name_en": "Exposure Estimator",
        "category": "估计", "method": "baseline", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.exposure_estimator", "config_group": "exposure",
        "doc_ref": "§4 sec:mod-exposure / §1.4 §2.6",
        "summary": "三层输出：披露事实+龄 / 系统估计区间 / 对估计的置信度。基线按披露龄衰减"
                   "置信、放宽区间。当前发行 INDUSTRY（有行业图时）与注入的 STYLE 维度。",
        "candidates": ["持仓漂移 / 规模 / 换手候选（延期 D-EXP-*）",
                       "CAP / ASSET_CLASS 维度（缺数据源，延期 D-EXP-3）"],
    },
    {
        "id": "market_state",
        "name_zh": "市场状态（四维）", "name_en": "Market State",
        "category": "市场状态", "method": "baseline", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.market_state", "config_group": "market_state",
        "doc_ref": "§4 sec:mod-market / §1.5 §2.7",
        "summary": "四个独立维度：方向(MA20/60 交叉+中性带)、波动(已实现波动分位)、"
                   "压力(成交额分位)、事件(政策/冲击 flag)。绝不塌缩成单一标签。",
        "candidates": ["估值/广度喂入以触达 RISING_RISK（延期 D47）"],
    },
    {
        "id": "horizon_forecaster",
        "name_zh": "多周期预测", "name_en": "Horizon Forecaster",
        "category": "预测", "method": "baseline", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.horizon_forecaster", "config_group": "forecaster",
        "doc_ref": "§4 sec:mod-forecast / §2.2 §3.3",
        "summary": "1–3 / 5–20 / 20–90 交易日的收益与风险『区间』（非标签）：历史基线 + "
                   "单向市场状态风险叠加（高波动/压力/冲击只放宽区间、抬高回撤概率）。",
        "candidates": ["动量 vs 均值回复再加权（延期 D-FC-5）", "分层预测模型（已实现，见下）"],
    },
    {
        "id": "hierarchical_forecaster",
        "name_zh": "分层预测模型", "name_en": "Hierarchical Forecaster",
        "category": "预测", "method": "baseline", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.hierarchical_forecaster", "config_group": "hierarchical",
        "doc_ref": "源文档 §3.3 / techdesign §5.3 sec:algo-hierarchical",
        "summary": "个股预测的分层结构（候选【推断】，非默认；先基线后候选）：大盘 + 行业偏差 + 个股偏差"
                   "可加分解，参数相互独立、复杂度按层递减（大盘更完整 / 行业中等 / 个股偏差保持克制）。"
                   "大盘负载 E 取基金净值对基准的点位滚动 beta（estimate_market_beta，账面权重仅作兜底），"
                   "大盘层与行业层均按 E 缩放；个股不确定性带按 sqrt(h) 随周期缩放。"
                   "点位读取（大盘 index_daily、行业 sw_daily、暴露），复用基线的市场状态叠加。"
                   "由 forecaster.use_hierarchical 开关切换，可与基线用回测/命中率比较（tuning）。"
                   "逐层评测见「诊断三层模型架构」（每层对比自身朴素基线，D-HF-7..10 / D-LAY-*）。",
        "candidates": ["个股层引入真实持仓 alpha（缺当前持仓数据，延期）",
                       "把归因（大盘/行业/个股）接入 explanation 输出（延期）",
                       "各层引入更丰富输入：估值/资金面/政策/景气度（数据缺口，延期）"],
    },
    {
        "id": "uncertainty",
        "name_zh": "不确定性 / 可信度", "name_en": "Uncertainty",
        "category": "不确定性", "method": "baseline", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.uncertainty", "config_group": None,
        "doc_ref": "§4 sec:mod-uncertainty / §3.3",
        "summary": "原始可信度 = 暴露置信 × 预测离散度 × 市场粗糙度（各∈[0,1]）。"
                   "空预测→保守下限。原始分随后经历史校准（见校准模型）。",
        "candidates": ["方法间一致性 method_agreement（候选，未启用）"],
    },
    {
        "id": "calibration",
        "name_zh": "可信度历史校准表", "name_en": "Confidence Calibration",
        "category": "校准", "method": "artifact", "mode": "analysis", "status": "DONE",
        "module": "qmt.calibration", "config_group": "calibration",
        "doc_ref": "§4 sec:mod-uncertainty / §5 row UncertaintyM / §7",
        "summary": "用 `qmt calibrate` 在历史窗口回放各阶段，按方向命中率把原始可信度分桶，"
                   "持久化到 data_cache/calibration.json；run_analysis 默认加载并套用。"
                   "无表时报告原始分并显式标注『未校准』。",
        "candidates": [],
    },
    {
        "id": "portfolio_lookthrough",
        "name_zh": "组合穿透", "name_en": "Portfolio Look-through",
        "category": "组合", "method": "baseline", "mode": "advice", "status": "DONE",
        "module": "qmt.pipeline.portfolio_lookthrough", "config_group": None,
        "doc_ref": "§4 sec:mod-lookthrough / §2.8",
        "summary": "穿透用户多只基金看真实行业/风格集中度：加权合并集中度、主题重叠、"
                   "暴露余弦重叠、风险贡献 RC。仅建议模式（读用户资产）。",
        "candidates": ["NAV 相关 / 因子协方差 RC（候选，未启用）"],
    },
    {
        "id": "risk_budget",
        "name_zh": "风险预算", "name_en": "Risk Budget",
        "category": "风险", "method": "baseline", "mode": "advice", "status": "DONE",
        "module": "qmt.pipeline.risk_budget", "config_group": "risk_budget",
        "doc_ref": "§4 sec:mod-riskbudget / §1.8-9",
        "summary": "按适当性等级(C1–C5)在软/硬集中度上限间线性插值出单桶上限，"
                   "判断用户是否已过度暴露；超阈禁止『加仓』类输出。",
        "candidates": ["风险平价预算（候选，未启用）"],
    },
    {
        "id": "action_strategy",
        "name_zh": "行动策略", "name_en": "Action Strategy",
        "category": "行动", "method": "baseline", "mode": "advice", "status": "DONE",
        "module": "qmt.pipeline.action_strategy", "config_group": "action",
        "doc_ref": "§4 sec:mod-action / §3.4 §5",
        "summary": "行动 ≠ 预测：阈值触发 + 仓位/集中度/成本/持有期硬约束，净费口径。"
                   "常见输出是『维持，不追高』。同时读预测风险侧(vol_est/drawdown_prob)。",
        "candidates": ["决策层（效用最大化）候选（未启用）"],
    },
    {
        "id": "product_view",
        "name_zh": "产品视图派生", "name_en": "Product View",
        "category": "派生", "method": "derivation", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.product_view", "config_group": "product_view",
        "doc_ref": "§3 sec:ds-action / §4",
        "summary": "把数值区间翻译成用户语言的 方向/趋势强度/波动/可信度——全部『派生』，"
                   "绝不主观贴标签。",
        "candidates": [],
    },
    {
        "id": "explanation",
        "name_zh": "解释卡派生", "name_en": "Explanation Card",
        "category": "派生", "method": "derivation", "mode": "both", "status": "DONE",
        "module": "qmt.pipeline.explanation", "config_group": None,
        "doc_ref": "§4 sec:mod-explain / §7 sec:eval-explain",
        "summary": "固定五段解释（乐观/谨慎/未知/新鲜度/何时改变）+ 证据结构；"
                   "区分『推动判断的因素』与『真实原因』，绝不写因果。",
        "candidates": [],
    },
    # --- backtest baseline strategies (evaluation models, §7) -------------------
    {
        "id": "bt_buy_and_hold",
        "name_zh": "长期持有", "name_en": "Buy & Hold",
        "category": "回测基线", "method": "strategy", "mode": "evaluation", "status": "DONE",
        "module": "qmt.backtest.strategy_buy_and_hold", "config_group": None,
        "doc_ref": "§7 sec:test-baselines (基线1)",
        "summary": "始终满仓持有。候选策略必须净费后稳定跑赢这些朴素基线才算有增量价值。",
        "candidates": [],
    },
    {
        "id": "bt_fixed_sip",
        "name_zh": "固定定投", "name_en": "Fixed SIP",
        "category": "回测基线", "method": "strategy", "mode": "evaluation", "status": "DONE",
        "module": "qmt.backtest (fixed_sip accounting)", "config_group": None,
        "doc_ref": "§7 sec:test-baselines (基线2)",
        "summary": "按固定周期等额买入的定投基线（货币加权）。",
        "candidates": [],
    },
    {
        "id": "bt_ma_cross",
        "name_zh": "简单均线", "name_en": "MA Cross",
        "category": "回测基线", "method": "strategy", "mode": "evaluation", "status": "DONE",
        "module": "qmt.backtest.strategy_ma_cross", "config_group": "market_state",
        "doc_ref": "§7 sec:test-baselines (基线3) / §5",
        "summary": "MA_fast 上穿 MA_slow 满仓、下穿空仓（用 market_state 的 ma_fast/slow_days）。",
        "candidates": [],
    },
    {
        "id": "bt_trend",
        "name_zh": "简单趋势", "name_en": "Trend",
        "category": "回测基线", "method": "strategy", "mode": "evaluation", "status": "DONE",
        "module": "qmt.backtest.strategy_trend", "config_group": "market_state",
        "doc_ref": "§7 sec:test-baselines (基线4) / §5",
        "summary": "按近端趋势方向决定满仓/空仓的朴素趋势基线。",
        "candidates": [],
    },
    {
        "id": "bt_vol_control",
        "name_zh": "简单波动控制", "name_en": "Vol Control",
        "category": "回测基线", "method": "strategy", "mode": "evaluation", "status": "DONE",
        "module": "qmt.backtest.strategy_vol_control", "config_group": "market_state",
        "doc_ref": "§7 sec:test-baselines (基线5) / §5",
        "summary": "目标波动 / 已实现波动 决定仓位比例（0..1），薄历史时保守持币。",
        "candidates": [],
    },
]

# Ordered kinds for stable grouping on the page.
CATEGORY_ORDER = [
    "准入", "数据", "估计", "市场状态", "预测", "不确定性", "校准",
    "组合", "风险", "行动", "派生", "回测基线",
]


def _group_params(group: Optional[str]) -> Optional[dict]:
    """Live values of the referenced CONFIG group (or ``None`` when no group applies)."""
    if not group:
        return None
    sub = getattr(CONFIG, group, None)
    if sub is None:
        return None
    return sub.model_dump(mode="json") if hasattr(sub, "model_dump") else dict(sub)


def models_catalog() -> dict:
    """The full model catalog with live parameters merged in.

    Returns ``{"models": [...], "categories": [...], "counts": {...}}`` where each model
    carries its curated metadata plus a live ``params`` snapshot from CONFIG.
    """
    models = []
    for m in _MODELS:
        params = _group_params(m.get("config_group"))
        models.append({**m, "params": params})

    counts: dict[str, int] = {}
    for m in models:
        counts[m["category"]] = counts.get(m["category"], 0) + 1

    categories = [c for c in CATEGORY_ORDER if c in counts]
    return {
        "models": models,
        "categories": categories,
        "counts": counts,
        "total": len(models),
    }
