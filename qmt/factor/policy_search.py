"""遗传进化决策树策略 — 环境构建 / GA / 基线对照 / 验证(用户设计的四步落地)。

纪律(全部保留,§7.3/§7.4):
  * 决策日全部严格早于 ``locked_start``(2025-05-08 的封存区一步不碰);
    步奖励窗口 = 相邻决策日之间,天然不越界。
  * 进化只看**训练区**的三段折适应度(均值 − 0.5·std − 复杂度惩罚);
    **验证区只在最后评估一次**。
  * 每棵被评估的树(含随机对照臂)都记入 SearchLedger;验证期 IC 的显著性
    门槛按总试验数通胀:sqrt(2·ln N)。
  * **同预算随机搜索对照臂**:随机树用与 GA 完全相同的评估预算与选择规则;
    若随机 ≈ GA,结论必须写「进化机制未产生实质价值」(§7.4)。
  * 本引擎不改变正式四道门的 NO-GO 结论 —— 输出是研究证据,不是上线许可。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore, TradingCalendar
from qmt.stats import assess_significance, spearman
from qmt.factor import panel as _panel
from qmt.factor.base_factors import BaseFactors
from qmt.factor.types import SafePanel, SearchLedger, SearchTrial
from qmt.factor.policy import (
    FEATURE_ZH, EpisodeResult, PolicyEnv, PolicyNode,
    copy_tree, crossover_trees, eval_tree, mutate_tree, random_tree,
    render_one_line, render_tree, seed_trees, tree_depth, tree_nodes,
)

__all__ = ["PolicySearchConfig", "build_policy_env", "run_policy_search", "run_stride_sweep"]


@dataclass
class PolicySearchConfig:
    population: int = 24
    generations: int = 8
    seed: int = 7
    max_depth: int = 4
    max_nodes: int = 15
    stride_days: int = 10
    buy_fee: float = 0.0015           # 申购费(买入腿)
    sell_fee: float = 0.005           # 短持赎回费(卖出腿;期末强制清算也按此计)
    top_k: int = 20
    hysteresis: float = 0.75          # 滞回带宽:已持仓分数 > -hysteresis 就保留,
                                      # 只在明显跌出(<= -hysteresis)时替换 —— 把换手
                                      # 从 ~0.40 压到 ~0.05,让毛信号有机会在费后存活
    train_frac: float = 0.6
    lambda_std: float = 0.5           # 三段折不稳定惩罚
    lambda_nodes: float = 0.0005      # 复杂度惩罚(每节点;费用已内生惩罚活跃度,
                                      # 0.004 时惩罚占优质树适应度 57%,搜索退化为
                                      # 「选择不作为」—— 怀疑者实证后调低一个量级)
    universe_min_history: str = "2024-12-31"
    locked_start: str = "2025-05-08"  # 封存区锚点 — 只能更晚,不能提前
    min_date: str = "2014-06-01"
    start_frac: float = 0.35
    random_budget: int | None = None  # None → 与 GA 总评估数相同


# --------------------------------------------------------------------------- #
# 环境构建
# --------------------------------------------------------------------------- #
def build_policy_env(
    store: PointInTimeStore,
    cfg: PolicySearchConfig,
    progress: Callable[[str], None] = lambda s: None,
) -> tuple[list[PolicyEnv], PolicyEnv, dict[str, Any]]:
    """构建训练三段折与验证环境(共享分位/奖励字典,省内存)。"""
    cal = TradingCalendar()
    nav = store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
    nav["nav_date"] = pd.to_datetime(nav["nav_date"])
    per = nav.groupby("fund_code")["nav_date"].agg(["min"])
    codes = sorted(per[per["min"] <= pd.Timestamp(cfg.universe_min_history)].index)
    del nav
    progress(f"宇宙 {len(codes)} 只基金;构建价格帧…")

    pf = _panel.usable_price_frame(store, "fund_nav", value_col="adj_nav",
                                   code_col="fund_code", date_col="nav_date",
                                   codes=codes, calendar=cal)
    pf = pf.loc[pf.index >= pd.Timestamp(cfg.min_date).date()]

    mkt = _panel.usable_price_frame(store, "index_daily", value_col="close",
                                    code_col="ts_code", date_col="trade_date", calendar=cal)
    market_px = mkt["000300.SH"] if "000300.SH" in mkt.columns else None

    progress(f"价格帧 {pf.shape};计算特征(vol/mom/dd/beta/drift/flow)…")
    sp = SafePanel(as_of=pf.index.max(), frame=pf, universe=list(pf.columns))
    extra: dict[str, pd.DataFrame] = {}
    if market_px is not None:
        extra["beta"] = _panel.rolling_beta_frame(pf, market_px)
        from qmt.factor.flows import build_drift_frame, build_flow_frame
        extra["drift"] = build_drift_frame(pf, market_px)
        fl = build_flow_frame(store, codes, pf.index)
        if not fl.empty and fl.notna().any().any():
            extra["flow"] = fl
    bpanel = BaseFactors(extra=extra).compute_base_factors(sp, pf.index.max())

    features = [f for f in FEATURE_ZH if f in bpanel.factors]
    progress(f"可用特征: {features}")

    # 决策日:锚定网格 —— 只用严格早于锁定区的行,固定预热 320 行后按 stride 取。
    # (此前用全样本长度的分位数取起点,封存区内多进几个月数据会平移整个网格,
    #  训练/验证切点随之漂移 —— 怀疑者实证 0/180 重合。锚定后网格与封存区后的
    #  数据量无关。)
    locked_start = pd.Timestamp(cfg.locked_start).date()
    pre_lock_rows = [d for d in pf.index if d < locked_start]
    dates = pre_lock_rows[320::cfg.stride_days]
    if len(dates) < 30:
        raise ValueError(f"决策日不足({len(dates)})——检查数据跨度")

    idx_arr = np.asarray(pf.index)

    # 步奖励:d 的**下一可执行**净值(p0+1)→ 下一决策日的下一可执行净值(p1+1)。
    # 按已公布净值建仓是不可成交的(怀疑者实证:提前一天使各策略验证收益偏移
    # −2.7%~+5.8%,足以翻转排序)。出场行必须仍在封存区之前。
    rel_fwd: dict[date, pd.Series] = {}
    n_rows = len(pf)
    for d0, d1 in zip(dates[:-1], dates[1:]):
        p0 = int(np.searchsorted(idx_arr, d0, side="right")) - 1 + 1   # 可执行入场
        p1 = int(np.searchsorted(idx_arr, d1, side="right")) - 1 + 1   # 可执行出场
        if p0 < 1 or p1 <= p0 or p1 >= n_rows:
            continue
        if pf.index[p1] >= locked_start:                               # 封存区守卫
            continue
        entry = pf.iloc[p0]
        exit_ = pf.iloc[p0:p1 + 1].ffill().iloc[-1]
        fwd = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if fwd.empty:
            continue
        rel_fwd[d0] = fwd - fwd.mean()

    # 截面分位:每个决策日只含**当日真实有数据**的基金(幽灵不进截面 —— 此前
    # NaN→0.5 让 72.8% 未成立基金参与打分并可入选组合,怀疑者实证的高危缺陷)。
    # 可选特征(beta/drift/flow)按「价格有效基金中的覆盖率」过滤:平均覆盖率
    # < 60% 的整体剔除(flow 实测 6–8% —— 分裂在它上面等于在选『有无披露』)。
    progress(f"决策日 {len(dates)} 个({dates[0]} → {dates[-1]},锚定网格,全部早于封存区 {locked_start});抽取真实截面…")
    core = [f for f in ("vol20", "vol60", "mom20", "mom60", "dd60") if f in features]
    optional = [f for f in features if f not in core]
    rows_cache: dict[str, dict[date, pd.Series]] = {}
    coverage: dict[str, float] = {}
    for f in features:
        frame = bpanel.factors[f]
        f_idx = np.asarray(frame.index)
        rows_cache[f] = {}
        covs = []
        for d in dates:
            pos = int(np.searchsorted(f_idx, d, side="right")) - 1
            row = frame.iloc[pos] if pos >= 0 else pd.Series(np.nan, index=frame.columns)
            rows_cache[f][d] = row
        # 覆盖率以核心价格宇宙为分母
        base = rows_cache.get(core[0]) if core else None
        for d in dates:
            uni = rows_cache[core[0]][d].dropna().index if core else frame.columns
            if len(uni):
                covs.append(float(rows_cache[f][d].reindex(uni).notna().mean()))
        coverage[f] = round(float(np.mean(covs)), 3) if covs else 0.0
    kept_optional = [f for f in optional if coverage[f] >= 0.6]
    dropped = [f for f in optional if f not in kept_optional]
    features_final = core + kept_optional
    if dropped:
        progress(f"  剔除低覆盖特征(<60%): {dropped}(覆盖率 {[coverage[f] for f in dropped]})")

    pct_by_date: dict[date, dict[str, pd.Series]] = {}
    for d in dates:
        uni = rows_cache[core[0]][d].dropna().index if core else pf.columns
        per: dict[str, pd.Series] = {}
        for f in features_final:
            row = rows_cache[f][d].reindex(uni)
            pct = row.rank(pct=True)
            per[f] = pct.fillna(0.5)      # 仅限已过覆盖率门槛特征的零星缺失
        pct_by_date[d] = per
    del bpanel, extra, sp, rows_cache

    # 中性并列打破:一次性种子随机优先序(与成立年代/代码序无关,粘性低换手)
    priority = pd.Series(
        np.random.default_rng(20240703).permutation(len(pf.columns)),
        index=pf.columns, dtype=float,
    )

    # 切分:训练(三段折)→ embargo 1 步 → 验证
    cut = int(len(dates) * cfg.train_frac)
    train_dates = dates[:cut]
    valid_dates = dates[cut + 1:]
    folds = [list(seg) for seg in np.array_split(np.array(train_dates, dtype=object), 3)]
    mk = lambda ds: PolicyEnv(dates=list(ds), pct_by_date=pct_by_date, rel_fwd=rel_fwd,
                              features=features_final, buy_fee=cfg.buy_fee,
                              sell_fee=cfg.sell_fee, top_k=cfg.top_k, priority=priority,
                              hysteresis=cfg.hysteresis)
    meta = {
        "n_funds": len(codes), "n_dates": len(dates),
        "train_span": [str(train_dates[0]), str(train_dates[-1])],
        "valid_span": [str(valid_dates[0]), str(valid_dates[-1])],
        "locked_start": str(locked_start), "features": features_final,
        "feature_coverage": coverage, "dropped_features": dropped,
        "tradable_per_date_median": int(np.median([len(v) for v in rel_fwd.values()])) if rel_fwd else 0,
    }
    return [mk(f) for f in folds], mk(valid_dates), meta


# --------------------------------------------------------------------------- #
# 进化
# --------------------------------------------------------------------------- #
def _fitness(tree: PolicyNode, folds: list[PolicyEnv], cfg: PolicySearchConfig,
             cache: dict[str, float], ledger: SearchLedger, note: str,
             counter: list[int]) -> float:
    key = render_one_line(tree)
    if key in cache:
        return cache[key]
    ledger.view_dataset()
    cums = [f.episode(tree).cum_log_reward for f in folds]
    fit = float(np.mean(cums) - cfg.lambda_std * np.std(cums)
                - cfg.lambda_nodes * tree_nodes(tree))
    cache[key] = fit
    counter[0] += 1
    ledger.record(SearchTrial(trial_id=counter[0], expr_text=key, passed_dsl=True,
                              select_ic=fit, note=note))
    return fit


def _evolve(folds: list[PolicyEnv], cfg: PolicySearchConfig, ledger: SearchLedger,
            progress: Callable[[str], None]) -> tuple[list[tuple[PolicyNode, float]], int]:
    rng = np.random.default_rng(cfg.seed)
    features = folds[0].features
    cache: dict[str, float] = {}
    counter = [0]

    pop: list[PolicyNode] = list(seed_trees(features).values())
    while len(pop) < cfg.population:
        pop.append(random_tree(rng, features, depth_budget=3))
    pop = [t for t in pop if tree_nodes(t) <= cfg.max_nodes]
    fits = {render_one_line(t): _fitness(t, folds, cfg, cache, ledger, "init", counter) for t in pop}

    for gen in range(1, cfg.generations + 1):
        ranked = sorted(pop, key=lambda t: fits[render_one_line(t)], reverse=True)
        elite = ranked[: max(2, cfg.population // 5)]
        new_pop = [copy_tree(t) for t in elite]
        guard = 0
        while len(new_pop) < cfg.population and guard < cfg.population * 30:
            guard += 1
            def tourney():
                cand = [pop[int(i)] for i in rng.integers(len(pop), size=3)]
                return max(cand, key=lambda t: fits[render_one_line(t)])
            child = (crossover_trees(rng, tourney(), tourney())
                     if rng.random() < 0.5 else mutate_tree(rng, tourney(), features, cfg.max_depth))
            if tree_nodes(child) > cfg.max_nodes or tree_depth(child) > cfg.max_depth + 1:
                continue
            fits[render_one_line(child)] = _fitness(child, folds, cfg, cache, ledger,
                                                    f"gen{gen}", counter)
            new_pop.append(child)
        pop = new_pop
        best = max(pop, key=lambda t: fits[render_one_line(t)])
        bf = fits[render_one_line(best)]
        progress(f"  第 {gen}/{cfg.generations} 代:最优适应度 {bf:+.4f},累计评估 {counter[0]} 棵")

    uniq: dict[str, PolicyNode] = {render_one_line(t): t for t in pop}
    ranked = sorted(uniq.values(), key=lambda t: fits[render_one_line(t)], reverse=True)
    return [(t, fits[render_one_line(t)]) for t in ranked], counter[0]


def _random_arm(folds: list[PolicyEnv], cfg: PolicySearchConfig, budget: int,
                ledger: SearchLedger, progress: Callable[[str], None]) -> tuple[PolicyNode, float, int]:
    """同预算随机搜索对照臂(§7.4)。独立 rng,同样的适应度与选择规则。"""
    rng = np.random.default_rng(cfg.seed + 1000)
    features = folds[0].features
    cache: dict[str, float] = {}
    counter = [0]
    best, best_fit = None, -1e18
    while counter[0] < budget:
        t = random_tree(rng, features, depth_budget=cfg.max_depth)
        if tree_nodes(t) > cfg.max_nodes:
            continue
        f = _fitness(t, folds, cfg, cache, ledger, "random_arm", counter)
        if f > best_fit:
            best, best_fit = t, f
    progress(f"  随机对照臂:{counter[0]} 棵,最优适应度 {best_fit:+.4f}")
    return best, best_fit, counter[0]


# --------------------------------------------------------------------------- #
# 验证与报告
# --------------------------------------------------------------------------- #
def _valid_metrics(tree: PolicyNode, env: PolicyEnv, min_abs_t: float) -> dict[str, Any]:
    ep = env.episode(tree)
    # 每步策略分数与下一期相对收益的 rank-IC(相邻窗口不重叠 → overlap=1)
    ics = []
    for d in env.dates:
        if d not in env.rel_fwd:
            continue
        s = eval_tree(tree, env.pct_by_date[d])
        joined = pd.concat([s.rename("s"), env.rel_fwd[d].rename("y")], axis=1).dropna()
        if joined["s"].std() > 0 and len(joined) >= 10:
            ics.append(spearman(joined["s"].to_numpy(), joined["y"].to_numpy()))
    ics = [v for v in ics if not math.isnan(v)]
    # overlap=2:相邻步窗口首尾相接虽不重叠,但步收益存在机制性序列相关;
    # overlap=1 会让三重校正全部退化为朴素 t(怀疑者实证)—— 保守取 HAC lag 1。
    sig = assess_significance(np.array(ics), overlap=2, min_abs_t=min_abs_t) if ics else None
    return {
        "cum_log_reward": round(ep.cum_log_reward, 4),
        "total_net_rel_return": round(float(np.expm1(ep.cum_log_reward)), 4),
        "avg_turnover": round(ep.avg_turnover, 3),
        "active_share": round(ep.active_share, 2),
        "n_steps": ep.n_steps,
        "mean_rank_ic": round(float(np.mean(ics)), 4) if ics else None,
        "ic_tstat_nw": round(sig.t_newey_west, 2) if sig else None,
        "ic_significant": bool(sig.significant) if sig else False,
    }


def run_policy_search(
    store: PointInTimeStore,
    cfg: PolicySearchConfig | None = None,
    progress: Callable[[str], None] = print,
    artifact_path: str | Path = "artifacts/policy_search.json",
) -> dict[str, Any]:
    cfg = cfg or PolicySearchConfig()
    t0 = time.time()
    folds, valid_env, meta = build_policy_env(store, cfg, progress)
    ledger = SearchLedger()
    ledger.lock_test_span((pd.Timestamp(cfg.locked_start).date(), date(2100, 1, 1)))

    progress(f"开始进化:种群 {cfg.population} × {cfg.generations} 代(基线播种 + 随机)…")
    ranked, n_ga = _evolve(folds, cfg, ledger, progress)
    budget = cfg.random_budget or n_ga
    progress(f"随机搜索对照臂(同预算 {budget} 棵)…")
    rnd_best, rnd_fit, n_rnd = _random_arm(folds, cfg, budget, ledger, progress)

    bar = max(2.0, float(np.sqrt(2.0 * np.log(max(ledger.n_trials, 2)))))
    progress(f"验证区一次性评估(显著性门槛 t>{bar:.2f},按 {ledger.n_trials} 次试验通胀)…")

    ga_best, ga_fit = ranked[0]
    results: dict[str, Any] = {}
    results["ga_best"] = {"tree": render_tree(ga_best), "one_line": render_one_line(ga_best),
                          "train_fitness": round(ga_fit, 4), "n_nodes": tree_nodes(ga_best),
                          **_valid_metrics(ga_best, valid_env, bar)}
    results["random_arm_best"] = {"one_line": render_one_line(rnd_best),
                                  "train_fitness": round(rnd_fit, 4),
                                  **_valid_metrics(rnd_best, valid_env, bar)}
    for name, tree in seed_trees(valid_env.features).items():
        results[name] = {"one_line": render_one_line(tree),
                         **_valid_metrics(tree, valid_env, bar)}

    # 诚实结论(§7.4)
    ga_v = results["ga_best"]["cum_log_reward"]
    rnd_v = results["random_arm_best"]["cum_log_reward"]
    seed_best = max((v["cum_log_reward"] for k, v in results.items() if k.startswith("seed:")),
                    default=0.0)
    verdicts = []
    if ga_v <= 0:
        verdicts.append("进化最优策略在验证区的累计净奖励不为正 —— 没有可用策略。")
    if ga_v <= rnd_v + 1e-6:
        verdicts.append("遗传进化未超过同预算随机搜索(§7.4)—— 进化机制本身未产生实质价值。")
    if ga_v <= seed_best + 1e-6:
        verdicts.append("进化结果未超过静态基线树(§5.6)—— 搜索没有找到超出已知规则的增量。")
    if not results["ga_best"]["ic_significant"]:
        verdicts.append(f"验证区 IC 未过通胀门槛(t 门槛 {bar:.2f})—— 统计上不能排除无效应。")
    if not verdicts:
        verdicts.append("进化策略在验证区同时超过零线、随机对照臂与静态基线,且 IC 过通胀门槛 —— "
                        "值得进入下一步(正式四道门与更长样本外观察);仍不改变当前 NO-GO。")

    report = {
        "config": asdict(cfg), "meta": meta,
        "n_trials_total": ledger.n_trials, "deflated_min_abs_t": round(bar, 2),
        "results": results, "verdict": " ".join(verdicts),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "本引擎输出为研究证据;不改变正式四道门的 NO-GO 结论。",
    }
    p = Path(artifact_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"完成({report['elapsed_s']}s),artifact → {p}")
    return report


def run_stride_sweep(
    store: PointInTimeStore,
    strides: list[int],
    base_cfg: PolicySearchConfig | None = None,
    progress: Callable[[str], None] = print,
    artifact_path: str | Path = "artifacts/policy_sweep.json",
) -> dict[str, Any]:
    """在多个调仓步长上跑同一套进化 + 基线,看信号能否在费后存活(费用随步长摊薄)。

    每个 stride 独立建环境、进化、评估;汇总为一张跨步长对照表。滞回(hysteresis)
    始终开启 —— 本扫描回答的正是「粘性持仓 + 更低频调仓」能否救回费后收益。
    """
    base = base_cfg or PolicySearchConfig()
    rows: list[dict[str, Any]] = []
    for s in strides:
        freq = {10: "双周", 20: "月频", 40: "季频"}.get(s, f"{s}日")
        progress(f"\n═══ 步长 {s}({freq}),滞回 {base.hysteresis} ═══")
        cfg = PolicySearchConfig(**{**asdict(base), "stride_days": s})
        rep = run_policy_search(store, cfg, progress,
                                artifact_path=f"artifacts/policy_sweep_stride{s}.json")
        r = rep["results"]
        def _row(name):
            v = r.get(name, {})
            return {"strategy": name, "stride": s, "freq": freq,
                    "cum_log_reward": v.get("cum_log_reward"),
                    "net_rel_return": v.get("total_net_rel_return"),
                    "avg_turnover": v.get("avg_turnover"),
                    "mean_rank_ic": v.get("mean_rank_ic"),
                    "ic_tstat": v.get("ic_tstat_nw"),
                    "ic_significant": v.get("ic_significant")}
        for name in ["ga_best", "random_arm_best", "seed:low_vol", "seed:low_drawdown"]:
            rows.append(_row(name))
        rows.append({"strategy": "_meta", "stride": s, "freq": freq,
                     "n_dates": rep["meta"]["n_dates"], "bar": rep["deflated_min_abs_t"],
                     "verdict": rep["verdict"]})

    # 跨步长最优:任一 stride 下 low_vol 净收益是否转正 + 显著
    lowvol = [x for x in rows if x["strategy"] == "seed:low_vol"]
    best_lv = max(lowvol, key=lambda x: (x.get("net_rel_return") or -1e9)) if lowvol else None
    any_survives = bool(best_lv and (best_lv.get("net_rel_return") or 0) > 0
                        and best_lv.get("ic_significant"))
    summary = {
        "strides": strides, "hysteresis": base.hysteresis, "rows": rows,
        "best_low_vol": best_lv,
        "verdict": (
            "在某个调仓频率下,低波动策略费后净收益转正且 IC 过通胀门槛 —— 值得进一步观察。"
            if any_survives else
            "跨所有调仓频率,费后净收益仍未同时满足(转正 + IC 显著);粘性 + 降频缩小了成本"
            "但未逆转结论 —— 仍为 NO-GO。"),
        "note": "研究证据;不改变四道门 NO-GO。",
    }
    p = Path(artifact_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"\n扫描完成,汇总 → {p}")
    return summary
