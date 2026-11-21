"""auto_search — 符号学习因子搜索（instruction §2.1/§5.6/§7.3/§7.4；tech doc ch.4 §sec:mod-search）。

纯 Python 遗传编程（GP），直接作用在 ``qmt.factor.dsl`` 的 AST 上，不引入 gplearn/deap。
遵循「基线优先阶梯」（§5.6/§7.4）：

    1. **人工因子播种**（seed_from_manual）：从 :func:`baselines.manual_exprs` 取经典
       因子做初始种群——符号学习必须先追平、再谈超越（§7.4）。
    2. **随机公式生成**（:meth:`random_expr`）：在 DSL 白名单算子 + ``cfg.allowed_windows``
       + ``panel.factors`` 叶子上随机生长 AST，受 ``max_complexity/max_depth`` 硬约束，
       ``dsl.validate`` 过滤非法。
    3. **小型遗传编程**：按 ``split.train_span`` 内固定时间分段的截面 IC 适应度
       做锦标赛选择 + 子树变异（:meth:`_mutate`）+ 子树交叉（:meth:`_crossover`），迭代
       ``cfg.generations`` 代。
    4. **全程记账**：每次尝试（含失败者、含 DSL 拒绝者）写 :class:`SearchTrial` 进 ledger，
       每次查看数据 :meth:`SearchLedger.view_dataset`；受 max_trials 账本预算限制，
       返回所有代中去重、复杂度受控的 top-K；可选训练行为相关性筛选。

关键不变量（§7.3）：
  * 只在 ``dsl.validate`` 放行的空间里变异/组合——搜索空间受 DSL 白名单约束。
  * 绝不用验证期或封存测试期评分——适应度仅取 ``split.train_span``，且 ``dsl.evaluate``
    在每个 ``as_of`` 只用 ``<= as_of`` 的行（点时点安全由 DSL 保证）。
  * 全部随机性来自 ``numpy`` 且以 ``cfg.seed`` 播种——同 seed 两次搜索得到相同 top 文本。
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from qmt.factor.dsl import (
    ALL_OPS,
    BIN_OPS,
    CS_OPS,
    EW_UNARY,
    TS_OPS,
    Binary,
    Const,
    Node,
    TsOp,
    Unary,
    Var,
    Where,
    fields_of,
    make_expr,
    walk,
)
from qmt.factor.baselines import manual_exprs
from qmt.factor.diversity import rank_signature, select_diverse
from qmt.factor.metrics import per_date_ic
from qmt.factor.panel import labels_by_date as _labels_by_date
from qmt.factor.types import (
    BaseFactorPanel,
    FactorDSL,
    FactorExpr,
    Label,
    SearchConfig,
    SearchLedger,
    SearchTrial,
    ValidationSplit,
)

__all__ = ["SymbolicSearch", "DEFAULT_TOP_K"]

# 默认返回的候选数量上限（去重后）。
DEFAULT_TOP_K = 10
# 无法评分（无数据/异常）时的哨兵适应度：排序时永远垫底。
_NEG_INF = float("-inf")
# 随机公式里可能出现的常数叶子（保守小集合，保证确定性）。
_CONSTS = (0.5, 1.0, 2.0)


class SymbolicSearch:
    """具体的 :class:`qmt.factor.types.AutoSearch` 实现（纯 Python GP）。"""

    def __init__(self, dsl: FactorDSL, industry_of: dict[str, str] | None = None) -> None:
        # 默认 DSL；:meth:`search` 会用调用方传入的 dsl 覆盖，二者应一致。
        self._dsl: FactorDSL = dsl
        self.industry_of = industry_of or {}
        # 由 _configure 填充的搜索空间（sorted → 确定性索引）。
        self._fields: list[str] = []
        self._ts: list[str] = []
        self._ew: list[str] = []
        self._cs: list[str] = []
        self._bin: list[str] = []
        self._windows: list[int] = []
        self._use_where = False
        # 运行期状态
        self._trial_counter = 0
        self._fit_cache: dict[str, float] = {}
        self._train_signatures: dict[str, dict] = {}
        self._diversity_enabled = False
        self._min_train_ic_coverage = 0.8
        self._candidates: dict[str, FactorExpr] = {}

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def search(
        self,
        cfg: SearchConfig,
        dsl: FactorDSL,
        panel: BaseFactorPanel,
        labels: Sequence[Label],
        split: ValidationSplit,
        ledger: SearchLedger,
    ) -> list[FactorExpr]:
        """搜索仅使用训练数据，返回预算内所有已试候选中的 top-K。"""
        ledger.assert_test_untouched()
        if cfg.max_trials < 0:
            raise ValueError("max_trials must be nonnegative")
        if cfg.population < 1:
            raise ValueError("population must be positive")
        self._dsl = dsl
        self._configure(cfg, panel)
        rng = np.random.default_rng(cfg.seed)
        self._trial_counter = ledger.n_trials
        self._fit_cache = {}
        self._train_signatures = {}
        self._diversity_enabled = cfg.diversity_max_corr is not None
        self._min_train_ic_coverage = cfg.min_train_ic_coverage
        if not 0 < self._min_train_ic_coverage <= 1:
            raise ValueError("min_train_ic_coverage must lie in (0, 1]")
        self._candidates = {}
        # Validate the pre-registered diversity control before any data views.
        select_diverse([], {}, limit=1, max_corr=cfg.diversity_max_corr,
                       min_dates=cfg.diversity_min_dates)

        lo, hi = split.train_span
        train_labels = [lab for lab in labels if lo <= lab.as_of <= hi]
        labmap = _labels_by_date(train_labels)
        # Fit on the TRAIN region only (never the validation window screening scores
        # on) — closes the previous in-sample leak. Fitness is cross-fold robust so
        # the search selects for consistency across sub-periods, not a lucky window.
        select_dates = self._fit_dates(split, labmap)
        self._assert_training_unsealed(panel, train_labels, select_dates, ledger,
                                       valid_start=split.valid_span[0])
        # Keep the causal full-frame fast path, but never pass validation or locked
        # feature values to it. Earlier history remains available for rolling ops.
        panel = self._training_panel(panel, max(select_dates))
        self._before_search(cfg, ledger)

        # --- 1) 人工因子播种（§5.6） ---------------------------------
        population: list[FactorExpr] = []
        seeds = manual_exprs(dsl, available_fields=set(panel.factors)) if cfg.seed_from_manual else {}
        for name, expr in seeds.items():
            if not self._consider(ledger, expr, cfg, panel, labmap, select_dates,
                                  parent_id=None, generation=0, note=f"seed:{name}",
                                  accept_into=population):
                break

        # --- 2) 随机公式填充至 population 规模 -------------------------
        attempts = 0
        max_attempts = max(50, cfg.population * 20)
        while (len(population) < cfg.population and attempts < max_attempts
               and ledger.n_trials < cfg.max_trials):
            attempts += 1
            expr = self._proposal_expr(cfg, rng)
            self._consider(ledger, expr, cfg, panel, labmap, select_dates,
                           parent_id=None, generation=0, note="random",
                           accept_into=population)

        population = self._dedup(population)
        if not population:
            return []
        population = sorted(population, key=self._fit_key, reverse=True)[:cfg.population]

        # --- 3) 小型遗传编程 ------------------------------------------
        for gen in range(1, cfg.generations + 1):
            if ledger.n_trials >= cfg.max_trials:
                break
            ranked = sorted(population, key=self._fit_key, reverse=True)
            n_elite = max(1, cfg.population // 5)
            new_pop = select_diverse(
                ranked, self._train_signatures, limit=n_elite,
                max_corr=cfg.diversity_max_corr, min_dates=cfg.diversity_min_dates,
            )
            guard = 0
            guard_cap = max(50, cfg.population * 30)
            while (len(new_pop) < cfg.population and guard < guard_cap
                   and ledger.n_trials < cfg.max_trials):
                guard += 1
                action = self._choose_operator(rng, population, cfg, gen)
                finite = [v for v in self._fit_cache.values() if np.isfinite(v)]
                before_best = max(finite) if finite else None
                before_population = list(population)
                if action == "crossover":
                    p1 = self._tournament(population, rng)
                    p2 = self._tournament(population, rng)
                    child = self._crossover(p1, p2, cfg, rng)
                    note, parent_id = "crossover", p1.text
                elif action == "mutate":
                    p1 = self._tournament(population, rng)
                    child = self._mutate(p1, cfg, rng)
                    note, parent_id = "mutate", p1.text
                elif action == "restart":
                    child = self._proposal_expr(cfg, rng)
                    note, parent_id = "restart", None
                else:
                    raise ValueError(f"unsupported evolutionary operator: {action}")
                was_known = child is not None and child.text in self._fit_cache
                if child is None:
                    # Failed construction is still a proposal. An adaptive
                    # controller cannot obtain uncharged retries of weak arms.
                    self._trial_counter += 1
                    ledger.record(SearchTrial(self._trial_counter, "<no_candidate>", False,
                                              parent_id=parent_id, generation=gen,
                                              note=note + ":no_candidate"))
                else:
                    self._consider(ledger, child, cfg, panel, labmap, select_dates,
                                   parent_id=parent_id, generation=gen, note=note,
                                   accept_into=new_pop)
                terminal = (ledger.n_trials >= cfg.max_trials or
                            (gen == cfg.generations and
                             (len(new_pop) >= cfg.population or guard >= guard_cap)))
                self._proposal_feedback(action=action, before_population=before_population,
                                        child=child, trial=ledger.trials[-1], cfg=cfg,
                                        generation=gen, terminal=terminal,
                                        before_best=before_best, was_known=was_known)
            population = self._dedup(new_pop)

        # --- 4) 汇总 top-K（去重、复杂度受控） -------------------------
        ranked = sorted(self._candidates.values(), key=self._fit_key, reverse=True)
        return self._select_candidates(ranked, cfg)

    def _before_search(self, cfg: SearchConfig, ledger: SearchLedger) -> None:
        """Reset an optional train-only proposal controller after safety checks."""

    def _choose_operator(self, rng, population, cfg, generation) -> str:
        """The fixed GP baseline keeps its 50/50 crossover/mutation policy."""
        return "crossover" if rng.random() < 0.5 else "mutate"

    def _proposal_feedback(self, **outcome) -> None:
        """Optional controller hook; fitness is already computed on discovery data."""

    def _proposal_expr(self, cfg: SearchConfig, rng: np.random.Generator) -> FactorExpr:
        """One raw random proposal, including DSL rejection in the trial budget.

        The public ``random_expr`` convenience helper still repairs until legal;
        search initialization/restarts do not use those hidden retries.
        Genetic subtree repair is label-blind and shared across every search arm.
        """
        depth = int(rng.integers(2, max(3, cfg.max_depth + 1)))
        return make_expr(self._grow(rng, depth, cfg))

    def _select_candidates(self, ranked: list[FactorExpr], cfg: SearchConfig) -> list[FactorExpr]:
        """Final selection hook; ``ranked`` contains every usable training candidate."""
        top = [e for e in ranked if e.complexity <= cfg.max_complexity]
        return select_diverse(
            top, self._train_signatures, limit=DEFAULT_TOP_K,
            max_corr=cfg.diversity_max_corr, min_dates=cfg.diversity_min_dates,
        )

    # ------------------------------------------------------------------
    # 随机公式生成
    # ------------------------------------------------------------------
    def random_expr(self, cfg: SearchConfig, rng: np.random.Generator) -> FactorExpr:
        """在 DSL 白名单空间里随机生长一棵合法 AST（不成功则退化为简单因子）。"""
        low, high = 2, max(3, cfg.max_depth + 1)
        for _ in range(40):
            depth = int(rng.integers(low, high))
            node = self._grow(rng, depth, cfg)
            if not fields_of(node):  # 常数-only 会被 validate 拒，提前剪枝
                continue
            expr = make_expr(node)
            if self._dsl.validate(expr, cfg).ok:
                return expr
        # 兜底：单叶子（或 rank(叶子)）——保证合法且非空
        leaf = Var(self._pick(rng, self._fields)) if self._fields else Var("ret")
        if self._cs:
            cand = make_expr(Unary(self._pick(rng, self._cs), leaf))
            if self._dsl.validate(cand, cfg).ok:
                return cand
        return make_expr(leaf)

    def _grow(self, rng: np.random.Generator, depth: int, cfg: SearchConfig) -> Node:
        """递归生长 AST；``depth`` 为剩余深度预算。"""
        if depth <= 1 or not self._fields:
            return Var(self._pick(rng, self._fields)) if self._fields else Var("ret")
        opts: list[str] = ["leaf"]
        if self._ts:
            opts.append("ts")
        if self._ew:
            opts.append("ew")
        if self._cs:
            opts.append("cs")
        if self._bin:
            opts.append("bin")
        if self._use_where and depth >= 3:
            opts.append("where")
        kind = self._pick(rng, opts)
        if kind == "leaf":
            return Var(self._pick(rng, self._fields))
        if kind == "ts":
            return TsOp(self._pick(rng, self._ts), self._pick(rng, self._windows),
                        self._grow(rng, depth - 1, cfg))
        if kind == "ew":
            return Unary(self._pick(rng, self._ew), self._grow(rng, depth - 1, cfg))
        if kind == "cs":
            return Unary(self._pick(rng, self._cs), self._grow(rng, depth - 1, cfg))
        if kind == "bin":
            left = self._grow(rng, depth - 1, cfg)
            if rng.random() < 0.15:
                right: Node = Const(float(self._pick(rng, list(_CONSTS))))
            else:
                right = self._grow(rng, depth - 1, cfg)
            return Binary(self._pick(rng, self._bin), left, right)
        # where
        return Where(self._grow(rng, depth - 1, cfg),
                     self._grow(rng, depth - 1, cfg),
                     self._grow(rng, depth - 1, cfg))

    # ------------------------------------------------------------------
    # 遗传算子
    # ------------------------------------------------------------------
    def _mutate(self, expr: FactorExpr, cfg: SearchConfig, rng: np.random.Generator) -> FactorExpr | None:
        """子树变异：随机选一个节点，替换为新生长的子树。"""
        root = expr.ast
        nodes = list(walk(root))
        target = int(rng.integers(len(nodes)))
        sub_depth = int(rng.integers(1, max(2, cfg.max_depth)))
        replacement = self._grow(rng, sub_depth + 1, cfg)
        new_root = self._rebuild(root, [0], target, replacement)
        if not fields_of(new_root):
            return None
        return make_expr(new_root)

    def _crossover(self, a: FactorExpr, b: FactorExpr, cfg: SearchConfig,
                   rng: np.random.Generator) -> FactorExpr | None:
        """子树交叉：把 b 的一个随机子树嫁接到 a 的一个随机节点上。"""
        root = a.ast
        donor_nodes = list(walk(b.ast))
        donor = self._copy(donor_nodes[int(rng.integers(len(donor_nodes)))])
        nodes = list(walk(root))
        target = int(rng.integers(len(nodes)))
        new_root = self._rebuild(root, [0], target, donor)
        if not fields_of(new_root):
            return None
        return make_expr(new_root)

    def _rebuild(self, node: Node, counter: list[int], target: int, replacement: Node) -> Node:
        """按 :func:`walk` 的前序把第 ``target`` 个节点替换为 ``replacement``（返回新树，不改原树）。"""
        if counter[0] == target:
            counter[0] += 1
            return self._copy(replacement)
        counter[0] += 1
        if isinstance(node, Var):
            return Var(node.name)
        if isinstance(node, Const):
            return Const(node.value)
        if isinstance(node, Unary):
            return Unary(node.op, self._rebuild(node.child, counter, target, replacement))
        if isinstance(node, Binary):
            return Binary(node.op,
                          self._rebuild(node.left, counter, target, replacement),
                          self._rebuild(node.right, counter, target, replacement))
        if isinstance(node, TsOp):
            return TsOp(node.op, node.window,
                        self._rebuild(node.child, counter, target, replacement))
        if isinstance(node, Where):
            return Where(self._rebuild(node.cond, counter, target, replacement),
                         self._rebuild(node.a, counter, target, replacement),
                         self._rebuild(node.b, counter, target, replacement))
        raise TypeError(f"cannot rebuild node {node!r}")

    def _copy(self, node: Node) -> Node:
        if isinstance(node, Var):
            return Var(node.name)
        if isinstance(node, Const):
            return Const(node.value)
        if isinstance(node, Unary):
            return Unary(node.op, self._copy(node.child))
        if isinstance(node, Binary):
            return Binary(node.op, self._copy(node.left), self._copy(node.right))
        if isinstance(node, TsOp):
            return TsOp(node.op, node.window, self._copy(node.child))
        if isinstance(node, Where):
            return Where(self._copy(node.cond), self._copy(node.a), self._copy(node.b))
        raise TypeError(f"cannot copy node {node!r}")

    # ------------------------------------------------------------------
    # 适应度（select 区间截面 IC 均值）
    # ------------------------------------------------------------------
    # 复杂度惩罚系数 & 跨子区间不稳定惩罚系数（简洁+稳健优先，§7.3）。
    _LAMBDA_CX = 0.003
    _LAMBDA_STD = 0.5

    def _fitness(self, expr: FactorExpr, panel: BaseFactorPanel, labmap: dict,
                 select_dates: list, ledger: SearchLedger) -> float:
        """跨子区间稳健适应度 (§7.3)：把 fit 区间切成最多三段，
        ``fit = mean(fold_ic) - 0.5*std(fold_ic) - 0.003*complexity``。

        各公式共享相同日期分段，每段可计算 IC 的覆盖率须达到预设下限。
        这是训练稳定性诊断，不是独立内层交叉验证或样本外证据。
        """
        cached = self._fit_cache.get(expr.text)
        if cached is not None:
            return cached
        ledger.view_dataset()  # 每评估一个新因子=查看一次数据（过拟合记账，§7.3）
        scores_by_date: dict = {}
        if hasattr(self._dsl, "evaluate_frame"):
            try:  # 因果全帧求值一次，按行读取（与逐日 evaluate 等价，见 dsl.evaluate_frame）
                from qmt.factor.metrics import scores_by_date_from_frame
                frame = self._dsl.evaluate_frame(expr, panel)
                scores_by_date = scores_by_date_from_frame(frame, list(select_dates))
            except Exception:
                scores_by_date = {}
        if not scores_by_date:
            for d in select_dates:
                try:
                    s = self._dsl.evaluate(expr, panel, d)  # 只用 <= d 的行（点时点安全）
                except Exception:
                    continue
                if s is not None and not s.empty:
                    scores_by_date[d] = s
        labs = {d: labmap[d] for d in select_dates if d in labmap}
        if not select_dates:
            self._fit_cache[expr.text] = _NEG_INF
            return _NEG_INF
        # Fold dates belong to the experiment, not to each expression's missing
        # pattern. A sparse factor cannot cherry-pick one lucky sub-period.
        n_folds = 3 if len(select_dates) >= 6 else 1
        fold_ics: list[float] = []
        for fold in np.array_split(np.array(select_dates, dtype=object), n_folds):
            sub_s = {d: scores_by_date[d] for d in fold if d in scores_by_date}
            sub_y = {d: labs[d] for d in fold if d in labs}
            _, rics = per_date_ic(sub_s, sub_y)
            if len(rics) < int(np.ceil(len(fold) * self._min_train_ic_coverage)):
                self._fit_cache[expr.text] = _NEG_INF
                return _NEG_INF
            fold_ics.append(float(np.mean(rics)))
        if not fold_ics:
            fit = _NEG_INF
        else:
            mean_ic = float(np.mean(fold_ics))
            std_ic = float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0
            fit = mean_ic - self._LAMBDA_STD * std_ic - self._LAMBDA_CX * expr.complexity
        self._fit_cache[expr.text] = fit
        if self._diversity_enabled and np.isfinite(fit):
            # Optional research mode only: ordinary GP retains its small float cache.
            self._train_signatures[expr.text] = rank_signature(scores_by_date)
        return fit

    def _fit_key(self, expr: FactorExpr) -> float:
        v = self._fit_cache.get(expr.text, _NEG_INF)
        return v if np.isfinite(v) else -1e18

    def _tournament(self, population: list[FactorExpr], rng: np.random.Generator,
                    k: int = 3) -> FactorExpr:
        k = min(k, len(population))
        idxs = rng.choice(len(population), size=k, replace=False)
        contenders = [population[int(i)] for i in np.atleast_1d(idxs)]
        return max(contenders, key=self._fit_key)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _consider(self, ledger: SearchLedger, expr: FactorExpr, cfg: SearchConfig,
                  panel: BaseFactorPanel, labmap: dict, select_dates: list,
                  *, parent_id: str | None, generation: int, note: str,
                  accept_into: list[FactorExpr]) -> bool:
        """预算检查→校验→评分→记账；非法、重复与人工播种同样消耗尝试预算。"""
        if ledger.n_trials >= cfg.max_trials:
            return False
        ok = self._dsl.validate(expr, cfg).ok
        select_ic: float | None = None
        if ok:
            fit = self._fitness(expr, panel, labmap, select_dates, ledger)
            select_ic = fit if np.isfinite(fit) else None
            accept_into.append(expr)
            if np.isfinite(fit):
                self._candidates.setdefault(expr.text, expr)
        self._trial_counter += 1
        ledger.record(SearchTrial(
            trial_id=self._trial_counter,
            expr_text=expr.text,
            passed_dsl=ok,
            select_ic=select_ic,
            parent_id=parent_id,
            generation=generation,
            note=note,
        ))
        return True

    def _configure(self, cfg: SearchConfig, panel: BaseFactorPanel) -> None:
        allowed = set(cfg.allowed_operators) if cfg.allowed_operators else set(ALL_OPS)
        # restrict leaves to the allowed field set (excludes e.g. raw px level) so the
        # search does not waste budget generating candidates validate would reject.
        if cfg.allowed_fields:
            self._fields = sorted(f for f in panel.factors.keys() if f in set(cfg.allowed_fields))
        else:
            self._fields = sorted(panel.factors.keys())
        self._ts = sorted(TS_OPS & allowed)
        self._ew = sorted(EW_UNARY & allowed)
        self._cs = sorted(CS_OPS & allowed)
        self._bin = sorted(BIN_OPS & allowed)
        self._windows = list(cfg.allowed_windows) or [20]
        self._use_where = "where" in allowed

    @staticmethod
    def _fit_dates(split: ValidationSplit, labmap: dict) -> list:
        """适应度区间 = 训练区间 ``train_span`` 内的决策日（绝不含验证区间）。

        若训练区间无可用标签，直接拒绝搜索；验证标签绝不能补作训练标签。
        """
        lo, hi = split.train_span
        dates = sorted(d for d in labmap if lo <= d <= hi)
        if not dates:
            raise ValueError("symbolic search requires labels inside train_span; validation labels cannot be used")
        return dates

    @staticmethod
    def _assert_training_unsealed(panel: BaseFactorPanel, labels: Sequence[Label],
                                  fit_dates: list, ledger: SearchLedger,
                                  valid_start: date | None = None) -> None:
        """Check decision dates and complete label windows against the sealed test."""
        locked = ledger.locked_test_span
        dates = sorted(set(pd.Timestamp(d).date() for d in panel.dates))
        horizons: dict = {}
        for lab in labels:
            horizons[lab.as_of] = max(horizons.get(lab.as_of, 0), lab.horizon_days)
        for d in fit_dates:
            if locked is not None and locked[0] <= d <= locked[1]:
                raise ValueError(f"training decision {d} overlaps locked_test_span")
            pos = bisect_right(dates, d) - 1
            end_pos = pos + horizons[d]
            if pos < 0 or end_pos >= len(dates):
                raise ValueError(f"training label at {d} has no complete horizon in panel dates")
            if valid_start is not None and dates[end_pos] >= valid_start:
                raise ValueError(f"training label at {d} reaches validation region")
            if locked is not None and d <= locked[1] and dates[end_pos] >= locked[0]:
                raise ValueError(f"training label at {d} reaches locked_test_span")

    @staticmethod
    def _training_panel(panel: BaseFactorPanel, cutoff) -> BaseFactorPanel:
        """Restrict feature evaluation to training while retaining past warmup rows."""
        frames = {}
        for name, frame in panel.factors.items():
            key = pd.Timestamp(cutoff) if isinstance(frame.index, pd.DatetimeIndex) else cutoff
            frames[name] = frame.loc[frame.index <= key]
        return BaseFactorPanel(
            dates=[d for d in panel.dates if pd.Timestamp(d).date() <= cutoff],
            funds=list(panel.funds), factors=frames, industry_of=panel.industry_of,
            meta=panel.meta,
        )

    @staticmethod
    def _select_dates(split: ValidationSplit) -> list:
        """Legacy helper: first half of validation dates; never used for fitting."""
        ser = list(split.as_of_series)
        if len(ser) <= 1:
            return ser
        return ser[: max(1, len(ser) // 2)]

    @staticmethod
    def _dedup(exprs: Sequence[FactorExpr]) -> list[FactorExpr]:
        seen: set[str] = set()
        out: list[FactorExpr] = []
        for e in exprs:
            if e.text not in seen:
                seen.add(e.text)
                out.append(e)
        return out

    @staticmethod
    def _pick(rng: np.random.Generator, seq):
        return seq[int(rng.integers(len(seq)))]
