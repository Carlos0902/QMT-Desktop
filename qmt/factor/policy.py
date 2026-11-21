"""决策树交易策略 + 回报环境 — 规则学习/程序综合引擎(instruction §2.2)。

用户设计思路的落地:
  1. **决策树表示策略**:内节点 = 「某特征的*截面分位* <= θ」(分位数比较,尺度无关,
     跨 14k 基金可比);叶节点 = 打分动作 ∈ {-1,-0.5,0,+0.5,+1}。对每个决策日,
     树对每只基金产生一个分数;正分入选组合,全非正 → **持现金**(策略可以选择空仓)。
  2. **累计奖励评价**:episode = 一段连续决策日;每步组合 = 正分前 top_k 等权;
     步奖励 = 组合下期**相对收益**(相对全宇宙均值,不奖励押方向)− 换手 × 费率;
     累计奖励 = Σ log(1+r)。换手成本内生 → 高频翻仓的策略自然低分。
  3. 遗传进化见 :mod:`qmt.factor.policy_search`。

点时点纪律:特征分位帧来自因果的 BaseFactorPanel(行 d 只含 <= d 信息);
奖励用 d→下一决策日的前向收益(标签侧,允许用未来);episode 全部落在
锁定区(locked_start)之前/训练区内,由上层保证并在 env 构建时断言。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Union

import numpy as np
import pandas as pd

__all__ = [
    "Leaf", "Cond", "PolicyNode", "eval_tree", "render_tree", "tree_nodes",
    "tree_depth", "copy_tree", "random_tree", "mutate_tree", "crossover_trees",
    "seed_trees", "PolicyEnv", "EpisodeResult",
    "FEATURE_ZH", "THETAS", "LEAF_VALUES",
]

# 特征的中文渲染(可审计输出用)
FEATURE_ZH = {
    "vol20": "20日波动分位", "vol60": "60日波动分位",
    "mom20": "20日动量分位", "mom60": "60日动量分位",
    "dd60": "60日回撤分位", "beta": "beta分位",
    "drift": "风格漂移分位", "flow": "资金流分位",
}
THETAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LEAF_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


# --------------------------------------------------------------------------- #
# 树结构
# --------------------------------------------------------------------------- #
@dataclass
class Leaf:
    value: float


@dataclass
class Cond:
    feature: str
    theta: float
    if_true: "PolicyNode"
    if_false: "PolicyNode"


PolicyNode = Union[Leaf, Cond]


def eval_tree(node: PolicyNode, pct: dict[str, pd.Series]) -> pd.Series:
    """在一个决策日上求值:``pct[feature]`` = (fund -> 截面分位∈[0,1])。

    返回 fund -> score。缺失特征分位以 0.5(中性)代入 —— fail-soft,并在
    环境构建时统计缺失率供审计。
    """
    universe = next(iter(pct.values())).index
    if isinstance(node, Leaf):
        return pd.Series(node.value, index=universe, dtype=float)
    f = pct.get(node.feature)
    if f is None:
        f = pd.Series(0.5, index=universe, dtype=float)
    f = f.reindex(universe).fillna(0.5)
    mask = f <= node.theta
    left = eval_tree(node.if_true, pct)
    right = eval_tree(node.if_false, pct)
    return left.where(mask, right)


def render_tree(node: PolicyNode, indent: int = 0) -> str:
    """渲染成人能读的 if-then 规则(可审计性是本方法的核心卖点)。"""
    pad = "  " * indent
    if isinstance(node, Leaf):
        return f"{pad}score = {node.value:+.1f}"
    zh = FEATURE_ZH.get(node.feature, node.feature)
    return (f"{pad}if {zh} <= {node.theta:.1f}:\n"
            f"{render_tree(node.if_true, indent + 1)}\n"
            f"{pad}else:\n"
            f"{render_tree(node.if_false, indent + 1)}")


def render_one_line(node: PolicyNode) -> str:
    if isinstance(node, Leaf):
        return f"{node.value:+.1f}"
    return (f"if({node.feature}<={node.theta:.1f}, "
            f"{render_one_line(node.if_true)}, {render_one_line(node.if_false)})")


def tree_nodes(node: PolicyNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return 1 + tree_nodes(node.if_true) + tree_nodes(node.if_false)


def tree_depth(node: PolicyNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return 1 + max(tree_depth(node.if_true), tree_depth(node.if_false))


def copy_tree(node: PolicyNode) -> PolicyNode:
    if isinstance(node, Leaf):
        return Leaf(node.value)
    return Cond(node.feature, node.theta, copy_tree(node.if_true), copy_tree(node.if_false))


# --------------------------------------------------------------------------- #
# 随机生成 / 变异 / 交叉(确定性:全部经由传入的 rng)
# --------------------------------------------------------------------------- #
def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def random_tree(rng: np.random.Generator, features: list[str],
                depth_budget: int = 3) -> PolicyNode:
    if depth_budget <= 1 or rng.random() < 0.3:
        return Leaf(_pick(rng, LEAF_VALUES))
    return Cond(_pick(rng, features), _pick(rng, THETAS),
                random_tree(rng, features, depth_budget - 1),
                random_tree(rng, features, depth_budget - 1))


def _all_subtrees(node: PolicyNode, path=()) -> list[tuple]:
    out = [(path, node)]
    if isinstance(node, Cond):
        out += _all_subtrees(node.if_true, path + ("t",))
        out += _all_subtrees(node.if_false, path + ("f",))
    return out


def _replace_at(node: PolicyNode, path: tuple, replacement: PolicyNode) -> PolicyNode:
    if not path:
        return copy_tree(replacement)
    assert isinstance(node, Cond)
    if path[0] == "t":
        return Cond(node.feature, node.theta,
                    _replace_at(node.if_true, path[1:], replacement), copy_tree(node.if_false))
    return Cond(node.feature, node.theta,
                copy_tree(node.if_true), _replace_at(node.if_false, path[1:], replacement))


def mutate_tree(rng: np.random.Generator, tree: PolicyNode, features: list[str],
                max_depth: int = 4) -> PolicyNode:
    """变异:改阈值 / 换特征 / 改叶值 / 子树替换 / 剪枝。"""
    subs = _all_subtrees(tree)
    path, node = subs[int(rng.integers(len(subs)))]
    kind = rng.random()
    if isinstance(node, Cond):
        if kind < 0.3:
            new = Cond(node.feature, _pick(rng, THETAS), copy_tree(node.if_true), copy_tree(node.if_false))
        elif kind < 0.55:
            new = Cond(_pick(rng, features), node.theta, copy_tree(node.if_true), copy_tree(node.if_false))
        elif kind < 0.8:
            new = random_tree(rng, features, depth_budget=max(2, max_depth - len(path)))
        else:
            new = Leaf(_pick(rng, LEAF_VALUES))          # 剪枝
    else:
        if kind < 0.5:
            new = Leaf(_pick(rng, LEAF_VALUES))
        else:
            new = random_tree(rng, features, depth_budget=max(2, max_depth - len(path)))
    return _replace_at(tree, path, new)


def crossover_trees(rng: np.random.Generator, a: PolicyNode, b: PolicyNode) -> PolicyNode:
    """交叉:把 b 的随机子树嫁接到 a 的随机位置。"""
    subs_a = _all_subtrees(a)
    subs_b = _all_subtrees(b)
    path, _ = subs_a[int(rng.integers(len(subs_a)))]
    _, donor = subs_b[int(rng.integers(len(subs_b)))]
    return _replace_at(a, path, donor)


def seed_trees(features: list[str]) -> dict[str, PolicyNode]:
    """基线播种(§5.6 先复现已知,再进化):低波动 / 动量 / 低回撤 静态策略树。"""
    seeds: dict[str, PolicyNode] = {}
    if "vol60" in features:
        seeds["seed:low_vol"] = Cond("vol60", 0.3, Leaf(1.0), Leaf(0.0))
    if "mom60" in features:
        seeds["seed:momentum"] = Cond("mom60", 0.7, Leaf(0.0), Leaf(1.0))
    if "dd60" in features:
        seeds["seed:low_drawdown"] = Cond("dd60", 0.7, Leaf(0.0), Leaf(1.0))
    if "vol60" in features and "mom60" in features:
        seeds["seed:lowvol_mom"] = Cond("vol60", 0.3, Cond("mom60", 0.5, Leaf(0.5), Leaf(1.0)), Leaf(0.0))
    return seeds


# --------------------------------------------------------------------------- #
# 环境:episode 与累计净奖励
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    cum_log_reward: float                 # Σ log(1 + 步净相对收益)
    step_rewards: list[float]
    avg_turnover: float
    active_share: float                   # 有持仓的步数占比(策略可选择空仓)
    n_steps: int


@dataclass
class PolicyEnv:
    """一段决策日序列上的回报环境。

    ``pct_by_date[d][feature]`` = 截面分位 Series(只含该日**真实有数据**的基金 ——
    幽灵基金不进截面,对抗审校确认过的缺陷);``rel_fwd[d]`` = d 的**可执行**净值 →
    下一决策日可执行净值的相对收益(清盘按最后净值结算)。

    费用分腿计:买入腿 ``buy_fee``(申购 ~0.15%)、卖出腿 ``sell_fee``(短持赎回
    ~0.5%),episode 末**强制清算**计赎回费(否则「持仓到期末」白赚一腿费用)。
    并列打破用 ``priority``(一次性种子随机序,与基金成立年代无关)—— 代码序
    并列打破被实测证伪(与成立日相关 +0.63,系统性买最老基金)。
    """

    dates: list[date]
    pct_by_date: dict[date, dict[str, pd.Series]]
    rel_fwd: dict[date, pd.Series]
    features: list[str]
    buy_fee: float = 0.0015
    sell_fee: float = 0.005
    top_k: int = 20
    priority: pd.Series | None = None      # fund -> 随机优先序(小者优先,粘性)
    hysteresis: float = 0.0                # 滞回带宽:已持仓的基金,分数只要 >= -hysteresis
                                           # 就保留(不因小幅下滑而卖出),只有明显跌出
                                           # (分数 < -hysteresis)才替换。0 = 无滞回。

    def _rank(self, cand: pd.Series) -> pd.Index:
        """按 (分数降序, 随机优先序升序) 排序候选,返回索引。"""
        if self.priority is not None:
            pri = self.priority.reindex(cand.index)
            order = pd.DataFrame({"s": cand, "p": pri}).sort_values(
                ["s", "p"], ascending=[False, True], kind="mergesort")
        else:
            order = cand.sort_values(ascending=False, kind="mergesort").to_frame("s")
        return order.index

    def _target(self, scores: pd.Series, held: pd.Index) -> pd.Series:
        """滞回选仓:先保留仍在退出带内的旧持仓,再从进入带补足到 top_k。

        进入带 = 分数 > 0(树给正分);退出带 = 分数 > -hysteresis。已持仓的基金
        只要还在退出带就留下(避免因分数在阈值附近抖动而反复换手);空出的名额
        按排名从进入带的新基金里补。hysteresis=0 时退化为每期重排前 top_k。
        """
        # 1) 保留:旧持仓中分数 > -hysteresis 且当日可交易的
        keep_mask = scores.reindex(held).fillna(-1e9) > -self.hysteresis
        keep = list(held[keep_mask.to_numpy()]) if len(held) else []
        keep = keep[: self.top_k]
        # 2) 补足:从进入带(分数>0)里,排除已保留的,按排名补到 top_k
        room = self.top_k - len(keep)
        if room > 0:
            enter = scores[(scores > 0) & (~scores.index.isin(keep))]
            if not enter.empty:
                keep += list(self._rank(enter)[:room])
        if not keep:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(keep), index=pd.Index(keep))

    def episode(self, tree: PolicyNode) -> EpisodeResult:
        prev_w = pd.Series(dtype=float)
        cum = 0.0
        rewards: list[float] = []
        turns: list[float] = []
        active = 0
        usable = [d for d in self.dates if d in self.rel_fwd]
        for d in usable:
            rel = self.rel_fwd[d]
            scores = eval_tree(tree, self.pct_by_date[d])
            scores = scores.reindex(rel.index).dropna()   # 只在可交易集合内选仓
            w = self._target(scores, prev_w.index)
            if len(w):
                active += 1
            delta = w.subtract(prev_w, fill_value=0.0)
            buys = float(delta[delta > 0].sum())
            sells = float(-delta[delta < 0].sum())
            cost = self.buy_fee * buys + self.sell_fee * sells
            gross = float((rel.reindex(w.index).fillna(0.0) * w).sum()) if len(w) else 0.0
            r = gross - cost
            cum += float(np.log1p(max(r, -0.95)))
            rewards.append(r)
            turns.append(0.5 * float(delta.abs().sum()))
            prev_w = w
        # 期末强制清算:结清最后持仓的赎回费
        if len(prev_w):
            final_cost = self.sell_fee * float(prev_w.abs().sum())
            cum += float(np.log1p(-final_cost))
            rewards.append(-final_cost)
        n = len(usable)
        return EpisodeResult(
            cum_log_reward=cum,
            step_rewards=rewards,
            avg_turnover=float(np.mean(turns)) if turns else 0.0,
            active_share=active / n if n else 0.0,
            n_steps=n,
        )
