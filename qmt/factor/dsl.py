"""factor_dsl — the constrained formula grammar (instruction §2.1/§2.3/§5.3).

The DSL is the first structural defense against overfitting: it bounds what
``auto_search`` may build. Every :class:`FactorExpr` is an AST of :class:`Node`s
drawn from a whitelist of operators over base-factor leaves, with a hard complexity
cap and guarded dangerous operations (div-by-~0, log/sqrt of negatives).

Evaluation is **frame-valued and point-in-time safe**: each node evaluates to a
``(date x fund)`` DataFrame restricted to rows ``<= as_of``; the factor's
cross-section at ``as_of`` is the last row. Time-series operators roll backward over
the date axis; cross-sectional operators (rank/zscore/neutralize) transform within a
single date across funds. Because the value at ``as_of`` only depends on rows
``<= as_of`` (and the base panel is itself PIT), the formula cannot see the future.

Four-layer building blocks (instruction §5.3):
  L1 base data (leaves)        — Var(name)
  L2 common calc              — ts_mean/ts_std/ts_delta/ts_drawdown/ts_min/ts_max/
                                 ts_rank, add/sub/mul/div_g, log_g/sqrt_g/sign/abs_/neg
  L3 cross-section            — rank, zscore, neutralize (industry-demean)
  L4 trading rule            — where(cond, a, b), vol_switch (checked in validate)
"""

from __future__ import annotations

import ast as _pyast
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from qmt.factor.types import (
    BaseFactorPanel,
    DSLVerdict,
    FactorExpr,
    GateStatus,
    SearchConfig,
)

# --- operator whitelist by DSL layer (instruction §5.3) -------------------

CS_OPS = {"rank", "zscore", "neutralize"}                       # L3 cross-section
TS_OPS = {"ts_mean", "ts_std", "ts_delta", "ts_drawdown",       # L2 time-series
          "ts_min", "ts_max", "ts_rank"}
EW_UNARY = {"log_g", "sqrt_g", "sign", "abs_", "neg"}            # L2 elementwise
BIN_OPS = {"add", "sub", "mul", "div_g", "min2", "max2"}        # L2 binary
TRADE_OPS = {"where", "vol_switch"}                             # L4 trading rule
GUARDED = {"div_g", "log_g", "sqrt_g"}                          # dangerous → guarded

ALL_OPS = CS_OPS | TS_OPS | EW_UNARY | BIN_OPS | TRADE_OPS

# python operator -> DSL binary op name
_PYOP = {_pyast.Add: "add", _pyast.Sub: "sub", _pyast.Mult: "mul", _pyast.Div: "div_g"}


def _cutoff_for(index: pd.Index, as_of: date):
    """Normalise ``as_of`` to the index dtype so ``index <= as_of`` never raises
    (a ``datetime.date`` vs ``DatetimeIndex`` comparison is a TypeError in
    pandas>=2; the silent-degradation failure mode is worse than the crash)."""
    if isinstance(index, pd.DatetimeIndex):
        return pd.Timestamp(as_of)
    return as_of


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass
class Node:
    def children(self) -> list["Node"]:
        return []


@dataclass
class Var(Node):
    name: str


@dataclass
class Const(Node):
    value: float


@dataclass
class Unary(Node):
    op: str                 # EW_UNARY | CS_OPS
    child: Node

    def children(self) -> list[Node]:
        return [self.child]


@dataclass
class Binary(Node):
    op: str                 # BIN_OPS
    left: Node
    right: Node

    def children(self) -> list[Node]:
        return [self.left, self.right]


@dataclass
class TsOp(Node):
    op: str                 # TS_OPS
    window: int
    child: Node

    def children(self) -> list[Node]:
        return [self.child]


@dataclass
class Where(Node):
    """Trading-rule gate (L4): where(cond > 0, a, b)."""

    cond: Node
    a: Node
    b: Node

    def children(self) -> list[Node]:
        return [self.cond, self.a, self.b]


# ---------------------------------------------------------------------------
# AST introspection
# ---------------------------------------------------------------------------


def walk(node: Node):
    yield node
    for c in node.children():
        yield from walk(c)


def complexity_of(node: Node) -> int:
    return sum(1 for _ in walk(node))


def depth_of(node: Node) -> int:
    kids = node.children()
    return 1 + (max(depth_of(c) for c in kids) if kids else 0)


def operators_of(node: Node) -> set[str]:
    ops: set[str] = set()
    for n in walk(node):
        if isinstance(n, Unary):
            ops.add(n.op)
        elif isinstance(n, Binary):
            ops.add(n.op)
        elif isinstance(n, TsOp):
            ops.add(n.op)
        elif isinstance(n, Where):
            ops.add("where")
    return ops


def fields_of(node: Node) -> list[str]:
    seen: list[str] = []
    for n in walk(node):
        if isinstance(n, Var) and n.name not in seen:
            seen.append(n.name)
    return seen


def windows_of(node: Node) -> list[int]:
    seen: list[int] = []
    for n in walk(node):
        if isinstance(n, TsOp) and n.window not in seen:
            seen.append(n.window)
    return seen


def to_text(node: Node) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Const):
        return repr(round(node.value, 4))
    if isinstance(node, Unary):
        return f"{node.op}({to_text(node.child)})"
    if isinstance(node, Binary):
        return f"{node.op}({to_text(node.left)}, {to_text(node.right)})"
    if isinstance(node, TsOp):
        return f"{node.op}({to_text(node.child)}, {node.window})"
    if isinstance(node, Where):
        return f"where({to_text(node.cond)}, {to_text(node.a)}, {to_text(node.b)})"
    raise TypeError(f"unknown node {node!r}")


def make_expr(node: Node) -> FactorExpr:
    """Wrap an AST into a fully-populated :class:`FactorExpr`."""
    ops = operators_of(node)
    return FactorExpr(
        text=to_text(node),
        ast=node,
        data_fields=fields_of(node),
        time_windows=windows_of(node),
        operators=frozenset(ops),
        complexity=complexity_of(node),
        guarded_ops=sorted(ops & GUARDED),
    )


# ---------------------------------------------------------------------------
# Guarded numeric helpers (frame-wise)
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _div_g(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    denom = b.where(b.abs() > _EPS, np.nan)
    return a / denom


def _log_g(a: pd.DataFrame) -> pd.DataFrame:
    return np.log(a.where(a > _EPS, np.nan))


def _sqrt_g(a: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(a.where(a >= 0.0, np.nan))


# ---------------------------------------------------------------------------
# The DSL implementation
# ---------------------------------------------------------------------------


class FormulaDSL:
    """Concrete :class:`qmt.factor.types.FactorDSL`.

    ``industry_of`` (fund/asset -> industry) is needed for the ``neutralize``
    operator; when absent, ``neutralize`` degrades to a plain cross-sectional demean.
    """

    def __init__(self, industry_of: dict[str, str] | None = None) -> None:
        self.industry_of = industry_of or {}

    # -- parse -------------------------------------------------------------
    def parse(self, expr_text: str) -> FactorExpr:
        """Parse a restricted-Python expression into a whitelisted AST."""
        try:
            tree = _pyast.parse(expr_text, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - defensive
            raise ValueError(f"cannot parse formula {expr_text!r}: {exc}") from exc
        node = self._convert(tree.body)
        return make_expr(node)

    def _convert(self, n: _pyast.AST) -> Node:
        if isinstance(n, _pyast.Name):
            return Var(n.id)
        if isinstance(n, _pyast.Constant) and isinstance(n.value, (int, float)):
            return Const(float(n.value))
        if isinstance(n, _pyast.UnaryOp) and isinstance(n.op, _pyast.USub):
            return Unary("neg", self._convert(n.operand))
        if isinstance(n, _pyast.BinOp):
            op = _PYOP.get(type(n.op))
            if op is None:
                raise ValueError(f"operator {type(n.op).__name__} not in DSL whitelist")
            return Binary(op, self._convert(n.left), self._convert(n.right))
        if isinstance(n, _pyast.Call) and isinstance(n.func, _pyast.Name):
            fname = n.func.id
            args = n.args
            if fname in TS_OPS:
                if len(args) != 2 or not isinstance(args[1], _pyast.Constant):
                    raise ValueError(f"{fname} expects (expr, window:int)")
                return TsOp(fname, int(args[1].value), self._convert(args[0]))
            if fname in CS_OPS or fname in EW_UNARY:
                if len(args) != 1:
                    raise ValueError(f"{fname} expects one arg")
                return Unary(fname, self._convert(args[0]))
            if fname in BIN_OPS:
                if len(args) != 2:
                    raise ValueError(f"{fname} expects two args")
                return Binary(fname, self._convert(args[0]), self._convert(args[1]))
            if fname == "where":
                if len(args) != 3:
                    raise ValueError("where expects (cond, a, b)")
                return Where(*(self._convert(a) for a in args))
            raise ValueError(f"function {fname!r} not in DSL whitelist")
        raise ValueError(f"unsupported syntax node {type(n).__name__}")

    # -- validate ----------------------------------------------------------
    def validate(self, expr: FactorExpr, cfg: SearchConfig) -> DSLVerdict:
        """Hard gate: reject anything outside the allowed space (tech doc §4.3)."""
        reasons: list[str] = []
        node = expr.ast
        cx = complexity_of(node)
        dp = depth_of(node)
        ops = operators_of(node)
        fields = fields_of(node)
        windows = windows_of(node)

        # 1) operator whitelist
        bad_ops = ops - ALL_OPS
        if bad_ops:
            reasons.append(f"operators not in whitelist: {sorted(bad_ops)}")
        if cfg.allowed_operators:
            unallowed = ops - set(cfg.allowed_operators)
            if unallowed:
                reasons.append(f"operators outside SearchConfig allow-set: {sorted(unallowed)}")

        # 2) data-field whitelist (must be PIT-checkable base factors)
        if cfg.allowed_fields:
            unknown = [f for f in fields if f not in set(cfg.allowed_fields)]
            if unknown:
                reasons.append(f"data_fields not allowed: {unknown}")
        if not fields:
            reasons.append("formula references no data field (constant-only)")

        # 3) time-window whitelist
        bad_w = [w for w in windows if w not in set(cfg.allowed_windows)]
        if bad_w:
            reasons.append(f"time windows not allowed: {bad_w}")

        # 4) complexity / depth caps — hard, never relaxed (§2.3, §7.3)
        if cx > cfg.max_complexity:
            reasons.append(f"complexity {cx} > max_complexity {cfg.max_complexity}")
        if dp > cfg.max_depth:
            reasons.append(f"depth {dp} > max_depth {cfg.max_depth}")

        status = GateStatus.PASS if not reasons else GateStatus.FAIL
        return DSLVerdict(status=status, reasons=reasons, complexity=cx)

    # -- evaluate ----------------------------------------------------------
    def evaluate(self, expr: FactorExpr, panel: BaseFactorPanel, as_of: date) -> pd.Series:
        """Cross-section (fund -> score) at ``as_of``; last row of the frame eval."""
        frame = self._eval_frame(expr.ast, panel, as_of)
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        return frame.iloc[-1].dropna()

    def evaluate_frame(self, expr: FactorExpr, panel: BaseFactorPanel) -> pd.DataFrame:
        """Full-history (date x fund) score frame, computed in ONE pass.

        Mathematically identical to calling :meth:`evaluate` at every date, because
        every DSL operator is *causal*: ``ts_*`` are backward-looking rolling windows,
        cross-sectional ops (rank/zscore/neutralize) act within a single row, and
        binary/where are pointwise. Hence row ``d`` of this frame equals
        ``evaluate(expr, panel, d)`` — but the whole history costs one rolling pass
        instead of one per decision date (removes the O(N²) evaluation bottleneck
        that previously forced short, low-power evaluation spans).
        """
        frame = self._eval_frame(expr.ast, panel, None)
        if frame is None:
            return pd.DataFrame()
        return frame

    def _eval_frame(self, node: Node, panel: BaseFactorPanel, as_of: date | None) -> pd.DataFrame:
        """``as_of=None`` evaluates the full history.

        Causality precondition: leaf frames must be date-ascending — rolling ops
        stream in storage order, so an unsorted frame would mix future rows into
        backward windows. Enforced here (sort if needed), not assumed.
        """
        funds = panel.funds
        if isinstance(node, Var):
            base = panel.factors.get(node.name)
            if base is None:
                raise KeyError(f"base factor {node.name!r} not in panel")
            if not base.index.is_monotonic_increasing:
                base = base.sort_index()
            if as_of is None:
                return base
            return base.loc[base.index <= _cutoff_for(base.index, as_of)]
        if isinstance(node, Const):
            # broadcast constant onto the reference frame's shape
            ref = self._reference_frame(panel, as_of)
            return pd.DataFrame(node.value, index=ref.index, columns=ref.columns)
        if isinstance(node, Unary):
            x = self._eval_frame(node.child, panel, as_of)
            return self._apply_unary(node.op, x)
        if isinstance(node, Binary):
            a = self._eval_frame(node.left, panel, as_of)
            b = self._eval_frame(node.right, panel, as_of)
            a, b = a.align(b, join="outer")
            return self._apply_binary(node.op, a, b)
        if isinstance(node, TsOp):
            x = self._eval_frame(node.child, panel, as_of)
            return self._apply_ts(node.op, node.window, x)
        if isinstance(node, Where):
            cond = self._eval_frame(node.cond, panel, as_of)
            a = self._eval_frame(node.a, panel, as_of)
            b = self._eval_frame(node.b, panel, as_of)
            cond, a = cond.align(a, join="outer")
            cond, b = cond.align(b, join="outer")
            return a.where(cond > 0, b)
        raise TypeError(f"cannot evaluate node {node!r}")

    def _reference_frame(self, panel: BaseFactorPanel, as_of: date | None) -> pd.DataFrame:
        any_factor = next(iter(panel.factors.values()))
        if not any_factor.index.is_monotonic_increasing:
            any_factor = any_factor.sort_index()
        if as_of is None:
            return any_factor
        return any_factor.loc[any_factor.index <= _cutoff_for(any_factor.index, as_of)]

    def _apply_unary(self, op: str, x: pd.DataFrame) -> pd.DataFrame:
        if op == "neg":
            return -x
        if op == "abs_":
            return x.abs()
        if op == "sign":
            return np.sign(x)
        if op == "log_g":
            return _log_g(x)
        if op == "sqrt_g":
            return _sqrt_g(x)
        if op == "rank":                      # cross-sectional rank -> [-0.5, 0.5]
            return x.rank(axis=1, pct=True) - 0.5
        if op == "zscore":                    # cross-sectional standardise
            mu = x.mean(axis=1)
            sd = x.std(axis=1).replace(0.0, np.nan)
            return x.sub(mu, axis=0).div(sd, axis=0)
        if op == "neutralize":                # demean within industry group
            return self._neutralize(x)
        raise ValueError(f"unknown unary op {op}")

    def _apply_binary(self, op: str, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "div_g":
            return _div_g(a, b)
        if op == "min2":
            return a.where(a < b, b)
        if op == "max2":
            return a.where(a > b, b)
        raise ValueError(f"unknown binary op {op}")

    def _apply_ts(self, op: str, window: int, x: pd.DataFrame) -> pd.DataFrame:
        # min_periods capped at window so a whitelisted window=1 stays evaluable
        mp = max(1, min(window, max(2, window // 2)))
        roll = x.rolling(window=window, min_periods=mp)
        if op == "ts_mean":
            return roll.mean()
        if op == "ts_std":
            return roll.std()
        if op == "ts_delta":
            return x - x.shift(window)
        if op == "ts_min":
            return roll.min()
        if op == "ts_max":
            return roll.max()
        if op == "ts_drawdown":               # (last - rolling max) / rolling max
            rmax = roll.max()
            return _div_g(x - rmax, rmax)
        if op == "ts_rank":                   # position of last value within window
            # Native rolling rank has the same average-tie / finite-count percentile
            # semantics, without constructing a Series for every date/fund window.
            return roll.rank(method="average", pct=True)
        raise ValueError(f"unknown ts op {op}")

    def _neutralize(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.industry_of:
            return x.sub(x.mean(axis=1), axis=0)
        groups: dict[str, list[str]] = {}
        for fund in x.columns:
            groups.setdefault(self.industry_of.get(fund, "_none"), []).append(fund)
        out = x.copy()
        for cols in groups.values():
            block = x[cols]
            out[cols] = block.sub(block.mean(axis=1), axis=0)
        return out


__all__ = [
    "Node", "Var", "Const", "Unary", "Binary", "TsOp", "Where",
    "FormulaDSL", "make_expr", "to_text", "complexity_of", "depth_of",
    "operators_of", "fields_of", "windows_of",
    "CS_OPS", "TS_OPS", "EW_UNARY", "BIN_OPS", "TRADE_OPS", "GUARDED", "ALL_OPS",
]
