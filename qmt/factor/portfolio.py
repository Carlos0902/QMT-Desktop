"""用户持仓账本 — 初始化真实持仓与流动资金,作为影子运行的决策基准。

功能:输入当前持有的基金与金额、当前可用流动资金(可为 0),初始化持仓状态;
此后每次影子快照(预测/换手/三档成本)都以这份真实持仓为基准,而非上一期的假想目标。

设计要点:
  * **金额 → 份额(点时点)**:初始化日用当日**可见**的最新复权净值把金额换算成份额
    (``available_at <= as_of``,绝不用未来净值);之后按任意日净值 mark-to-market,
    权重随净值自然漂移。
  * **持有天数**:每笔持仓记 ``purchase_date``(缺省=初始化日),赎回费档位按真实
    持有期计算(qmt.fees 的分档)。
  * **合规定位**:这是本地影子模拟的状态输入,输出为诊断信息(换手/成本/分位),
    不构成个性化投资建议(CLAUDE.md 合规边界)。

存储:``<ledger_dir>/portfolio.json``(与影子账本同目录);``portfolio-init`` 重复
执行即覆盖(重新初始化),历史快照不受影响(账本只追加)。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qmt.pit import PointInTimeStore
from qmt.factor.types import PortfolioState

__all__ = ["Position", "PortfolioBook", "init_book", "load_book", "mark_to_market",
           "to_portfolio_state", "buy", "sell", "set_cash", "set_purchase_dates",
           "BOOK_FILENAME"]

BOOK_FILENAME = "portfolio.json"


@dataclass
class Position:
    fund_code: str
    units: float                     # 份额 = 初始金额 / 初始化日可见净值
    init_amount: float               # 初始化时输入的金额(元),仅留档
    init_nav: float                  # 换算用的净值(点时点),仅留档
    purchase_date: str               # YYYY-MM-DD,持有天数从这天起算


@dataclass
class PortfolioBook:
    initialized_at: str              # YYYY-MM-DD
    cash: float                      # 可用流动资金(元),>= 0
    positions: dict[str, Position] = field(default_factory=dict)

    def to_json(self) -> str:
        obj = {
            "initialized_at": self.initialized_at,
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)


def _latest_visible_nav(store: PointInTimeStore, fund_code: str, as_of: date) -> float | None:
    """初始化日**可见**的最新复权净值(available_at <= as_of,点时点)。"""
    df = store.get_asof("fund_nav", pd.Timestamp(as_of))
    df = df[df["fund_code"].astype(str) == str(fund_code)]
    if df.empty:
        return None
    row = df.sort_values("nav_date").iloc[-1]
    nav = row.get("adj_nav")
    if nav is None or (isinstance(nav, float) and (np.isnan(nav) or nav <= 0)):
        nav = row.get("unit_nav")
    try:
        nav = float(nav)
    except (TypeError, ValueError):
        return None
    return nav if nav > 0 else None


def init_book(
    store: PointInTimeStore,
    holdings: dict[str, float],
    cash: float,
    *,
    as_of: date | None = None,
    ledger_dir: str | Path,
    purchase_dates: dict[str, str] | None = None,
    auto_fetch: bool = True,
) -> PortfolioBook:
    """初始化(或覆盖)持仓账本。

    ``holdings``: {6 位基金代码: 金额(元)};``cash``: 可用流动资金(元,可为 0)。
    金额按 ``as_of``(缺省=今天)可见的最新净值换算成份额;店里没有该基金净值时,
    ``auto_fetch=True`` 会尝试即时拉取(需 TUSHARE_TOKEN),仍拿不到则报错拒收
    (fail-closed,绝不默认净值=1)。
    """
    as_of = as_of or date.today()
    if cash < 0:
        raise ValueError(f"可用流动资金不能为负: {cash}")
    if not holdings and cash <= 0:
        raise ValueError("持仓与现金都为空 — 没有可初始化的状态")
    purchase_dates = purchase_dates or {}

    positions: dict[str, Position] = {}
    missing: list[str] = []
    for fc, amount in holdings.items():
        fc = str(fc).strip()
        if not (len(fc) == 6 and fc.isdigit()):
            raise ValueError(f"基金代码须为 6 位数字: {fc!r}")
        if amount <= 0:
            raise ValueError(f"{fc} 的金额须为正: {amount}")
        nav = _latest_visible_nav(store, fc, as_of)
        if nav is None and auto_fetch:
            try:
                from qmt.data import sources
                from qmt.data.tushare_client import fund_code_to_ts_code
                sources.fund_nav(fund_code_to_ts_code(fc), store=store, persist=True)
                nav = _latest_visible_nav(store, fc, as_of)
            except Exception:
                nav = None
        if nav is None:
            missing.append(fc)
            continue
        positions[fc] = Position(
            fund_code=fc,
            units=float(amount) / nav,
            init_amount=float(amount),
            init_nav=nav,
            purchase_date=purchase_dates.get(fc, str(as_of)),
        )
    if missing:
        raise ValueError(
            f"以下基金在 {as_of} 没有可见净值,无法换算份额(先 ingest 或检查代码): {missing}"
        )

    book = PortfolioBook(initialized_at=str(as_of), cash=float(cash), positions=positions)
    path = Path(ledger_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / BOOK_FILENAME).write_text(book.to_json(), encoding="utf-8")
    return book


def set_purchase_dates(
    ledger_dir: str | Path,
    new_date: date,
    funds: list[str] | None = None,
) -> PortfolioBook:
    """把持仓的买入日改为 ``new_date``(缺省全部;影响赎回费档位与持有天数)。

    只改 ``purchase_date``,份额/金额/现金不动;写回 ``portfolio.json``。
    """
    book = load_book(ledger_dir)
    if book is None:
        raise ValueError("尚未初始化持仓 — 先运行 portfolio-init")
    targets = set(funds) if funds else set(book.positions)
    unknown = targets - set(book.positions)
    if unknown:
        raise ValueError(f"账本中没有这些基金: {sorted(unknown)}")
    for fc in targets:
        book.positions[fc].purchase_date = str(new_date)
    (Path(ledger_dir) / BOOK_FILENAME).write_text(book.to_json(), encoding="utf-8")
    return book


def _save(book: PortfolioBook, ledger_dir: str | Path) -> None:
    (Path(ledger_dir) / BOOK_FILENAME).write_text(book.to_json(), encoding="utf-8")


def buy(store: PointInTimeStore, ledger_dir: str | Path, fund_code: str, amount: float,
        *, as_of: date | None = None, purchase_date: str | None = None,
        from_cash: bool = True, auto_fetch: bool = True) -> PortfolioBook:
    """买入 ``amount`` 元的某基金(按当日可见净值换算份额,加进现有持仓)。

    ``from_cash=True`` 时从可用现金里扣(现金不足则报错);已持有则加仓并重置
    买入日为本次(赎回费档位从新算,保守)。
    """
    as_of = as_of or date.today()
    fc = str(fund_code).strip()
    if not (len(fc) == 6 and fc.isdigit()):
        raise ValueError(f"基金代码须为 6 位数字: {fc!r}")
    if amount <= 0:
        raise ValueError(f"买入金额须为正: {amount}")
    book = load_book(ledger_dir)
    if book is None:
        raise ValueError("尚未初始化持仓 — 先运行 portfolio-init")
    if from_cash and amount > book.cash + 1e-6:
        raise ValueError(f"现金不足:可用 ¥{book.cash:,.0f},需 ¥{amount:,.0f}(用 --no-from-cash 允许外部资金)")
    nav = _latest_visible_nav(store, fc, as_of)
    if nav is None and auto_fetch:
        try:
            from qmt.data import sources
            from qmt.data.tushare_client import fund_code_to_ts_code
            sources.fund_nav(fund_code_to_ts_code(fc), store=store, persist=True)
            nav = _latest_visible_nav(store, fc, as_of)
        except Exception:
            nav = None
    if nav is None:
        raise ValueError(f"{fc} 在 {as_of} 没有可见净值,无法换算份额")
    new_units = amount / nav
    pd_str = purchase_date or str(as_of)
    if fc in book.positions:
        p = book.positions[fc]
        p.units += new_units
        p.init_amount += float(amount)
        p.init_nav = nav
        p.purchase_date = pd_str      # 加仓 → 持有期从本次重算(赎回费保守)
    else:
        book.positions[fc] = Position(fund_code=fc, units=new_units,
                                      init_amount=float(amount), init_nav=nav, purchase_date=pd_str)
    if from_cash:
        book.cash -= float(amount)
    _save(book, ledger_dir)
    return book


def sell(store: PointInTimeStore, ledger_dir: str | Path, fund_code: str,
         amount: float | None = None, *, all: bool = False, as_of: date | None = None,
         to_cash: bool = True) -> PortfolioBook:
    """卖出某基金:``--all`` 全部赎回,或 ``amount`` 元(按当日可见净值折算份额)。

    ``to_cash=True`` 时卖出所得(按估值,未扣赎回费)回到可用现金。
    """
    as_of = as_of or date.today()
    fc = str(fund_code).strip()
    book = load_book(ledger_dir)
    if book is None or fc not in book.positions:
        raise ValueError(f"账本中没有持仓 {fc}")
    pos = book.positions[fc]
    nav = _latest_visible_nav(store, fc, as_of)
    if nav is None:
        raise ValueError(f"{fc} 在 {as_of} 没有可见净值,无法折算")
    cur_value = pos.units * nav
    if all or amount is None or amount >= cur_value - 1e-6:
        proceeds = cur_value
        del book.positions[fc]
    else:
        if amount <= 0:
            raise ValueError(f"卖出金额须为正: {amount}")
        proceeds = float(amount)
        pos.units -= amount / nav
    if to_cash:
        book.cash += float(proceeds)
    _save(book, ledger_dir)
    return book


def set_cash(ledger_dir: str | Path, cash: float) -> PortfolioBook:
    """把可用流动资金设为 ``cash`` 元(>= 0)。"""
    if cash < 0:
        raise ValueError("可用流动资金不能为负")
    book = load_book(ledger_dir)
    if book is None:
        raise ValueError("尚未初始化持仓 — 先运行 portfolio-init")
    book.cash = float(cash)
    _save(book, ledger_dir)
    return book


def load_book(ledger_dir: str | Path) -> PortfolioBook | None:
    p = Path(ledger_dir) / BOOK_FILENAME
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    positions = {k: Position(**v) for k, v in obj.get("positions", {}).items()}
    return PortfolioBook(
        initialized_at=obj["initialized_at"],
        cash=float(obj.get("cash", 0.0)),
        positions=positions,
    )


def mark_to_market(
    book: PortfolioBook, store: PointInTimeStore, as_of: date
) -> dict[str, Any]:
    """按 ``as_of`` 可见净值估值:每只基金市值、权重、持有天数、总资产。"""
    rows: dict[str, dict[str, Any]] = {}
    total = book.cash
    for fc, pos in book.positions.items():
        nav = _latest_visible_nav(store, fc, as_of)
        value = pos.units * nav if nav is not None else None
        if value is not None:
            total += value
        rows[fc] = {
            "units": round(pos.units, 4),
            "nav": nav,
            "value": (round(value, 2) if value is not None else None),
            # clamp >= 0: an init dated after the latest NAV date (e.g. today vs
            # yesterday's NAV) must not produce negative holding periods
            "hold_days": max(0, (pd.Timestamp(as_of) - pd.Timestamp(pos.purchase_date)).days),
            "stale_nav": nav is None,
        }
    for fc, r in rows.items():
        r["weight"] = (round(r["value"] / total, 4) if (r["value"] is not None and total > 0) else None)
    return {
        "as_of": str(as_of),
        "total_value": round(total, 2),
        "cash": round(book.cash, 2),
        "cash_weight": (round(book.cash / total, 4) if total > 0 else None),
        "positions": rows,
    }


def to_portfolio_state(
    book: PortfolioBook, store: PointInTimeStore, as_of: date
) -> PortfolioState:
    """把账本转成 ExecSim 的 :class:`PortfolioState`(权重 + 真实持有天数)。

    净值缺失的持仓按 0 权重进入(fail-closed),并保留在 hold_days 里以便赎回费
    档位仍按真实持有期计。
    """
    mtm = mark_to_market(book, store, as_of)
    weights = {fc: (r["weight"] or 0.0) for fc, r in mtm["positions"].items()}
    hold_days = {fc: int(r["hold_days"]) for fc, r in mtm["positions"].items()}
    return PortfolioState(
        as_of=as_of,
        holdings=pd.Series(weights, dtype=float),
        hold_days=hold_days,
        cash=float(mtm["cash_weight"] or 0.0),
    )
