"""factor_archive —— 因子档案库（tech doc ch.4 §sec:mod-archive / instruction §5.1 L6, §5.2）。

存放 ``FactorRecord`` 的全生命周期档案。核心设计约束（§5.2 / §7.3 审计）：

*不只保留胜者*。被淘汰的候选（``status`` 为 ``retired`` / ``archived``，带
``reject_reason``）与失败方案同样入库、可查——否则搜索本身带上幸存者偏差，
DSR / PBO 无法对「一共尝试了多少次才挑出赢家」做出正确校正。

本模块提供两个实现，接口一致（均满足 :class:`qmt.factor.types.FactorArchive`）：

* :class:`InMemoryFactorArchive` —— 纯内存字典，进程内使用。
* :class:`DuckDBFactorArchive` —— 惰性 import ``duckdb`` 持久化到磁盘文件；
  ``duckdb`` 不可用或建连失败时**退化**为内存实现（不抛出，保证接口可用）。

``put`` 以 ``factor_id`` 为主键：同 id 覆盖即为状态流转（candidate→screened→…→retired）。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Union

from .dsl import FormulaDSL
from .types import FactorExpr, FactorRecord, FactorStatus

__all__ = ["InMemoryFactorArchive", "DuckDBFactorArchive"]

# 用于过滤的顶层字段（query(**filters) 支持的键）——其余键按同名标量字段过滤。
_FILTERABLE = frozenset(
    {"status", "parent_id", "search_config_id", "factor_id", "compliance_passed",
     "data_source_checked", "split_id"}
)

_dsl = FormulaDSL()  # 无 industry_of 也够用：仅用其 parse 重建 AST


# ---------------------------------------------------------------------------
# 序列化：FactorRecord <-> 纯 JSON 可写 dict
# ---------------------------------------------------------------------------

def _expr_to_dict(expr: FactorExpr) -> dict[str, Any]:
    """把 FactorExpr 拆成可 JSON 化的 dict（AST 不直接序列化，靠 text 重建）。"""
    return {
        "text": expr.text,
        "data_fields": list(expr.data_fields),
        "time_windows": list(expr.time_windows),
        "operators": sorted(expr.operators),
        "complexity": expr.complexity,
        "guarded_ops": list(expr.guarded_ops),
    }


def _expr_from_dict(d: dict[str, Any]) -> FactorExpr:
    """从 dict 重建 FactorExpr；AST 优先用 text 重解析，失败则保留元数据、ast=None。"""
    text = d.get("text", "")
    try:
        # 重解析可同时校正 data_fields/operators 等派生字段，最忠实。
        return _dsl.parse(text)
    except Exception:  # noqa: BLE001 —— 无法解析时退化为仅元数据的壳
        return FactorExpr(
            text=text,
            ast=None,
            data_fields=list(d.get("data_fields", [])),
            time_windows=list(d.get("time_windows", [])),
            operators=frozenset(d.get("operators", [])),
            complexity=int(d.get("complexity", 0)),
            guarded_ops=list(d.get("guarded_ops", [])),
        )


def _span_to_list(span: tuple[date, date] | None) -> list[str] | None:
    if span is None:
        return None
    a, b = span
    return [a.isoformat(), b.isoformat()]


def _span_from_list(v: Any) -> tuple[date, date] | None:
    if v is None:
        return None
    return (date.fromisoformat(v[0]), date.fromisoformat(v[1]))


def record_to_dict(record: FactorRecord) -> dict[str, Any]:
    """FactorRecord -> 纯 JSON 可序列化 dict（保真：所有字段可回程）。"""
    return {
        "factor_id": record.factor_id,
        "expr": _expr_to_dict(record.expr),
        "parent_id": record.parent_id,
        "search_config_id": record.search_config_id,
        "train_span": _span_to_list(record.train_span),
        "valid_span": _span_to_list(record.valid_span),
        "split_id": record.split_id,
        "ic": record.ic,
        "rank_ic": record.rank_ic,
        "turnover": record.turnover,
        "net_return_by_tier": dict(record.net_return_by_tier),
        "capacity_est": record.capacity_est,
        "industry_exposure": dict(record.industry_exposure),
        "style_exposure": dict(record.style_exposure),
        "market_exposure": record.market_exposure,
        "compliance_passed": record.compliance_passed,
        "data_source_checked": record.data_source_checked,
        "status": record.status.value,
        "score": record.score,
        "reject_reason": record.reject_reason,
        "created_at": record.created_at.isoformat(),
    }


def record_from_dict(d: dict[str, Any]) -> FactorRecord:
    """纯 dict -> FactorRecord（record_to_dict 的逆）。"""
    return FactorRecord(
        factor_id=d["factor_id"],
        expr=_expr_from_dict(d["expr"]),
        parent_id=d.get("parent_id"),
        search_config_id=d.get("search_config_id", ""),
        train_span=_span_from_list(d.get("train_span")),
        valid_span=_span_from_list(d.get("valid_span")),
        split_id=d.get("split_id", ""),
        ic=float(d.get("ic", 0.0)),
        rank_ic=float(d.get("rank_ic", 0.0)),
        turnover=float(d.get("turnover", 0.0)),
        net_return_by_tier=dict(d.get("net_return_by_tier", {})),
        capacity_est=d.get("capacity_est"),
        industry_exposure=dict(d.get("industry_exposure", {})),
        style_exposure=dict(d.get("style_exposure", {})),
        market_exposure=float(d.get("market_exposure", 0.0)),
        compliance_passed=bool(d.get("compliance_passed", False)),
        data_source_checked=bool(d.get("data_source_checked", False)),
        status=FactorStatus(d.get("status", FactorStatus.CANDIDATE.value)),
        score=d.get("score"),
        reject_reason=d.get("reject_reason"),
        created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime(1970, 1, 1),
    )


def _matches(record: FactorRecord, filters: dict[str, Any]) -> bool:
    """记录是否命中全部过滤条件（AND 语义）；未知键按 record 的同名属性比较。"""
    for key, want in filters.items():
        have = getattr(record, key, None)
        # 允许用枚举的字符串值过滤 status
        if key == "status" and isinstance(want, str):
            have = have.value if isinstance(have, FactorStatus) else have
        if have != want:
            return False
    return True


# ---------------------------------------------------------------------------
# 内存实现
# ---------------------------------------------------------------------------

class InMemoryFactorArchive:
    """纯内存的 :class:`qmt.factor.types.FactorArchive`（tech doc §sec:mod-archive）。

    同 ``factor_id`` 的 ``put`` 覆盖旧记录——用于状态流转与结果回填。
    失败/淘汰候选一律入库，``query`` 可按 ``status`` / ``parent_id`` 等回捞（§5.2 审计）。
    """

    def __init__(self) -> None:
        self._records: dict[str, FactorRecord] = {}

    def put(self, record: FactorRecord) -> str:
        """写入或覆盖一条记录，返回其 ``factor_id``。"""
        if not record.factor_id:
            raise ValueError("FactorRecord.factor_id 不能为空——它是档案主键。")
        self._records[record.factor_id] = record
        return record.factor_id

    def get(self, factor_id: str) -> FactorRecord:
        """按 id 取记录；缺失 raise ``KeyError``。"""
        return self._records[factor_id]

    def query(self, **filters: Any) -> list[FactorRecord]:
        """按 ``status`` / ``parent_id`` / ``search_config_id`` 等过滤；无 filter 返回全部。

        返回按 ``created_at`` 升序（同刻按 ``factor_id``）稳定排序，便于审计回放。
        """
        out = [r for r in self._records.values() if _matches(r, filters)]
        out.sort(key=lambda r: (r.created_at, r.factor_id))
        return out

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# DuckDB 持久化实现（惰性 import + 退化容错）
# ---------------------------------------------------------------------------

class DuckDBFactorArchive:
    """DuckDB 文件持久化的 :class:`qmt.factor.types.FactorArchive`。

    表 ``factor_archive`` 保存 ``factor_id``(PK) + 若干可索引过滤列 + ``payload``(JSON)。
    ``put`` 用 delete+insert 实现 upsert（同 id 覆盖=状态流转）。

    容错：``duckdb`` 不可用、或建连/建表失败时，**退化**为内存字典
    （``self._degraded=True``），接口不变、不抛出——仿 ts_model_bench 里 torch 的惰性写法。
    """

    _TABLE = "factor_archive"

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        self._degraded = False
        self._con: Any = None
        self._mem: dict[str, FactorRecord] = {}  # 退化时的后备存储
        self._path = str(path) if path is not None else ":memory:"
        try:
            import duckdb  # noqa: PLC0415 —— 惰性 import，重依赖容错
        except Exception:  # noqa: BLE001
            self._degraded = True
            return
        try:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(database=self._path)
            self._con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    factor_id VARCHAR PRIMARY KEY,
                    status VARCHAR,
                    parent_id VARCHAR,
                    search_config_id VARCHAR,
                    created_at VARCHAR,
                    payload VARCHAR
                )
                """
            )
        except Exception:  # noqa: BLE001 —— 任何建连/建表失败都退化，不影响调用方
            self._degraded = True
            self._con = None

    # -- 写 ----------------------------------------------------------------
    def put(self, record: FactorRecord) -> str:
        if not record.factor_id:
            raise ValueError("FactorRecord.factor_id 不能为空——它是档案主键。")
        if self._degraded:
            self._mem[record.factor_id] = record
            return record.factor_id
        payload = json.dumps(record_to_dict(record), ensure_ascii=False)
        # delete + insert 实现幂等 upsert（同 id 覆盖）
        self._con.execute(f"DELETE FROM {self._TABLE} WHERE factor_id = ?", [record.factor_id])
        self._con.execute(
            f"INSERT INTO {self._TABLE} "
            f"(factor_id, status, parent_id, search_config_id, created_at, payload) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            [
                record.factor_id,
                record.status.value,
                record.parent_id,
                record.search_config_id,
                record.created_at.isoformat(),
                payload,
            ],
        )
        return record.factor_id

    # -- 读 ----------------------------------------------------------------
    def get(self, factor_id: str) -> FactorRecord:
        if self._degraded:
            return self._mem[factor_id]
        rows = self._con.execute(
            f"SELECT payload FROM {self._TABLE} WHERE factor_id = ?", [factor_id]
        ).fetchall()
        if not rows:
            raise KeyError(factor_id)
        return record_from_dict(json.loads(rows[0][0]))

    def query(self, **filters: Any) -> list[FactorRecord]:
        if self._degraded:
            out = [r for r in self._mem.values() if _matches(r, filters)]
            out.sort(key=lambda r: (r.created_at, r.factor_id))
            return out
        # 可下推到 SQL 的列走 WHERE；其余（如 compliance_passed）读回后在 Python 过滤。
        pushdown_cols = {"status", "parent_id", "search_config_id", "factor_id"}
        clauses: list[str] = []
        params: list[Any] = []
        residual: dict[str, Any] = {}
        for key, want in filters.items():
            if key in pushdown_cols:
                val = want.value if isinstance(want, FactorStatus) else want
                if val is None:
                    clauses.append(f"{key} IS NULL")
                else:
                    clauses.append(f"{key} = ?")
                    params.append(val)
            else:
                residual[key] = want
        sql = f"SELECT payload FROM {self._TABLE}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, factor_id"
        rows = self._con.execute(sql, params).fetchall()
        out = [record_from_dict(json.loads(r[0])) for r in rows]
        if residual:
            out = [r for r in out if _matches(r, residual)]
        return out

    def close(self) -> None:
        """关闭底层连接（持久化到文件后可安全释放）。"""
        if self._con is not None:
            try:
                self._con.close()
            except Exception:  # noqa: BLE001
                pass
            self._con = None

    def __len__(self) -> int:
        if self._degraded:
            return len(self._mem)
        return int(self._con.execute(f"SELECT COUNT(*) FROM {self._TABLE}").fetchone()[0])
