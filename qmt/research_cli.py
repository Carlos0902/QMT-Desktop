"""Data acquisition and immutable evaluation control surfaces."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import typer

cache_app = typer.Typer(help="全基金原始缓存、采集队列与覆盖情况。", add_completion=False)
evaluation_app = typer.Typer(help="冻结数据版本与评测协议，重复比较固定基线。", add_completion=False)
research_app = typer.Typer(help="本地研究平台。", add_completion=False)
forward_app = typer.Typer(help="不可回填的前瞻信号登记、冻结与评分。", add_completion=False)
evaluation_app.add_typer(forward_app, name="forward")


def _root(path):
    from qmt.config import get_settings
    return path or get_settings().data_cache_dir / "fund_lake"


def _emit(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))


@cache_app.command("status")
def cache_status(lake: Optional[Path] = typer.Option(None, "--lake")):
    from qmt.data.fund_lake import DatasetLake
    from qmt.cache_worker import _publish
    root = _root(lake)
    inventory = DatasetLake(root).inventory()
    _publish(Path(root) / "inventory.json", inventory)
    collector = Path(root) / "collector.json"
    _emit({**inventory, "collector": json.loads(collector.read_text()) if collector.exists() else None})


@cache_app.command("import-legacy")
def import_legacy(lake: Optional[Path] = typer.Option(None, "--lake"),
                  source: Optional[Path] = typer.Option(None, "--source")):
    """复制旧缓存为可追溯的归档版本；不覆盖旧文件。"""
    from qmt.config import get_settings
    from qmt.data.fund_lake import DatasetLake
    _emit(DatasetLake(_root(lake)).import_legacy(source or get_settings().data_cache_dir))


@cache_app.command("plan")
def plan(lake: Optional[Path] = typer.Option(None, "--lake"),
         start: str = typer.Option("1998-01-01", "--start"),
         end: Optional[str] = typer.Option(None, "--end"),
         markets: str = typer.Option("O,E", "--markets"),
         page_size: int = typer.Option(4000, "--page-size", min=1, max=15000),
         supplements: bool = typer.Option(True, "--supplements/--no-supplements")):
    """登记全类型、全生命周期基金及逐日净值任务，重复登记幂等。"""
    from qmt.data.fund_lake import DatasetLake
    end = end or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    start = datetime.fromisoformat(start).date().isoformat()
    end = datetime.fromisoformat(end).date().isoformat()
    selected = tuple(s.strip().upper() for s in markets.split(",") if s.strip())
    if not selected or set(selected) - {"O", "E"}:
        raise typer.BadParameter("markets must be O,E")
    store = DatasetLake(_root(lake))
    result = {"basic": store.plan_basic(markets=selected, page_size=page_size),
              "nav": store.plan_nav_dates(start, end, markets=selected, page_size=page_size)}
    store.enqueue("fund_company", {}, priority=100)
    first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
    for year in range(first.year, last.year + 1):
        lo, hi = max(start, f"{year:04d}-01-01"), min(end, f"{year:04d}-12-31")
        store.enqueue("trade_cal", {"exchange": "SSE", "start_date": lo.replace("-", ""),
                                     "end_date": hi.replace("-", "")}, priority=100)
        store.enqueue("index_daily", {"ts_code": "000300.SH", "start_date": lo.replace("-", ""),
                                       "end_date": hi.replace("-", "")}, priority=90)
    if supplements:
        quarters = []
        for year in range(first.year, last.year + 1):
            for suffix in ("03-31", "06-30", "09-30", "12-31"):
                period = f"{year:04d}-{suffix}"
                if start <= period <= end:
                    quarters.append({"period": period.replace("-", "")})
        days = [(first + timedelta(days=n)).strftime("%Y%m%d") for n in range((last-first).days + 1)]
        result["supplements"] = {
            "holdings": store.enqueue_many("fund_portfolio", quarters, page_size=page_size, priority=30),
            "dividends": store.enqueue_many("fund_div", ({"ann_date": d} for d in days), page_size=page_size, priority=25),
            "listed_shares": store.enqueue_many("fund_share", ({"trade_date": d} for d in days), page_size=2000, priority=20),
        }
        store.enqueue("fund_manager", {}, page_size=2000, priority=80)
    result["inventory"] = store.inventory()
    _emit(result)


@cache_app.command("refresh")
def refresh(lake: Optional[Path] = typer.Option(None, "--lake"),
            days: int = typer.Option(14, "--days", min=1),
            end: Optional[str] = typer.Option(None, "--end")):
    """追加近期修订和当天元数据刷新任务，保留旧版本与旧任务。"""
    from qmt.data.fund_lake import DatasetLake
    store = DatasetLake(_root(lake))
    generation = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    store.plan_basic(generation=generation)
    _emit(store.refresh_recent(days=days, end_date=end, generation=generation))


@cache_app.command("collect")
def collect(lake: Optional[Path] = typer.Option(None, "--lake"),
            hours: float = typer.Option(24., "--hours", min=.001)):
    """持续处理采集队列，自动处理限频；支持暂停及重启续传。"""
    from qmt.cache_worker import run_worker
    root = _root(lake)
    _emit(run_worker(root, seconds=hours * 3600, resume=True))


@cache_app.command("pause")
def pause(lake: Optional[Path] = typer.Option(None, "--lake")):
    """在当前批次完成后暂停托管采集，保留全部断点。"""
    root = Path(_root(lake))
    root.mkdir(parents=True, exist_ok=True)
    (root / ".collector.pause").touch()
    _emit({"status": "pause_requested", "resume": "qmt cache collect"})


@cache_app.command("sync")
def sync(lake: Optional[Path] = typer.Option(None, "--lake"),
         queries: int = typer.Option(500, "--max-queries", min=1),
         seconds: float = typer.Option(900, "--max-seconds", min=1),
         interval: float = typer.Option(.4, "--interval", min=.3)):
    """按已登记队列采集，页级断点和权限失败保留在目录数据库。"""
    from qmt.data.fund_lake import DatasetLake
    from qmt.data.tushare_client import TushareClient
    _emit(DatasetLake(_root(lake)).run_budget(TushareClient(timeout=25, max_retries=0),
          max_queries=queries, max_seconds=seconds, min_interval=interval))


@cache_app.command("retry-interface")
def retry_interface(api: str, lake: Optional[Path] = typer.Option(None, "--lake")):
    """权限变更后显式恢复被阻断的接口任务。"""
    from qmt.data.fund_lake import DatasetLake
    _emit({"tasks_reset": DatasetLake(_root(lake)).reset_interface(api)})


@cache_app.command("retry-failed")
def retry_failed(lake: Optional[Path] = typer.Option(None, "--lake"),
                 api: Optional[str] = typer.Option(None, "--api"),
                 include_truncated: bool = typer.Option(False, "--include-truncated")):
    """恢复失败任务；分页契约问题需显式选择重试并修正原因。"""
    from qmt.data.fund_lake import DatasetLake
    _emit({"tasks_reset": DatasetLake(_root(lake)).retry_tasks(api_name=api,
                                                           include_truncated=include_truncated)})


@cache_app.command("quality")
def quality(lake: Optional[Path] = typer.Option(None, "--lake"),
            output: Optional[Path] = typer.Option(None, "--output")):
    """公告日期凭证校验、精确去重与冲突隔离，生成独立不可变清洗版本。"""
    from qmt.data.fund_lake import _atomic_bytes
    from qmt.data.quality import build_quality_bundle
    root = Path(_root(lake))
    output = output or root / "quality"
    result = build_quality_bundle(root, output)
    _atomic_bytes(root / "quality" / "latest.json", json.dumps(
        {"quality_path": result["quality_path"], "quality_id": result["quality_id"],
         "manifest_hash": result["manifest_hash"]}, ensure_ascii=False, indent=2).encode())
    _emit(result)


@evaluation_app.command("snapshot")
def snapshot(lake: Optional[Path] = typer.Option(None, "--lake"),
             output: Path = typer.Option(Path("artifacts/evaluation_platform/snapshots"), "--output"),
             seal: str = typer.Option("2024-12-10", "--seal"),
             fund_type: str = typer.Option("股票型", "--fund-type"),
             quality_dir: Optional[Path] = typer.Option(None, "--quality-dir")):
    """固定封存前净值与元数据版本；采集可继续，快照保持不变。"""
    from qmt.evaluation import prepare_snapshot
    from qmt.data.quality import quality_evaluation_inputs
    if quality_dir is None:
        pointer = Path(_root(lake)) / "quality" / "latest.json"
        if not pointer.exists():
            raise typer.BadParameter("先运行 qmt cache quality，评测仅使用已校验的清洗版本。")
        quality_dir = Path(json.loads(pointer.read_text())["quality_path"])
    nav, basic, calendar, quality_report = quality_evaluation_inputs(quality_dir, seal=seal, fund_type=fund_type)
    _emit(prepare_snapshot(nav, basic, calendar, output, sealed_from=seal,
                           metadata_basis="current_snapshot", source="tushare_curated_vendor_announcements",
                           provenance={"quality_id": quality_report["quality_id"],
                                       "quality_manifest_hash": quality_report["manifest_hash"],
                                       "source_hash": quality_report["source_hash"]}))


@evaluation_app.command("register")
def register(snapshot_path: Path = typer.Argument(...),
             registry: Path = typer.Option(Path("artifacts/evaluation_platform/protocols"), "--registry"),
             config: Optional[Path] = typer.Option(None, "--config")):
    """评测前冻结基线、候选、日期、覆盖率与统计口径。"""
    from qmt.evaluation import register_protocol
    protocol = json.loads(config.read_text()) if config else None
    _emit(register_protocol(snapshot_path, registry, protocol=protocol))


@evaluation_app.command("run")
def evaluate(snapshot_path: Path = typer.Argument(...), protocol_path: Path = typer.Argument(...),
             output: Path = typer.Option(Path("artifacts/evaluation_platform/runs"), "--output")):
    """统一基线与候选的日期和基金池；结果为排序研究，不自动认证交易收益。"""
    from qmt.evaluation import run_evaluation
    _emit(run_evaluation(snapshot_path, protocol_path, output))


@evaluation_app.command("search")
def search_benchmark(snapshot_path: Path = typer.Argument(...),
                     protocol_path: Path = typer.Argument(...),
                     output: Path = typer.Option(..., "--output"),
                     budget: int = typer.Option(120, "--budget", min=20)):
    """清洗快照上的年度 GP/Q 学习及随机、bandit 对照；仅认证排序研究。"""
    from qmt.evaluation.search_benchmark import run
    _emit(run(snapshot_path, protocol_path, output, budget=budget))


@evaluation_app.command("improve-search")
def improve_search(snapshot_path: Path = typer.Argument(...),
                   protocol_path: Path = typer.Argument(...),
                   output: Path = typer.Option(..., "--output"),
                   budget: int = typer.Option(120, "--budget", min=20)):
    """基线补充信号、独立长验证和配对证据门槛的 GP/Q 消融研究。"""
    from qmt.evaluation.search_improvement import run
    _emit(run(snapshot_path, protocol_path, output, budget=budget))


@evaluation_app.command("features")
def information_features(snapshot_path: Path, protocol_path: Path,
                         quality_dir: Path = typer.Option(..., "--quality-dir"),
                         output: Path = typer.Option(..., "--output"),
                         lake: Optional[Path] = typer.Option(None, "--lake")):
    """审核持仓版本并构建六个新信息特征；原始净值快照不变。"""
    from qmt.evaluation.information_features import build
    _emit(build(snapshot_path, protocol_path, quality_dir, _root(lake), output))


@evaluation_app.command("feature-audit")
def information_audit(feature_path: Path):
    """校验特征快照哈希及原定决策日期的覆盖门槛。"""
    from qmt.evaluation.information_features import load
    _emit(load(feature_path)["manifest"])


@evaluation_app.command("feature-study")
def information_study(feature_path: Path, output: Path = typer.Option(..., "--output")):
    """冻结并执行十组新信息实验；预算上限180次搜索。"""
    from qmt.evaluation.information_study import run
    _emit(run(feature_path, output))


@forward_app.command("register")
def forward_register(study_path: Path, output: Path = typer.Option(..., "--output"),
                     lake: Optional[Path] = typer.Option(None, "--lake")):
    from qmt.evaluation.forward import register
    _emit(register(study_path, output, _root(lake)))


@forward_app.command("record")
def forward_record(registry: Path, lake: Optional[Path] = typer.Option(None, "--lake")):
    from qmt.evaluation.forward import record
    _emit(record(registry, _root(lake)))


@forward_app.command("score")
def forward_score(registry: Path, lake: Optional[Path] = typer.Option(None, "--lake")):
    from qmt.evaluation.forward import score
    _emit(score(registry, _root(lake)))


@forward_app.command("tick")
def forward_tick(registry: Path, lake: Optional[Path] = typer.Option(None, "--lake"),
                 acquire: bool = typer.Option(True, "--acquire/--no-acquire")):
    """定时入口：追加近期缓存、冻结当期信号并评分已成熟标签。"""
    from qmt.evaluation.forward import tick
    _emit(tick(registry, _root(lake), acquire=acquire))


@research_app.command("serve")
def serve(lake: Optional[Path] = typer.Option(None, "--lake"),
          artifacts: Path = typer.Option(Path("artifacts/evaluation_platform"), "--artifacts"),
          host: str = typer.Option("127.0.0.1", "--host"),
          port: int = typer.Option(8766, "--port")):
    """查看本地缓存任务、数据版本与评测记录。"""
    import uvicorn
    from qmt.research_api import create_app
    uvicorn.run(create_app(_root(lake), artifacts), host=host, port=port)
