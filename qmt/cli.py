"""QMT command-line control surface.

A thin Typer wrapper over the pipeline so the system can be driven, debugged and
refined from a terminal — the CLI half of the control plane (the other half is the
FastAPI dashboard served by ``qmt serve``).

Every command reads data only through the point-in-time store (``ingest`` writes,
the run/backtest paths read ``available_at <= as_of``), so the CLI cannot peek at
the future any more than the library can.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="QMT 控制台 CLI — 控制 / 调试 / 调参 场外权益基金分析流水线。")
config_app = typer.Typer(help="查看有效配置（可调阈值）。")
app.add_typer(config_app, name="config")

# 符号学习因子生产平台（新架构，见 doc/techdesign/）：`qmt factor demo`
from qmt.factor.cli import app as factor_app  # noqa: E402

app.add_typer(factor_app, name="factor")

from qmt.research_cli import cache_app, evaluation_app, research_app  # noqa: E402

app.add_typer(cache_app, name="cache")
app.add_typer(evaluation_app, name="evaluate")
app.add_typer(research_app, name="research")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _store():
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore

    return PointInTimeStore(get_settings().ensure_data_cache())


def _enum(v):
    return getattr(v, "value", v)


def _echo_json(obj) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# --------------------------------------------------------------------------- #
@app.command()
def analyze(
    fund_code: str = typer.Argument(..., help="6 位基金代码，如 011133。"),
    as_of: str = typer.Option(..., "--as-of", help="决策时点 YYYY-MM-DD。"),
    no_ingest: bool = typer.Option(False, "--no-ingest", help="不重新入库，复用已缓存数据。"),
    raw: bool = typer.Option(False, "--raw", help="打印完整 JSON。"),
) -> None:
    """分析模式：ingest + run_analysis，打印逐阶段轨迹（不读取用户资产）。"""
    from qmt.compliance import ComplianceViolation
    from qmt.data.ingest import ingest_fund
    from qmt.pipeline.run import run_analysis

    d = _parse_date(as_of)
    store = _store()
    if not no_ingest:
        rep = ingest_fund(fund_code, d, store)
        typer.echo(f"入库：{rep.nav_rows} NAV · {rep.holdings_reports} 持仓报告 · 行业图 {rep.industry_map_size} · gaps={rep.gaps or '无'}")
    try:
        res = run_analysis(fund_code, d, store)
    except ComplianceViolation as exc:
        typer.secho(f"✗ 已拒绝（fail-closed）门={exc.result.gate}：{exc.result.reason}", fg=typer.colors.RED)
        raise typer.Exit(2)

    pv, ms, sc = res.product_view, res.market_state, res.scope
    _TIER_ZH = {"full": "✓ 完整", "limited": "◐ 有限", "deferred": "○ 暂缓"}
    tier = sc.support_tier.value
    typer.secho(f"\n{fund_code} @ {as_of}", bold=True)
    typer.echo(f"  分类     : {_TIER_ZH.get(tier, tier)} [{_enum(sc.category)}] {sc.reason}")
    if not sc.in_scope:
        # Doc §4 lst:scope / §3.5-2/3: non-FULL types get a DEGRADED output — classified,
        # disclosed facts + risk drivers + disclaimer, no direction / trend / strong judgment.
        typer.echo(f"  分析口径 : 视角={'、'.join(sc.caliber.applicable_lenses) or '—'} · 风险来源={'、'.join(sc.caliber.risk_drivers)}")
        typer.secho(f"  {sc.caliber.disclaimer}", fg=typer.colors.YELLOW)
        return
    typer.echo(f"  产品视图 : 方向={_enum(pv.direction)} 趋势={pv.trend_strength:.3f} 波动={_enum(pv.volatility)} 可信度={pv.confidence:.3f}")
    typer.echo(f"  市场状态 : {_enum(ms.direction)} / {_enum(ms.volatility)} / {_enum(ms.stress)} / {_enum(ms.event)}")
    exps = res.exposure or []
    if exps:
        typer.echo("  暴露     :")
        for e in exps[:8]:
            typer.echo(f"    - {_enum(e.dimension)}:{e.label}  {e.low:.0%}–{e.high:.0%} (mid {e.point_estimate:.0%}) 龄{e.disclosure_age_days}d 置信{e.confidence:.2f}")
    ec = res.explanation
    if ec is not None:
        typer.echo("  解释卡   :")
        for lbl, val in (("为什么偏乐观", ec.why_optimistic), ("为什么仍需谨慎", ec.why_cautious),
                         ("我们不知道什么", ec.unknowns), ("数据有多新", ec.data_freshness),
                         ("什么会改变判断", ec.what_would_change)):
            if val:
                v = "；".join(val) if isinstance(val, list) else val
                typer.echo(f"    · {lbl}：{v}")
    if raw:
        _echo_json({"scope": _enum(sc.category), "product_view": pv.model_dump() if hasattr(pv, "model_dump") else str(pv)})


@app.command()
def advice(
    fund_code: str = typer.Argument(...),
    as_of: str = typer.Option(..., "--as-of"),
    portfolio: Path = typer.Option(..., "--portfolio", help="用户组合 JSON 文件（仅建议模式）。"),
    position: float = typer.Option(0.0, "--position", help="当前该基金仓位 0..1。"),
) -> None:
    """建议模式：读取用户组合，run_advice（穿透 + 风险预算 + 行动建议）。"""
    from qmt.compliance import ComplianceViolation
    from qmt.data.ingest import ingest_fund
    from qmt.pipeline.run import run_advice
    from qmt.types import FeeSchedule, SuitabilityProfile, UserPortfolio

    d = _parse_date(as_of)
    data = json.loads(Path(portfolio).read_text(encoding="utf-8"))
    up = UserPortfolio.model_validate(data["portfolio"]) if "portfolio" in data else UserPortfolio.model_validate(data)
    suit = SuitabilityProfile.model_validate(data.get("suitability", {"risk_tolerance": "C3", "investment_horizon_days": 365}))
    fees = FeeSchedule.model_validate(data["fees"]) if "fees" in data else FeeSchedule(fund_code=fund_code, purchase_fee_rate=0.0015, redemption_tiers=[], sales_service_fee_rate=0.0)
    store = _store()
    ingest_fund(fund_code, d, store)
    try:
        res = run_advice(fund_code, d, up, suit, fees, position=position, store=store)
    except (ComplianceViolation, ValueError) as exc:
        typer.secho(f"✗ 已拒绝：{exc}", fg=typer.colors.RED)
        raise typer.Exit(2)
    typer.secho(f"\n建议动作：{_enum(res.action.action)}", bold=True, fg=typer.colors.CYAN)
    typer.echo(f"  gated_by: {list(res.action.gated_by)}")
    if res.lookthrough is not None:
        ic = getattr(res.lookthrough, "industry_conc", {}) or {}
        top = sorted(ic.items(), key=lambda kv: -kv[1])[:5]
        typer.echo(f"  穿透行业集中: {', '.join(f'{k} {v:.0%}' for k, v in top) or '—'}")
    if res.risk_budget is not None:
        typer.echo(f"  风险预算: usage={res.risk_budget.budget_usage:.2f} over_budget={res.risk_budget.over_budget}")


@app.command()
def ingest(
    fund_code: str = typer.Argument(...),
    as_of: str = typer.Option(..., "--as-of"),
) -> None:
    """把某基金在 as_of 可见的数据预热进持久化 PIT 存储。"""
    from qmt.data.ingest import ingest_fund

    rep = ingest_fund(fund_code, _parse_date(as_of), _store())
    typer.echo(f"入库完成：NAV={rep.nav_rows} 持仓={rep.holdings_reports} 行业图={rep.industry_map_size} 在universe={rep.in_universe} gaps={rep.gaps or '无'}")


@app.command()
def backtest(
    fund_code: str = typer.Argument(...),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    strategy: str = typer.Option("baseline", "--strategy"),
) -> None:
    """按文档 10 条回测协议跑一段回测，对比五个朴素基线（净费后）。"""
    from qmt.backtest import run_backtest
    from qmt.fees import default_fee_schedule

    # Pass a realistic, non-zero fee schedule (§7 rules 6/7: 不能假设零成本).
    res = run_backtest(
        fund_code,
        _parse_date(start),
        _parse_date(end),
        strategy=strategy,
        fees=default_fee_schedule(fund_code),
        store=_store(),
    )
    typer.secho(f"\n回测 {fund_code} {start}→{end} [{res.strategy_name}]", bold=True)
    typer.echo(f"  净收益={res.net_return:.2%} 最大回撤={res.max_drawdown:.2%} 总成本={res.total_cost:.2%} 决策={res.n_decisions}")
    for name, delta in (res.vs_baseline or {}).items():
        if name == res.self_baseline:  # this "baseline" IS the strategy — a self-comparison
            typer.secho(f"  vs {name}: 策略自身（非独立基线，差值恒为 0）", fg=typer.colors.YELLOW)
            continue
        typer.echo(f"  vs {name}: {'+' if delta >= 0 else ''}{delta:.2%}")


@app.command()
def calibrate(
    fund_code: list[str] = typer.Argument(..., help="一个或多个 6 位基金代码（观测合并入同一张表）。"),
    start: str = typer.Option(..., "--start", help="历史回放起始 YYYY-MM-DD。"),
    end: str = typer.Option(..., "--end", help="历史回放结束 YYYY-MM-DD。"),
) -> None:
    """构建历史可信度校准表并持久化到 data_cache/calibration.json（run_analysis 默认加载并应用）。

    回放分析各阶段（暴露→市场→预测→不确定性→产品方向），把每个原始 confidence 与
    「方向判断后来是否正确」配对，按历史命中率分桶（§4 uncertainty / §5）。信号读取严格
    点位（14:30），仅结果用后续净值衡量（历史结果本就已知）。需先 ingest 好基准指数与净值。
    """
    from qmt.calibration import build_calibration_from_history, calibration_cache_path

    s, e = _parse_date(start), _parse_date(end)
    table = build_calibration_from_history(fund_code, s, e, _store())
    total = sum(b.sample_count for b in table.buckets)
    typer.secho(
        f"\n校准表已构建：{len(table.buckets)} 桶 · {total} 条观测 → {calibration_cache_path()}",
        bold=True,
    )
    for b in table.buckets:
        if b.sample_count:
            typer.echo(f"  [{b.lower:.1f},{b.upper:.1f}] 命中率 {b.hit_rate:.2f} (n={b.sample_count})")
    if total == 0:
        typer.secho(
            "  提示：0 条观测——请确认已 ingest 基准指数/净值且窗口内有足够前向数据。",
            fg=typer.colors.YELLOW,
        )


@app.command()
def tune(
    model_id: str = typer.Argument(..., help="可调模型 id：timing_ma / vol_control / direction。"),
    fund_code: str = typer.Argument(..., help="6 位基金代码。"),
    train_start: str = typer.Option(..., "--train-start", help="训练窗口起始 YYYY-MM-DD。"),
    train_end: str = typer.Option(..., "--train-end", help="训练窗口结束 YYYY-MM-DD。"),
    valid_start: str = typer.Option(..., "--valid-start", help="验证窗口起始（须晚于 train-end）。"),
    valid_end: str = typer.Option(..., "--valid-end", help="验证窗口结束 YYYY-MM-DD。"),
    metric: Optional[str] = typer.Option(None, "--metric", help="目标指标（默认取该模型首选）。"),
    apply: bool = typer.Option(False, "--apply", help="把最优参数写入本进程 CONFIG。"),
) -> None:
    """【诊断用】单切分参数搜索：在训练窗口择优、验证窗口确认（§5/§7）。

    ⚠ 不要用它来真正选参数：实测在沪深300 上，按『训练窗口目标』排名与样本外表现**负相关**
    （Spearman ≈ -0.36），会系统性选到泛化最差的候选。产品训练路径已改为走步式交叉验证 +
    留出确认 + 『稳定优于基线』验收门（见网页『训练』页 / qmt.tuning.stream_walk_forward_search）。
    此命令保留为单切分诊断工具。
    """
    from qmt.tuning import run_parameter_search, tuning_spec

    typer.secho(
        "⚠ 单切分择优仅供诊断：按训练窗口排名与样本外表现负相关，勿据此改线上参数；"
        "请用网页『训练』页的走步式验证 + 验收门。",
        fg=typer.colors.YELLOW,
    )

    try:
        spec = tuning_spec(model_id)
    except KeyError as exc:
        typer.secho(f"✗ {exc}", fg=typer.colors.RED)
        raise typer.Exit(2)
    res = run_parameter_search(
        spec, fund_code,
        _parse_date(train_start), _parse_date(train_end),
        _parse_date(valid_start), _parse_date(valid_end),
        _store(), metric=metric,
    )
    typer.secho(f"\n训练 {model_id} · {fund_code} · 目标={res.metric}（{res.objective}）", bold=True)
    base = res.baseline
    if base is not None:
        typer.echo(f"  基线（当前配置）: 训练 {base.objective_value:.4f}"
                   + (f" · 验证 {base.valid.objective(res.metric):.4f}" if base.valid else ""))
    if res.best is None:
        typer.secho("  无可测候选（窗口内数据不足）。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    b = res.best
    typer.secho(f"  最优参数: {b.params}", fg=typer.colors.CYAN)
    typer.echo(f"    训练 {res.metric}={b.objective_value:.4f}"
               + (f" · 验证（样本外）{b.valid.objective(res.metric):.4f}" if b.valid else " · 验证：未确认"))
    if res.objective == "backtest" and isinstance(b.train, object) and hasattr(b.train, "sharpe"):
        m = b.train
        typer.echo(f"    净收益={m.net_return:.2%} 夏普={m.sharpe:.2f} 换手={m.turnover:.2f} "
                   f"最大回撤={m.max_drawdown:.2%} 成本={m.total_cost:.2%}")
    if res.capped:
        typer.secho(f"  注：网格 {res.total_candidates} 个候选（原始 {spec.grid_size()}，已等距抽样封顶）。",
                    fg=typer.colors.YELLOW)
    if apply and b.params:
        from qmt.controlplane.api import apply_config_override

        for path, val in b.params.items():
            apply_config_override(path, val)
        typer.secho(f"  已应用到 CONFIG：{b.params}", fg=typer.colors.GREEN)


@config_app.command("show")
def config_show() -> None:
    """打印有效配置（所有可调阈值）。"""
    from qmt.config import CONFIG

    dump = CONFIG.model_dump() if hasattr(CONFIG, "model_dump") else vars(CONFIG)
    _echo_json(dump)


@app.command("layer-curves")
def layer_curves(
    out_dir: str = typer.Option("artifacts", "--out-dir", help="PNG 输出目录。"),
    horizon: int = typer.Option(20, "--industry-horizon", help="行业层曲线的周期（交易日）。"),
    extremes: int = typer.Option(2, "--extremes", help="行业层自动取相关性最高/最低各 N 个。"),
) -> None:
    """渲染大盘层与行业层的『预测 vs 真实』曲线（点位口径），观察训练后的差异。"""
    import matplotlib

    matplotlib.use("Agg")
    from qmt.pipeline.layer_diagnostics import build_grid, load_industry_tapes, load_market_tape
    from qmt.viz import plot_industry_curves, plot_market_curves

    store = _store()
    grid = build_grid(load_market_tape(store), load_industry_tapes(store))
    if grid.n == 0:
        typer.echo("无可用指数数据（先 ingest 大盘/行业指数）。")
        raise typer.Exit(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = out / "market_curves.png"
    q = out / "industry_curves.png"
    plot_market_curves(grid, out_path=str(m))
    plot_industry_curves(grid, horizon=horizon, auto_extremes=extremes, out_path=str(q))
    typer.echo(f"已渲染：{m}\n已渲染：{q}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8770, "--port"),
) -> None:
    """启动控制台：FastAPI + 网页仪表盘（qmt.controlplane.api:app）。"""
    import uvicorn

    typer.echo(f"控制台: http://{host}:{port}/   (API 文档 /docs)")
    uvicorn.run("qmt.controlplane.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
