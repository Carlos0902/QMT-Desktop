"""CLI for the symbolic-learning factor platform.

    python -m qmt.factor demo            # industry cross-section (always available)
    python -m qmt.factor funds           # real off-exchange fund cross-section (needs ingested NAV)
    qmt factor funds --max-funds 200     # (once registered on the main `qmt` app)

Each command runs the full ten-layer pipeline and prints an auditable JSON summary
plus per-factor diagnostics (full-history rank-IC, overlap-corrected t-stat,
sub-period stability, and the first gate each top factor fails) and an honest verdict.
"""

from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="符号学习因子生产平台 — 端到端流水线 CLI。")


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _report(res, top: int) -> None:
    s = res.summary()
    diag = s.pop("diagnostics", [])
    _echo(s)
    if diag:
        typer.secho("\n诊断（为什么通过/被否）：", bold=True)
        typer.echo("  factor                          origin    validIC  fullIC   t     sig    gate_fail")
        for d in diag:
            typer.echo(
                f"  {d['factor'][:30]:30s} {d['origin']:8s} "
                f"{d['valid_rank_ic']:+.3f}  {d['full_rank_ic']:+.3f}  {d['full_tstat']:+5.2f} "
                f"{str(d['significant']):5s}  {d['first_gate_fail'] or '-'}"
            )
    typer.secho("\nTop 人工基线因子（OOS rank-IC）：", bold=True)
    for sf in sorted(res.manual, key=lambda x: -x.score.rank_ic)[:top]:
        gate = "PASS" if sf.gate_passed else "fail"
        typer.echo(f"  {sf.name:14s} rankIC={sf.score.rank_ic:+.4f} gate={gate} turnover={sf.score.turnover:.2f}")
    typer.secho("Top 搜索因子（公式 + OOS rank-IC）：", bold=True)
    for sf in sorted(res.searched, key=lambda x: -x.score.rank_ic)[:top]:
        typer.echo(f"  rankIC={sf.score.rank_ic:+.4f} cx={sf.expr.complexity:2d}  {sf.expr.text[:60]}")
    typer.secho(f"\n结论：{res.verdict}", fg=typer.colors.YELLOW)


@app.command()
def demo(
    horizon: int = typer.Option(40, "--horizon", help="预测窗口（交易日），默认 40（20–60 区间）。"),
    stride: int = typer.Option(20, "--stride", help="调仓步长（交易日）。"),
    max_dates: int = typer.Option(100, "--max-dates", help="参与评估的决策日上限（控制运行时长）。"),
    top: int = typer.Option(5, "--top", help="打印前 N 个因子。"),
) -> None:
    """在真实 31 行业横截面上跑通十层流水线（始终可用）。"""
    from qmt.factor.pipeline import run_industry_demo
    typer.secho("运行因子流水线（行业横截面：safe_data → … → execution_sim）…", fg=typer.colors.CYAN)
    _report(run_industry_demo(horizon_days=horizon, stride_days=stride, max_decision_dates=max_dates), top)


@app.command()
def funds(
    horizon: int = typer.Option(40, "--horizon"),
    stride: int = typer.Option(20, "--stride"),
    max_dates: int = typer.Option(80, "--max-dates"),
    max_funds: int = typer.Option(300, "--max-funds", help="参与横截面的基金数上限。"),
    top: int = typer.Option(5, "--top"),
) -> None:
    """在真实场外基金横截面上跑通十层流水线（需先入库 fund_nav）。"""
    from qmt.factor.pipeline import run_fund_demo
    typer.secho("运行因子流水线（基金横截面）…", fg=typer.colors.CYAN)
    try:
        res = run_fund_demo(horizon_days=horizon, stride_days=stride,
                            max_decision_dates=max_dates, max_funds=max_funds)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    _report(res, top)


@app.command("cost-benchmark")
def cost_benchmark(
    max_funds: int = typer.Option(80, "--max-funds", min=5),
    budget: int = typer.Option(120, "--budget", min=1),
    seeds: str = typer.Option("7,19,43", "--seeds"),
    output: str = typer.Option("artifacts/cost_benchmark.json", "--output"),
    fees_file: str = typer.Option(None, "--fees-file", help="带公告可知时间的基金/渠道费表 JSON"),
) -> None:
    """现金与份额回测：训练内组合增量筛选，费后比较简单基线（封存区前）。"""
    from pathlib import Path
    from qmt.config import get_settings
    from qmt.factor.cost_benchmark import run_cost_benchmark, json_ready
    path = Path(output)
    try:
        result = run_cost_benchmark(get_settings().data_cache_dir, max_funds=max_funds,
                                    budget=budget, seeds=tuple(int(s) for s in seeds.split(",")),
                                    fee_history=fees_file,
                                    output_dir=path.parent / "cost_paths")
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = json_ready(result)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")
    _echo({"output": str(path.resolve()), "results": result["results"],
           "paired_vs_frozen_manual": result["paired_vs_frozen_manual"],
           "verdict": result["verdict"]})


@app.command("walk-forward")
def walk_forward(
    history: str = typer.Option("data_cache/execution_history/paper_research", "--history"),
    budget: int = typer.Option(120, "--budget", min=1),
    seeds: str = typer.Option("7,19,43", "--seeds"),
    output_dir: str = typer.Option("artifacts/walk_forward", "--output-dir"),
    search_method: str = typer.Option("gp", "--search-method", help="gp / uniform / bandit / q"),
) -> None:
    """严格历史数据滚动验证：缺少实际费用/状态/分红覆盖时不计算收益。"""
    from pathlib import Path
    from qmt.config import get_settings
    from qmt.factor.rolling_cli import run_reviewed_walk_forward
    try:
        report = run_reviewed_walk_forward(Path(history), get_settings().data_cache_dir,
                                           budget=budget, seeds=tuple(int(s) for s in seeds.split(",")),
                                           output_dir=Path(output_dir), search_method=search_method)
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    _echo({"status": report["status"], "alpha_approved": report["alpha_approved"],
           "segments": [{"fold_id": f["fold_id"], "status": f["status"],
                          "test_results": f["test_results"]} for f in report["segments"]],
           "output": str((Path(output_dir) / "walk_forward_report.json").resolve())})


@app.command("shadow-step")
def shadow_step(
    as_of: str = typer.Option(None, "--as-of", help="快照日 YYYY-MM-DD;缺省=数据里最新可用日。"),
    mode: str = typer.Option("live", "--mode", help="live（锁定区之后）| rehearsal（锁定区之前的排练）。"),
    raw: bool = typer.Option(False, "--raw", help="额外打印完整 JSON。"),
) -> None:
    """S5 影子运行：取一个快照（观察名单打分 → 观察组合 → 三档成本 → 记账 → 中文解读）。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner, explain_snapshot

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        if as_of is None:
            nav = store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
            d = pd.to_datetime(nav["nav_date"]).max().date()
        else:
            d = pd.Timestamp(as_of).date()
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        snap = ShadowRunner(store, cfg).step(d, mode=mode)
        typer.echo(explain_snapshot(snap))
        # 强制调仓提示(用户要求):每期必给结论,含「不建议调仓」的情况
        from qmt.factor.advice import rebalance_note
        typer.echo("")
        typer.echo(rebalance_note(snap, store))
        if raw:
            _echo({k: v for k, v in snap.items() if k != "target_weights"})
        typer.secho(f"\n快照已入账（{len(snap['target_weights'])} 只观察持仓）。", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("shadow-explain")
def shadow_explain() -> None:
    """用普通话重看最近一次影子快照的解读（不取新快照）。"""
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        typer.echo(ShadowRunner(store, cfg).explain_last())
    finally:
        store.close()


@app.command("shadow-monitor")
def shadow_monitor(
    now: str = typer.Option(None, "--now", help="当前日 YYYY-MM-DD;缺省=数据里最新可用日。"),
) -> None:
    """路 2 一键闭环:补拍快照(若数据推进)→ 结算到期快照 → 输出实盘证据自裁决。

    每周跑一次即可;系统自动积累 low_vol 等观察因子的实盘 rank-IC,
    并在其穿越通胀显著性门槛时明确告诉你。封存区不碰。
    """
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        if now is None:
            nav = store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
            d = pd.to_datetime(nav["nav_date"]).max().date()
        else:
            d = pd.Timestamp(now).date()
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        runner = ShadowRunner(store, cfg)
        res = runner.monitor(d)
        typer.secho(f"补拍快照:{'是' if res['stepped'] else '否(数据未推进或在封存区内)'}"
                    f" · 本次新结算:{res['n_matured_now']} 个", fg=typer.colors.GREEN)
        typer.echo("")
        typer.echo(runner.explain_verdict())
    finally:
        store.close()


@app.command("shadow-verdict")
def shadow_verdict() -> None:
    """只看实盘证据自裁决(不补拍、不结算)。"""
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        typer.echo(ShadowRunner(store, cfg).explain_verdict())
    finally:
        store.close()


@app.command("shadow-mature")
def shadow_mature(
    now: str = typer.Option(None, "--now", help="结算日 YYYY-MM-DD;缺省=数据里最新可用日。"),
) -> None:
    """S5 影子运行：结算已到期的快照（真实前向相对收益 + 观察名单的实盘 rank-IC）。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        if now is None:
            nav = store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
            d = pd.to_datetime(nav["nav_date"]).max().date()
        else:
            d = pd.Timestamp(now).date()
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        matured = ShadowRunner(store, cfg).mature(d)
        _echo(matured)
        typer.secho(f"{len(matured)} 个快照完成结算。", fg=typer.colors.GREEN)
    finally:
        store.close()


@app.command("shadow-report")
def shadow_report() -> None:
    """S5 影子运行报告：数据时效 / 信号稳定性 / 成本符合度 / go-no-go。"""
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        _echo(ShadowRunner(store, cfg).report())
    finally:
        store.close()


@app.command("portfolio-init")
def portfolio_init(
    holding: list[str] = typer.Option(..., "--holding", "-H",
                                      help="一笔持仓,格式 基金代码=金额(元),可多次:-H 011133=10000 -H 161725=5000。"),
    cash: float = typer.Option(0.0, "--cash", help="当前可用流动资金(元),可为 0。"),
    as_of: str = typer.Option(None, "--as-of", help="初始化日 YYYY-MM-DD,缺省=今天;按该日可见净值换算份额。"),
    purchase: list[str] = typer.Option(None, "--purchase",
                                       help="可选:某笔持仓的买入日,格式 代码=YYYY-MM-DD(影响赎回费档位)。"),
) -> None:
    """初始化真实持仓与流动资金;此后每次影子快照以此为决策基准(重复执行即覆盖)。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.portfolio import init_book, mark_to_market

    def _parse_pairs(items, val_type):
        out = {}
        for it in items or []:
            if "=" not in it:
                raise typer.BadParameter(f"格式应为 代码=值: {it!r}")
            k, v = it.split("=", 1)
            out[k.strip()] = val_type(v.strip())
        return out

    holdings = _parse_pairs(holding, float)
    purchase_dates = _parse_pairs(purchase, str)
    d = pd.Timestamp(as_of).date() if as_of else None

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        ledger_dir = get_settings().data_cache_dir / "shadow"
        book = init_book(store, holdings, cash, as_of=d, ledger_dir=ledger_dir,
                         purchase_dates=purchase_dates)
        mtm = mark_to_market(book, store, pd.Timestamp(book.initialized_at).date())
        typer.secho(f"持仓账本已初始化 @ {book.initialized_at}(影子模拟基准,非投资建议):",
                    fg=typer.colors.GREEN)
        _echo(mtm)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("policy-search")
def policy_search(
    generations: int = typer.Option(8, "--generations", help="进化代数。"),
    population: int = typer.Option(24, "--population", help="种群规模。"),
    seed: int = typer.Option(7, "--seed", help="随机种子(同种子结果可复现)。"),
    top_k: int = typer.Option(20, "--top-k", help="策略组合的最大持仓数。"),
    buy_fee: float = typer.Option(0.0015, "--buy-fee", help="买入腿费率(申购)。"),
    sell_fee: float = typer.Option(0.005, "--sell-fee", help="卖出腿费率(短持赎回;期末清算同)。"),
    stride: int = typer.Option(10, "--stride", help="决策步长(交易日):10双周/20月频/40季频。"),
    hysteresis: float = typer.Option(0.75, "--hysteresis", help="滞回带宽(粘性持仓):0=每期重排,越大越粘。"),
) -> None:
    """遗传进化决策树策略(强化学习式):在扩展宇宙上用累计净奖励进化 if-then 交易规则。

    纪律:训练/验证分区、封存区(2025-05-08+)不碰、同预算随机对照臂(§7.4)、
    试验全记账、显著性按试验数通胀。输出为研究证据,不改变四道门的 NO-GO。
    """
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.policy_search import PolicySearchConfig, run_policy_search

    cfg = PolicySearchConfig(population=population, generations=generations, seed=seed,
                             top_k=top_k, buy_fee=buy_fee, sell_fee=sell_fee,
                             stride_days=stride, hysteresis=hysteresis)
    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        report = run_policy_search(store, cfg, progress=lambda s: typer.secho(s, fg=typer.colors.CYAN))
    finally:
        store.close()

    r = report["results"]
    typer.secho("\n━━━ 进化最优策略(可审计 if-then 规则)━━━", bold=True)
    typer.echo(r["ga_best"]["tree"])
    typer.secho("\n━━━ 验证区对照(累计净相对奖励,越高越好)━━━", bold=True)
    typer.echo(f"{'策略':24s} {'累计log奖励':>10s} {'净相对收益':>9s} {'均换手':>7s} {'IC':>7s} {'t':>6s} {'显著':>4s}")
    for name in ["ga_best", "random_arm_best"] + [k for k in r if k.startswith("seed:")]:
        v = r[name]
        ic = f"{v['mean_rank_ic']:+.3f}" if v.get("mean_rank_ic") is not None else "—"
        t = f"{v['ic_tstat_nw']:+.2f}" if v.get("ic_tstat_nw") is not None else "—"
        typer.echo(f"{name:24s} {v['cum_log_reward']:>+10.4f} {v['total_net_rel_return']:>+9.2%} "
                   f"{v['avg_turnover']:>7.2f} {ic:>7s} {t:>6s} {str(v['ic_significant']):>4s}")
    typer.echo(f"\n试验总数 {report['n_trials_total']}(全部入账)· 显著性门槛 t>{report['deflated_min_abs_t']}"
               f" · 决策日 {report['meta']['n_dates']} 个 · 封存区 {report['meta']['locked_start']} 之后未触碰")
    typer.secho(f"\n结论:{report['verdict']}", fg=typer.colors.YELLOW)
    typer.echo(f"完整报告:artifacts/policy_search.json")


@app.command("policy-sweep")
def policy_sweep(
    strides: str = typer.Option("10,20,40", "--strides", help="逗号分隔的调仓步长(交易日)。"),
    generations: int = typer.Option(8, "--generations"),
    population: int = typer.Option(24, "--population"),
    seed: int = typer.Option(7, "--seed"),
    hysteresis: float = typer.Option(0.75, "--hysteresis", help="滞回带宽(粘性持仓)。"),
) -> None:
    """跨调仓频率扫描(双周/月频/季频):粘性持仓 + 降频能否让信号在费后存活。"""
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.policy_search import PolicySearchConfig, run_stride_sweep

    slist = [int(x) for x in strides.split(",") if x.strip()]
    base = PolicySearchConfig(population=population, generations=generations,
                              seed=seed, hysteresis=hysteresis)
    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        rep = run_stride_sweep(store, slist, base, progress=lambda s: typer.secho(s, fg=typer.colors.CYAN))
    finally:
        store.close()

    typer.secho("\n━━━ 跨调仓频率对照(滞回持仓,费后净相对收益)━━━", bold=True)
    typer.echo(f"{'策略':18s} {'频率':>5s} {'净收益':>9s} {'均换手':>7s} {'IC':>7s} {'t':>6s} {'显著':>4s}")
    for x in rep["rows"]:
        if x["strategy"] == "_meta":
            continue
        ic = f"{x['mean_rank_ic']:+.3f}" if x.get("mean_rank_ic") is not None else "—"
        t = f"{x['ic_tstat']:+.2f}" if x.get("ic_tstat") is not None else "—"
        nr = f"{x['net_rel_return']:+.2%}" if x.get("net_rel_return") is not None else "—"
        tu = f"{x['avg_turnover']:.3f}" if x.get("avg_turnover") is not None else "—"
        typer.echo(f"{x['strategy']:18s} {x['freq']:>5s} {nr:>9s} {tu:>7s} {ic:>7s} {t:>6s} {str(x['ic_significant']):>4s}")
    typer.secho(f"\n结论:{rep['verdict']}", fg=typer.colors.YELLOW)
    typer.echo("完整报告:artifacts/policy_sweep.json")


@app.command("portfolio-set-purchase")
def portfolio_set_purchase(
    purchase_date: str = typer.Option(..., "--date", help="新买入日 YYYY-MM-DD(影响赎回费档位)。"),
    fund: list[str] = typer.Option(None, "--fund", help="只改这些基金(可多次);缺省=全部持仓。"),
) -> None:
    """把持仓的买入日改为指定日期(只改日期,份额/金额/现金不动)。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.factor.portfolio import set_purchase_dates

    try:
        d = pd.Timestamp(purchase_date).date()
        book = set_purchase_dates(get_settings().data_cache_dir / "shadow", d, list(fund) if fund else None)
        n = len(fund) if fund else len(book.positions)
        typer.secho(f"已把 {n} 只持仓的买入日改为 {d}。", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("portfolio-rebalance")
def portfolio_rebalance() -> None:
    """基于你采用的因子(low_vol_60)+ 最新快照,生成一次**具体到元**的调仓方案。

    先取一个新鲜快照(基于你真实持仓),再按季频粘性纪律输出卖出/买入/保持订单。
    规则化输出,非持牌投顾建议。
    """
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.shadow import ShadowConfig, ShadowRunner
    from qmt.factor.advice import rebalance_plan, explain_plan

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        nav = store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
        d = pd.to_datetime(nav["nav_date"]).max().date()
        cfg = ShadowConfig.load(get_settings().data_cache_dir / "shadow")
        if not cfg.primary:
            typer.secho("未设采用因子(primary)—— 先用 shadow 采用一个因子再生成方案。",
                        fg=typer.colors.YELLOW)
            raise typer.Exit(1)
        snap = ShadowRunner(store, cfg).step(d, mode="live")
        plan = rebalance_plan(snap, store)
        typer.echo(explain_plan(plan, store))
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("portfolio-buy")
def portfolio_buy(
    fund: str = typer.Argument(..., help="6 位基金代码。"),
    amount: float = typer.Argument(..., help="买入金额(元)。"),
    from_cash: bool = typer.Option(True, "--from-cash/--no-from-cash", help="从现金扣(默认)还是外部资金。"),
    as_of: str = typer.Option(None, "--as-of", help="成交日 YYYY-MM-DD,缺省=今天。"),
) -> None:
    """买入:把某基金加进持仓(默认从可用现金扣款)。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.portfolio import buy, mark_to_market
    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        d = pd.Timestamp(as_of).date() if as_of else None
        book = buy(store, get_settings().data_cache_dir / "shadow", fund, amount,
                   as_of=d, from_cash=from_cash)
        typer.secho(f"已买入 {fund} ¥{amount:,.0f}"
                    f"{'(从现金)' if from_cash else '(外部资金)'};现金余 ¥{book.cash:,.0f}。",
                    fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED); raise typer.Exit(1)
    finally:
        store.close()


@app.command("portfolio-sell")
def portfolio_sell(
    fund: str = typer.Argument(..., help="6 位基金代码。"),
    amount: float = typer.Argument(None, help="卖出金额(元);省略或加 --all 则全部赎回。"),
    all_: bool = typer.Option(False, "--all", help="全部赎回该基金。"),
    to_cash: bool = typer.Option(True, "--to-cash/--no-to-cash", help="卖出所得回到现金(默认)。"),
    as_of: str = typer.Option(None, "--as-of", help="成交日 YYYY-MM-DD,缺省=今天。"),
) -> None:
    """卖出:全部赎回(--all)或按金额减仓;所得默认回到可用现金。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.portfolio import sell
    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        d = pd.Timestamp(as_of).date() if as_of else None
        book = sell(store, get_settings().data_cache_dir / "shadow", fund,
                    amount=amount, all=all_, as_of=d, to_cash=to_cash)
        held = "已清仓" if fund not in book.positions else "已减仓"
        typer.secho(f"{held} {fund};现金余 ¥{book.cash:,.0f}。", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED); raise typer.Exit(1)
    finally:
        store.close()


@app.command("portfolio-cash")
def portfolio_cash(
    amount: float = typer.Argument(..., help="把可用流动资金设为该金额(元,>=0)。"),
) -> None:
    """直接设定可用流动资金(例如你外部转入/转出后同步一下)。"""
    from qmt.config import get_settings
    from qmt.factor.portfolio import set_cash
    try:
        book = set_cash(get_settings().data_cache_dir / "shadow", amount)
        typer.secho(f"可用现金已设为 ¥{book.cash:,.0f}。", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED); raise typer.Exit(1)


@app.command("portfolio-show")
def portfolio_show(
    as_of: str = typer.Option(None, "--as-of", help="估值日 YYYY-MM-DD,缺省=数据里最新净值日。"),
) -> None:
    """查看当前持仓账本(按最新可见净值 mark-to-market)。"""
    import pandas as pd
    from qmt.config import get_settings
    from qmt.pit import PointInTimeStore
    from qmt.factor.portfolio import load_book, mark_to_market

    store = PointInTimeStore(base_dir=get_settings().data_cache_dir)
    try:
        book = load_book(get_settings().data_cache_dir / "shadow")
        if book is None:
            typer.secho("尚未初始化持仓 — 先运行 qmt factor portfolio-init。", fg=typer.colors.YELLOW)
            raise typer.Exit(1)
        if as_of:
            d = pd.Timestamp(as_of).date()
        else:
            nav = store.get_asof("fund_nav", pd.Timestamp("2100-01-01"))
            d = pd.to_datetime(nav["nav_date"]).max().date()
        _echo(mark_to_market(book, store, d))
    finally:
        store.close()


if __name__ == "__main__":
    app()
