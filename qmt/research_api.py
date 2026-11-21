"""Read-only local view of acquisition progress and frozen evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _entries(root, kind):
    directory = root / kind
    entries = []
    if not directory.exists():
        return entries
    for path in sorted(directory.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (kind == "snapshots" and path.name != "manifest.json" or
                kind == "runs" and path.name != "report.json" or
                kind == "protocols" and path.parent != directory):
            continue
        payload = _read(path)
        if not isinstance(payload, dict):
            continue
        annotation = _read(path.parent / "diagnostic_annotation.json") if kind == "runs" else None
        entries.append({"id": path.stem if kind == "protocols" else path.parent.name,
                        "file": str(path.relative_to(root)),
                        "status": (annotation or {}).get("status") or payload.get("status") or payload.get("verdict"),
                        "snapshot_id": payload.get("snapshot_id"),
                        "protocol_id": payload.get("protocol_id"),
                        "hash": payload.get("snapshot_hash") or payload.get("data_hash") or payload.get("protocol_hash"),
                        "details": payload})
    return entries


def create_app(lake_dir: str | Path, artifacts_dir: str | Path) -> FastAPI:
    lake = Path(lake_dir).resolve()
    artifacts = Path(artifacts_dir).resolve()
    app = FastAPI(title="QMT 数据与评测平台", version="0.1")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/api/health")
    def health():
        return {"status": "ok", "read_only": True, "lake": str(lake), "artifacts": str(artifacts)}

    @app.get("/api/cache/status")
    def status():
        # The collector publishes an atomic counts-only snapshot. No long-lived
        # DuckDB reader contends with a collecting process or blocks an HTTP call.
        cached = _read(lake / "inventory.json")
        if cached is None:
            return {"status": "NO_STATUS_SNAPSHOT", "datasets": {}, "tasks": {},
                    "message": "请运行 qmt cache status 或采集任务生成状态快照。"}
        collector = _read(lake / "collector.json")
        pointer = _read(lake / "quality" / "latest.json")
        curation = _read(Path(pointer["quality_path"]) / "manifest.json") if pointer else None
        return {**cached, "collector": collector or {}, "curation": curation}

    @app.get("/api/evaluation/{kind}")
    def listing(kind: str):
        if kind not in {"snapshots", "protocols", "runs"}:
            raise HTTPException(404, "unknown artifact kind")
        return {"items": _entries(artifacts, kind)}

    @app.get("/api/evaluation/{kind}/{artifact_id}")
    def detail(kind: str, artifact_id: str):
        if kind not in {"snapshots", "protocols", "runs"} or not re.fullmatch(r"[A-Za-z0-9_.-]+", artifact_id):
            raise HTTPException(404, "unknown artifact")
        for entry in _entries(artifacts, kind):
            if entry["id"] == artifact_id:
                return entry["details"]
        raise HTTPException(404, "artifact not found")

    return app


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QMT 数据与评测平台</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#e7edf7;background:#0d1522;line-height:1.6}
body{max-width:1240px;margin:32px auto;padding:0 24px}h1{font-size:28px;margin:0}h2{font-size:19px}
.muted{color:#9baabe}.tag{display:inline-block;padding:3px 10px;border-radius:20px;background:#20384b;color:#a4dddb}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:20px 0}
.card,section{background:#152132;border:1px solid #29394e;border-radius:12px;padding:20px}.number{font-size:29px;font-weight:650}
section{margin:18px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid #29394e}
button{background:#25465c;color:white;border:0;border-radius:7px;padding:7px 14px;cursor:pointer}pre{overflow:auto;color:#b7d4e9;white-space:pre-wrap;font-size:13px}
summary{cursor:pointer;color:#a4dddb}.warn{border-left:4px solid #d9ac63;padding-left:14px}code{background:#203047;padding:2px 5px;border-radius:4px}
</style></head><body>
<span class="tag">本地研究平台 · v0.1</span><h1>基金数据与固定评测</h1>
<p class="muted">持续采集的数据仓库，与冻结版本的评测结果分别记录。</p>
<div class="grid" id="cards"></div>
<section><h2>数据覆盖</h2><p class="muted" id="stamp"></p><div style="overflow:auto"><table><thead><tr><th>数据集</th><th>归档记录量</th><th>代码数下界</th><th>说明</th></tr></thead><tbody id="datasets"></tbody></table></div>
<details><summary>采集任务与接口状态</summary><pre id="tasks"></pre></details></section>
<section><h2>清洗版本</h2><div id="quality"></div><p class="muted">原始归档保留不变。缺乏公告证据或内容冲突的记录单独隔离，评测使用已校验版本。</p></section>
<section class="warn"><strong>评测口径</strong><p>全量缓存覆盖多种基金。当前排序研究按基金类型筛选，并固定日期、共同基金池和简单基线。
历史费率、申赎状态、分红及清盘结算尚未认证的样本，不获得可交易费后收益结论。</p></section>
<section><h2>数据快照</h2><div id="snapshots"></div></section>
<section><h2>固定协议</h2><div id="protocols"></div></section>
<section><h2>评测记录</h2><div id="runs"></div></section>
<section><h2>操作入口</h2><pre>qmt cache status
qmt cache plan --start 1998-01-01 --markets O,E
qmt cache sync --max-queries 500 --max-seconds 900
qmt evaluate snapshot
qmt evaluate register SNAPSHOT_DIRECTORY
qmt evaluate run SNAPSHOT_DIRECTORY PROTOCOL_FILE</pre>
<p class="muted">本页只读取状态。采集可断点续跑；新采集不会改写既有评测快照。</p></section>
<script>
const fmt=n=>typeof n==='number'?n.toLocaleString('zh-CN'):String(n??'—');
const labels={fund_basic:'基金基本信息',fund_nav:'基金净值',fund_portfolio:'披露持仓',fund_div:'基金分红',fund_share:'场内基金份额',fund_company:'基金管理公司',fund_manager:'基金经理',trade_cal:'交易日历',index_daily:'基准指数',sw_daily:'申万行业指数'};
const baselines={low_vol_60:'60日低波动',mom_60:'60日动量',mom_20:'20日动量',reversal_5:'5日反转',trend_60:'60日风险调整趋势',equal_weight:'等权（无排序IC）'};
const el=(tag,text)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;return x};
async function refresh(){try{
 const s=await fetch('/api/cache/status').then(r=>r.json());
 const tasks=s.tasks||{}, cards=document.getElementById('cards');cards.replaceChildren();
 const pending=(tasks.pending||0)+(tasks.retry||0),done=tasks.done||0,failed=(tasks.failed||0)+(tasks.truncated||0)+(tasks.permission_blocked||0);
 for(const [label,value] of [['已完成任务',done],['待采集任务',pending],['需处理任务',failed],['原始数据页',s.shards??s.raw_shards??'—']]){const c=el('div');c.className='card';c.append(el('div',label));const n=el('div',fmt(value));n.className='number';c.append(n);cards.append(c)}
 document.getElementById('stamp').textContent=s.generated_at||s.updated_at||s.message||'状态来自最近一次原子保存的采集清单';
 document.getElementById('tasks').textContent=JSON.stringify({tasks,interfaces:s.interfaces||[],collector:s.collector||{}},null,2);
 const quality=document.getElementById('quality');quality.replaceChildren();
 if(!s.curation)quality.append(el('p','尚无已提交的清洗版本'));
 else{quality.append(el('p',s.curation.quality_id));const table=el('table'),head=el('tr');
  for(const t of ['数据集','保留的唯一版本','去重/合并','隔离记录','补回公告日期'])head.append(el('th',t));table.append(head);
  for(const [name,q] of Object.entries(s.curation.datasets||{})){const tr=el('tr');for(const v of [labels[name]||name,fmt(q.clean_distinct_versions),fmt((q.exact_duplicates_removed||0)+(q.complementary_duplicates_collapsed||0)),fmt(q.quarantined_rows),fmt(q.repaired_announcement_rows)])tr.append(el('td',v));table.append(tr)}quality.append(table)}
 const body=document.getElementById('datasets');body.replaceChildren();
 for(const [name,d] of Object.entries(s.datasets||{})){const tr=el('tr');for(const value of [labels[name]||name,fmt(d.rows??d.archived_rows??d.raw_rows),fmt(d.funds_or_codes??d.funds_or_codes_lower_bound??d.codes),d.note||'包含历史版本与重叠；评测按冻结快照去重'])tr.append(el('td',value));body.append(tr)}
 for(const kind of ['snapshots','protocols','runs']){const out=document.getElementById(kind),result=await fetch('/api/evaluation/'+kind).then(r=>r.json());const expanded=new Set([...out.querySelectorAll('details[open]')].map(x=>x.dataset.artifactId));out.replaceChildren();
  if(!result.items.length){out.append(el('p','尚无记录'));continue}
  for(const item of result.items){const data=item.details;
   if(kind==='snapshots')out.append(el('p',fmt(data.funds)+' 只基金 · '+fmt(data.rows)+' 条净值 · 截至 '+(data.last_nav_date||'—')));
   const diagnostic=kind==='runs'&&String(item.status||'').startsWith('DIAGNOSTIC_');
   if(diagnostic)out.append(el('p',item.status==='DIAGNOSTIC_ACQUISITION_GAP'?'诊断记录：存在已定位的采集缺口，补齐后另行评测。':'诊断记录：旧引擎的迟披露处理存在问题，仅保留供排错。'));
   if(kind==='runs'&&!diagnostic){
    out.append(el('p','引擎 '+data.engine_version+' · Rank IC '+data.scored_ic_dates+'/'+data.planned_dates+' 个时点 · 组合净值代理 '+data.scored_portfolio_proxy_dates+'/'+data.planned_dates+' 个时点'));
    const table=el('table'),head=el('tr');for(const t of ['基线','平均 Rank IC','95%区间','有效时点'])head.append(el('th',t));table.append(head);
    for(const [arm,m] of Object.entries(data.aggregate||{})){const tr=el('tr'),ic=m.rank_ic||{},ci=ic.block_bootstrap_ci95;
     for(const t of [baselines[arm]||arm,ic.mean==null?'—':ic.mean.toFixed(4),ci?ci.map(x=>Number(x).toFixed(4)).join(' ～ '):'—',fmt(ic.n)])tr.append(el('td',t));table.append(tr)}out.append(table);
   }
   const details=el('details');details.dataset.artifactId=item.id;details.open=expanded.has(item.id);details.append(el('summary',item.id+(item.status?' · '+item.status:'')));details.append(el('pre',JSON.stringify(data,null,2)));out.append(details)
  }
 }
}catch(e){document.getElementById('stamp').textContent='状态暂不可用：'+e.message}}
refresh();setInterval(refresh,15000);
</script></body></html>"""
