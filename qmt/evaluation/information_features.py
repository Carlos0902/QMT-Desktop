"""Immutable published-holdings and market-risk features, with an ex-ante coverage gate."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from qmt.evaluation.platform import LEAVES,_hash_file,_json,verify_snapshot
from qmt.evaluation.search_benchmark import dump
from qmt.evaluation.search_diagnostics import load_inputs
from qmt.data.quality import verify_quality_bundle

MARKET=('market_beta120','market_resmom60','market_resvol60')
HOLDINGS=('hold_hhi10','hold_top1','hold_change10')
GROUPS={'F0':LEAVES,'F1':LEAVES+MARKET,'F2':LEAVES+HOLDINGS,'F3':LEAVES+MARKET+HOLDINGS}


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    return {str(root/p):_hash_file(root/p) for p in ('qmt/evaluation/information_features.py',
      'qmt/evaluation/search_diagnostics.py','qmt/evaluation/platform.py','qmt/evaluation/search_benchmark.py')}


def curate_reports(frame,basic,tainted=None):
    """One complete code/period/announcement package; no latest-per-security mixing."""
    rows=[];counts={};keys=['ts_code','end_date','ann_date']
    found=basic.set_index('fund_code').found_date
    found=pd.to_datetime(found.astype('string'),format='mixed',errors='coerce')
    tainted=set(tainted or [])
    for key,g in frame.groupby(keys,sort=True,dropna=False):
        code,end,ann=key;reason=None
        start=found.get(code,pd.NaT)
        if pd.isna(start):reason='unknown_identity_or_inception'
        elif pd.isna(end) or pd.isna(ann) or ann<end:reason='invalid_publication'
        elif end<start:reason='before_recorded_inception'
        elif (end.month not in (3,6,9,12)) or not end.is_month_end:reason='not_quarter_end'
        elif (code,end,ann) in tainted:reason='quarantined_source_package'
        elif g.groupby('symbol').payload_hash.nunique().gt(1).any():reason='conflicting_disclosure_versions'
        elif g.symbol.isna().any() or not (np.isfinite(g.mkv)&g.mkv.gt(0)).all():reason='invalid_security_or_market_value'
        unique=g.drop_duplicates('symbol').sort_values(['mkv','symbol'],ascending=[False,True])
        if reason is None and len(unique)<10:reason='fewer_than_ten_securities'
        weights={}
        if reason is None:
            top=unique.head(10);weights=dict(zip(top.symbol.astype(str),top.mkv/top.mkv.sum()))
        counts[reason or 'valid']=counts.get(reason or 'valid',0)+len(g)
        payload={'ts_code':str(code),'end_date':end,'ann_date':ann,'weights':weights,
                 'reason':reason,'row_count':len(g),'security_count':len(unique),
                 'source_payload_hashes':sorted(g.payload_hash.astype(str).unique())}
        package_hash=hashlib.sha256(_json(payload).encode()).hexdigest()
        rows.append({**{k:payload[k] for k in ('ts_code','end_date','ann_date','reason','row_count','security_count')},
                     'weights_json':_json(weights),'package_hash':package_hash,
                     'source_payload_hashes':_json(payload['source_payload_hashes'])})
    columns=['ts_code','end_date','ann_date','reason','row_count','security_count','weights_json','package_hash','source_payload_hashes']
    return pd.DataFrame(rows,columns=columns),counts


def project_holdings(reports,sessions,decisions,codes):
    if not decisions.is_monotonic_increasing:raise ValueError('decision dates must be ascending')
    arrays={key:np.full((len(decisions),len(codes)),np.nan) for key in HOLDINGS};selected=[]
    if reports.empty:
        return {k:pd.DataFrame(v,index=decisions,columns=codes) for k,v in arrays.items()},pd.DataFrame(columns=['decision_date','fund_code'])
    positions={c:i for i,c in enumerate(codes)}
    reports=reports.copy()
    reports['end_date']=pd.to_datetime(reports.end_date);reports['ann_date']=pd.to_datetime(reports.ann_date)
    available=sessions.searchsorted(reports.ann_date,side='right')
    reports['usable_date']=[sessions[p] if p<len(sessions) else pd.NaT for p in available]
    for code,parts in reports.groupby('ts_code',sort=False):
        if code not in positions:continue
        j=positions[code]
        packets=parts[parts.usable_date.notna()].sort_values(['usable_date','end_date','ann_date']).to_dict('records')
        cursor=0;public={};latest_end=None
        for i,d in enumerate(decisions):
            while cursor<len(packets) and packets[cursor]['usable_date']<=d:
                packet=packets[cursor];public[packet['end_date']]=packet;cursor+=1
                latest_end=max(public)
            if latest_end is None:continue
            current=public[latest_end]
            if pd.notna(current['reason']) or (d-current['end_date']).days>210:continue
            w=json.loads(current['weights_json'])
            arrays['hold_hhi10'][i,j]=sum(v*v for v in w.values())
            arrays['hold_top1'][i,j]=max(w.values())
            prev_end=current['end_date']-pd.offsets.QuarterEnd(1)
            previous=public.get(prev_end)
            if previous is not None:
                if pd.isna(previous['reason']) and (d-previous['end_date']).days<=210:
                    old=json.loads(previous['weights_json'])
                    arrays['hold_change10'][i,j]=.5*sum(abs(w.get(k,0)-old.get(k,0)) for k in set(w)|set(old))
            selected.append({'decision_date':d,'fund_code':code,'report_end':current['end_date'],
                'ann_date':current['ann_date'],'usable_date':current['usable_date'],'package_hash':current['package_hash'],
                'previous_package_hash':previous['package_hash'] if previous is not None else None})
    return {k:pd.DataFrame(v,index=decisions,columns=codes) for k,v in arrays.items()},pd.DataFrame(selected)


def market_series(group,sessions,decisions,market_close,*,lag=2):
    indexed=group.set_index('nav_date').reindex(sessions)
    px=indexed.adj_nav;ret=px.pct_change(fill_method=None)
    mret=market_close.reindex(sessions).pct_change(fill_method=None)
    usable=pd.to_datetime(indexed.usable_date)
    pub=pd.Series(sessions.get_indexer(usable),index=sessions,dtype=float).replace(-1,np.nan)
    positions=sessions.get_indexer(decisions)
    visible=pub.rolling(121,min_periods=121).max().shift(lag).iloc[positions].to_numpy()<=positions
    complete=ret.rolling(120,min_periods=120).count().eq(120)&mret.rolling(120,min_periods=120).count().eq(120)
    mmean=mret.rolling(120,min_periods=120).mean();rmean=ret.rolling(120,min_periods=120).mean()
    cov=(ret*mret).rolling(120,min_periods=120).mean()-rmean*mmean
    var=(mret*mret).rolling(120,min_periods=120).mean()-mmean*mmean
    beta=cov/var.where(var>1e-16)
    r60=ret.rolling(60,min_periods=60).mean();m60=mret.rolling(60,min_periods=60).mean()
    cov60=(ret*mret).rolling(60,min_periods=60).mean()-r60*m60
    # Population residual moments for a fixed beta estimated at the window end.
    vr=ret.rolling(60,min_periods=60).var(ddof=0);vm=mret.rolling(60,min_periods=60).var(ddof=0)
    residual_var=(vr+beta*beta*vm-2*beta*cov60).clip(lower=0)
    values=(beta,r60-beta*m60,np.sqrt(residual_var))
    return {k:np.where(visible,v.where(complete).shift(lag).iloc[positions].to_numpy(),np.nan) for k,v in zip(MARKET,values)}


def project_market(nav_path,sessions,decisions,codes,index):
    index=index.copy();index.trade_date=pd.to_datetime(index.trade_date)
    if index.trade_date.duplicated().any():raise ValueError('duplicate market sessions')
    m=index.set_index('trade_date').close
    if not (np.isfinite(m)&m.gt(0)).all():raise ValueError('invalid market close')
    arrays={k:np.full((len(decisions),len(codes)),np.nan) for k in MARKET};lookup={c:i for i,c in enumerate(codes)}
    carry=pd.DataFrame()
    def consume(g):
        code=g.fund_code.iloc[0]
        if code not in lookup:return
        vals=market_series(g,sessions,decisions,m)
        for k,a in vals.items():arrays[k][:,lookup[code]]=a
    for batch in pq.ParquetFile(nav_path).iter_batches(batch_size=131072):
        frame=pd.concat([carry,batch.to_pandas()],ignore_index=True)
        last=frame.fund_code.iloc[-1];carry=frame[frame.fund_code.eq(last)]
        for _,g in frame[~frame.fund_code.eq(last)].groupby('fund_code',sort=False):consume(g)
    if len(carry):consume(carry)
    return {k:pd.DataFrame(a,index=decisions,columns=codes) for k,a in arrays.items()}


def build(snapshot,baseline_protocol,quality,lake,output):
    from qmt.data.fund_lake import DatasetLake
    output=Path(output);quality=Path(quality);snapshot=Path(snapshot)
    nav_manifest=verify_snapshot(snapshot);quality_manifest=verify_quality_bundle(quality)
    output.mkdir(parents=True,exist_ok=False)
    rules={'snapshot':str(snapshot.resolve()),'snapshot_hash':nav_manifest['snapshot_hash'],
      'baseline_protocol':str(Path(baseline_protocol).resolve()),'quality':str(quality.resolve()),
      'quality_manifest_hash':quality_manifest['manifest_hash'],'created_at':datetime.now(timezone.utc).isoformat(),
      'groups':GROUPS,'min_funds':20,'min_date_coverage':.8,'report_max_age_days':210,
      'report_scope':'normalized disclosed top ten by mkv; not total portfolio exposure or actual turnover',
      'code_hashes':source_hashes()}
    dump(output/'protocol.json',rules)
    x=load_inputs(snapshot,baseline_protocol,output/'nav_inputs',first_year=2009)
    codes=list(x['outcomes'].columns);decisions=x['outcomes'].index;sessions=x['sessions']
    store=DatasetLake(lake)
    index=store.read_table('index_daily',cutoff=nav_manifest['sealed_from'],columns=['ts_code','trade_date','close'],include_provenance=True)
    index=index[index.ts_code.eq('000300.SH')].copy()
    if index.get('_same_vintage_conflict',pd.Series(False,index=index.index)).any():raise ValueError('conflicting index quotes')
    index=index.sort_values('trade_date');index.to_parquet(output/'index.parquet',index=False)
    market=project_market(snapshot/'nav.parquet',sessions,decisions,codes,index)
    db=duckdb.connect();db.execute("SET threads=2");db.execute("SET memory_limit='2GB'")
    # The older quality bundle may have been captured mid-pagination. A new,
    # independent holdings receipt includes ONLY fully completed acquisitions.
    from qmt.data.quality import _register_raw
    with store._db() as catalog:
        complete=catalog.execute("SELECT s.shard_id,s.dataset,s.raw_path,s.fetched_at,s.ingested_at,s.row_count,t.task_id,t.updated_at,s.fingerprint FROM shards s JOIN tasks t ON s.task_id=t.task_id WHERE s.dataset='fund_portfolio' AND s.row_count>0 AND t.status='done'").fetchall()
    names=['shard_id','dataset','raw_path','fetched_at','ingested_at','row_count','task_id','task_completed_at','payload_fingerprint']
    sources=[dict(zip(names,row)) for row in complete]
    for source in sources:source['file_sha256']=_hash_file(store.root/source['raw_path'])
    dump(output/'holdings_source.json',{'acquired_before':datetime.now(timezone.utc).isoformat(),
         'only_completed_tasks':True,'sources':sources})
    normalized=_register_raw(db,'fund_portfolio',{'lake_root':str(store.root),'sources':sources})
    db.execute('CREATE VIEW holdings AS '+normalized)
    db.register('cohort',pd.DataFrame({'ts_code':codes}))
    frame=db.execute("SELECT h.ts_code,cast(end_date AS TIMESTAMP) end_date,cast(ann_date AS TIMESTAMP) ann_date,symbol,mkv,sha256(to_json(row(symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio))) payload_hash FROM holdings h JOIN cohort USING(ts_code) WHERE ann_date<DATE '2024-12-10' AND end_date<DATE '2024-12-10'").fetchdf()
    db.close()
    basic=pd.read_parquet(snapshot/'fund_basic.parquet')
    reports,counts=curate_reports(frame,basic)
    reports.to_parquet(output/'holdings_reports.parquet',index=False)
    holdings,selected=project_holdings(reports,sessions,decisions,codes)
    selected.to_parquet(output/'holdings_selection.parquet',index=False)
    features={**x['features'],**market,**holdings}
    common=pd.DataFrame(np.logical_and.reduce([np.isfinite(f) for f in features.values()]),index=decisions,columns=codes)
    files=['index.parquet','holdings_reports.parquet','holdings_selection.parquet','holdings_source.json']
    for key,frame in features.items():frame.to_parquet(output/f'{key}.parquet');files.append(f'{key}.parquet')
    common.to_parquet(output/'common.parquet');files.append('common.parquet')
    outer_dates=pd.DatetimeIndex(sorted(d for ds in x['outer'].values() for d in ds))
    coverage=pd.DataFrame({'base_funds':x['common'].sum(axis=1),'common_funds':common.sum(axis=1)}).reindex(outer_dates)
    coverage['feature_scored']=coverage.common_funds.ge(20);coverage.index.name='decision_date'
    coverage.to_parquet(output/'coverage.parquet');files.append('coverage.parquet')
    passing=int(coverage.feature_scored.sum());ok=passing/len(coverage)>=.8
    manifest={**rules,'kind':'new_information_feature_snapshot_v1','quality_gate':bool(ok),
       'feature_date_coverage':passing/len(coverage),'passing_decisions':passing,'planned_decisions':len(coverage),
       'holdings_rows_by_reason':counts,'reports':len(reports),'valid_reports':int(reports.reason.isna().sum()),
       'files':{n:_hash_file(output/n) for n in files},'nav_inputs_manifest_hash':_hash_file(output/'nav_inputs/manifest.json'),
       'evidence_tier':'revised_vendor_history_with_announcements_not_certified_original_vintages'}
    if rules['code_hashes']!=source_hashes():raise RuntimeError('feature code changed during build')
    manifest['manifest_hash']=hashlib.sha256(_json(manifest).encode()).hexdigest()
    dump(output/'manifest.json',manifest)
    print('FEATURES',passing,'/',len(coverage),'quality_gate',ok,flush=True)
    return manifest


def load(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    raw={k:v for k,v in m.items() if k!='manifest_hash'}
    if hashlib.sha256(_json(raw).encode()).hexdigest()!=m['manifest_hash']:raise ValueError('feature manifest modified')
    for n,d in m['files'].items():
        if Path(n).name!=n or _hash_file(path/n)!=d:raise ValueError(f'feature file modified: {n}')
    if _hash_file(path/'nav_inputs/manifest.json')!=m['nav_inputs_manifest_hash']:raise ValueError('NAV input manifest modified')
    x=load_inputs(m['snapshot'],m['baseline_protocol'],path/'nav_inputs',first_year=2009)
    x['features']={k:pd.read_parquet(path/f'{k}.parquet') for k in LEAVES+MARKET+HOLDINGS}
    x['full_common']=x['common'];x['common']=pd.read_parquet(path/'common.parquet');x['manifest']=m
    return x
