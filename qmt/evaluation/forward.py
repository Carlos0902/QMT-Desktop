"""Append-only prospective shadow evaluation. No historical-seal override or trading."""
from __future__ import annotations
import portalocker
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib,json,os,shutil,tempfile
from importlib.metadata import version
import platform
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb
import numpy as np
import pandas as pd
from qmt.evaluation.platform import LEAVES,_hash_file,_json,_clean
from qmt.evaluation.search_benchmark import dump,joint_test
from qmt.evaluation.information_features import MARKET,HOLDINGS,curate_reports,project_holdings,market_series
from qmt.evaluation.information_study import ARMS,model_scores
from qmt.stats import spearman

TZ=ZoneInfo('Asia/Shanghai')


def _now():return datetime.now(timezone.utc)


def _runtime():
    return {'python':platform.python_version(),**{name:version(name) for name in ('numpy','pandas','pyarrow','duckdb','scikit-learn','scipy')}}


def _exclusive_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name,delete=False) as h:
        temporary=Path(h.name)
        h.write(json.dumps(_clean(value),ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n');h.flush();os.fsync(h.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()


@contextmanager
def _lock(root):
    with (Path(root)/'.lock').open('a') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        try:yield
        finally:portalocker.unlock(f)


def calendar_from_lake(lake):
    from qmt.data.fund_lake import DatasetLake
    frame=DatasetLake(lake).read_table('trade_cal',include_provenance=True)
    if frame.empty:raise ValueError('missing exchange calendar')
    frame=frame[frame.exchange.eq('SSE')].copy()
    frame['cal_date']=pd.to_datetime(frame.cal_date)
    if frame.cal_date.duplicated().any():raise ValueError('conflicting calendar dates')
    return frame.sort_values('cal_date')


def refresh(lake,*,calendar_only=False,with_holdings=True):
    from qmt.data.fund_lake import DatasetLake
    from qmt.data.tushare_client import TushareClient
    store=DatasetLake(lake);today=_now().astimezone(TZ).date();generation='forward-'+today.isoformat()
    required=[]
    for year in (today.year,today.year+1):
        required.append(store.enqueue('trade_cal',{'exchange':'SSE','start_date':f'{year}0101','end_date':f'{year}1231'},priority=220,generation=generation))
    if not calendar_only:
        for offset in range(14):
            day=pd.Timestamp(today)-pd.Timedelta(days=offset)
            required.append(store.enqueue('fund_nav',{'nav_date':day.strftime('%Y%m%d'),'market':'O'},priority=190,generation=generation))
        required.append(store.enqueue('index_daily',{'ts_code':'000300.SH','start_date':(pd.Timestamp(today)-pd.Timedelta(days=30)).strftime('%Y%m%d'),'end_date':today.strftime('%Y%m%d')},priority=200,generation=generation))
        if with_holdings:
            for status in ('L','I','D'):
                required.append(store.enqueue('fund_basic',{'market':'O','status':status},priority=210,generation=generation))
            end=pd.Timestamp(today)-pd.offsets.QuarterEnd()
            for _ in range(3):
                required.append(store.enqueue('fund_portfolio',{'period':end.strftime('%Y%m%d')},priority=180,generation=generation))
                end-=pd.offsets.QuarterEnd()
    result=store.run_budget(TushareClient(timeout=20,max_retries=0),max_queries=1500,max_seconds=900 if not calendar_only else 80,min_interval=.4)
    with store._db() as db:
        states=db.execute('SELECT task_id,status,last_error FROM tasks WHERE task_id IN (SELECT unnest(?))',[required]).fetchall()
    incomplete=[{'task_id':i,'status':s,'error':e} for i,s,e in states if s!='done']
    return {**result,'required_complete':len(states)==len(required) and not incomplete,'incomplete_tasks':incomplete}


def register(study,output,lake,*,update_calendar=True):
    study=Path(study);output=Path(output)
    report=json.loads((study/'report.json').read_text())
    if report['status']!='COMPLETE_DEVELOPMENT_STUDY':raise ValueError('forward registration requires a completed historical study')
    for name,digest in report['files'].items():
        if _hash_file(study/name)!=digest:raise ValueError('historical result modified')
    protocol=json.loads((study/'protocol.json').read_text())
    for name,digest in protocol['code_hashes'].items():
        if _hash_file(Path(name))!=digest:raise ValueError('historical model code modified')
    feature_path=Path(protocol['feature_path']);fm=json.loads((feature_path/'manifest.json').read_text())
    if fm['manifest_hash']!=report['feature_manifest_hash']:raise ValueError('feature/model identity mismatch')
    if update_calendar:
        if not refresh(lake,calendar_only=True)['required_complete']:raise ValueError('calendar acquisition incomplete')
    calendar=calendar_from_lake(lake);now=_now();day=pd.Timestamp(now.astimezone(TZ).date())
    open_dates=pd.DatetimeIndex(calendar.loc[pd.to_numeric(calendar.is_open).eq(1),'cal_date'])
    future=open_dates[open_dates>day]
    if not len(future):raise ValueError('no verified future exchange session; cannot guess weekdays')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(study/'frozen_models.json',output/'models.json')
    snapshot=Path(fm['snapshot']);basic=pd.read_parquet(snapshot/'fund_basic.parquet')
    codes=sorted(pd.read_parquet(snapshot/'nav.parquet',columns=['fund_code']).fund_code.unique().tolist())
    basic[basic.fund_code.isin(codes)].to_parquet(output/'basic.parquet',index=False)
    calendar.to_parquet(output/'calendar.parquet',index=False)
    root=Path(__file__).resolve().parents[2]
    files=['qmt/evaluation/forward.py','qmt/evaluation/information_features.py','qmt/evaluation/information_study.py',
           'qmt/evaluation/search_improvement.py','qmt/evaluation/search_diagnostics.py','qmt/evaluation/search_benchmark.py',
           'qmt/factor/dsl.py','qmt/factor/types.py','qmt/stats.py','qmt/data/fund_lake.py',
           'qmt/data/quality.py','qmt/data/tushare_client.py']
    p={'kind':'prospective_shadow_v1','registered_at':now.isoformat(),'timezone':'Asia/Shanghai',
      'start_date':future[0].strftime('%Y-%m-%d'),'stride_sessions':40,'horizon_sessions':40,'deadline_local':'08:30',
      'historical_study_protocol_hash':report['protocol_hash'],'feature_manifest_hash':fm['manifest_hash'],
      'historical_advance_arms':[a for a,v in report.get('summary',{}).items() if v.get('advance')],
      'shadow_only':True,
      'primary':'ridge_F3','secondary':[a for a in ARMS if a!='ridge_F3'],'fixed_codes':codes,
      'model_training_fold':2024,'refit_with_forward_labels':False,'label_policy':'first observed valid endpoint NAV; no zero-fill or truncation',
      'frozen_files':{name:_hash_file(output/name) for name in ('models.json','basic.parquet','calendar.parquet')},
      'code_hashes':{str(root/f):_hash_file(root/f) for f in files},
      'runtime_versions':_runtime(),
      'confirmation':{'min_dates':12,'min_calendar_years':3,'min_mean_delta':.01,'family_lower95_above':0,'positive_year_fraction':2/3},
      'historical_seal_unchanged':True,'current_context_only':'post-seal prices may form contemporaneous features; never historical return selection',
      'alpha_approved':False}
    p['protocol_hash']=hashlib.sha256(_json(p).encode()).hexdigest()
    _exclusive_json(output/'protocol.json',p)
    return p


def _verify(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    if hashlib.sha256(_json({k:v for k,v in p.items() if k!='protocol_hash'}).encode()).hexdigest()!=p['protocol_hash']:
        raise ValueError('forward protocol modified')
    if any(_hash_file(root/n)!=h for n,h in p['frozen_files'].items()):raise ValueError('frozen forward artifact modified')
    if any(_hash_file(Path(n))!=h for n,h in p['code_hashes'].items()):raise ValueError('forward code changed; register a new version')
    if 'runtime_versions' in p and p['runtime_versions']!=_runtime():raise ValueError('forward runtime changed; register a new version')
    return p


def _calendar(root,lake):
    frozen=pd.read_parquet(Path(root)/'calendar.parquet');fresh=calendar_from_lake(lake)
    past=pd.Timestamp(_now().astimezone(TZ).date())
    old=frozen.set_index('cal_date').is_open;new=fresh.set_index('cal_date').is_open
    overlap=old.index.intersection(new.index);overlap=overlap[overlap<=past]
    if not pd.to_numeric(old.loc[overlap]).eq(pd.to_numeric(new.loc[overlap])).all():raise ValueError('previously published calendar revised')
    return fresh


def _schedule(p,calendar):
    sessions=pd.DatetimeIndex(calendar.loc[pd.to_numeric(calendar.is_open).eq(1),'cal_date']).sort_values()
    start=sessions.searchsorted(pd.Timestamp(p['start_date']))
    return sessions,sessions[start::p['stride_sessions']]


def _validate_receipt(p,receipt,anchors,folder):
    day=pd.Timestamp(receipt['decision_date'])
    if receipt.get('protocol_hash')!=p['protocol_hash'] or day not in anchors or folder.name!=day.strftime('%Y-%m-%d'):
        raise ValueError('prediction receipt does not belong to this forward protocol and schedule')
    if receipt['status']=='RECORDED':
        stamp=pd.Timestamp(receipt['recorded_at'])
        if stamp.tzinfo is None:raise ValueError('prediction timestamp must include timezone')
        local=stamp.tz_convert(TZ)
        deadline=pd.Timestamp(datetime.combine(day.date(),datetime.strptime(p['deadline_local'],'%H:%M').time(),TZ))
        if local.date()!=day.date() or local>=deadline:
            raise ValueError('prediction was not frozen before the registered cutoff')
        if 'registered_at' in p and stamp<pd.Timestamp(p['registered_at']):
            raise ValueError('prediction predates prospective registration')


def raw_context(lake,dataset,since,observed_before,codes):
    """Only acquired pages observed before capture time, bounded by source query dates."""
    from qmt.data.fund_lake import DatasetLake
    from qmt.data.quality import _register_raw
    store=DatasetLake(lake);pages=[]
    with store._db() as db:
        records=db.execute("SELECT s.shard_id,s.dataset,s.raw_path,s.fetched_at,s.ingested_at,s.row_count,s.params_json FROM shards s JOIN tasks t ON s.task_id=t.task_id WHERE s.dataset=? AND s.row_count>0 AND t.status='done' AND cast(t.updated_at AS TIMESTAMPTZ)<=cast(? AS TIMESTAMPTZ)",[dataset,observed_before]).fetchall()
    for shard,ds,path,fetched,ingested,count,params in records:
        if pd.Timestamp(ingested)>pd.Timestamp(observed_before):continue
        params=json.loads(params);end=params.get('nav_date') or params.get('period') or params.get('end_date')
        if end and pd.to_datetime(str(end),format='mixed')<pd.Timestamp(since):continue
        pages.append({'shard_id':shard,'dataset':ds,'raw_path':path,'fetched_at':fetched,'ingested_at':ingested,'row_count':count})
    if not pages:return pd.DataFrame(),[]
    db=duckdb.connect();db.execute("SET threads=2");db.execute("SET memory_limit='2GB'")
    normalized=_register_raw(db,dataset,{'lake_root':str(store.root),'sources':pages})
    db.execute('CREATE VIEW normalized AS '+normalized)
    db.register('cohort',pd.DataFrame({'ts_code':codes}))
    datecol='nav_date' if dataset=='fund_nav' else 'end_date'
    frame=db.execute(f'SELECT n.* FROM normalized n JOIN cohort USING(ts_code) WHERE {datecol}>=cast(? AS DATE) AND ann_date IS NOT NULL AND ingested_at<=cast(? AS TIMESTAMPTZ)',[str(pd.Timestamp(since).date()),observed_before]).fetchdf()
    db.close()
    used=set(frame.source_shard_id) if len(frame) else set()
    receipts=[{**v,'file_sha256':_hash_file(store.root/v['raw_path'])} for v in pages if v['shard_id'] in used]
    return frame,receipts


def nav_quotes(frame,*,first_observation=False):
    if frame.empty:return frame
    frame=frame.copy();frame.nav_date=pd.to_datetime(frame.nav_date);frame.ann_date=pd.to_datetime(frame.ann_date)
    frame['value_hash']=[hashlib.sha256(_json([a,b]).encode()).hexdigest() for a,b in zip(frame.unit_nav,frame.adj_nav)]
    conflict=frame.groupby(['ts_code','nav_date','ann_date','fetched_at']).value_hash.transform('nunique').gt(1)
    bad_keys=set(zip(frame.loc[conflict,'ts_code'],frame.loc[conflict,'nav_date']))
    def valid_rows(f):
        valid=np.isfinite(f.adj_nav)&f.adj_nav.gt(0)&np.isfinite(f.unit_nav)&f.unit_nav.gt(0)&f.ann_date.ge(f.nav_date)
        return valid & pd.Series([(c,d) not in bad_keys for c,d in zip(f.ts_code,f.nav_date)],index=f.index,dtype=bool)
    if first_observation:
        frame=frame.loc[valid_rows(frame)]
        order=['ts_code','nav_date','ingested_at','fetched_at','ann_date']
        return frame.sort_values(order).drop_duplicates(['ts_code','nav_date']).copy()
    order=['ts_code','nav_date','ann_date','fetched_at','ingested_at']
    frame=frame.sort_values(order,ascending=[True,True,False,False,False]).drop_duplicates(['ts_code','nav_date'])
    return frame.loc[valid_rows(frame)].copy()


def identity_mask(frozen,current,codes):
    old=pd.to_datetime(frozen.set_index('fund_code').found_date.astype('string'),format='mixed',errors='coerce')
    if current.empty:return pd.Series(False,index=codes)
    now=current.drop_duplicates('ts_code',keep=False).set_index('ts_code')
    found=pd.to_datetime(now.found_date.astype('string'),format='mixed',errors='coerce')
    valid=old.reindex(codes).notna() & found.reindex(codes).eq(old.reindex(codes))
    if '_same_vintage_conflict' in now:valid&=~now._same_vintage_conflict.reindex(codes).fillna(True).astype(bool)
    return valid


def contemporaneous_features(root,lake,p,day,sessions):
    from qmt.data.fund_lake import DatasetLake
    observed=_now().isoformat();codes=p['fixed_codes'];since=day-pd.Timedelta(days=420)
    raw,nav_sources=raw_context(lake,'fund_nav',since,observed,codes)
    raw=raw[raw.ann_date<day] if len(raw) else raw
    nav=nav_quotes(raw)
    if nav.empty:raise ValueError('no observed published NAV context')
    usable=sessions.searchsorted(pd.DatetimeIndex(nav.ann_date),side='right')
    nav['usable_date']=[sessions[i] if i<len(sessions) else pd.NaT for i in usable]
    h,hold_sources=raw_context(lake,'fund_portfolio',since,observed,codes)
    if h.empty:raise ValueError('no observed holdings context')
    h=h[h.ann_date<day].copy()
    h['payload_hash']=[hashlib.sha256(_json([s,m,a,b,c]).encode()).hexdigest() for s,m,a,b,c in zip(h.symbol,h.mkv,h.amount,h.stk_mkv_ratio,h.stk_float_ratio)]
    basic=pd.read_parquet(Path(root)/'basic.parquet')
    reports,_=curate_reports(h,basic)
    ds=pd.DatetimeIndex([day]);features,_=project_holdings(reports,sessions,ds,codes)
    store=DatasetLake(lake)
    index=store.read_table('index_daily',cutoff=observed,acquired_before=observed,columns=['ts_code','trade_date','close'],include_provenance=True)
    index=index[index.ts_code.eq('000300.SH')].copy();index.trade_date=pd.to_datetime(index.trade_date)
    if index.trade_date.duplicated().any() or index._same_vintage_conflict.any():raise ValueError('conflicting current index history')
    market=index.set_index('trade_date').close
    arrays={k:np.full((1,len(codes)),np.nan) for k in LEAVES+MARKET};lookup={c:i for i,c in enumerate(codes)}
    pos=sessions.get_loc(day)
    for code,g in nav.groupby('ts_code',sort=False):
        px=g.set_index('nav_date').adj_nav.reindex(sessions);ret=px.pct_change(fill_method=None)
        pub=pd.Series(sessions.get_indexer(pd.to_datetime(g.set_index('nav_date').usable_date.reindex(sessions))),index=sessions,dtype=float).replace(-1,np.nan)
        visible=pub.rolling(61,min_periods=61).max().shift(2).iloc[pos]<=pos
        complete=ret.rolling(60,min_periods=60).count().eq(60)
        mean20=ret.rolling(20,min_periods=20).mean();mean60=ret.rolling(60,min_periods=60).mean()
        vol20=ret.rolling(20,min_periods=20).std();vol60=ret.rolling(60,min_periods=60).std()
        vals={'mom20':mean20,'mom60':mean60,'vol20':vol20,'vol60':vol60,
              'reversal5':-ret.rolling(5,min_periods=5).mean(),'trend60':mean60/vol60.where(vol60>1e-12),
              'dd60':px/px.rolling(60,min_periods=60).max()-1}
        j=lookup[code]
        for key,val in vals.items():arrays[key][0,j]=val.where(complete).shift(2).iloc[pos] if visible else np.nan
        mg=g.rename(columns={'ts_code':'fund_code'})
        for key,values in market_series(mg,sessions,ds,market).items():arrays[key][0,j]=values[0]
    features.update({k:pd.DataFrame(v,index=ds,columns=codes) for k,v in arrays.items()})
    common=pd.DataFrame(np.logical_and.reduce([np.isfinite(v) for v in features.values()]),index=ds,columns=codes)
    metadata=store.read_table('fund_basic',cutoff=observed,acquired_before=observed,codes=codes,
                             columns=['ts_code','found_date'],include_provenance=True)
    identity=identity_mask(basic,metadata,codes)
    common&=identity.to_numpy()[None,:]
    return features,common,{'observed_before':observed,'NAV':nav_sources,'holdings':hold_sources,
        'index_rows':index.to_dict('records'),'identity_checked_funds':int(identity.sum()),
        'identity_metadata':metadata.to_dict('records')}


def record(registry,lake,*,context_provider=None):
    root=Path(registry);p=_verify(root);now=_now().astimezone(TZ);today=pd.Timestamp(now.date())
    with _lock(root):
        calendar=_calendar(root,lake);sessions,anchors=_schedule(p,calendar)
        if today>pd.Timestamp(calendar.cal_date.max()):return {'status':'MISSING_CALENDAR','attention':True}
        missed=[]
        for d in anchors[anchors<today]:
            receipt=root/'decisions'/d.strftime('%Y-%m-%d')/'receipt.json'
            if not receipt.exists():
                _exclusive_json(receipt,{'status':'MISSED_DEADLINE','decision_date':str(d.date()),'observed_at':_now().isoformat(),'protocol_hash':p['protocol_hash']});missed.append(str(d.date()))
        if today not in anchors:return {'status':'NOT_DUE','missed':missed,'attention':bool(missed)}
        target=root/'decisions'/today.strftime('%Y-%m-%d');receipt=target/'receipt.json'
        if receipt.exists():
            existing=json.loads(receipt.read_text());_validate_receipt(p,existing,anchors,receipt.parent)
            return {'status':'ALREADY_RECORDED','receipt':existing,'missed':missed,'attention':bool(missed)}
        deadline=datetime.combine(today.date(),datetime.strptime(p['deadline_local'],'%H:%M').time(),TZ)
        def failure(status,detail=None):
            value={'status':status,'decision_date':str(today.date()),'observed_at':_now().isoformat(),'protocol_hash':p['protocol_hash'],'detail':detail}
            _exclusive_json(receipt,value);return {**value,'attention':True}
        if now>=deadline:return failure('MISSED_DEADLINE')
        try:
            features,common,sources=(context_provider or contemporaneous_features)(root,lake,p,today,sessions)
            if int(common.iloc[0].sum())<20:return failure('MISSING_DATA','fewer than 20 common published feature scores')
            models=json.loads((root/'models.json').read_text())['models'];rows=[]
            eligible=common.columns[common.iloc[0]]
            for arm,bundle in models.items():
                for seed,model in bundle.items():
                    scores,reasons=model_scores(model,features,common)
                    for code in eligible:rows.append({'fund_code':code,'arm':arm,'seed':int(seed),'score':float(scores.loc[today,code]),
                        'baseline_score':float(-features['vol60'].loc[today,code]),'active':bool(model.get('active',False)),
                        'fallback_reason':reasons.loc[today]})
            predictions=pd.DataFrame(rows)
            if not np.isfinite(predictions[['score','baseline_score']]).all(axis=None):return failure('MISSING_DATA','nonfinite final prediction')
            if _now().astimezone(TZ)>=deadline:return failure('MISSED_DEADLINE','computation completed after cutoff')
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
                tmp=Path(temporary);predictions.to_parquet(tmp/'predictions.parquet',index=False)
                values=pd.DataFrame({k:v.loc[today] for k,v in features.items()});values.index.name='fund_code'
                values.to_parquet(tmp/'features.parquet');dump(tmp/'source_receipts.json',sources)
                value={'status':'RECORDED','decision_date':str(today.date()),'recorded_at':_now().isoformat(),
                    'protocol_hash':p['protocol_hash'],'eligible_funds':len(eligible),'deadline':deadline.isoformat(),
                    'files':{n:_hash_file(tmp/n) for n in ('predictions.parquet','features.parquet','source_receipts.json')}}
                if _now().astimezone(TZ)>=deadline:return failure('MISSED_DEADLINE','artifact serialization crossed cutoff')
                os.rename(tmp,target)
                committed=_now().astimezone(TZ)
                if committed>=deadline:
                    diagnostics=root/'diagnostics';diagnostics.mkdir(exist_ok=True)
                    os.rename(target,diagnostics/(today.strftime('%Y-%m-%d')+'-late'))
                    return failure('MISSED_DEADLINE','atomic publication completed after cutoff')
                value['recorded_at']=committed.isoformat()
                _exclusive_json(target/'receipt.json',value)
            return {**value,'missed':missed,'attention':bool(missed)}
        except Exception as exc:
            return failure('MISSING_DATA',f'{type(exc).__name__}: {exc}')


def score(registry,lake):
    root=Path(registry);p=_verify(root);calendar=_calendar(root,lake);sessions,anchors=_schedule(p,calendar)
    now=_now();today=pd.Timestamp(now.astimezone(TZ).date());new=[];pending=[]
    with _lock(root):
        receipts=sorted((root/'decisions').glob('*/receipt.json')) if (root/'decisions').exists() else []
        for receipt in receipts:
            r=json.loads(receipt.read_text());day=pd.Timestamp(r['decision_date'])
            _validate_receipt(p,r,anchors,receipt.parent)
            if r['status']!='RECORDED':continue
            if any(_hash_file(receipt.parent/n)!=h for n,h in r['files'].items()):raise ValueError('frozen prediction changed')
            result=receipt.parent/'score.json'
            if result.exists():continue
            pos=sessions.get_indexer([day])[0]
            if pos<0 or pos+40>=len(sessions):continue
            end=sessions[pos+40]
            if end>=today:continue
            raw,source=raw_context(lake,'fund_nav',day,now.isoformat(),p['fixed_codes'])
            raw=raw[raw.ann_date<today] if len(raw) else raw
            quotes=nav_quotes(raw,first_observation=True)
            if quotes.empty:
                pending.append({'decision_date':str(day.date()),'outcome_end':str(end.date()),'reason':'NO_OBSERVED_ENDPOINT_QUOTES'})
                continue
            px=quotes.pivot(index='nav_date',columns='ts_code',values='adj_nav')
            if day not in px.index or end not in px.index:
                pending.append({'decision_date':str(day.date()),'outcome_end':str(end.date()),'reason':'ENDPOINT_NOT_PUBLISHED'})
                continue
            y=px.loc[end]/px.loc[day]-1
            predictions=pd.read_parquet(receipt.parent/'predictions.parquet')
            universe=predictions.fund_code.unique();known=y.reindex(universe).replace([np.inf,-np.inf],np.nan).dropna()
            if len(known)<20 or len(known)/len(universe)<.95:
                pending.append({'decision_date':str(day.date()),'outcome_end':str(end.date()),
                    'reason':'INSUFFICIENT_LABEL_COVERAGE','observed_fraction':len(known)/len(universe)})
                continue
            rows=[]
            for (arm,seed),g in predictions.groupby(['arm','seed']):
                g=g.set_index('fund_code').reindex(known.index)
                base=spearman(g.baseline_score.to_numpy(),known.to_numpy());ic=spearman(g.score.to_numpy(),known.to_numpy())
                rows.append({'arm':arm,'seed':int(seed),'rank_ic':ic,'baseline_ic':base,'delta_ic':ic-base})
            value={'status':'SCORED','decision_date':str(day.date()),'outcome_end':str(end.date()),'scored_at':now.isoformat(),
              'protocol_hash':p['protocol_hash'],'observed_label_fraction':len(known)/len(universe),'metrics':rows,
              'label_source_receipts':source,'label_vintage':'first_observed_valid_endpoint'}
            _exclusive_json(result,value);new.append(str(day.date()))
        results=[]
        for file in sorted((root/'decisions').glob('*/score.json')) if (root/'decisions').exists() else []:
            r=json.loads(file.read_text())
            receipt=json.loads((file.parent/'receipt.json').read_text())
            _validate_receipt(p,receipt,anchors,file.parent)
            if receipt['status']!='RECORDED' or r.get('protocol_hash')!=p['protocol_hash'] or r['decision_date']!=receipt['decision_date']:
                raise ValueError('score does not match a frozen prediction in this protocol')
            results.extend({'decision_date':r['decision_date'],**row} for row in r['metrics'])
        if results:
            f=pd.DataFrame(results);f.decision_date=pd.to_datetime(f.decision_date)
            f['delta_ic']=pd.to_numeric(f.delta_ic,errors='coerce')
            temporal=f.groupby(['decision_date','arm']).delta_ic.agg(
                lambda s:s.mean() if s.notna().all() else np.nan).unstack().dropna()
            inference=joint_test(temporal)
            years=temporal.index.year.nunique()
            enough=len(temporal)>=12 and years>=3
            confirmed=[]
            for arm,v in inference.get('arms',{}).items():
                annual=temporal[arm].groupby(temporal.index.year).mean()
                seeds=f[f.arm.eq(arm)].groupby('seed').delta_ic.mean()
                if enough and v['mean_delta_ic']>=.01 and v['simultaneous_lower95']>0 and v['family_adjusted_one_sided_p']<.05 and annual.gt(0).sum()>=np.ceil(2*years/3) and seeds.gt(0).all():confirmed.append(arm)
            summary={'status':'EVALUATED' if enough else 'INSUFFICIENT_PROSPECTIVE_EVIDENCE','scored_dates':len(temporal),'calendar_years':years,'family':inference,'confirmed_arms':confirmed,'primary':p['primary'],'alpha_approved':False}
        else:summary={'status':'WAITING_FOR_MATURE_LABELS','scored_dates':0,'confirmed_arms':[],'alpha_approved':False}
        summary['pending_mature_decisions']=pending
        # Versioned summaries, never overwrite a previous score or prediction.
        stamp=hashlib.sha256(_json(summary).encode()).hexdigest()[:20]
        path=root/'summaries'/f'{stamp}.json'
        changed=not path.exists()
        if changed:_exclusive_json(path,summary)
        return {**summary,'new_scored_decisions':new,'attention':bool(new or (pending and changed))}


def tick(registry,lake,*,acquire=True):
    root=Path(registry);p=_verify(root)
    now=_now().astimezone(TZ);today=pd.Timestamp(now.date())
    if today<pd.Timestamp(p['start_date']):
        return {'status':'WAITING_START','start_date':p['start_date'],'attention':False}
    calendar=_calendar(root,lake);sessions,anchors=_schedule(p,calendar)
    if acquire and (today>pd.Timestamp(calendar.cal_date.max()) or int((sessions>today).sum())<45):
        try:
            if not refresh(lake,calendar_only=True)['required_complete']:raise ValueError('calendar acquisition incomplete')
        except Exception as exc:return {'status':'CALENDAR_ACQUISITION_FAILED','detail':str(exc),'attention':True}
        calendar=_calendar(root,lake);sessions,anchors=_schedule(p,calendar)
    if today>pd.Timestamp(calendar.cal_date.max()):return {'status':'MISSING_CALENDAR','attention':True}
    day_row=calendar[calendar.cal_date.eq(today)]
    if day_row.empty:return {'status':'MISSING_CALENDAR_DAY','attention':True}
    if len(day_row) and not bool(pd.to_numeric(day_row.is_open).iloc[0]):
        result=record(root,lake)
        return {'status':'NON_TRADING_DAY','record':result,'attention':result.get('attention',False)}
    due=today in anchors
    if due and now.time()>=datetime.strptime(p['deadline_local'],'%H:%M').time():
        result=record(root,lake)
        return {'record':result,'score':score(root,lake),'attention':result.get('attention',False)}
    if acquire:
        try:
            if not refresh(lake,calendar_only=False,with_holdings=due)['required_complete']:raise ValueError('required input acquisition incomplete')
        except Exception as exc:
            detail=f'{type(exc).__name__}: {exc}'
            if due:
                path=root/'decisions'/today.strftime('%Y-%m-%d')/'receipt.json'
                with _lock(root):
                    if not path.exists():_exclusive_json(path,{'status':'MISSING_DATA','decision_date':str(today.date()),
                        'observed_at':_now().isoformat(),'protocol_hash':p['protocol_hash'],'detail':detail})
            return {'status':'ACQUISITION_FAILED','detail':detail,'attention':True}
    result=record(root,lake);grading=score(root,lake)
    return {'record':result,'score':grading,'attention':bool(result.get('attention') or grading.get('attention'))}
