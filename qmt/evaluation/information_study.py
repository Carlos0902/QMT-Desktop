"""Ten pre-registered new-information arms; same ex-ante universe and safe fallback."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from qmt.evaluation.platform import LEAVES,LIMITATIONS,_hash_file,_json,_interval,_runtime_versions
from qmt.evaluation.information_features import GROUPS,load
from qmt.evaluation.search_benchmark import UniverseDSL,SEEDS,fit_config,dump
from qmt.evaluation.search_improvement import anchored_scores,IncrementalGP,IncrementalRL,phase_dates,paired_gate
from qmt.evaluation.search_diagnostics import rank_ic_frame
from qmt.factor.rl_operators import OperatorLearningConfig
from qmt.factor.types import BaseFactorPanel,SearchLedger,ValidationSplit,ValidationScheme
from qmt.evaluation.search_benchmark import bounded_labels,joint_test

ARMS=tuple(f'ridge_{g}' for g in GROUPS)+tuple(f'{method}_{g}' for method in ('gp','uniform','q') for g in ('F0','F3'))


def safe_anchor(raw,baseline):
    score=anchored_scores(raw,baseline)
    eligible=baseline.notna()
    nonfinite=((~np.isfinite(raw))&eligible).any(axis=1)
    flat=raw.where(eligible).nunique(axis=1).le(1)
    bad=nonfinite|flat|((~np.isfinite(score))&eligible).any(axis=1)
    reason=pd.Series('CANDIDATE',index=baseline.index)
    reason.loc[flat]='CONSTANT_SCORE';reason.loc[nonfinite]='INVALID_SCORE'
    score.loc[bad]=baseline.loc[bad]
    return score,reason


class SafeDSL(UniverseDSL):
    def evaluate_frame(self,expr,panel):
        raw=super().evaluate_frame(expr,panel)
        return safe_anchor(raw,-panel.factors['vol60'])[0]
    def evaluate(self,expr,panel,as_of):
        frame=self.evaluate_frame(expr,panel)
        return frame.loc[frame.index<=pd.Timestamp(as_of)].iloc[-1].dropna()


def rank_inputs(features,fields,common):
    return {k:features[k].where(common).rank(axis=1,pct=True).sub(
        features[k].where(common).rank(axis=1,pct=True).mean(axis=1),axis=0) for k in fields}


def fit_ridge(features,fields,outcomes,common,dates):
    ranked=rank_inputs(features,fields,common);xx=[];yy=[];ww=[]
    for d in dates:
        y=outcomes.loc[d].where(common.loc[d]).dropna()
        if len(y)<20:continue
        values=np.column_stack([ranked[k].loc[d].reindex(y.index).to_numpy() for k in fields])
        if not np.isfinite(values).all():raise ValueError('nonfinite linear model training features')
        target=y.rank(pct=True);target=target-target.mean()
        xx.append(values);yy.append(target.to_numpy());ww.append(np.full(len(y),1/len(y)))
    if not xx:raise ValueError('no mature linear training samples')
    weights=np.concatenate(ww)/len(xx)
    model=Ridge(alpha=1.,fit_intercept=False).fit(np.concatenate(xx),np.concatenate(yy),sample_weight=weights)
    return {'kind':'ridge','fields':list(fields),'coefficients':model.coef_.tolist(),'alpha':1.,'training_dates':len(xx)}


def predict_raw(model,features,common):
    if model['kind']=='ridge':
        ranked=rank_inputs(features,model['fields'],common)
        values=sum(ranked[k]*v for k,v in zip(model['fields'],model['coefficients']))
        return values
    fields=model['fields'];panel=BaseFactorPanel([d.date() for d in common.index],list(common.columns),
        {k:features[k].where(common) for k in fields})
    dsl=UniverseDSL()
    return dsl.evaluate_frame(dsl.parse(model['formula']),panel)


def model_scores(model,features,common):
    base=-features['vol60'].where(common)
    if model.get('kind')=='baseline' or not model.get('active',False):
        return base,pd.Series('NOT_ADOPTED',index=base.index)
    return safe_anchor(predict_raw(model,features,common),base)


def prepare_protocol(feature_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    feature_manifest=json.loads((Path(feature_path)/'manifest.json').read_text())
    root=Path(__file__).resolve().parents[2]
    files=['qmt/evaluation/information_study.py','qmt/evaluation/information_features.py',
      'qmt/evaluation/search_improvement.py','qmt/evaluation/search_diagnostics.py','qmt/evaluation/search_benchmark.py',
      'qmt/factor/auto_search.py','qmt/factor/rl_search.py','qmt/factor/rl_operators.py','qmt/factor/dsl.py',
      'qmt/factor/types.py','qmt/stats.py']
    protocol={'kind':'new_information_study_v1','created_at':datetime.now(timezone.utc).isoformat(),
      'feature_path':str(Path(feature_path).resolve()),'feature_manifest_hash':feature_manifest['manifest_hash'],
      'arms':ARMS,'seeds':SEEDS,'groups':GROUPS,'max_searches':180,'max_proposals':21600,'proposal_budget':120,
      'outer_years':list(range(2015,2025)),'discovery_years':3,'independent_validation_years':3,
      'horizon_sessions':40,'signal_lag':2,'anchor_weight':.25,'linear_alpha':1.,'linear_date_equal_weight':True,
      'adoption':{'min_discovery_dates':12,'min_validation_dates':12,'min_validation_years':3,
         'candidate_family_lower95':True,'validation_positive_year_fraction':2/3,'bootstrap_reps':499,'block':2},
      'all_arms_same_ex_ante_union_feature_intersection':True,'formula_invalid_or_flat':'fallback baseline before seeing outcomes',
      'primary_metric':'paired delta rank IC','beta_adjusted_metric':'diagnostic only; never used for fitting/adoption',
      'advance':{'min_mean_delta':.01,'family_lower95_above':0,'positive_years':7,'all_search_seed_means_positive':True,'min_date_coverage':.8},
      'forward_primary':'ridge_F3','forward_models':'reuse last historical fold; no extra searches or forward retraining',
      'code_hashes':{str(root/f):_hash_file(root/f) for f in files},'runtime':_runtime_versions(),'limitations':LIMITATIONS}
    protocol['protocol_hash']=hashlib.sha256(_json(protocol).encode()).hexdigest()
    dump(output/'protocol.json',protocol)
    return protocol


def fit_fold(x,year,*,journal_dir=None,seed_only=None):
    features=x['features'];common=x['common'];y=x['outcomes'];known=x['known_at'];sessions=x['sessions'];config=x['config']
    ds=y.index;codes=list(y.columns);start=sessions[sessions.year==year-3][0];test=sessions[sessions.year==year][0]
    train_deadline=sessions[sessions.get_loc(start)-2];valid_deadline=sessions[sessions.get_loc(test)-2]
    train=phase_dates(ds,sessions,year-6,year-4,train_deadline)
    valid=phase_dates(ds,sessions,year-3,year-1,valid_deadline)
    labels,good=bounded_labels(y,known,train,train_deadline,common,config)
    mature_train=y.where(known.le(train_deadline))
    mature_valid=y.loc[valid].where(known.loc[valid].le(valid_deadline))
    base=-features['vol60'].where(common)
    valid_base=rank_ic_frame(base.loc[valid],mature_valid,common.loc[valid],config)
    adequate=len(good)>=12 and valid_base.notna().sum()>=12 and valid_base.dropna().index.year.nunique()>=3
    models={};trials=0;searches=0
    for group,fields in GROUPS.items():
        arm=f'ridge_{group}'
        if not adequate:model={'kind':'baseline','active':False,'reason':'INSUFFICIENT_DISCOVERY_OR_VALIDATION'}
        else:
            model=fit_ridge(features,fields,mature_train,common,good)
            model['active']=True
            score,_=model_scores(model,{k:v.loc[valid] for k,v in features.items()},common.loc[valid])
            ic=rank_ic_frame(score,mature_valid,common.loc[valid],config)
            chosen,audit=paired_gate(pd.DataFrame({arm:ic-valid_base}).loc[valid_base.notna()])
            model.update(active=chosen is not None,adoption=audit)
        models[arm]={'0':model}
        if journal_dir:dump(Path(journal_dir)/f'{year}-{arm}.json',model)
    for method in ('gp','uniform','q'):
        for group in ('F0','F3'):
            fields=GROUPS[group];arm=f'{method}_{group}';models[arm]={}
            for seed in ([seed_only] if seed_only is not None else SEEDS):
                if not adequate:
                    model={'kind':'baseline','active':False,'reason':'INSUFFICIENT_DISCOVERY_OR_VALIDATION'}
                    audit={'trials':[],'controller':None}
                else:
                    dsl=SafeDSL();cls=IncrementalGP if method=='gp' else IncrementalRL
                    searcher=cls(dsl) if method=='gp' else cls(dsl,mode=method,learning_config=OperatorLearningConfig(novelty_bonus=0))
                    searcher._evaluation_config=config
                    cfg=replace(fit_config(seed,120),allowed_fields=list(fields))
                    panel=BaseFactorPanel([d.date() for d in sessions[(sessions.year>=year-6)&(sessions<start)]],codes,
                        {k:features[k].where(common).loc[train] for k in fields})
                    split=ValidationSplit(str(year),ValidationScheme.ROLLING,(good[0].date(),good[-1].date()),
                        (start.date(),valid_deadline.date()),embargo_days=40)
                    ledger=SearchLedger(locked_test_span=(test.date(),sessions[sessions.year==year][-1].date()))
                    candidates=searcher.search(cfg,dsl,panel,labels,split,ledger)
                    if ledger.n_trials!=120:raise RuntimeError('proposal budget not exhausted')
                    searches+=1;trials+=ledger.n_trials
                    vp=BaseFactorPanel([d.date() for d in valid],codes,{k:features[k].where(common).loc[valid] for k in fields})
                    pool={e.text:rank_ic_frame(dsl.evaluate_frame(e,vp),mature_valid,common.loc[valid],config) for e in candidates}
                    delta=pd.DataFrame(pool,index=valid).sub(valid_base,axis=0).loc[valid_base.notna()]
                    chosen,adoption=paired_gate(delta)
                    model={'kind':'formula','fields':list(fields),'formula':chosen or 'neg(vol60)','active':chosen is not None,'adoption':adoption}
                    audit={'trials':[t.__dict__ for t in ledger.trials],
                           'controller':searcher.policy_audit() if hasattr(searcher,'policy_audit') else None}
                models[arm][str(seed)]=model
                if journal_dir:dump(Path(journal_dir)/f'{year}-{arm}-{seed}.json',{'model':model,**audit})
                print(year,arm,seed,'active',model['active'],flush=True)
    return models,{'train_dates':len(good),'validation_dates':int(valid_base.notna().sum()),'searches':searches,'proposals':trials}


def _fold_job(feature_path,year,output):
    return fit_fold(load(feature_path),year,journal_dir=output)


def run(feature_path,output,*,workers=3):
    output=Path(output);p=prepare_protocol(feature_path,output);x=load(feature_path)
    if not x['manifest']['quality_gate']:
        report={'status':'BLOCKED_DATA_QUALITY','protocol_hash':p['protocol_hash'],'searches':0,'proposals':0,
          'arms':{a:'NOT_RUN_DATA_GATE' for a in ARMS},'coverage':x['manifest']['feature_date_coverage'],'alpha_approved':False}
        dump(output/'report.json',report);return report
    rows=[];choices=[];folds=[];last_models=None
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) if workers>1 else None
    jobs={year:pool.submit(_fold_job,str(feature_path),year,str(output)) for year in x['outer']} if pool else {}
    for year,test_dates in x['outer'].items():
        models,audit=jobs[year].result() if pool else fit_fold(x,year,journal_dir=output)
        folds.append({'year':year,**audit});last_models=models
        features={k:v.loc[test_dates] for k,v in x['features'].items()};common=x['common'].loc[test_dates]
        base=rank_ic_frame(-features['vol60'].where(common),x['outcomes'].loc[test_dates],common,x['config'])
        broad=rank_ic_frame(-features['vol60'],x['outcomes'].loc[test_dates],x['full_common'].loc[test_dates],x['config'])
        for arm,seed_models in models.items():
            for seed,model in seed_models.items():
                scores,reasons=model_scores(model,features,common)
                ic=rank_ic_frame(scores,x['outcomes'].loc[test_dates],common,x['config'])
                # Market-adjusted future labels are diagnostic, not a training target.
                index=pd.read_parquet(Path(feature_path)/'index.parquet').set_index('trade_date').close.reindex(x['sessions'])
                positions=x['sessions'].get_indexer(test_dates)
                market_outcome=index.iloc[positions+40].to_numpy()/index.iloc[positions].to_numpy()-1
                adjusted=x['outcomes'].loc[test_dates]-features['market_beta120'].mul(pd.Series(market_outcome,index=test_dates),axis=0)
                adjusted_ic=rank_ic_frame(scores,adjusted,common,x['config'])
                for d in test_dates:
                    rows.append({'decision_date':d,'year':year,'arm':arm,'seed':int(seed),'rank_ic':ic.loc[d],
                        'baseline_ic':base.loc[d],'broad_baseline_ic':broad.loc[d],'delta_ic':ic.loc[d]-base.loc[d],
                        'beta_adjusted_ic_diagnostic':adjusted_ic.loc[d],'baseline_retained':not model.get('active',False),
                        'fallback_reason':reasons.loc[d],'eligible_funds':int(common.loc[d].sum())})
                choices.append({'year':year,'arm':arm,'seed':int(seed),'active':model.get('active',False),**audit})
        pd.DataFrame(rows).to_parquet(output/'metrics.partial.parquet',index=False)
    if pool:pool.shutdown()
    metrics=pd.DataFrame(rows);metrics.to_parquet(output/'metrics.parquet',index=False)
    pd.DataFrame(choices).to_parquet(output/'selections.parquet',index=False)
    dump(output/'frozen_models.json',{'models':last_models,'training_fold':2024,'primary':'ridge_F3','protocol_hash':p['protocol_hash'],
        'no_additional_forward_searches':True,'feature_manifest_hash':x['manifest']['manifest_hash']})
    temporal=metrics.groupby(['decision_date','arm']).agg(
      rank_ic=('rank_ic',lambda s:s.mean() if s.notna().all() else np.nan),
      delta_ic=('delta_ic',lambda s:s.mean() if s.notna().all() else np.nan)).reset_index()
    family=joint_test(temporal.pivot(index='decision_date',columns='arm',values='delta_ic'))
    summary={}
    for arm,part in temporal.groupby('arm'):
        years=part.groupby(part.decision_date.dt.year).delta_ic.mean()
        seeds=metrics[metrics.arm.eq(arm)].groupby('seed').delta_ic.mean()
        f=family.get('arms',{}).get(arm,{})
        coverage=part.delta_ic.notna().sum()/59
        passed=(part.delta_ic.mean()>=.01 and years.gt(1e-12).sum()>=7 and seeds.gt(0).all() and coverage>=.8
                and f.get('simultaneous_lower95',-1)>0 and f.get('family_adjusted_one_sided_p',1)<.05)
        summary[arm]={'rank_ic':_interval(part.rank_ic,x['config']),'paired_delta_ic':_interval(part.delta_ic,x['config']),
          'positive_years':int(years.gt(1e-12).sum()),'seed_delta_ic':seeds.to_dict(),
          'adopted_seed_years':sum(c['active'] for c in choices if c['arm']==arm),'date_coverage':coverage,
          'advance':bool(passed),**f}
    if any(_hash_file(Path(n))!=h for n,h in p['code_hashes'].items()):raise RuntimeError('study code changed')
    report={'status':'COMPLETE_DEVELOPMENT_STUDY','protocol_hash':p['protocol_hash'],'feature_manifest_hash':p['feature_manifest_hash'],
      'summary':summary,'family_test':family,'folds':folds,'searches':sum(f['searches'] for f in folds),
      'proposals':sum(f['proposals'] for f in folds),'alpha_approved':False,'historical_periods_reused':True,
      'files':{n:_hash_file(output/n) for n in ('metrics.parquet','selections.parquet','frozen_models.json')}}
    dump(output/'report.json',report);(output/'metrics.partial.parquet').unlink()
    return report
