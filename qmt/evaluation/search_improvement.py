"""Opt-in, baseline-anchored GP/RL ablations motivated by archived diagnostics.

Historical development study, not untouched confirmation or fee-net certification.
Original search_benchmark and production defaults are not changed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmt.evaluation.platform import LEAVES,LIMITATIONS,_hash_file,_interval,_json,_load_protocol,_runtime_versions,verify_snapshot
from qmt.evaluation.search_benchmark import CacheGP,CacheRL,UniverseDSL,SEEDS,SEED_FORMULAS,bounded_labels,dump,fit_config,joint_test
from qmt.evaluation.search_diagnostics import load_inputs,rank_ic_frame
from qmt.factor.rl_operators import OperatorLearningConfig
from qmt.factor.types import BaseFactorPanel,SearchLedger,ValidationScheme,ValidationSplit

METHODS=('gp','q','q_no_novelty','gp_incremental','q_incremental')
ANCHOR_WEIGHT=.25
MIN_RESIDUAL_RATIO=.05


def anchored_scores(scores,baseline,weight=ANCHOR_WEIGHT,min_residual_ratio=MIN_RESIDUAL_RATIO):
    """Add only contemporaneously orthogonal rank information to low-volatility.

    No labels or rolling sparse-row operators. A copy, inverse, constant, or
    near-clone of baseline contributes zero; invalid eligible scores stay invalid.
    """
    eligible=pd.DataFrame(np.isfinite(baseline),index=baseline.index,columns=baseline.columns)
    b=baseline.where(eligible).rank(axis=1,pct=True)
    finite=pd.DataFrame(np.isfinite(scores),index=scores.index,columns=scores.columns)
    c=scores.where(eligible & finite).rank(axis=1,pct=True)
    b=b.sub(b.mean(axis=1),axis=0);c=c.sub(c.mean(axis=1),axis=0)
    b2=(b*b).sum(axis=1)
    beta=(b*c).sum(axis=1)/b2.where(b2>0)
    residual=c-b.mul(beta,axis=0)
    r2=(residual*residual).sum(axis=1)
    c2=(c*c).sum(axis=1)
    ratio=np.sqrt(r2/c2.where(c2>0))
    active=(ratio>=min_residual_ratio)&(r2>1e-16)&(b2>0)
    scale=np.sqrt(b2/r2.where(active)).where(active,0.)
    contribution=residual.mul(scale,axis=0)
    result=b+weight*contribution
    # Inactive constant/clone candidates fall back to baseline, but missing
    # values on actual eligible assets cannot be repaired by the anchor.
    complete=(finite|~eligible).all(axis=1)
    # Explicit broadcasting avoids pandas constructing one Series per fund for
    # a row mask. The boolean condition and floating-point results are identical.
    row_mask=np.broadcast_to(complete.to_numpy()[:,None],result.shape)
    return result.where(eligible & row_mask)


class AnchoredDSL(UniverseDSL):
    def evaluate_frame(self,expr,panel):
        return anchored_scores(super().evaluate_frame(expr,panel),-panel.factors['vol60'])

    def evaluate(self,expr,panel,as_of):
        frame=self.evaluate_frame(expr,panel)
        return frame.loc[frame.index<=pd.Timestamp(as_of)].iloc[-1].dropna()


class IncrementalFitness:
    """Score actual anchored IC gain with variability of PAIRED fold differences."""
    def _before_search(self,cfg,ledger):
        super()._before_search(cfg,ledger)
        self._increment_context=None

    def _fitness(self,expr,panel,labmap,select_dates,ledger):
        if expr.text in self._fit_cache:return self._fit_cache[expr.text]
        ledger.view_dataset()
        if self._increment_context is None:
            ds=pd.DatetimeIndex(select_dates)
            y=pd.DataFrame([labmap[d] for d in select_dates],index=ds).reindex(columns=panel.funds)
            common=panel.factors['vol60'].reindex(ds).notna()
            config=self._evaluation_config
            base=rank_ic_frame(-panel.factors['vol60'].reindex(ds),y,common,config)
            self._increment_context=(ds,y,common,base)
        ds,y,common,base=self._increment_context
        scores=self._dsl.evaluate_frame(expr,panel).reindex(ds)
        delta=rank_ic_frame(scores,y,common,self._evaluation_config)-base
        values=[]
        for positions in np.array_split(np.arange(len(ds)),3 if len(ds)>=6 else 1):
            part=delta.iloc[positions]
            if part.notna().sum()<int(np.ceil(len(part)*self._min_train_ic_coverage)):
                self._fit_cache[expr.text]=-np.inf;return -np.inf
            values.append(part.mean())
        score=float(np.mean(values)-.5*np.std(values)-.003*max(0,expr.complexity-2))
        self._fit_cache[expr.text]=score
        return score


class IncrementalGP(IncrementalFitness,CacheGP):pass
class IncrementalRL(IncrementalFitness,CacheRL):pass


def phase_dates(decisions,sessions,start_year,end_year,deadline,horizon=40):
    """Annual anchors with full same-year labels; no year-end overlapping add-ons."""
    ds=decisions[(decisions.year>=start_year)&(decisions.year<=end_year)]
    positions=sessions.get_indexer(ds)+horizon
    valid=positions<len(sessions)
    ds,positions=ds[valid],positions[valid]
    exits=sessions[positions]
    return ds[(exits.year==ds.year)&(exits<=deadline)]


def paired_gate(delta,*,min_dates=12,repetitions=499,block=2,seed=20261016):
    """Independent-validation simultaneous lower bound, plus calendar consistency.

    Columns are frozen discovery candidates; all use the same dates and labels.
    Candidate construction never sees validation outcomes. Near-zero/negative
    improvements cannot pass simply by being stable.
    """
    if len(delta)<min_dates or delta.index.year.nunique()<3:
        return None,{'status':'INSUFFICIENT_INDEPENDENT_VALIDATION','n_dates':len(delta),'min_dates':min_dates}
    valid=delta.notna().all(axis=0)
    candidates=delta.loc[:,valid]
    if candidates.empty:return None,{'status':'NO_COMPLETE_CANDIDATE','n_dates':len(delta)}
    data=candidates.to_numpy(dtype=float);n,m=data.shape
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-block+1,size=(repetitions,int(np.ceil(n/block))))
    idx=(starts[...,None]+np.arange(block)).reshape(repetitions,-1)[:,:n]
    boot=data[idx].mean(axis=1);mu=data.mean(axis=0);se=boot.std(axis=0,ddof=1)
    variable=se>1e-12
    centered=np.divide(boot-mu,se,out=np.zeros_like(boot),where=variable)
    critical=float(np.quantile(centered.max(axis=1),.95))
    lower=mu-critical*se
    # Degenerate distributions are not sufficient inference evidence.
    lower[~variable]=np.minimum(0.,mu[~variable])
    years=candidates.groupby(candidates.index.year).mean()
    consistent=years.gt(1e-12).sum(axis=0)>=int(np.ceil(2*len(years)/3))
    passed=(lower>1e-12)&consistent.to_numpy()
    winner=int(np.argmax(np.where(passed,mu,-np.inf))) if passed.any() else None
    audit={'status':'ACCEPT' if winner is not None else 'KEEP_BASELINE','n_dates':n,
           'years':len(years),'family_size':m,'block_decisions':block,'bootstrap_repetitions':repetitions,
           'candidates':[{'formula':name,'mean_paired_delta':mu[j],'simultaneous_lower95':lower[j],
                          'positive_years':int(years[name].gt(1e-12).sum()),'accepted':bool(passed[j])}
                         for j,name in enumerate(candidates.columns)]}
    return candidates.columns[winner] if winner is not None else None,audit


def select_candidates(pool_ics,base_ic,valid_dates,mode):
    ds=pd.DatetimeIndex(valid_dates)
    base=base_ic.reindex(ds).dropna()
    pool=pool_ics.reindex(base.index)
    if mode in ('short_control','long_control'):
        if mode=='short_control':
            pool=pool[pool.index.year==ds.year.max()];base=base.reindex(pool.index)
        if len(base)<3:return None,{'status':'INSUFFICIENT_VALIDATION','n_dates':len(base)}
        best=None;value=float(base.mean()-.5*base.std(ddof=0));audit=[]
        for name,part in pool.items():
            score=float(part.mean()-.5*part.std(ddof=0)) if part.notna().all() else -np.inf
            audit.append({'formula':name,'robust_validation_ic':score})
            if score>value+1e-12:best,value=name,score
        return best,{'status':'ACCEPT' if best else 'KEEP_BASELINE','n_dates':len(base),'candidates':audit}
    return paired_gate(pool.sub(base,axis=0))


def run(snapshot,baseline_protocol,output,*,budget=120,seeds=SEEDS,methods=METHODS):
    output=Path(output);manifest=verify_snapshot(snapshot)
    frozen,config=_load_protocol(baseline_protocol,manifest['snapshot_hash'])
    output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[2]
    code_files=[source_root/p for p in ('qmt/evaluation/search_improvement.py','qmt/evaluation/search_diagnostics.py',
      'qmt/evaluation/search_benchmark.py','qmt/evaluation/platform.py','qmt/factor/auto_search.py',
      'qmt/factor/rl_search.py','qmt/factor/rl_operators.py','qmt/factor/types.py','qmt/factor/dsl.py',
      'qmt/factor/diversity.py','qmt/factor/metrics.py','qmt/factor/panel.py','qmt/stats.py')]
    registered={'created_at':datetime.now(timezone.utc).isoformat(),'snapshot_hash':manifest['snapshot_hash'],
      'baseline_protocol_hash':frozen['protocol_hash'],'methods':list(methods),'seeds':list(seeds),'budget':budget,
      'search_config':{**asdict(fit_config(seeds[0],budget)),'allowed_operators':sorted(fit_config(seeds[0],budget).allowed_operators)},
      'discovery':'three calendar years outer-6..outer-4; no validation features/labels in search',
      'validation':'three independent years outer-3..outer-1; short control uses only latest one of same independent years',
      'date_policy':'same-year full-horizon anchors only; 40-session purge, one-session embargo, endpoint publication cutoff',
      'reward_ablation':'q vs q_no_novelty differ only novelty_bonus .01 vs 0; q_incremental also uses actual anchored paired IC fitness',
      'anchoring':{'residual_weight':ANCHOR_WEIGHT,'minimum_residual_std_ratio':MIN_RESIDUAL_RATIO,
                   'policy':'cross-sectional rank residual orthogonal to baseline; normalize residual RMS; no outcome inputs'},
      'selectors':['short_control','long_control','gated','anchored_gated (raw-fitness methods only)'],
      'gate':{'min_dates':12,'min_years':3,'bootstrap':499,'block':2,'family_lower_bound':.95,'positive_year_fraction':2/3},
      'stability':'date coverage>=.8; at least ceil(.7*year_count) positive years; all seed means positive; joint p<.05 and simultaneous lower95>0',
      'evidence':'historical development ablation; reused viewed years; no untouched confirmation or fee-net approval',
      'code_hashes':{str(p):_hash_file(p) for p in code_files},'runtime':_runtime_versions(),'limitations':LIMITATIONS}
    registered['protocol_hash']=hashlib.sha256(_json(registered).encode()).hexdigest()
    dump(output/'protocol.json',registered)
    print('REGISTERED',registered['protocol_hash'],flush=True)
    x=load_inputs(snapshot,baseline_protocol,output/'inputs',first_year=2009)
    features,y,known,common,sessions,outer=(x[k] for k in ('features','outcomes','known_at','common','sessions','outer'))
    decisions=y.index;codes=list(y.columns)
    base_ic=rank_ic_frame(-features['vol60'],y,common,config)
    rows=[];selections=[]
    for year,test_dates in outer.items():
        valid_start=sessions[sessions.year==year-3][0];test_start=sessions[sessions.year==year][0]
        train_deadline=sessions[sessions.get_loc(valid_start)-2];valid_deadline=sessions[sessions.get_loc(test_start)-2]
        train_ds=phase_dates(decisions,sessions,year-6,year-4,train_deadline)
        valid_ds=phase_dates(decisions,sessions,year-3,year-1,valid_deadline)
        labels,good_train=bounded_labels(y,known,train_ds,train_deadline,common,config)
        if len(good_train)<6:raise ValueError(f'{year}: insufficient discovery dates')
        fit_panel=BaseFactorPanel([d.date() for d in sessions[(sessions.year>=year-6)&(sessions<valid_start)]],codes,
                                 {k:v.loc[train_ds] for k,v in features.items()})
        split=ValidationSplit(str(year),ValidationScheme.ROLLING,(good_train[0].date(),good_train[-1].date()),
                              (valid_start.date(),valid_deadline.date()),embargo_days=40)
        # Independent validation and outer score features are prepared only for
        # evaluation; neither frame is given to any searcher or controller.
        score_dates=valid_ds.union(test_dates)
        score_panel=BaseFactorPanel([d.date() for d in score_dates],codes,{k:v.loc[score_dates] for k,v in features.items()})
        valid_y=y.loc[valid_ds].where(known.loc[valid_ds].le(valid_deadline))
        valid_base=rank_ic_frame(-features['vol60'].loc[valid_ds],valid_y,common.loc[valid_ds],config)
        for method in methods:
            for seed in seeds:
                incremental=method.endswith('incremental')
                dsl=AnchoredDSL() if incremental else UniverseDSL()
                learning=OperatorLearningConfig(novelty_bonus=0.) if method in ('q_no_novelty','q_incremental') else None
                if method in ('gp','gp_incremental'):
                    cls=IncrementalGP if incremental else CacheGP;searcher=cls(dsl)
                else:
                    cls=IncrementalRL if incremental else CacheRL;searcher=cls(dsl,mode='q',learning_config=learning)
                searcher._evaluation_config=config
                ledger=SearchLedger(locked_test_span=(test_start.date(),sessions[sessions.year==year][-1].date()))
                candidates=searcher.search(fit_config(seed,budget),dsl,fit_panel,labels,split,ledger)
                if ledger.n_trials!=budget:raise RuntimeError('proposal budget was not exhausted')
                exprs={e.text:e for e in candidates if e.text!='neg(vol60)'}
                pool={name:dsl.evaluate_frame(expr,score_panel) for name,expr in exprs.items()}
                pool_ics=pd.DataFrame({name:rank_ic_frame(s.loc[valid_ds],valid_y,common.loc[valid_ds],config) for name,s in pool.items()},index=valid_ds)
                variants=['short_control','long_control','gated']+([] if incremental else ['anchored_gated'])
                trace={'year':year,'method':method,'seed':seed,'train_dates':[str(d.date()) for d in good_train],
                       'validation_dates':[str(d.date()) for d in valid_ds],
                       'train_label_deadline':train_deadline,'validation_label_deadline':valid_deadline,
                       'trials':[asdict(t) for t in ledger.trials],
                       'controller':searcher.policy_audit() if hasattr(searcher,'policy_audit') else None,'selectors':{}}
                for variant in variants:
                    variant_pool=pool;variant_ics=pool_ics
                    if variant=='anchored_gated':
                        variant_pool={name:anchored_scores(s,-features['vol60'].loc[score_dates]) for name,s in pool.items()}
                        variant_ics=pd.DataFrame({name:rank_ic_frame(s.loc[valid_ds],valid_y,common.loc[valid_ds],config) for name,s in variant_pool.items()},index=valid_ds)
                    chosen,audit=select_candidates(variant_ics,valid_base,valid_ds,variant)
                    trace['selectors'][variant]={'selected':chosen or 'neg(vol60)',**audit}
                    arm=f'{method}__{variant}'
                    scores=variant_pool[chosen].loc[test_dates] if chosen else -features['vol60'].loc[test_dates]
                    ic=rank_ic_frame(scores,y.loc[test_dates],common.loc[test_dates],config)
                    selections.append({'year':year,'method':method,'seed':seed,'selector':variant,'arm':arm,
                                       'selected':chosen or 'neg(vol60)','baseline_retained':chosen is None,
                                       'train_dates':len(good_train),'validation_ic_dates':audit['n_dates']})
                    for d in test_dates:
                        rows.append({'decision_date':d,'year':year,'method':method,'selector':variant,'arm':arm,'seed':seed,
                                     'rank_ic':ic.loc[d],'baseline_ic':base_ic.loc[d],
                                     'delta_ic':ic.loc[d]-base_ic.loc[d],'selected':chosen or 'neg(vol60)',
                                     'baseline_retained':chosen is None,'eligible_funds':int(common.loc[d].sum())})
                dump(output/f'{year}-{method}-{seed}.json',trace)
                pd.DataFrame(rows).to_parquet(output/'metrics.partial.parquet',index=False)
                print(f'{year} {method} seed={seed} trials={ledger.n_trials}',flush=True)
    metrics=pd.DataFrame(rows);metrics.to_parquet(output/'metrics.parquet',index=False)
    selections=pd.DataFrame(selections);selections.to_parquet(output/'selections.parquet',index=False)
    temporal=metrics.groupby(['decision_date','arm']).agg(
        rank_ic=('rank_ic',lambda a:a.mean() if a.notna().all() else np.nan),
        delta_ic=('delta_ic',lambda a:a.mean() if a.notna().all() else np.nan)).reset_index()
    family=joint_test(temporal.pivot(index='decision_date',columns='arm',values='delta_ic'))
    summary={}
    for arm,part in temporal.groupby('arm'):
        yearly=part.groupby(part.decision_date.dt.year).delta_ic.mean()
        seed_means=metrics[metrics.arm.eq(arm)].groupby('seed').delta_ic.mean()
        inference=family.get('arms',{}).get(arm,{})
        coverage=part.delta_ic.notna().sum()/sum(map(len,outer.values()))
        enough=coverage>=.8 and yearly.gt(1e-12).sum()>=int(np.ceil(.7*len(outer))) and seed_means.gt(0).all()
        stable=enough and inference.get('family_adjusted_one_sided_p',1)<.05 and inference.get('simultaneous_lower95',-1)>0
        summary[arm]={'rank_ic':_interval(part.rank_ic,config),'paired_delta_ic':_interval(part.delta_ic,config),
                      'positive_years':int(yearly.gt(1e-12).sum()),'year_delta_ic':yearly.to_dict(),
                      'seed_delta_ic':seed_means.to_dict(),'baseline_retained_year_fraction':selections.loc[selections.arm.eq(arm),'baseline_retained'].mean(),
                      'stable_ranking_improvement':bool(stable),'date_coverage':coverage,**inference}
    report={'status':'COMPLETE_DEVELOPMENT_ABLATION','protocol_hash':registered['protocol_hash'],
            'snapshot_hash':manifest['snapshot_hash'],'baseline_ic':_interval(metrics.drop_duplicates('decision_date').baseline_ic,config),
            'summary':summary,'family_test':family,'searches':len(outer)*len(methods)*len(seeds),
            'proposals':len(outer)*len(methods)*len(seeds)*budget,
            'alpha_approved':False,'fee_net_status':'NOT_RUN_MISSING_REVIEWED_EXECUTION_INPUTS',
            'historical_periods_reused':True,'limitations':LIMITATIONS,
            'files':{n:_hash_file(output/n) for n in ('metrics.parquet','selections.parquet')}}
    if registered['code_hashes']!={str(p):_hash_file(p) for p in code_files}:raise RuntimeError('code changed during study')
    dump(output/'report.json',report);(output/'metrics.partial.parquet').unlink()
    print('COMPLETE',flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('protocol');p.add_argument('output')
    p.add_argument('--budget',type=int,default=120);a=p.parse_args();run(a.snapshot,a.protocol,a.output,budget=a.budget)

if __name__=='__main__':main()
