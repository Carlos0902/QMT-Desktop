"""Post-hoc attribution of the frozen search study; oracle results are diagnostics only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmt.evaluation.platform import LEAVES, _features_and_outcomes, _hash_file, _load_protocol, verify_snapshot
from qmt.evaluation.search_benchmark import UniverseDSL, dump, label_availability
from qmt.factor.types import BaseFactorPanel


def load_inputs(snapshot_path, protocol_path, cache_dir, *, first_year=2010):
    path, cache = Path(snapshot_path), Path(cache_dir)
    manifest = verify_snapshot(path)
    frozen, config = _load_protocol(protocol_path, manifest['snapshot_hash'])
    sessions = pd.DatetimeIndex(pd.read_parquet(path/'calendar.parquet').trade_date)
    outer = {int(f['fold_id']): pd.DatetimeIndex(f['decisions']) for f in frozen['folds']}
    if cache.exists():
        receipt = json.loads((cache/'manifest.json').read_text())
        if receipt['snapshot_hash'] != manifest['snapshot_hash'] or receipt['baseline_protocol_hash'] != frozen['protocol_hash'] or receipt['first_year'] != first_year:
            raise ValueError('feature-cache identity mismatch')
        if any(_hash_file(cache/name) != digest for name,digest in receipt['files'].items()):
            raise ValueError('feature-cache artifact hash mismatch')
        features = {key: pd.read_parquet(cache/f'{key}.parquet') for key in LEAVES}
        outcomes, known = (pd.read_parquet(cache/f'{key}.parquet') for key in ('outcomes','known_at'))
    else:
        cache.mkdir(parents=True)
        dates = set(d for ds in outer.values() for d in ds)
        for year in range(first_year, max(outer)+1):
            annual = sessions[(sessions.year==year) & (np.arange(len(sessions))+config.horizon_sessions<len(sessions))]
            dates.update(annual[::config.decision_stride_sessions])
        decisions = pd.DatetimeIndex(sorted(dates))
        basic = pd.read_parquet(path/'fund_basic.parquet')
        from qmt.factor.cost_benchmark import domestic_fund_mask
        basic = basic[basic.market.isin(config.markets) & basic.fund_type.isin(config.fund_types) & domestic_fund_mask(basic['name'])]
        codes = sorted(basic.fund_code.astype(str))
        features, outcomes = _features_and_outcomes(path,sessions,decisions,codes,config)
        known = label_availability(path,sessions,decisions,codes,config.horizon_sessions)
        common = np.logical_and.reduce([np.isfinite(frame) for frame in features.values()])
        features = {k:v.where(common) for k,v in features.items()}
        names=[]
        for key,frame in {**features,'outcomes':outcomes,'known_at':known}.items():
            name=f'{key}.parquet'; frame.to_parquet(cache/name); names.append(name)
        receipt={'snapshot_hash':manifest['snapshot_hash'],'baseline_protocol_hash':frozen['protocol_hash'],
                 'first_year':first_year,'files':{name:_hash_file(cache/name) for name in names}}
        dump(cache/'manifest.json',receipt)
    return dict(features=features,outcomes=outcomes,known_at=known,common=features['vol60'].notna(),
                sessions=sessions,outer=outer,config=config,frozen=frozen,cache_receipt=receipt)


def rank_ic_frame(scores,outcomes,common,config):
    """Vectorized exact average-tie Spearman, same complete-score universe as runner."""
    usable = outcomes.notna() & np.isfinite(outcomes) & common
    n = usable.sum(axis=1)
    permitted = (n>=config.min_cross_section) & (n/common.sum(axis=1).replace(0,np.nan)>=config.min_outcome_coverage)
    permitted &= ((~np.isfinite(scores)) & common).sum(axis=1).eq(0)
    a=scores.where(usable).rank(axis=1,method='average')
    b=outcomes.where(usable).rank(axis=1,method='average')
    a=a.sub(a.mean(axis=1),axis=0);b=b.sub(b.mean(axis=1),axis=0)
    denominator=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return ((a*b).sum(axis=1)/denominator.where(denominator>0)).where(permitted)


def row_correlation(a,b):
    a=a.where(b.notna()).rank(axis=1,pct=True);b=b.where(a.notna()).rank(axis=1,pct=True)
    a=a.sub(a.mean(axis=1),axis=0);b=b.sub(b.mean(axis=1),axis=0)
    denominator=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return (a*b).sum(axis=1)/denominator.where(denominator>0)


def diagnose(snapshot,baseline_protocol,study,output,*,resume=False):
    study,output=Path(study),Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    inputs=load_inputs(snapshot,baseline_protocol,output/'inputs')
    features,y,common,config=(inputs[k] for k in ('features','outcomes','common','config'))
    dates=y.index;codes=list(y.columns)
    panel=BaseFactorPanel([d.date() for d in dates],codes,features)
    dsl=UniverseDSL()
    baseline=-features['vol60']
    baseline_ic=rank_ic_frame(baseline,y,common,config)
    original=pd.read_parquet(study/'metrics.parquet')
    expected=original.drop_duplicates('decision_date').set_index('decision_date').baseline_ic
    pd.testing.assert_series_equal(baseline_ic.reindex(expected.index),expected,check_names=False,rtol=1e-12,atol=1e-14)
    records=[json.loads(p.read_text()) for p in sorted(study.glob('20*.json'))]
    formulas=set()
    for r in records:
        formulas.update(v['formula'] for v in r['validation'].get('candidates',[]))
        formulas.update(t['expr_text'] for t in r['trials'] if t['passed_dsl'] and t['select_ic'] is not None)
    if resume and (output/'formula_ic.parquet').exists():
        ic=pd.read_parquet(output/'formula_ic.parquet')
        correlations=pd.read_parquet(output/'baseline_rank_correlation.parquet')
        if set(ic.columns)!=formulas:raise ValueError('cached diagnostic formula identity mismatch')
    else:
        ic_cache={};similarity={}
        for i,text in enumerate(sorted(formulas)):
            scores=dsl.evaluate_frame(dsl.parse(text),panel)
            ic_cache[text]=rank_ic_frame(scores,y,common,config)
            similarity[text]=row_correlation(scores,baseline)
            if i%100==0:print('Scored diagnostic formulas',i,'/',len(formulas),flush=True)
        ic=pd.DataFrame(ic_cache);correlations=pd.DataFrame(similarity)
        ic.to_parquet(output/'formula_ic.parquet');correlations.to_parquet(output/'baseline_rank_correlation.parquet')
    trials=[];rewards=[];selection=[]
    for r in records:
        year,method,seed=(r[k] for k in ('year','method','seed'))
        test_dates=inputs['outer'][year]
        test_ic=ic.loc[test_dates];test_base=baseline_ic.loc[test_dates]
        candidate_texts=[v['formula'] for v in r['validation'].get('candidates',[])]
        chosen=r['formula']
        pool_values=test_ic.loc[test_base.notna(),candidate_texts]
        pool_means=pool_values.mean().where(pool_values.notna().all(axis=0)).dropna()
        baseline_entry=next((v for v in r['validation'].get('candidates',[]) if v['formula']=='neg(vol60)'),{})
        selected_entry=next((v for v in r['validation'].get('candidates',[]) if v['formula']==chosen),{})
        best=pool_means.idxmax()
        train_exprs={t['expr_text']:t['select_ic'] for t in r['trials'] if t['passed_dsl'] and t['select_ic'] is not None}
        train_best=max(train_exprs,key=train_exprs.get)
        selection.append({'year':year,'method':method,'seed':seed,'train_dates':len(r['discovery_dates']),
                          'validation_dates':r['validation']['validation_dates'],'selected':chosen,
                          'outer_scored_dates':int(test_base.notna().sum()),
                          'validation_robust_gain':selected_entry.get('robust_validation_ic',np.nan)-baseline_entry.get('robust_validation_ic',np.nan),
                          'outer_delta':float((test_ic[chosen]-test_base).mean()),
                          'oracle_pool_delta_DIAGNOSTIC_ONLY':float(pool_means.max()-test_base.mean()),
                          'oracle_formula_DIAGNOSTIC_ONLY':best,
                          'discovery_best_outer_delta':float((test_ic[train_best]-test_base).mean()),
                          'selected_baseline_corr':float(correlations.loc[test_dates,chosen].mean()),
                          'oracle_beats_baseline':bool(pool_means.max()>test_base.mean()+1e-12)})
        for t in r['trials']:
            trials.append({'year':year,'method':method,'seed':seed,**t})
        if r['controller']:
            for t in r['controller']['transitions']:
                rewards.append({'year':year,'method':method,'seed':seed,**t})
    selection=pd.DataFrame(selection);trials=pd.DataFrame(trials);rewards=pd.DataFrame(rewards)
    selection.to_parquet(output/'selection_diagnostics.parquet',index=False)
    trials.to_parquet(output/'trial_diagnostics.parquet',index=False)
    rewards.to_parquet(output/'reward_diagnostics.parquet',index=False)
    by_method={}
    for m,s in selection.groupby('method'):
        t=trials[trials.method.eq(m)];rr=rewards[rewards.method.eq(m)]
        gains=rr.training_best_fitness_gain if len(rr) else pd.Series(dtype=float)
        by_method[m]={'mean_train_dates':s.train_dates.mean(),'mean_validation_dates':s.validation_dates.mean(),
                      'mean_selected_validation_robust_gain':s.validation_robust_gain.mean(),
                      'mean_selected_outer_delta':original.loc[original.method.eq(m),'delta_ic'].mean(),
                      'mean_discovery_best_outer_delta':s.discovery_best_outer_delta.mean(),
                      'mean_oracle_top10_outer_delta_DIAGNOSTIC_ONLY':np.average(s.oracle_pool_delta_DIAGNOSTIC_ONLY,weights=s.outer_scored_dates),
                      'oracle_pool_positive_fraction_DIAGNOSTIC_ONLY':s.oracle_beats_baseline.mean(),
                      'nonbaseline_selected_fraction':s.selected.ne('neg(vol60)').mean(),
                      'validation_outer_delta_correlation':s[['validation_robust_gain','outer_delta']].corr().iloc[0,1],
                      'reward_transitions':len(rr),'positive_training_gain_fraction':gains.gt(1e-12).mean(),
                      'novel_without_fitness_gain_fraction':((rr.novel_usable_formula)&gains.le(1e-12)).mean() if len(rr) else None,
                      'total_positive_fitness_gain':gains.sum(),
                      'total_novelty_bonus':float((rr.novel_usable_formula & ~rr.was_known).sum()*.01) if len(rr) else None}
    leaves={}
    for k,frame in features.items():
        oriented= -frame if k.startswith('vol') else frame
        leaf_ic=rank_ic_frame(oriented,y,common,config)
        outer_dates=pd.DatetimeIndex(sorted(set(d for ds in inputs['outer'].values() for d in ds)))
        leaves[k]={'outer_mean_ic':leaf_ic.reindex(outer_dates).mean(),
                   'mean_rank_corr_with_lowvol':row_correlation(oriented,baseline).reindex(outer_dates).mean()}
    leaf_corr={}
    for a in features:
        leaf_corr[a]={b:row_correlation(features[a],features[b]).mean() for b in features}
    report={'status':'COMPLETE_POSTHOC_DIAGNOSIS','snapshot_hash':inputs['cache_receipt']['snapshot_hash'],
            'original_study_protocol_hash':json.loads((study/'protocol.json').read_text())['protocol_hash'],
            'oracle_use':'diagnostic upper bound only; uses outer outcomes and must never select deployed formulas or tune thresholds',
            'by_method':by_method,'leaf_summary':leaves,'mean_leaf_rank_correlations':leaf_corr,
            'lowvol_year_ic':baseline_ic.reindex(expected.index).groupby(expected.index.year).mean().to_dict(),
            'temporal_sample_warning':'fund count is cross-sectional breadth, not independent time samples; formulas/seeds share outcomes',
            'outcome_status':'NAV ranking only; no certified fee-net reward','distinct_formulas_scored':len(formulas),'diagnostic_code_hash':_hash_file(Path(__file__))}
    dump(output/'report.json',report)
    print(json.dumps(report,default=str,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('protocol');p.add_argument('study');p.add_argument('output')
    p.add_argument('--resume',action='store_true');args=p.parse_args();diagnose(args.snapshot,args.protocol,args.study,args.output,resume=args.resume)

if __name__=='__main__':main()
