"""Assemble completed seed shards only when their frozen rules match exactly."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from qmt.evaluation.platform import LIMITATIONS,_hash_file,_interval,_json,_load_protocol
from qmt.evaluation.search_benchmark import dump,joint_test


def _comparable_protocol(protocol):
    ignored={'created_at','seeds','protocol_hash'}
    value={k:v for k,v in protocol.items() if k not in ignored}
    search=dict(value['search_config'])
    if search.pop('seed')!=protocol['seeds'][0]:
        raise ValueError('representative config seed differs from declared seed')
    value['search_config']=search
    return value


def assemble(root,shards,baseline_protocol):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    _,config=_load_protocol(baseline_protocol,protocol['snapshot_hash'])
    expected=_comparable_protocol(protocol)
    seeds=[];metrics=[];selections=[];receipts=[];replayed=0
    checkpoint=root/'sequential_checkpoint'
    checkpoint.mkdir(exist_ok=True)
    for p in list(root.glob('20*.json'))+list(root.glob('*.partial.parquet')):
        shutil.move(str(p),checkpoint/p.name)
    for directory in shards:
        directory=Path(directory);child=json.loads((directory/'protocol.json').read_text())
        report=json.loads((directory/'report.json').read_text())
        if _comparable_protocol(child)!=expected:
            raise ValueError(f'shard protocol mismatch: {directory}')
        if report['status']!='COMPLETE_DEVELOPMENT_ABLATION' or len(child['seeds'])!=1:
            raise ValueError('only complete single-seed shards can be assembled')
        seed=child['seeds'][0]
        if seed in seeds:raise ValueError('duplicate seed shard')
        seeds.append(seed)
        for name,digest in report['files'].items():
            if _hash_file(directory/name)!=digest:raise ValueError('shard output modified')
        for name,digest in child['code_hashes'].items():
            if _hash_file(Path(name))!=digest:raise ValueError('shard code changed')
        frames=pd.read_parquet(directory/'metrics.parquet');choice=pd.read_parquet(directory/'selections.parquet')
        metrics.append(frames);selections.append(choice)
        records=list(directory.glob('20*.json'))
        if len(records)!=50:raise ValueError('expected 50 searches in ten-year five-method seed shard')
        for record in records:
            data=json.loads(record.read_text())
            if len(data['trials'])!=protocol['budget']:raise ValueError('proposal budget mismatch')
            old=checkpoint/record.name
            if old.exists():
                if json.loads(old.read_text())!=data:raise ValueError(f'replay mismatch: {record.name}')
                replayed+=1
            shutil.copy2(record,root/record.name)
        receipts.append({'directory':str(directory),'seed':seed,'protocol_hash':child['protocol_hash'],
                         'report_hash':_hash_file(directory/'report.json')})
    if sorted(seeds)!=sorted(protocol['seeds']):raise ValueError('missing registered seeds')
    metrics=pd.concat(metrics,ignore_index=True).sort_values(['year','method','seed','selector','decision_date']).reset_index(drop=True)
    selections=pd.concat(selections,ignore_index=True).sort_values(['year','method','seed','selector']).reset_index(drop=True)
    if len(metrics)!=3186 or len(selections)!=540:raise ValueError('incomplete 18-arm ten-year experiment')
    if not metrics.groupby(['arm','seed']).size().eq(59).all():raise ValueError('date-grid mismatch')
    if metrics.groupby('decision_date').eligible_funds.nunique().ne(1).any():raise ValueError('universe mismatch')
    metrics.to_parquet(root/'metrics.parquet',index=False);selections.to_parquet(root/'selections.parquet',index=False)
    temporal=metrics.groupby(['decision_date','arm']).agg(
      rank_ic=('rank_ic',lambda a:a.mean() if a.notna().all() else np.nan),
      delta_ic=('delta_ic',lambda a:a.mean() if a.notna().all() else np.nan)).reset_index()
    family=joint_test(temporal.pivot(index='decision_date',columns='arm',values='delta_ic'))
    summary={}
    for arm,part in temporal.groupby('arm'):
        yearly=part.groupby(part.decision_date.dt.year).delta_ic.mean()
        seed_means=metrics[metrics.arm.eq(arm)].groupby('seed').delta_ic.mean()
        inference=family.get('arms',{}).get(arm,{})
        coverage=part.delta_ic.notna().sum()/59
        enough=coverage>=.8 and yearly.gt(1e-12).sum()>=7 and seed_means.gt(0).all()
        stable=enough and inference.get('family_adjusted_one_sided_p',1)<.05 and inference.get('simultaneous_lower95',-1)>0
        summary[arm]={'rank_ic':_interval(part.rank_ic,config),'paired_delta_ic':_interval(part.delta_ic,config),
          'positive_years':int(yearly.gt(1e-12).sum()),'year_delta_ic':yearly.to_dict(),'seed_delta_ic':seed_means.to_dict(),
          'baseline_retained_year_fraction':selections.loc[selections.arm.eq(arm),'baseline_retained'].mean(),
          'stable_ranking_improvement':bool(stable),'date_coverage':coverage,**inference}
    report={'status':'COMPLETE_DEVELOPMENT_ABLATION','execution':'registered seed shards; one joint family test after seed averaging',
      'protocol_hash':protocol['protocol_hash'],'snapshot_hash':protocol['snapshot_hash'],
      'baseline_ic':_interval(metrics.drop_duplicates('decision_date').baseline_ic,config),
      'summary':summary,'family_test':family,'searches':150,'proposals':150*protocol['budget'],
      'alpha_approved':False,'fee_net_status':'NOT_RUN_MISSING_REVIEWED_EXECUTION_INPUTS',
      'historical_periods_reused':True,'limitations':LIMITATIONS,'shards':receipts,
      'sequential_checkpoint_exact_replay_searches':replayed,
      'assembly_code_hash':_hash_file(Path(__file__)),
      'files':{name:_hash_file(root/name) for name in ('metrics.parquet','selections.parquet')}}
    dump(root/'report.json',report)
    return report
