"""Frozen-cache rolling GP/operator-RL benchmark on disclosure-safe daily-window leaves.

The GP and Q-controller implementations are reused. This adapter deliberately
registers pointwise/cross-sectional compositions of the platform's 5/20/60-session
features: rolling 20 sparse decision rows would incorrectly mean 800 sessions.
This is a new common search space, not a reproduction of the legacy daily DSL or
its fee-net reward. No execution/fee-net approval can be issued by this module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qmt.evaluation.platform import (
    LEAVES, LIMITATIONS, EvaluationProtocol, _clean, _features_and_outcomes,
    _hash_file, _interval, _json, _load_protocol, _runtime_versions, _weights,
    verify_snapshot,
)
from qmt.factor.auto_search import SymbolicSearch
from qmt.factor.dsl import BIN_OPS, CS_OPS, EW_UNARY, FormulaDSL
from qmt.factor.rl_search import RLGuidedSearch
from qmt.factor.types import BaseFactorPanel, Label, SearchConfig, SearchLedger, ValidationScheme, ValidationSplit
from qmt.stats import spearman

METHODS = ('gp', 'uniform', 'bandit', 'q')
SEEDS = (7, 19, 43)
SEED_FORMULAS = ('neg(vol60)', 'mom60', 'mom20', 'reversal5', 'trend60', 'dd60', 'neg(vol20)')
OPS = frozenset((BIN_OPS | CS_OPS | EW_UNARY) - {'neutralize'})


class UniverseDSL(FormulaDSL):
    """Keep constants/guarded operators from resurrecting unavailable assets.

    Mask every intermediate node, before a parent rank/zscore can observe it.
    All leaf frames supplied by this adapter share the vol60 eligibility mask.
    """
    def _eval_frame(self, node, panel, as_of):
        result = super()._eval_frame(node, panel, as_of)
        if result is not None:
            result = result.where(panel.factors['vol60'].notna().reindex(
                index=result.index, columns=result.columns))
        return result


def dump(path, value):
    Path(path).write_text(json.dumps(_clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


class SeededProposals:
    """Identical seven initial proposals for every arm; all consume trial budget."""
    def _before_search(self, cfg, ledger):
        super()._before_search(cfg, ledger)
        self._seed_queue = iter(SEED_FORMULAS)

    def _proposal_expr(self, cfg, rng):
        text = next(self._seed_queue, None)
        return self._dsl.parse(text) if text else super()._proposal_expr(cfg, rng)


class CacheGP(SeededProposals, SymbolicSearch):
    pass


class CacheRL(SeededProposals, RLGuidedSearch):
    pass


def fit_config(seed, budget):
    return SearchConfig(config_id='clean-cache-compositions-v1', allowed_fields=list(LEAVES),
                        allowed_operators=OPS, allowed_windows=(), max_complexity=10,
                        max_depth=4, max_trials=budget, population=20, generations=budget,
                        seed=seed, seed_from_manual=False, min_train_ic_coverage=.8)


def label_availability(path, sessions, decisions, codes, horizon):
    """Endpoint publication dates, not merely valuation ends, bound label access."""
    exits = sessions[sessions.get_indexer(decisions) + horizon]
    needed = set(decisions) | set(exits)
    blocks = []
    for batch in pq.ParquetFile(Path(path) / 'nav.parquet').iter_batches(
            columns=['fund_code', 'nav_date', 'usable_date']):
        block = batch.to_pandas()
        block = block[block.nav_date.isin(needed) & block.fund_code.isin(codes)]
        if len(block):
            blocks.append(block)
    raw = pd.concat(blocks, ignore_index=True).set_index(['nav_date', 'fund_code']).usable_date
    def matrix(dates):
        idx = pd.MultiIndex.from_product([dates, codes], names=['nav_date', 'fund_code'])
        return raw.reindex(idx).to_numpy(dtype='datetime64[ns]').reshape(len(dates), len(codes))
    a, b = matrix(decisions), matrix(exits)
    return pd.DataFrame(np.maximum(a, b), index=decisions, columns=codes)


def bounded_labels(outcomes, known_at, dates, deadline, common, config):
    """Never remove an asset to repair outcome coverage; return only mature labels."""
    labels, accepted = [], []
    for d in dates:
        eligible = common.loc[d]
        y = outcomes.loc[d].where(eligible & known_at.loc[d].le(deadline)).dropna()
        if len(y) < config.min_cross_section or len(y) / max(1, int(eligible.sum())) < config.min_outcome_coverage:
            continue
        accepted.append(d)
        labels.extend(Label(c, d.date(), config.horizon_sessions, float(v), 0.) for c, v in y.items())
    return labels, pd.DatetimeIndex(accepted)


def section_ic(score, outcome, eligible, config):
    """Every candidate must score the complete same ex-ante baseline universe."""
    s = score[eligible]
    y = outcome[eligible].replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) < config.min_cross_section or len(y) / max(1, len(s)) < config.min_outcome_coverage:
        return np.nan
    if not np.isfinite(s.to_numpy()).all():
        return np.nan
    return spearman(s.reindex(y.index).to_numpy(), y.to_numpy())


def choose_expression(candidates, dsl, panel, outcomes, known_at, valid_dates, deadline, common, config):
    """Select one top-ten discovery candidate on independent preceding year.

    Fixed low-volatility is always a candidate and wins exact ties. No outer-test
    scores are accessed, and missing candidate dates cannot improve its average.
    """
    exprs = [dsl.parse(SEED_FORMULAS[0])] + list(candidates)
    valid_outcomes = outcomes.reindex(valid_dates).where(known_at.reindex(valid_dates).le(deadline))
    baseline = -panel.factors['vol60']
    base_ics = pd.Series({d: section_ic(baseline.loc[d], valid_outcomes.loc[d], common.loc[d], config)
                          for d in valid_dates}).dropna()
    if len(base_ics) < 3:
        return exprs[0], {'status': 'INSUFFICIENT_VALIDATION', 'validation_dates': len(base_ics)}
    best, best_score, audit = exprs[0], -np.inf, []
    seen = set()
    for expr in exprs:
        if expr.text in seen:
            continue
        seen.add(expr.text)
        scores = dsl.evaluate_frame(expr, panel)
        ics = pd.Series({d: section_ic(scores.loc[d], valid_outcomes.loc[d], common.loc[d], config)
                          for d in base_ics.index})
        value = float(ics.mean() - .5 * ics.std(ddof=0)) if ics.notna().all() else -np.inf
        audit.append({'formula': expr.text, 'robust_validation_ic': value, 'n': int(ics.notna().sum())})
        if value > best_score + 1e-12:
            best, best_score = expr, value
    return best, {'status': 'SELECTED', 'validation_dates': len(base_ics), 'candidates': audit}


def joint_test(differences, reps=1999, block=2, seed=20261016):
    """Common temporal blocks for all methods and seeds; seeds are not new dates.

    Single-step centered max-t family test, with four method means after averaging
    seeds on each date. No winning-seed selection. Descriptive historical evidence.
    """
    data = differences.dropna().to_numpy(dtype=float)
    if len(data) < 12:
        return {'status': 'INSUFFICIENT_COMMON_DATES', 'n': len(data)}
    n, m = data.shape
    means = data.mean(axis=0)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-block+1, size=(reps, int(np.ceil(n/block))))
    idx = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
    boot = data[idx].mean(axis=1)
    se = boot.std(axis=0, ddof=1)
    valid = se > 1e-12
    observed = np.divide(means, se, out=np.zeros(m), where=valid)
    centered = np.divide(boot-means, se, out=np.zeros_like(boot), where=valid)
    maximum = centered.max(axis=1)
    critical = float(np.quantile(maximum, .95))
    return {'status': 'COMPLETE', 'n': n, 'block_decisions': block, 'repetitions': reps,
            'arms': {name: {'mean_delta_ic': means[j],
                            'family_adjusted_one_sided_p': (1+int((maximum >= observed[j]).sum()))/(reps+1) if valid[j] else 1.,
                            'simultaneous_lower95': means[j]-critical*se[j] if valid[j] else means[j]}
                     for j, name in enumerate(differences.columns)}}


def run(snapshot_path, baseline_protocol, output, budget=120, seeds=SEEDS, methods=METHODS):
    path, output = Path(snapshot_path), Path(output)
    manifest = verify_snapshot(path)
    frozen, config = _load_protocol(baseline_protocol, manifest['snapshot_hash'])
    output.mkdir(parents=True, exist_ok=False)
    sessions = pd.DatetimeIndex(pd.read_parquet(path/'calendar.parquet').trade_date)
    outer = {int(f['fold_id']): pd.DatetimeIndex(f['decisions']) for f in frozen['folds']}
    all_dates = set(d for dates in outer.values() for d in dates)
    # Same real-session stride, fresh annual anchor, including pre-2015 discovery.
    for year in range(min(outer)-4, max(outer)+1):
        annual = sessions[(sessions.year == year) & (np.arange(len(sessions))+config.horizon_sessions < len(sessions))]
        all_dates.update(annual[::config.decision_stride_sessions])
    decisions = pd.DatetimeIndex(sorted(all_dates))
    basic = pd.read_parquet(path/'fund_basic.parquet')
    from qmt.factor.cost_benchmark import domestic_fund_mask
    basic = basic[basic.market.isin(config.markets) & basic.fund_type.isin(config.fund_types) & domestic_fund_mask(basic['name'])]
    codes = sorted(basic.fund_code.astype(str))
    import qmt.factor.auto_search, qmt.factor.dsl, qmt.factor.rl_search, qmt.factor.rl_operators, qmt.factor.metrics
    import qmt.factor.types, qmt.factor.diversity, qmt.stats
    modules = [Path(__file__), Path(qmt.factor.auto_search.__file__), Path(qmt.factor.dsl.__file__),
               Path(qmt.factor.rl_search.__file__), Path(qmt.factor.rl_operators.__file__),
               Path(qmt.factor.metrics.__file__), Path(qmt.factor.types.__file__),
               Path(qmt.factor.diversity.__file__), Path(qmt.stats.__file__),
               Path(__file__).with_name('platform.py')]
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'snapshot_hash': manifest['snapshot_hash'],
                'snapshot_path': str(path.resolve()), 'baseline_protocol_hash': frozen['protocol_hash'],
                'runtime_versions': _runtime_versions(), 'code_hashes': {str(p.resolve()): _hash_file(p) for p in modules},
                'methods': list(methods), 'seeds': list(seeds), 'budget': budget,
                'search_config': {**asdict(fit_config(seeds[0], budget)), 'allowed_operators': sorted(OPS)},
                'initial_proposals': SEED_FORMULAS, 'folds': frozen['folds'],
                'training': 'three preceding calendar years of discovery, then one independent validation year; full 40-session label purge and one-session embargo, including actual endpoint usable dates',
                'selection': 'top ten discovery candidates plus fixed low-vol; maximize mean validation IC minus 0.5 population std; baseline wins ties; insufficient validation falls back to baseline',
                'space': 'pointwise/cross-sectional compositions of seven causal daily-window leaves; NO temporal DSL operators on sparse decision rows; adapted common feature space, not exact legacy daily DSL',
                'reward': 'existing IC-fitness improvement and novelty; not fee-net reward',
                'universe_policy': 'mask every AST intermediate node to the same ex-ante complete-leaf universe; constants cannot resurrect missing or not-yet-observed funds',
                'net_reward_status': 'NOT_RUN_MISSING_BROAD_HISTORICAL_EXECUTION_INPUTS',
                'stability_gate': 'coverage>=0.8 of 59 frozen dates; >=7/10 positive yearly seed-mean deltas; all seed overall deltas positive; joint one-sided p<0.05 and simultaneous lower95>0',
                'limitations': LIMITATIONS}
    protocol['protocol_hash'] = hashlib.sha256(_json(protocol).encode()).hexdigest()
    dump(output/'protocol.json', protocol)  # registration precedes feature/label scoring
    print('Registered', protocol['protocol_hash'], flush=True)
    features, outcomes = _features_and_outcomes(path, sessions, decisions, codes, config)
    common = pd.DataFrame(True, index=decisions, columns=codes)
    for f in features.values():
        common &= np.isfinite(f)
    features = {k: v.where(common) for k, v in features.items()}
    known_at = label_availability(path, sessions, decisions, codes, config.horizon_sessions)
    print('Features ready:', len(decisions), 'dates;', len(codes), 'metadata codes', flush=True)
    rows, selection_rows = [], []
    dsl = UniverseDSL()
    for year, test_dates in outer.items():
        validation_start = sessions[sessions.year == year-1][0]
        test_start = sessions[sessions.year == year][0]
        train_deadline = sessions[sessions.get_loc(validation_start)-2]
        valid_deadline = sessions[sessions.get_loc(test_start)-2]
        train_dates = decisions[(decisions.year >= year-4) & (decisions.year <= year-2)]
        train_dates = train_dates[sessions.get_indexer(train_dates)+config.horizon_sessions <= sessions.get_loc(train_deadline)]
        valid_dates = decisions[(decisions.year == year-1)]
        valid_dates = valid_dates[sessions.get_indexer(valid_dates)+config.horizon_sessions <= sessions.get_loc(valid_deadline)]
        labels, good_train = bounded_labels(outcomes, known_at, train_dates, train_deadline, common, config)
        if len(good_train) < 6:
            raise ValueError(f'{year}: insufficient discovery dates: {len(good_train)}')
        fit_dates = decisions[(decisions.year >= year-4) & (decisions < validation_start)]
        panel = BaseFactorPanel(dates=[d.date() for d in sessions[(sessions.year >= year-4) & (sessions < validation_start)]],
                                funds=codes, factors={k:v.loc[fit_dates] for k,v in features.items()})
        validation_panel = BaseFactorPanel(dates=[d.date() for d in valid_dates], funds=codes,
                                          factors={k:v.loc[valid_dates] for k,v in features.items()})
        split = ValidationSplit(str(year), ValidationScheme.ROLLING,
                                (good_train[0].date(), good_train[-1].date()),
                                (validation_start.date(), valid_deadline.date()), embargo_days=40)
        for method in methods:
            for seed in seeds:
                ledger = SearchLedger(locked_test_span=(test_start.date(), sessions[sessions.year==year][-1].date()))
                searcher = CacheGP(dsl) if method=='gp' else CacheRL(dsl, mode=method)
                candidates = searcher.search(fit_config(seed,budget), dsl, panel, labels, split, ledger)
                expr, audit = choose_expression(candidates, dsl, validation_panel, outcomes, known_at,
                                                valid_dates, valid_deadline, common, config)
                test_panel = BaseFactorPanel(dates=[d.date() for d in test_dates], funds=codes,
                                             factors={k:v.loc[test_dates] for k,v in features.items()})
                scores = dsl.evaluate_frame(expr, test_panel)
                record = {'year':year,'method':method,'seed':seed,'formula':expr.text,'n_trials':ledger.n_trials,
                          'discovery_dates':[str(d.date()) for d in good_train], 'train_label_deadline':train_deadline,
                          'validation_label_deadline':valid_deadline, 'validation':audit,
                          'trials':[asdict(t) for t in ledger.trials],
                          'controller':searcher.policy_audit() if hasattr(searcher,'policy_audit') else None}
                dump(output/f'{year}-{method}-{seed}.json', record)
                selection_rows.append({k:v for k,v in record.items() if k not in ('trials','controller','validation','discovery_dates')})
                for d in test_dates:
                    base = -features['vol60'].loc[d]
                    ic = section_ic(scores.loc[d], outcomes.loc[d], common.loc[d], config)
                    bic = section_ic(base, outcomes.loc[d], common.loc[d], config)
                    eligible = common.loc[d]
                    y = outcomes.loc[d][eligible]
                    portfolio_ok = len(y)>=config.min_cross_section and y.notna().all() and np.isfinite(scores.loc[d][eligible]).all()
                    proxy = bproxy = None
                    if portfolio_ok:
                        w = _weights(scores.loc[d][eligible], config.top_k)
                        bw = _weights(base[eligible], config.top_k)
                        proxy, bproxy = float((w*y.reindex(w.index)).sum()), float((bw*y.reindex(bw.index)).sum())
                    rows.append({'decision_date':d,'year':year,'method':method,'seed':seed,'rank_ic':ic,
                                 'baseline_ic':bic,'delta_ic':ic-bic,'eligible_funds':int(eligible.sum()),
                                 'gross_nav_return_proxy':proxy,'baseline_gross_proxy':bproxy,'formula':expr.text})
                pd.DataFrame(rows).to_parquet(output/'metrics.partial.parquet', index=False)
                print(f'{year} {method} seed={seed} trials={ledger.n_trials} selected={expr.text}',flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(output/'metrics.parquet',index=False)
    pd.DataFrame(selection_rows).to_parquet(output/'selections.parquet',index=False)
    method_date = metrics.groupby(['decision_date','method']).agg(
        delta_ic=('delta_ic', lambda x: x.mean() if x.notna().all() else np.nan),
        rank_ic=('rank_ic',lambda x:x.mean() if x.notna().all() else np.nan)).reset_index()
    differences = method_date.pivot(index='decision_date',columns='method',values='delta_ic')
    family = joint_test(differences)
    summary = {}
    for method in methods:
        arm = metrics[metrics.method.eq(method)]
        temporal = method_date[method_date.method.eq(method)]
        yearly = temporal.groupby(temporal.decision_date.dt.year).delta_ic.mean()
        seed_deltas = arm.groupby('seed').delta_ic.mean()
        f = family.get('arms',{}).get(method,{})
        coverage = temporal.delta_ic.notna().sum()/sum(map(len,outer.values()))
        stable = (coverage>=.8 and int(yearly.gt(1e-12).sum())>=7 and seed_deltas.gt(0).all()
                  and f.get('family_adjusted_one_sided_p',1)<.05 and f.get('simultaneous_lower95',-1)>0)
        summary[method] = {'rank_ic':_interval(temporal.rank_ic,config),
                           'paired_delta_ic':_interval(temporal.delta_ic,config),
                           'positive_years':int(yearly.gt(1e-12).sum()), 'year_delta_ic':yearly.to_dict(),
                           'seed_delta_ic':seed_deltas.to_dict(),'date_coverage':coverage,
                           'selected_baseline_fraction':float(arm.formula.eq(SEED_FORMULAS[0]).mean()),
                           'stable_ranking_improvement':bool(stable), **f}
    report = {'status':'COMPLETE','protocol_hash':protocol['protocol_hash'],'snapshot_hash':manifest['snapshot_hash'],
              'baseline_rank_ic':_interval(metrics.drop_duplicates('decision_date').baseline_ic,config),
              'summary':summary,'family_test':family,'outer_dates':sum(map(len,outer.values())),
              'complete_portfolio_dates':int(metrics.drop_duplicates('decision_date').baseline_gross_proxy.notna().sum()),
              'searches':len(selection_rows),'total_proposals':sum(r['n_trials'] for r in selection_rows),
              'alpha_approved':False,'fee_net_status':protocol['net_reward_status'],'limitations':LIMITATIONS,
              'output_hashes':{n:_hash_file(output/n) for n in ('metrics.parquet','selections.parquet')}}
    # Code/environment changes during a long run invalidate comparison rather than quietly passing.
    if protocol['code_hashes'] != {str(p.resolve()):_hash_file(p) for p in modules}:
        raise RuntimeError('benchmark code changed during run')
    dump(output/'report.json',report)
    (output/'metrics.partial.parquet').unlink()
    print(json.dumps(_clean(report),ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot');p.add_argument('baseline_protocol');p.add_argument('output')
    p.add_argument('--budget',type=int,default=120)
    args=p.parse_args()
    run(args.snapshot,args.baseline_protocol,args.output,args.budget)

if __name__=='__main__':
    main()
