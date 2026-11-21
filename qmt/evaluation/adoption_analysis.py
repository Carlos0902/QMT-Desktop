"""Read-only replay of archived candidates, separating selection from adoption.

No training/search entry point is called. These are post-hoc development
diagnostics, not new model approvals or independent confirmation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmt.evaluation.information_features import load, MARKET, HOLDINGS
from qmt.evaluation.information_study import model_scores
from qmt.evaluation.platform import _hash_file, _interval, _json
from qmt.evaluation.search_benchmark import UniverseDSL, dump, joint_test
from qmt.evaluation.search_diagnostics import rank_ic_frame

POLICIES = ('gated', 'ungated_best', 'positive_mean', 'gate_on_best')


def validation_best(model):
    """Use saved validation means only; preserve the archived order on ties."""
    pool = model.get('adoption', {}).get('candidates', [])
    if not pool:
        return None
    if any(not np.isfinite(c['mean_paired_delta']) for c in pool):
        raise ValueError('archived validation pool contains nonfinite means')
    return max(pool, key=lambda c: c['mean_paired_delta'])


def activate(model, candidate):
    """Recover the rejected formula from its audit, never the stored baseline."""
    result = deepcopy(model)
    if candidate is None or result['kind'] == 'baseline':
        return {'kind': 'baseline', 'active': False}
    result['active'] = True
    if result['kind'] == 'formula':
        result['formula'] = candidate['formula']
    return result


def policy_models(model):
    best = validation_best(model)
    return {
        'gated': deepcopy(model),
        'ungated_best': activate(model, best),
        'positive_mean': activate(model, best if best and best['mean_paired_delta'] > 1e-12 else None),
        'gate_on_best': activate(model, best if best and best['accepted'] else None),
    }


def rejection_reason(model, candidate):
    if candidate is None:
        return model.get('reason', model.get('adoption', {}).get('status', 'NO_SAVED_CANDIDATE'))
    if candidate['accepted']:
        return 'ACCEPTED'
    reasons = []
    if candidate['mean_paired_delta'] <= 1e-12:
        reasons.append('NONPOSITIVE_VALIDATION_MEAN')
    if candidate['simultaneous_lower95'] <= 1e-12:
        reasons.append('FAMILY_LOWER_BOUND')
    years = model['adoption']['years']
    if candidate['positive_years'] < int(np.ceil(2 * years / 3)):
        reasons.append('YEAR_CONSISTENCY')
    return '|'.join(reasons) or 'OTHER_ARCHIVED_REJECTION'


def used_new_fields(model):
    if model['kind'] == 'baseline':
        return []
    if model['kind'] == 'ridge':
        fields = {k for k, c in zip(model['fields'], model['coefficients']) if c != 0}
    else:
        fields = set(UniverseDSL().parse(model['formula']).data_fields)
    return sorted(fields & set(MARKET + HOLDINGS))


def temporal_metrics(metrics):
    # Seeds share market outcomes: each date gets one vote after averaging seeds.
    def complete_mean(s):
        return s.mean() if s.notna().all() else np.nan
    return metrics.groupby(['decision_date', 'arm', 'policy'], sort=True).agg(
        rank_ic=('rank_ic', complete_mean), delta_ic=('delta_ic', complete_mean)
    ).reset_index()


def _summary(series, config):
    years = series.groupby(series.index.year).mean()
    return {**_interval(series, config), 'positive_years': int(years.gt(1e-12).sum()),
            'negative_years': int(years.lt(-1e-12).sum()), 'yearly': years.to_dict()}


def _inputs(study, features):
    paths = [study / 'protocol.json', study / 'report.json', study / 'metrics.parquet',
             study / 'selections.parquet', features / 'manifest.json']
    paths += sorted(study.glob('20*.json'))
    return {str(p.resolve()): _hash_file(p) for p in paths}


def replay(feature_path, study_path, output):
    feature_path, study, output = map(Path, (feature_path, study_path, output))
    output.mkdir(parents=True, exist_ok=False)
    source_report = json.loads((study / 'report.json').read_text())
    source_protocol = json.loads((study / 'protocol.json').read_text())
    for name, digest in source_report['files'].items():
        if _hash_file(study / name) != digest:
            raise ValueError(f'original study file modified: {name}')
    if source_report['protocol_hash'] != source_protocol['protocol_hash']:
        raise ValueError('study protocol identity mismatch')
    if hashlib.sha256(_json({k: v for k, v in source_protocol.items() if k != 'protocol_hash'}).encode()).hexdigest() != source_protocol['protocol_hash']:
        raise ValueError('study protocol modified')
    for name, digest in source_protocol['code_hashes'].items():
        if _hash_file(Path(name)) != digest:
            raise ValueError(f'original scoring code modified: {name}')
    source_hashes = _inputs(study, feature_path)
    protocol = {
        'kind': 'archived_adoption_diagnostic_v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_protocol_hash': source_protocol['protocol_hash'],
        'feature_manifest_hash': source_protocol['feature_manifest_hash'],
        'input_hashes': source_hashes, 'analysis_source_hash': _hash_file(Path(__file__)),
        'policies': {
            'gated': 'exact original deployed model and gate',
            'ungated_best': 'highest saved independent-validation mean in original audited pool; first tie wins',
            'positive_mean': 'same as ungated_best, baseline if validation mean <= 1e-12',
            'gate_on_best': 'same ungated-best candidate only if its saved gate passed; otherwise baseline',
        },
        'unchanged': ['yearly fitted parameters', 'original top-ten validation pool', 'fund/date intersection',
                      '0.25 residual supplement', 'invalid/constant score fallbacks', 'label horizon and lag'],
        'not_fitted': 'keep original baseline; do not fabricate a 2015 candidate',
        'oracle': 'best full-year OOS mean from original validation pool only; hindsight diagnostic, never a policy',
        'inference': 'average seeds per date; 2-date moving blocks; joint one-sided max-t over ALL policy gains, gate contrasts and feature contrasts',
        'evidence': 'post-hoc development; no threshold selection or prospective approval',
        'new_searches': 0, 'new_proposals': 0, 'new_fits': 0,
    }
    protocol['protocol_hash'] = hashlib.sha256(_json(protocol).encode()).hexdigest()
    dump(output / 'protocol.json', protocol)
    print('REGISTERED REPLAY', protocol['protocol_hash'], flush=True)
    x = load(feature_path)
    if x['manifest']['manifest_hash'] != source_protocol['feature_manifest_hash']:
        raise ValueError('study/feature identity mismatch')
    original = pd.read_parquet(study / 'metrics.parquet').set_index(['year', 'arm', 'seed', 'decision_date']).sort_index()
    choices = pd.read_parquet(study / 'selections.parquet')
    rows, candidate_rows, selections, oracle_rows = [], [], [], []
    max_replay_error = 0.
    for year, ds in sorted(x['outer'].items()):
        features = {k: v.loc[ds] for k, v in x['features'].items()}
        common, outcomes = x['common'].loc[ds], x['outcomes'].loc[ds]
        baseline = -features['vol60'].where(common)
        base_ic = rank_ic_frame(baseline, outcomes, common, x['config'])
        cache = {}

        def score(model):
            # Include coefficients/fields, not only text, in the cache key.
            key = _json({k: v for k, v in model.items() if k != 'adoption'})
            if key not in cache:
                values, reasons = model_scores(model, features, common)
                ic = rank_ic_frame(values, outcomes, common, x['config'])
                identical = values.rank(axis=1).eq(baseline.rank(axis=1)) | ~common
                cache[key] = (ic, reasons, identical.all(axis=1))
            return cache[key]

        for entry in choices.loc[choices.year.eq(year)].itertuples(index=False):
            arm, seed = entry.arm, int(entry.seed)
            name = f'{year}-{arm}' + ('' if arm.startswith('ridge_') else f'-{seed}') + '.json'
            journal = json.loads((study / name).read_text())
            model = journal.get('model', journal)
            best = validation_best(model)
            policies = policy_models(model)
            reference = original.loc[(year, arm, seed)].reindex(ds)
            pd.testing.assert_series_equal(base_ic, reference.baseline_ic, check_names=False, rtol=1e-12, atol=1e-14)
            for policy, pm in policies.items():
                ic, reasons, unchanged = score(pm)
                if policy == 'gated':
                    pd.testing.assert_series_equal(ic, reference.rank_ic, check_names=False, rtol=1e-12, atol=1e-14)
                    if reasons.tolist() != reference.fallback_reason.tolist():
                        raise ValueError('original fallback reasons did not reproduce')
                    max_replay_error = max(max_replay_error, float((ic - reference.rank_ic).abs().max()))
                for d in ds:
                    rows.append({'decision_date': d, 'year': year, 'arm': arm, 'seed': seed, 'policy': policy,
                        'rank_ic': ic.loc[d], 'baseline_ic': base_ic.loc[d], 'delta_ic': ic.loc[d] - base_ic.loc[d],
                        'fitted': model['kind'] != 'baseline', 'active': pm.get('active', False),
                        'fallback_reason': reasons.loc[d], 'same_ranks_as_baseline': bool(unchanged.loc[d]),
                        'uses_new_fields': bool(used_new_fields(pm)) if pm.get('active') else False,
                        'eligible_funds': int(common.loc[d].sum())})
            selection = {'year': year, 'arm': arm, 'seed': seed, 'fitted': model['kind'] != 'baseline',
                'gated_active': model.get('active', False), 'best_formula': best['formula'] if best else None,
                'best_validation_delta': best['mean_paired_delta'] if best else None,
                'best_gate_accepted': best['accepted'] if best else False,
                'best_gate_reason': rejection_reason(model, best),
                'best_new_fields': ','.join(used_new_fields(policies['ungated_best'])),
                'gated_formula': model.get('formula') if model.get('active') else None,
                'gated_new_fields': ','.join(used_new_fields(model)) if model.get('active') else '',
                'best_lower95': best['simultaneous_lower95'] if best else None,
                'best_positive_validation_years': best['positive_years'] if best else None,
                'n_scored_dates': int(base_ic.notna().sum())}
            for policy, pm in policies.items():
                selection[policy + '_oos_delta'] = (score(pm)[0] - base_ic).mean()
            pool_ics = {}
            for candidate in model.get('adoption', {}).get('candidates', []):
                cm = activate(model, candidate)
                ic, reasons, unchanged = score(cm)
                pool_ics[candidate['formula']] = ic
                for d in ds:
                    candidate_rows.append({'decision_date': d, 'year': year, 'arm': arm, 'seed': seed,
                        **candidate, 'uses_new_fields': bool(used_new_fields(cm)), 'rank_ic': ic.loc[d],
                        'delta_ic': ic.loc[d] - base_ic.loc[d], 'fallback_reason': reasons.loc[d],
                        'same_ranks_as_baseline': bool(unchanged.loc[d])})
            if pool_ics:
                pool = pd.DataFrame(pool_ics).loc[base_ic.notna()]
                if not pool.notna().all().all():
                    raise ValueError('candidate missing on an original scoring date; do not delete difficult dates')
                oracle = pool.mean().idxmax()
                selection['oracle_formula_DIAGNOSTIC_ONLY'] = oracle
                oracle_ic = pool[oracle].reindex(ds)
            else:
                oracle_ic = base_ic
            for d in ds:
                oracle_rows.append({'decision_date': d, 'arm': arm, 'seed': seed,
                                    'delta_ic': oracle_ic.loc[d] - base_ic.loc[d]})
            selections.append(selection)
        print('REPLAYED', year, 'unique archived scores', len(cache), flush=True)
    metrics = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    selections = pd.DataFrame(selections)
    temporal = temporal_metrics(metrics)
    wide = temporal.pivot(index='decision_date', columns=['arm', 'policy'], values='delta_ic')
    hypotheses = {f'gain/{arm}/{policy}': wide[arm, policy] for arm, policy in wide.columns}
    feature_contrasts = {}
    gate_contrasts = {}
    for arm in wide.columns.get_level_values(0).unique():
        for other in ('ungated_best', 'positive_mean'):
            key = f'gate_effect/{arm}/gated-minus-{other}'
            gate_contrasts[key] = wide[arm, 'gated'] - wide[arm, other]
        for policy in POLICIES:
            if arm.endswith('_F3'):
                a0 = arm[:-2] + 'F0'
                feature_contrasts[f'feature/{arm[:-3]}/F3-minus-F0/{policy}'] = wide[arm, policy] - wide[a0, policy]
            if arm in ('ridge_F1', 'ridge_F2'):
                feature_contrasts[f'feature/ridge/{arm[-2:]}-minus-F0/{policy}'] = wide[arm, policy] - wide['ridge_F0', policy]
    hypotheses.update(gate_contrasts)
    hypotheses.update(feature_contrasts)
    family = joint_test(pd.DataFrame(hypotheses))
    summary = {}
    for (arm, policy), part in temporal.groupby(['arm', 'policy']):
        s = part.set_index('decision_date')
        raw = metrics[metrics.arm.eq(arm) & metrics.policy.eq(policy)]
        sel = selections[selections.arm.eq(arm)]
        key = f'gain/{arm}/{policy}'
        summary.setdefault(arm, {})[policy] = {
            'rank_ic': _interval(s.rank_ic, x['config']), 'delta': _summary(s.delta_ic, x['config']),
            'fitted_only_delta': _summary(s.loc[s.index.year > 2015, 'delta_ic'], x['config']),
            'seed_means': raw.groupby('seed').delta_ic.mean().to_dict(),
            'fitted_seed_years': int(sel.fitted.sum()),
            'active_seed_years': int(raw.drop_duplicates(['year', 'seed']).active.sum()),
            'scored_seed_dates': int(raw.delta_ic.notna().sum()),
            'candidate_score_fallbacks': int((raw.delta_ic.notna() & raw.fallback_reason.isin(['INVALID_SCORE', 'CONSTANT_SCORE'])).sum()),
            'same_ranks_seed_dates': int((raw.delta_ic.notna() & raw.same_ranks_as_baseline).sum()),
            **family.get('arms', {}).get(key, {})}
    contrasts = {k: {**_summary(s, x['config']), **family.get('arms', {}).get(k, {})}
                 for k, s in {**gate_contrasts, **feature_contrasts}.items()}
    decomposition = {}
    for arm, sel in selections.groupby('arm'):
        mr = metrics[metrics.arm.eq(arm)]
        mw = mr.pivot(index=['year', 'seed', 'decision_date'], columns='policy', values='delta_ic').reset_index()
        # Invalid original label/coverage dates remain missing, never zero gains.
        mw = mw.loc[mw[list(POLICIES)].notna().all(axis=1)].copy()
        mw = mw.merge(sel[['year', 'seed', 'fitted', 'best_validation_delta', 'best_gate_accepted', 'best_new_fields']],
                      on=['year', 'seed'], validate='many_to_one')
        rejected = mw.fitted & ~mw.best_gate_accepted
        positive_rejected = rejected & mw.best_validation_delta.gt(1e-12)
        def contribution(values):
            # Full original scoring grid denominator, seeds averaged first.
            return values.groupby(mw.decision_date).mean().mean()
        decomposition[arm] = {
            'positive_validation_rejected_seed_years': int((sel.fitted & ~sel.best_gate_accepted & sel.best_validation_delta.gt(1e-12)).sum()),
            'rejected_best_would_gain_seed_years': int((sel.fitted & ~sel.best_gate_accepted & sel.ungated_best_oos_delta.gt(1e-12)).sum()),
            'rejected_best_would_lose_seed_years': int((sel.fitted & ~sel.best_gate_accepted & sel.ungated_best_oos_delta.lt(-1e-12)).sum()),
            'missed_positive_contribution': contribution(mw.ungated_best.clip(lower=0).where(rejected, 0.)),
            'avoided_negative_contribution': -contribution(mw.ungated_best.clip(upper=0).where(rejected, 0.)),
            'positive_validation_rejected_oos_contribution': contribution(mw.ungated_best.where(positive_rejected, 0.)),
            'redirection_contribution': contribution(mw.gated - mw.gate_on_best),
            'gate_net_contribution': contribution(mw.gated - mw.ungated_best),
            'best_uses_new_fields_seed_years': int(sel.best_new_fields.ne('').sum()),
            'gated_uses_new_fields_seed_years': int(sel.gated_new_fields.ne('').sum()),
            'best_rejection_reasons': sel.best_gate_reason.value_counts().to_dict(),
        }
        expected = (decomposition[arm]['avoided_negative_contribution']
                    - decomposition[arm]['missed_positive_contribution'] + decomposition[arm]['redirection_contribution'])
        if not np.isclose(expected, decomposition[arm]['gate_net_contribution'], atol=1e-12):
            raise ValueError('gate contribution accounting did not reconcile')
    oracle = pd.DataFrame(oracle_rows).groupby(['decision_date', 'arm']).delta_ic.mean().reset_index()
    oracle_summary = {arm: _summary(p.set_index('decision_date').delta_ic, x['config']) for arm, p in oracle.groupby('arm')}
    for filename, frame in [('metrics.parquet', metrics), ('candidate_metrics.parquet', candidates),
                            ('selections.parquet', selections), ('temporal_metrics.parquet', temporal),
                            ('oracle_DIAGNOSTIC_ONLY.parquet', oracle)]:
        frame.to_parquet(output / filename, index=False)
    if _inputs(study, feature_path) != source_hashes:
        raise ValueError('input files changed during replay')
    if _hash_file(Path(__file__)) != protocol['analysis_source_hash']:
        raise ValueError('analysis code changed during replay')
    report = {'status': 'COMPLETE_POSTHOC_ADOPTION_REPLAY', 'protocol_hash': protocol['protocol_hash'],
        'new_searches': 0, 'new_proposals': 0, 'new_fits': 0, 'gated_replay_max_abs_error': max_replay_error,
        'original_rows_reproduced': len(original), 'original_scored_dates': int(original.reset_index().dropna(subset=['rank_ic']).decision_date.nunique()),
        'summary': summary, 'contrasts': contrasts, 'gate_decomposition': decomposition,
        'oracle_DIAGNOSTIC_ONLY': oracle_summary, 'family_test': family,
        'limitations': ['Viewed development years; no new independent evidence',
                        'Oracle uses future outcomes and cannot be traded',
                        'Syntactic new-field use is not proof of causal feature value',
                        'NAV ranking, not a fee-net portfolio return'],
        'files': {p.name: _hash_file(p) for p in output.glob('*.parquet')}}
    dump(output / 'report.json', report)
    print('COMPLETE', output.resolve(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features')
    parser.add_argument('study')
    parser.add_argument('output')
    args = parser.parse_args()
    replay(args.features, args.study, args.output)


if __name__ == '__main__':
    main()
