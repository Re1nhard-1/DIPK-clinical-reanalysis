from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.metrics import roc_auc_score

P = Path(__file__).resolve().parent
D = P / 'method_screen/DIPK'
E = D / 'extension'
SCHEMES = {
    'outcome': ['y'],
    'outcome_source': ['y', 'source'],
    'outcome_source_er': ['y', 'source', 'er'],
    'outcome_source_pam50': ['y', 'source', 'pam50'],
    'outcome_source_er_pam50': ['y', 'source', 'er', 'pam50'],
}
PERM_SEED, BOOT_SEED = 202609301, 202609302
PERM_DRAWS, BOOT_DRAWS = 10000, 2000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_auc(labels, ranks):
    m, n = labels.sum(), len(labels)
    return float((labels @ ranks - m * (m + 1) / 2) / (m * (n - m))) if 0 < m < n else np.nan


def reference_mean(labels, ranks, codes):
    totals = np.bincount(codes)
    positives = np.bincount(codes, weights=labels, minlength=len(totals))
    q = np.divide(positives, totals, out=np.zeros_like(positives), where=totals > 0)[codes]
    m, n = labels.sum(), len(labels)
    return float((q @ ranks - m * (m + 1) / 2) / (m * (n - m))) if 0 < m < n else np.nan


def pair_reference(labels, scores, codes):
    groups = [np.flatnonzero(codes == g) for g in np.unique(codes)]
    q = np.empty(len(labels))
    for ix in groups:
        q[ix] = labels[ix].mean()
    membership = q[:, None] * (1 - q[None, :])
    for ix in groups:
        n, m = len(ix), labels[ix].sum()
        membership[np.ix_(ix, ix)] = m * (n - m) / (n * (n - 1)) if n > 1 else 0
    np.fill_diagonal(membership, 0)
    delta = scores[:, None] - scores[None, :]
    kernel = (delta > 0) + .5 * (delta == 0)
    return float((membership * kernel).sum() / (labels.sum() * (len(labels) - labels.sum())))


def components(labels, scores, ix, weights):
    pos, neg = ix[labels[ix] == 1], ix[labels[ix] == 0]
    delta = scores[pos, None] - scores[None, neg]
    kernel = (delta > 0) + .5 * (delta == 0)
    wp, wn = weights[:, pos], weights[:, neg]
    numerator = ((wp @ kernel) * wn).sum(axis=1)
    denominator = wp.sum(axis=1) * wn.sum(axis=1)
    return numerator, denominator


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)


protected = {
    D / 'GSE25055_fixed_prediction_crosswalk.csv': 'd769210ea5d2be25d840ceea5b667abf4e0fb4a37712cee7c012146245b58c10',
    D / 'sensitivity/analysis_input.csv': '1cf667165bb726fcd43beb7909f055b910be599c87e1a68f74c78a177027c276',
    D / 'third_cohort/GSE20194_verified_array_crosswalk.csv': '424f6fe537cf230203b9ffdae0bb1f1cb7c7eb08ead5aa9ffced679a8957b3b0',
}
for path, expected in protected.items():
    assert digest(path) == expected
paths = list(protected) + [P / 'GSE25055_metadata.csv', E / 'source_label_input.csv',
                           E / 'proxy_alignment_results.json', E / 'proxy_alignment_permutations.csv']
hashes = {str(p.relative_to(P)): digest(p) for p in paths}
frame = pd.read_csv(E / 'source_label_input.csv')
assert len(frame) == 306 and frame.GSM.is_unique
sensitivity = pd.read_csv(D / 'sensitivity/analysis_input.csv')
sensitivity = sensitivity[sensitivity.cohort.eq('GSE25055')].set_index('GSM').loc[frame.GSM]
metadata = pd.read_csv(P / 'GSE25055_metadata.csv', keep_default_na=False).set_index('GSM').loc[frame.GSM]
crosswalk = pd.read_csv(D / 'GSE25055_fixed_prediction_crosswalk.csv').set_index('GSM').loc[frame.GSM]
assert np.array_equal(frame.y, sensitivity.y) and np.array_equal(frame.z, sensitivity.stored)
assert np.array_equal(frame.score, sensitivity.score) and np.array_equal(frame.score, crosswalk.score)
assert np.array_equal(frame.y, crosswalk.observed_pcr_rd.eq('pCR').astype(int))
assert np.array_equal(frame.z, crosswalk.dlda30_prediction.eq('pCR').astype(int))
assert np.array_equal(frame.source, metadata.source)
er_map = {'P': 'positive', 'N': 'negative', 'I': 'indeterminate', 'NA': 'missing'}
assert set(metadata.er_status_ihc) == set(er_map)
frame['er_source'] = metadata.er_status_ihc.to_numpy()
frame['er'] = metadata.er_status_ihc.map(er_map).to_numpy()
frame['pam50'] = metadata.pam50_class.to_numpy()
assert np.array_equal(frame.er.replace({'indeterminate': 'unknown', 'missing': 'unknown'}), sensitivity.er)
assert np.array_equal(frame.pam50, sensitivity.pam50)
frame.to_csv(E / 'proxy_background_input.csv', index=False)
y, z, scores = frame.y.to_numpy(), frame.z.to_numpy(), frame.score.to_numpy()
ranks = rankdata(scores)
actual = rank_auc(z, ranks)
assert abs(actual - roc_auc_score(z, scores)) < 1e-12
prior = json.loads((E / 'proxy_alignment_results.json').read_text(encoding='utf-8'))
assert abs(actual - prior['points']['archived_proxy']) < 1e-12
assert int((y != z).sum()) == 90
codes = {name: frame.groupby(columns, sort=True).ngroup().to_numpy() for name, columns in SCHEMES.items()}
rng = np.random.default_rng(PERM_SEED)
strata, summaries, permutation_rows = [], [], []
for name, columns in SCHEMES.items():
    groups = [np.flatnonzero(codes[name] == g) for g in np.unique(codes[name])]
    mixed_n, mixed_groups = 0, 0
    for g, ix in enumerate(groups):
        positive, n = int(z[ix].sum()), len(ix)
        mixed = 0 < positive < n
        mixed_n += n if mixed else 0
        mixed_groups += int(mixed)
        strata.append({'scheme': name, 'stratum': g, **{col: frame.iloc[ix[0]][col] for col in columns},
                       'n': n, 'proxy_positive': positive, 'proxy_negative': n - positive, 'mixed_labels': mixed})
    exact = reference_mean(z, ranks, codes[name])
    assert abs(exact - pair_reference(z, scores, codes[name])) < 1e-12
    if name in ['outcome', 'outcome_source']:
        summary = dict(next(s for s in prior['permutation_summaries'] if s['scheme'] == name))
        assert abs(exact - summary['expected_auc']) < 1e-12
    else:
        vals = np.empty(PERM_DRAWS)
        for b in range(PERM_DRAWS):
            perm = z.copy()
            for ix in groups:
                perm[ix] = rng.permutation(z[ix])
                assert perm[ix].sum() == z[ix].sum()
            assert int((perm != y).sum()) == 90
            vals[b] = rank_auc(perm, ranks)
            if b in [0, 1, PERM_DRAWS - 1]:
                assert abs(vals[b] - roc_auc_score(perm, scores)) < 1e-12
        se = float(vals.std(ddof=1) / np.sqrt(PERM_DRAWS))
        assert abs(vals.mean() - exact) < 6 * se
        exceedances = int((vals >= actual).sum())
        summary = {'scheme': name, 'draws': PERM_DRAWS, 'expected_auc': exact,
                   'permutation_mean': float(vals.mean()), 'monte_carlo_mean_se': se,
                   'reference_low': float(np.quantile(vals, .025)), 'reference_high': float(np.quantile(vals, .975)),
                   'observed_proxy_auc': actual, 'observed_minus_reference_mean': actual - exact,
                   'exceedances': exceedances, 'plus_one_upper_tail_fraction': (1 + exceedances) / (PERM_DRAWS + 1)}
        permutation_rows.extend({'scheme': name, 'draw': b, 'auc': val} for b, val in enumerate(vals))
    summary.update(n_strata=len(groups), mixed_label_strata=mixed_groups, patients_in_mixed_strata=mixed_n,
                   patients_in_constant_strata=len(frame) - mixed_n)
    summaries.append(summary)
pd.DataFrame(strata).to_csv(E / 'proxy_background_strata.csv', index=False)
pd.DataFrame(permutation_rows).to_csv(E / 'proxy_background_permutations.csv', index=False)

rng = np.random.default_rng(BOOT_SEED)
indices = rng.integers(0, len(frame), size=(BOOT_DRAWS, len(frame)))
weights = np.array([np.bincount(ix, minlength=len(frame)) for ix in indices], dtype=float)
weights = np.vstack([np.ones(len(frame)), weights])
conditional_specs, conditional_values = [], {}
for fixed, target in [('y', 'z'), ('z', 'y')]:
    labels = frame[target].to_numpy()
    for value in [0, 1]:
        ix = np.flatnonzero(frame[fixed].eq(value))
        key = f'{target}_within_{fixed}{value}'
        num, den = components(labels, scores, ix, weights)
        conditional_values[key] = ratio(num, den)
        conditional_specs.append({'metric': key, 'fixed_field': fixed, 'fixed_value': value, 'target': target,
                                  'source': 'all', 'n': len(ix), 'positive': int(labels[ix].sum()),
                                  'negative': int(len(ix) - labels[ix].sum()), 'eligible_pairs': int(den[0])})
    sum_num, sum_den = np.zeros(BOOT_DRAWS + 1), np.zeros(BOOT_DRAWS + 1)
    for (value, source), group in frame.groupby([fixed, 'source'], sort=True):
        ix = group.index.to_numpy()
        key = f'{target}_within_{fixed}{value}_{source}'
        num, den = components(labels, scores, ix, weights)
        conditional_values[key] = ratio(num, den)
        sum_num += num
        sum_den += den
        conditional_specs.append({'metric': key, 'fixed_field': fixed, 'fixed_value': int(value), 'target': target,
                                  'source': source, 'n': len(ix), 'positive': int(labels[ix].sum()),
                                  'negative': int(len(ix) - labels[ix].sum()), 'eligible_pairs': int(den[0])})
    key = f'{target}_within_{fixed}_source_pooled'
    conditional_values[key] = ratio(sum_num, sum_den)
    conditional_specs.append({'metric': key, 'fixed_field': fixed, 'fixed_value': 'pooled', 'target': target,
                              'source': 'pair_weighted', 'n': len(frame), 'eligible_pairs': int(sum_den[0])})

verification_max_error = 0.
for spec in conditional_specs:
    key, fixed, target = spec['metric'], spec['fixed_field'], spec['target']
    for b in [0, 1, 2, 19, 100, BOOT_DRAWS]:
        sub = frame if b == 0 else frame.iloc[indices[b - 1]]
        if spec['source'] == 'pair_weighted':
            numerator, denominator = 0., 0
            for _, group in sub.groupby([fixed, 'source']):
                labels, values = group[target].to_numpy(), group.score.to_numpy()
                pairs = int(labels.sum() * (len(labels) - labels.sum()))
                if pairs:
                    delta = values[labels == 1, None] - values[None, labels == 0]
                    numerator += ((delta > 0) + .5 * (delta == 0)).sum()
                    denominator += pairs
            check = numerator / denominator if denominator else np.nan
        else:
            sub = sub[sub[fixed].eq(spec['fixed_value'])]
            if spec['source'] != 'all':
                sub = sub[sub.source.eq(spec['source'])]
            check = roc_auc_score(sub[target], sub.score) if sub[target].nunique() == 2 else np.nan
        value = conditional_values[key][b]
        assert np.isnan(check) == np.isnan(value)
        if np.isfinite(check):
            verification_max_error = max(verification_max_error, abs(check - value))
            assert abs(check - value) < 1e-12

boot = pd.DataFrame({key: val[1:] for key, val in conditional_values.items()})
boot.insert(0, 'draw', np.arange(BOOT_DRAWS))
reference_values = {name: np.empty(BOOT_DRAWS) for name in SCHEMES}
proxy_values = np.empty(BOOT_DRAWS)
for b, ix in enumerate(indices):
    zr, sr = z[ix], scores[ix]
    rr = rankdata(sr)
    proxy_values[b] = rank_auc(zr, rr)
    for name in SCHEMES:
        value = reference_mean(zr, rr, codes[name][ix])
        reference_values[name][b] = value
        if b in [0, 1, 18, 99, BOOT_DRAWS - 1]:
            check = pair_reference(zr, sr, codes[name][ix])
            assert abs(value - check) < 1e-12
            verification_max_error = max(verification_max_error, abs(value - check))
for name, values in reference_values.items():
    boot[f'{name}_reference'] = values
    boot[f'{name}_gap'] = proxy_values - values
boot['proxy_auc'] = proxy_values
boot.to_csv(E / 'proxy_background_bootstrap.csv', index=False)
saved_boot = pd.read_csv(E / 'proxy_background_bootstrap.csv')


def interval(key):
    vals = saved_boot[key].dropna().to_numpy()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    native = boot[key].dropna().quantile([.025, .975]).to_numpy()
    assert np.allclose([lo, hi], native, atol=1e-14, rtol=0)
    return {'low': float(lo), 'high': float(hi), 'defined_draws': len(vals), 'undefined_draws': BOOT_DRAWS - len(vals)}


for spec in conditional_specs:
    spec.update(auc=float(conditional_values[spec['metric']][0]), **interval(spec['metric']))
for summary in summaries:
    name = summary['scheme']
    summary['reference_bootstrap'] = interval(f'{name}_reference')
    summary['gap_bootstrap'] = interval(f'{name}_gap')
for path in paths:
    assert digest(path) == hashes[str(path.relative_to(P))]
pd.DataFrame(conditional_specs).to_csv(E / 'proxy_background_conditional.csv', index=False)
result = {
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'permutation_seed': PERM_SEED, 'permutation_draws_per_new_scheme': PERM_DRAWS,
    'bootstrap_seed': BOOT_SEED, 'bootstrap_draws': BOOT_DRAWS,
    'input_hashes': hashes, 'analysis_input_sha256': digest(E / 'proxy_background_input.csv'),
    'script_sha256': digest(Path(__file__)), 'plan_recorded_before_new_effects': True,
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
    'n': len(frame), 'er_counts': frame.er.value_counts().to_dict(), 'pam50_counts': frame.pam50.value_counts().to_dict(),
    'observed_proxy_auc': actual, 'observed_pathology_auc': rank_auc(y, ranks),
    'permutation_summaries': summaries, 'conditional_aucs': conditional_specs,
    'verification': {'source_metadata_join_exact': True, 'original_and_prior_analysis_files_unchanged': True,
                     'all_permutation_counts_preserved': True, 'rank_and_pair_reference_agreement': True,
                     'weighted_bootstrap_vs_explicit_duplication_checks': 84, 'bootstrap_pair_reference_checks': 25,
                     'maximum_verification_error': verification_max_error, 'saved_percentiles_recalculated': True}
}
(E / 'proxy_background_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'schemes': summaries, 'conditional_aucs': conditional_specs, 'verification': result['verification']}, indent=2))
