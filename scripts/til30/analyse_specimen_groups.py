from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from analyse_saved_outputs import metrics, verify


ROOT = Path(__file__).resolve().parent
METADATA = ROOT.parents[1] / 'GSE41998_metadata.csv'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    d = pd.read_csv(ROOT / 'GSE41998_fixed_score_crosswalk.csv', float_precision='round_trip')
    m = pd.read_csv(METADATA, dtype=str).set_index('GSM').loc[d.GSM]
    d['specimen_name'] = m['specimen name'].to_numpy()
    d['candidate_group'] = d.specimen_name.str.replace(r'-\d+$', '', regex=True)
    groups, code = np.unique(d.candidate_group, return_inverse=True)
    sizes = d.groupby('candidate_group').size()
    repeated = sizes[sizes > 1].to_dict()
    assert len(groups) == 118 and repeated == {'A6432766': 3, 'A7347583': 3}
    for group in repeated:
        ids = d.loc[d.candidate_group == group, 'GSM']
        values = m.loc[ids]
        for col in ['gender', 'age', 'treatment arm', 'ac response', 'pcr', 'er', 'pr', 'her2stat']:
            assert values[col].nunique(dropna=False) == 1, (group, col)
    d.to_csv(ROOT / 'GSE41998_specimen_groups.csv', index=False)
    a, p, s = d.source_response.to_numpy(), d.observed_pcr.to_numpy(), d.saved_score.to_numpy()
    rng = np.random.default_rng(202610012)
    cols = ['auc_AC', 'ap_AC', 'auc_pathology', 'ap_pathology', 'delta_auc', 'delta_ap']
    actual = np.concatenate([metrics(a, s), metrics(p, s), metrics(p, s) - metrics(a, s)])
    draws = np.empty((5000, 6))
    for b in range(len(draws)):
        group_ids = rng.integers(0, len(groups), len(groups))
        weights = np.bincount(group_ids, minlength=len(groups))[code]
        ac, pc = metrics(a, s, weights), metrics(p, s, weights)
        draws[b] = np.concatenate([ac, pc, pc - ac])
        if b in [0, 1, 10, 999, 4999]:
            ids = np.repeat(np.arange(len(d)), weights)
            verify(a[ids], s[ids])
            verify(p[ids], s[ids])
            assert np.allclose(ac, metrics(a[ids], s[ids]), rtol=0, atol=1e-14)
            assert np.allclose(pc, metrics(p[ids], s[ids]), rtol=0, atol=1e-14)
    table = pd.DataFrame(draws, columns=cols)
    table.to_csv(ROOT / 'GSE41998_group_bootstrap.csv', index=False)
    summary = []
    for i, col in enumerate(cols):
        finite = np.isfinite(draws[:, i])
        lo, hi = np.quantile(draws[finite, i], [.025, .975])
        summary.append({'metric': col, 'estimate': float(actual[i]), 'lower': float(lo), 'upper': float(hi),
                        'defined_draws': int(finite.sum()), 'undefined_draws': int((~finite).sum())})
    pd.DataFrame(summary).to_csv(ROOT / 'GSE41998_group_results.csv', index=False)
    saved = pd.read_csv(ROOT / 'GSE41998_group_bootstrap.csv', float_precision='round_trip')
    assert np.array_equal(saved.to_numpy(), draws)
    for row in summary:
        assert np.array_equal(np.quantile(saved[row['metric']], [.025, .975]), [row['lower'], row['upper']])
    files = [ROOT / 'analyse_specimen_groups.py',
             ROOT / 'analyse_saved_outputs.py', ROOT / 'GSE41998_fixed_score_crosswalk.csv', METADATA]
    result = {'date': '2026-10-01', 
              'records': len(d), 'candidate_groups': len(groups), 'repeated_groups': repeated,
              
              'bootstrap_draws': len(draws), 'seed': 202610012, 
              'estimates': summary, 'verification': {'five_explicit_replication_draws': True, 'saved_intervals_exact': True},
              'input_sha256': {str(f.relative_to(ROOT.parents[1])): sha(f) for f in files}}
    (ROOT / 'GSE41998_group_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
