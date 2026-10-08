import hashlib
import io
import json
import pickle
import pickletools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
P = ROOT.parents[1]


class NumericReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'),
            ('numpy._core.multiarray', '_reconstruct'),
            ('numpy', 'ndarray'), ('numpy', 'dtype'),
            ('pandas.core.series', 'Series'),
            ('pandas.core.internals.managers', 'SingleBlockManager'),
            ('pandas.core.indexes.base', '_new_Index'),
            ('pandas.core.indexes.base', 'Index'), ('builtins', 'slice'),
        }
        if (module, name) not in allowed:
            raise ValueError((module, name))
        return super().find_class(module, name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_saved(cohort):
    ypath = ROOT / 'saved_outputs' / f'y_test_{cohort}.pkl'
    spath = ROOT / 'saved_outputs' / f'y_probs_{cohort}.pkl'
    y = NumericReader(io.BytesIO(ypath.read_bytes())).load()
    s = NumericReader(io.BytesIO(spath.read_bytes())).load()
    blocks = [v for _, v, _ in pickletools.genops(spath.read_bytes())
              if isinstance(v, bytes) and len(v) > 1]
    assert len(blocks) == 1
    assert np.array_equal(s, np.frombuffer(blocks[0], dtype='<f8'))
    assert isinstance(y, pd.Series) and isinstance(s, np.ndarray)
    assert s.ndim == 1 and len(s) == len(y) and np.isfinite(s).all()
    assert set(y.unique()) == {0, 1} and np.all((s >= 0) & (s <= 1))
    return y, s


def metrics(y, s, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w)
    _, group = np.unique(s, return_inverse=True)
    positive = np.bincount(group, weights=w * y)
    negative = np.bincount(group, weights=w * (1 - y))
    n1, n0 = positive.sum(), negative.sum()
    if not n1 or not n0:
        return np.array([np.nan, np.nan])
    auc = np.sum(positive * (np.cumsum(negative) - negative / 2)) / (n1 * n0)
    p, n = positive[::-1], negative[::-1]
    den = np.cumsum(p + n)
    precision = np.divide(np.cumsum(p), den, out=np.zeros_like(den), where=den > 0)
    ap = np.sum((p / n1) * precision)
    return np.array([auc, ap])


def verify(y, s):
    pair_auc = ((s[y == 1, None] > s[None, y == 0]).mean()
                + .5 * (s[y == 1, None] == s[None, y == 0]).mean())
    n1 = y.sum()
    rank_auc = (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    own = metrics(y, s)
    assert np.allclose([own[0]] * 3, [pair_auc, rank_auc, roc_auc_score(y, s)], atol=1e-14, rtol=0)
    assert abs(own[1] - average_precision_score(y, s)) < 1e-14


def main():
    d = pd.read_csv(ROOT / 'GSE41998_source_outcomes.csv')
    ystored, score = read_saved('41998')
    assert list(ystored.index) == d.GSM.tolist()
    yac = (d.source_label == 0).to_numpy(dtype=int)
    assert np.array_equal(yac, ystored.to_numpy())
    assert np.array_equal(yac, (d['ac response'] == 'complete response').astype(int))
    assert set(d.pcr) == {'Yes', 'No'}
    ypcr = (d.pcr == 'Yes').to_numpy(dtype=int)
    assert round(roc_auc_score(yac, score), 2) == .67
    verify(yac, score)
    verify(ypcr, score)
    d['saved_score'] = score
    d['source_response'] = yac
    d['observed_pcr'] = ypcr
    d.to_csv(ROOT / 'GSE41998_fixed_score_crosswalk.csv', index=False)
    assert np.array_equal(pd.read_csv(ROOT / 'GSE41998_fixed_score_crosswalk.csv', float_precision='round_trip').saved_score, score)
    actual = np.concatenate([metrics(yac, score), metrics(ypcr, score), metrics(ypcr, score) - metrics(yac, score)])
    cols = ['auc_AC', 'ap_AC', 'auc_pathology', 'ap_pathology', 'delta_auc', 'delta_ap']
    rng = np.random.default_rng(202610011)
    draws = np.empty((5000, 6))
    for b in range(len(draws)):
        ids = rng.integers(0, len(d), len(d))
        weights = np.bincount(ids, minlength=len(d))
        ac, pc = metrics(yac, score, weights), metrics(ypcr, score, weights)
        draws[b] = np.concatenate([ac, pc, pc - ac])
        if b in [0, 1, 10, 999, 4999]:
            assert np.allclose(ac, metrics(yac[ids], score[ids]), equal_nan=True)
            assert np.allclose(pc, metrics(ypcr[ids], score[ids]), equal_nan=True)
            verify(yac[ids], score[ids])
            verify(ypcr[ids], score[ids])
    pd.DataFrame(draws, columns=cols).to_csv(ROOT / 'GSE41998_paired_bootstrap.csv', index=False)
    summary = []
    for i, col in enumerate(cols):
        finite = np.isfinite(draws[:, i])
        lo, hi = np.quantile(draws[finite, i], [.025, .975])
        summary.append({'metric': col, 'estimate': actual[i], 'lower': lo, 'upper': hi,
                        'defined_draws': int(finite.sum()), 'undefined_draws': int((~finite).sum())})
    pd.DataFrame(summary).to_csv(ROOT / 'GSE41998_target_results.csv', index=False)

    t = pd.read_csv(ROOT / 'GSE20194_input_identity.csv')
    yt, st = read_saved('20194')
    assert list(yt.index) == (t.source_row + 1).tolist()
    assert np.array_equal(yt, (t.source_label == 'pCR').astype(int))
    v = pd.read_csv(P / 'method_screen/DIPK/third_cohort/GSE20194_verified_array_crosswalk.csv', dtype=str)
    v = t.merge(v[['GSM', 'patient_key', 'pcr_vs_rd', 'raw_basename']], on='GSM', validate='one_to_one')
    assert len(v) == 71 and (v.source_label == v.pcr_vs_rd).all()
    links = pd.read_csv(ROOT / 'GSE20194_links_to_training.csv', dtype=str)
    assert links.GSM_20194.nunique() == len(links) == 60
    v['linked_training_GSM'] = v.GSM.map(links.set_index('GSM_20194').GSM_25055)
    v['linked_to_training'] = v.linked_training_GSM.notna()
    v.to_csv(ROOT / 'GSE20194_validation_patient_support.csv', index=False)
    verify(yt.to_numpy(), st)
    auc = metrics(yt.to_numpy(), st)[0]
    notebook = json.loads((ROOT / 'source/external_validation_11_test_set.ipynb').read_text(encoding='utf-8'))
    cached = '\n'.join(''.join(o.get('text', [])) + ''.join(o.get('data', {}).get('text/plain', [])) for o in notebook['cells'][10].get('outputs', []))
    assert f'{auc:.6f}' in cached and '0.670281' in cached
    support = {'validation_records': len(v), 'source_patients': int(v.patient_key.nunique()),
               'linked_records': int(v.linked_to_training.sum()),
               'linked_source_patients': int(v[v.linked_to_training].patient_key.nunique()),
               'unlinked_records': int((~v.linked_to_training).sum()),
               'unlinked_source_patients': int(v[~v.linked_to_training].patient_key.nunique()),
               'source_auc': auc, 'notebook_auc_match_at_six_decimals': True,
               'published_table2_auc': .96, 'direct_two_decimal_rounding_matches_table2': bool(round(auc, 2) == .96)}
    files = [ROOT / 'analyse_saved_outputs.py',
             ROOT / 'GSE41998_source_outcomes.csv', ROOT / 'GSE20194_input_identity.csv',
             ROOT / 'train_input_identity.csv', ROOT / 'GSE20194_links_to_training.csv',
             ROOT / 'source/external_validation_11_test_set.ipynb']
    files += list((ROOT / 'saved_outputs').glob('*.pkl'))
    result = {'date': '2026-10-01', 
              'repository_commit': '55cbe09820c7c679995630b65e682b1c98f4b869',
              'GSE41998': {'n': len(d), 'AC_positive': int(yac.sum()), 'pathology_positive': int(ypcr.sum()),
                           'discordant': int(np.count_nonzero(yac != ypcr)), 'bootstrap_seed': 202610011,
                           'bootstrap_draws': 5000, 'estimates': summary,
                           'source_GSM_order_and_labels_verified': True, 'published_table1_n': 125},
              'GSE20194': support,
              'verification': {'pair_count_rank_and_sklearn_auc': True, 'tied_score_AP_vs_sklearn': True,
                               'five_draws_verified_by_patient_replication': True,
                               'raw_score_payload_vs_decoded_array_exact': True,
                               'source_notebook_numeric_agreement': True},
              'input_sha256': {str(f.relative_to(ROOT)): sha(f) for f in files}}
    (ROOT / 'saved_output_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'GSE41998': result['GSE41998'], 'GSE20194': support}, indent=2))


if __name__ == '__main__':
    main()
