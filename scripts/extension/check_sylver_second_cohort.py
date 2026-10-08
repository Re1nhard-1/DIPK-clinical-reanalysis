import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from check_dipk_sylver_comparison import metrics


ROOT = Path(__file__).resolve().parent
D = ROOT / 'method_screen/DIPK'
S = ROOT / 'method_screen/SYLVER'
OUT = D / 'extension'
SEED = 202609304
DRAWS = 5000


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(f):
    result = {}
    for group in ['all', 'without_RPS_flags']:
        sub = f if group == 'all' else f.loc[~f.rps_flag]
        for target in ['pathology', 'proxy']:
            a, p = metrics(sub[target].to_numpy(), sub.SLMHRD.to_numpy())
            result[f'{group}.auc.{target}'] = a
            result[f'{group}.ap.{target}'] = p
        for metric in ['auc', 'ap']:
            result[f'{group}.{metric}.pathology_minus_proxy'] = result[f'{group}.{metric}.pathology'] - result[f'{group}.{metric}.proxy']
    return result


def verify(f, actual):
    errors = []
    for group in ['all', 'without_RPS_flags']:
        sub = f if group == 'all' else f.loc[~f.rps_flag]
        for target in ['pathology', 'proxy']:
            y, s = sub[target].to_numpy(), sub.SLMHRD.to_numpy()
            a, p = actual[f'{group}.auc.{target}'], actual[f'{group}.ap.{target}']
            diff = s[y == 1, None] - s[None, y == 0]
            errors.extend([abs(a-roc_auc_score(y, s)), abs(p-average_precision_score(y, s)), abs(a-((diff>0)+0.5*(diff==0)).mean())])
    assert max(errors) < 1e-12, errors
    return errors


def main():
    paths = [S/'hatzis2_released_scores.csv', S/'source_figure4.xlsx', Path(__file__), ROOT/'check_dipk_sylver_comparison.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    protected = [D/'GSE25055_fixed_prediction_crosswalk.csv', D/'sensitivity/analysis_input.csv',
                 D/'third_cohort/GSE20194_verified_array_crosswalk.csv', OUT/'sylver_comparison_results.json',
                 OUT/'sylver_comparison_input.csv', OUT/'sylver_comparison_bootstrap.csv']
    protected_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    f = pd.read_csv(paths[0], keep_default_na=False, float_precision='round_trip')
    assert len(f) == f.GSM.nunique() == f.patient_key.nunique() == 181
    assert f.rps_flag.dtype == bool and f.rps_flag.sum() == 28
    assert f.pathology.sum() == 42 and f.proxy.sum() == 69
    point = evaluate(f)
    errors = verify(f, point)
    rng = np.random.default_rng(SEED)
    samples = []
    for i in range(DRAWS):
        sub = f.iloc[rng.integers(0, len(f), len(f))]
        result = evaluate(sub)
        samples.append(result)
        if i in [0, 1, 99, 999, 4999]:
            errors.extend(verify(sub, result))
    boot = pd.DataFrame(samples)
    estimates = {key: {'estimate': value, 'ci_low': float(boot[key].quantile(.025)),
                       'ci_high': float(boot[key].quantile(.975)), 'valid_draws': int(boot[key].notna().sum())}
                 for key, value in point.items()}
    unique = np.unique(f.SLMHRD)
    positions = np.sort(np.r_[unique[0]-1, unique, (unique[:-1]+unique[1:])/2, unique[-1]+1])
    bounds = []
    for s in positions:
        row = {'candidate_score_position': float(s)}
        for target in ['pathology', 'proxy']:
            y = np.r_[f[target].to_numpy(), 0]
            score = np.r_[f.SLMHRD.to_numpy(), s]
            row[target] = float(roc_auc_score(y, score))
        row['pathology_minus_proxy'] = row['pathology'] - row['proxy']
        bounds.append(row)
    rank = pd.DataFrame(bounds)
    counts = {}
    for name, sub in [('all', f), ('without_RPS_flags', f.loc[~f.rps_flag])]:
        counts[name] = {'n': len(sub), 'pathology_pcr': int(sub.pathology.sum()), 'proxy_pcr': int(sub.proxy.sum()),
                        'label_disagreements': int((sub.pathology != sub.proxy).sum()),
                        'source_counts': {str(k):int(v) for k,v in sub.source.value_counts().items()}}
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 
              'seed': SEED, 'draws': DRAWS, 'counts': counts, 'estimates': estimates,
              'missing_score_auc_rank_bounds': {k: {'min':float(rank[k].min()), 'max':float(rank[k].max())} for k in ['pathology','proxy','pathology_minus_proxy']},
              'missing_score_rank_positions': len(rank), 
              'input_sha256': before, 'protected_sha256': protected_hashes,
              'numerical_verification': {'checks':len(errors), 'max_error':max(errors), 'bootstrap_draw_indices_checked':[0,1,99,999,4999], 'undefined_draws':int(boot.isna().sum().sum())},
              'environment': {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    assert all(sha(p) == before[str(p.relative_to(ROOT))] for p in paths)
    assert all(sha(p) == protected_hashes[str(p.relative_to(ROOT))] for p in protected)
    for name in ['sylver_second_cohort_results.json','sylver_second_cohort_bootstrap.csv','sylver_second_cohort_missing_rank_bounds.csv']:
        assert not (OUT/name).exists(), 'Do not overwrite completed results'
    boot.to_csv(OUT/'sylver_second_cohort_bootstrap.csv',index=False)
    rank.to_csv(OUT/'sylver_second_cohort_missing_rank_bounds.csv',index=False)
    (OUT/'sylver_second_cohort_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'counts':counts, 'estimates':estimates,'bounds':result['missing_score_auc_rank_bounds'],'verification':result['numerical_verification']},indent=2))


if __name__ == '__main__':
    main()
