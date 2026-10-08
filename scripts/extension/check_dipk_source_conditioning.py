from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

P = Path(__file__).resolve().parent
D = P / 'method_screen/DIPK'
O = D / 'extension'
files = [P / 'GSE25055_metadata.csv', P / 'GSE25055_matrix_header.txt',
         D / 'GSE25055_fixed_prediction_crosswalk.csv', O / 'clinical_expanded_input.csv',
         Path(__file__)]
hashes = {str(f.relative_to(P)).replace('\\', '/'): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
meta = pd.read_csv(P / 'GSE25055_metadata.csv', keep_default_na=False)
crosswalk = pd.read_csv(D / 'GSE25055_fixed_prediction_crosswalk.csv')
raw = [r for r in csv.reader((P / 'GSE25055_matrix_header.txt').read_text(encoding='utf-8-sig').splitlines(), delimiter='\t') if r]
ids = next(r[1:] for r in raw if r[0] == '!Sample_geo_accession')
sources = next(r[1:] for r in raw if r[0] == '!Sample_characteristics_ch1' and all(v.startswith('source: ') for v in r[1:]))
lookup = dict(zip(ids, [v.removeprefix('source: ') for v in sources]))
assert meta.GSM.map(lookup).equals(meta.source)
data = crosswalk[crosswalk.observed_known].merge(meta[['GSM', 'source']], on='GSM', validate='one_to_one')
data['y'] = data.observed_pcr_rd.eq('pCR').astype(int)
data['z'] = data.dlda30_prediction.eq('pCR').astype(int)
assert len(data) == 306 and data.GSM.is_unique and data.y.sum() == 57
groups = {name: data[data.source == name].sort_values('GSM').reset_index(drop=True) for name in ['MDACC', 'ISPY']}
assert [(len(g), int(g.y.sum())) for g in groups.values()] == [(227, 43), (79, 14)]
weights = {name: len(g) / len(data) for name, g in groups.items()}


def auc(y, score):
    count = int(y.sum())
    other = len(y) - count
    if not count or not other:
        return np.nan
    return float((rankdata(score)[y == 1].sum() - count * (count + 1) / 2) / (count * other))


def pairs(y, score):
    positive, negative = score[y == 1], score[y == 0]
    return float(((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative)).mean())


points = []
for name, g in groups.items():
    old, observed = auc(g.z.to_numpy(), g.score.to_numpy()), auc(g.y.to_numpy(), g.score.to_numpy())
    for field in ['y', 'z']:
        target, score = g[field].to_numpy(), g.score.to_numpy()
        assert np.allclose([auc(target, score)] * 2, [roc_auc_score(target, score), pairs(target, score)], atol=1e-12, rtol=0)
    points.append(dict(source=name, n=len(g), observed_pcr=int(g.y.sum()), archived_pcr=int(g.z.sum()),
                       disagreements=int((g.y != g.z).sum()), archived_auc=old, observed_auc=observed,
                       difference=observed - old, fixed_weight=weights[name]))
points.append(dict(source='Fixed-source-weighted', n=306, observed_pcr=57, archived_pcr=int(data.z.sum()),
                   disagreements=int((data.y != data.z).sum()), fixed_weight=1,
                   **{field: sum(weights[r['source']] * r[field] for r in points) for field in ['archived_auc', 'observed_auc', 'difference']}))
rng = np.random.default_rng(202609296)
draws = []
invalid = []
for draw in range(2000):
    current = []
    for name, g in groups.items():
        index = rng.integers(len(g), size=len(g))
        y, z, score = g.y.to_numpy()[index], g.z.to_numpy()[index], g.score.to_numpy()[index]
        old, observed = auc(z, score), auc(y, score)
        if not np.isfinite(old + observed):
            invalid.append(dict(draw=draw, source=name, reason='Undefined outcome class'))
        if draw in [0, 1, 999, 1999] and np.isfinite(old + observed):
            assert abs(observed - pairs(y, score)) < 1e-12
            assert abs(old - roc_auc_score(z, score)) < 1e-12
        current.append(dict(draw=draw, source=name, archived_auc=old, observed_auc=observed, difference=observed-old))
    draws.extend(current)
    draws.append(dict(draw=draw, source='Fixed-source-weighted',
                      **{field: sum(weights[r['source']] * r[field] for r in current) for field in ['archived_auc', 'observed_auc', 'difference']}))
boot = pd.DataFrame(draws)
summary = pd.DataFrame(points)
for i, row in summary.iterrows():
    selected = boot[boot.source == row.source]
    for metric in ['archived_auc', 'observed_auc', 'difference']:
        low, high = selected[metric].quantile([.025, .975])
        summary.loc[i, metric + '_low'] = low
        summary.loc[i, metric + '_high'] = high
    summary.loc[i, 'valid_draws'] = int(np.isfinite(selected.difference).sum())
summary.to_csv(O / 'source_label_summary.csv', index=False)
boot.to_csv(O / 'source_label_bootstrap.csv', index=False)
data[['GSM', 'source', 'y', 'z', 'score']].to_csv(O / 'source_label_input.csv', index=False)
expanded = pd.read_csv(O / 'clinical_expanded_input.csv').query("cohort=='GSE25055'")
expanded = expanded.merge(meta[['GSM','source']], on='GSM', validate='one_to_one')
audit = []
for name in groups:
    all_source = meta[meta.source == name]
    eligible = expanded[expanded.source == name]
    audit.append(dict(source=name, total_records=len(all_source), primary_known_outcomes=len(groups[name]),
                      primary_pcr=int(groups[name].y.sum()), clinical_eligible=len(eligible),
                      clinical_pcr=int(eligible.y.sum()), pr_missing=int(eligible.pr_positive.isna().sum()),
                      her2_missing=int(eligible.her2_positive.isna().sum()), grade_missing=int(eligible.grade.isna().sum()),
                      her2_positive=int(eligible.her2_positive.eq(1).sum())))
pd.DataFrame(audit).to_csv(O / 'source_eligibility_audit.csv', index=False)
pooled = {'archived_auc':auc(data.z.to_numpy(),data.score.to_numpy()),'observed_auc':auc(data.y.to_numpy(),data.score.to_numpy())}
prior = pd.read_csv(O / 'label_metrics.csv').set_index('evaluation')
assert abs(pooled['archived_auc'] - prior.loc['DIPK_archived','auc']) < 1e-12
assert abs(pooled['observed_auc'] - prior.loc['DIPK_observed','auc']) < 1e-12
assert all(hashlib.sha256((P / f).read_bytes()).hexdigest() == value for f, value in hashes.items())
report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), seed=202609296, draws=2000,
              input_hashes=hashes, invalid_draws=invalid, pooled_arithmetic_check=pooled,
              direct_pair_and_sklearn_agreement=True, fixed_source_weights=weights,
              source_model_fits=0)
(O / 'source_label_verification.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
print(summary.to_string(index=False))
print(pd.DataFrame(audit).to_string(index=False))
print('COMPLETE',len(boot),'draw rows;',len(invalid),'undefined source draws')
