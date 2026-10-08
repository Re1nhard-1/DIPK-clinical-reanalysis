from pathlib import Path
import hashlib, json, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

P = Path(__file__).resolve().parent
D = P / 'method_screen' / 'DIPK'
O = D / 'sensitivity'
O.mkdir(exist_ok=True)
SEED, B = 20260927, 2000
inputs = [P/'GSE25055_metadata.csv', P/'GSE32646_metadata.csv',
          D/'GSE25055_fixed_prediction_crosswalk.csv',
          D/'GSE32646_fixed_prediction_crosswalk.csv', D/'label_repair_results.json']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(P)).replace('\\', '/'): sha(p) for p in inputs}
prior = json.loads((D/'label_repair_results.json').read_text(encoding='utf-8'))

models = []
for acc in ['GSE25055', 'GSE32646']:
    x = pd.read_csv(D/f'{acc}_fixed_prediction_crosswalk.csv', keep_default_na=False)
    m = pd.read_csv(P/f'{acc}_metadata.csv', keep_default_na=False)
    assert x.GSM.is_unique and m.GSM.is_unique and set(x.GSM) == set(m.GSM)
    if acc == 'GSE25055':
        m = m[['GSM','age_years','er_status_ihc','pam50_class','pathologic_response_pcr_rd']]
        x = x.merge(m, on='GSM', validate='one_to_one', how='left')

        assert x.observed_pcr_rd.eq(x.pathologic_response_pcr_rd.replace({'NA':''})).all()
        x = x[x.observed_pcr_rd.isin(['pCR', 'RD'])].copy()
        y = x.observed_pcr_rd.eq('pCR')
        er = x.er_status_ihc.map({'P':'positive','N':'negative'}).fillna('unknown')
        age = pd.to_numeric(x.age_years, errors='raise')
        pam = x.pam50_class
        assert len(x) == 306 and y.sum() == 57 and er.ne('unknown').sum() == 301
        assert set(pam) == {'Basal','Her2','LumA','LumB','Normal'}
    else:
        x = x.merge(m[['GSM','age','er status ihc']], on='GSM', validate='one_to_one')
        y = x['pathologic response pcr ncr'].eq('pCR')
        er, age = x['er status ihc'], pd.to_numeric(x.age, errors='raise')
        pam = pd.Series('unavailable', index=x.index)
        assert len(x) == 115 and y.sum() == 27 and set(er) == {'positive','negative'}
    assert np.isfinite(x.score).all() and np.isfinite(age).all()
    assert np.array_equal(x.score.to_numpy(), -x.ln_ic50.to_numpy())
    models.append(pd.DataFrame(dict(cohort=acc, GSM=x.GSM, y=y.astype(int),
        stored=x.archive_group.eq('pCR').astype(int), score=x.score,
        age10=age/10, er=er, pam50=pam)))
data = pd.concat(models, ignore_index=True)
assert data.GSM.is_unique
data.to_csv(O/'analysis_input.csv', index=False)

def auc(y, score):
    y = np.asarray(y, dtype=bool)
    n, m = int(y.sum()), int((~y).sum())
    return float((rankdata(score)[y].sum()-n*(n+1)/2)/(n*m)) if n*m else np.nan

def within(y, score, group):
    numerator, denominator = 0.0, 0
    for g in np.unique(group):
        ix = group == g
        yy, ss = y[ix], score[ix]
        pairs = int(yy.sum()) * int((1-yy).sum())
        if pairs:
            numerator += auc(yy, ss)*pairs
            denominator += pairs
    return numerator/denominator if denominator else np.nan

def interval(v):
    v = np.asarray(v)
    valid = np.isfinite(v)
    q = np.quantile(v[valid], [.025, .975]) if valid.any() else [np.nan,np.nan]
    return dict(ci_low=float(q[0]), ci_high=float(q[1]),
                valid_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))

rng = np.random.default_rng(SEED)
strata, conditional, boot = [], [], []
for acc in ['GSE25055', 'GSE32646']:
    a = data[data.cohort.eq(acc)]
    for grouping in (['er','pam50'] if acc == 'GSE25055' else ['er']):
        a0 = a[a[grouping].ne('unknown')].copy()
        y, old, s, g = [a0[k].to_numpy() for k in ['y','stored','score',grouping]]
        for level in sorted(np.unique(g)):
            ix = g == level
            yy, ss = y[ix], s[ix]
            bs = []
            for b in range(B):
                ids = rng.integers(0, len(yy), len(yy))
                val = auc(yy[ids], ss[ids])
                bs.append(val)
                boot.append([acc,grouping,level,'observed_auc',b,val])
            strata.append(dict(cohort=acc,grouping=grouping,stratum=level,
                n=int(ix.sum()),pcr=int(yy.sum()),rd=int((1-yy).sum()),
                auc=auc(yy,ss),**interval(bs)))
        ys = {'observed':y}
        if acc == 'GSE25055': ys['DLDA30'] = old
        vals = {target:within(yy,s,g) for target,yy in ys.items()}
        draws = {target:[] for target in ys}
        if len(ys) == 2:
            vals['observed_minus_DLDA30'] = vals['observed']-vals['DLDA30']
            draws['observed_minus_DLDA30'] = []
        for b in range(B):
            ids = rng.integers(0,len(y),len(y))
            one = {target:within(yy[ids],s[ids],g[ids]) for target,yy in ys.items()}
            if len(ys) == 2: one['observed_minus_DLDA30'] = one['observed']-one['DLDA30']
            for target,v in one.items():
                draws[target].append(v)
                boot.append([acc,grouping,'within_stratum',target,b,v])
        for target,v in vals.items():
            pairs = sum(int(yy[g==k].sum())*int((1-yy[g==k]).sum()) for k in np.unique(g)) if (yy:=ys.get(target)) is not None else None
            conditional.append(dict(cohort=acc,grouping=grouping,target=target,
                n=len(y),within_pairs=pairs,estimate=v,**interval(draws[target])))

pd.DataFrame(strata).to_csv(O/'subgroup_auc.csv', index=False)
pd.DataFrame(conditional).to_csv(O/'within_stratum_auc.csv', index=False)
pd.DataFrame(boot,columns=['cohort','grouping','stratum','target','draw','estimate']).to_csv(O/'bootstrap_draws.csv',index=False)
assert hashes == {str(p.relative_to(P)).replace('\\','/'):sha(p) for p in inputs}
result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
    
    input_hashes=hashes, seed=SEED, draws=B,
    subgroup_auc=strata, within_stratum_auc=conditional,
    runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
    script_sha256=sha(Path(__file__)),inputs_unchanged=True)
(O/'sensitivity_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(pd.DataFrame(strata).to_string(index=False))
print(pd.DataFrame(conditional).to_string(index=False))
