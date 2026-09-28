from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind,rankdata
P=Path(__file__).resolve().parent;D=P/'method_screen'/'DIPK';A=D/'archive_clinical'/'Task5'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def scores(g):
    a=pd.read_csv(A/f'test_geo{g}/predict_pCR.csv').rename(columns={'Cell':'GSM','pCR':'ln_ic50'})
    b=pd.read_csv(A/f'test_geo{g}/predict_RD.csv').rename(columns={'Cell':'GSM','RD':'ln_ic50'})
    a['archive_group']='pCR';b['archive_group']='RD'
    d=pd.concat([a,b],ignore_index=True);assert d.GSM.is_unique and np.isfinite(d.ln_ic50).all()
    d['score']=-d.ln_ic50
    f=pd.read_csv(D/f'Figure/Task5/result_geo{g}.csv')
    assert len(d)==len(f)
    assert np.max(np.abs(d.score.to_numpy()-f.IC50.to_numpy()))<5.1e-10
    assert d.archive_group.eq(f.type.replace({'nCR':'RD'})).all()
    return d
def auc(y,s):
    n=int(y.sum());m=len(y)-n
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m))
def metrics(y,s):
    return {'n':len(y),'pcr':int(y.sum()),'RD':int((~y).sum()),'auc':auc(y,s),
            'mean_difference':float(s[y].mean()-s[~y].mean()),'welch_p':float(ttest_ind(s[y],s[~y],equal_var=False).pvalue)}
x=scores('01')
labels=pd.read_csv(A/'DataPreprocess/GEO/1/GSE25055.csv',header=None,names=['GSM','raw_archive_label'])
assert labels.GSM.is_unique and labels.raw_archive_label.str.startswith('dlda30_prediction: ').all()
truth=pd.read_csv(P/'GSE25055_label_crosswalk.csv')[['GSM','observed_pcr_rd','dlda30_prediction']]
assert set(x.GSM)==set(labels.GSM)==set(truth.GSM)
x=x.merge(labels,on='GSM',validate='one_to_one').merge(truth,on='GSM',validate='one_to_one')
assert x.archive_group.eq(x.dlda30_prediction).all()
assert x.raw_archive_label.str.split(': ',n=1).str[1].eq(x.dlda30_prediction).all()
x['observed_known']=x.observed_pcr_rd.isin(['pCR','RD'])
x.to_csv(D/'GSE25055_fixed_prediction_crosswalk.csv',index=False)
v=x.loc[x.observed_known].copy();assert len(v)==306
s=v.score.to_numpy();old=v.archive_group.eq('pCR').to_numpy();new=v.observed_pcr_rd.eq('pCR').to_numpy()
assert (old!=new).sum()==90
result={'date':'2026-09-26','scope':'Saved DIPK prediction recalculation with exact GSM joins; no model training/inference.',
 'score':'Negative saved predicted LN IC50, fixed from source figure axis and verified against published figure CSV to 5.1e-10.',
 'identity_checks':{'all_310_GSM_unique_and_equal':True,'all_310_archive_labels_equal_DLDA30':True,'paired_known':306,'label_disagreements':90},
 'original_310_provided_labels':metrics(x.archive_group.eq('pCR').to_numpy(),x.score.to_numpy()),
 'same_306_stored_DLDA30':metrics(old,s),'same_306_observed_pathology':metrics(new,s)}
rng=np.random.default_rng(20260926);boot=[];failed=0
for k in range(2000):
    ids=rng.integers(0,len(s),size=len(s));a=old[ids];b=new[ids];t=s[ids]
    if min(a.sum(),(~a).sum(),b.sum(),(~b).sum())==0:failed+=1;continue
    ao=auc(a,t);an=auc(b,t);mo=float(t[a].mean()-t[~a].mean());mn=float(t[b].mean()-t[~b].mean())
    boot.append((ao,an,an-ao,mo,mn,mn-mo))
cols=['auc_DLDA30','auc_observed','auc_delta_observed_minus_DLDA30','mean_difference_DLDA30','mean_difference_observed','mean_difference_delta']
bs=pd.DataFrame(boot,columns=cols);bs.to_csv(D/'paired_label_bootstrap.csv',index=False)
result['bootstrap']={'seed':20260926,'requested':2000,'valid':len(bs),'missing_class_draws':failed,'percentile_95':{c:[float(z) for z in np.quantile(bs[c],[.025,.975])] for c in cols}}
result['auc_delta']=result['same_306_observed_pathology']['auc']-result['same_306_stored_DLDA30']['auc']
z=scores('02');zt=pd.read_csv(P/'GSE32646_metadata.csv')[['GSM','pathologic response pcr ncr']]
zl=pd.read_csv(A/'DataPreprocess/GEO/2/GSE32646.csv',sep='\t',header=None,names=['GSM','raw_archive_label'])
assert set(z.GSM)==set(zt.GSM)==set(zl.GSM) and zl.GSM.is_unique
z=z.merge(zt,on='GSM',validate='one_to_one').merge(zl,on='GSM',validate='one_to_one')
observed=z['pathologic response pcr ncr'].replace({'nCR':'RD'})
assert z.archive_group.eq(observed).all()
assert z.raw_archive_label.str.split(': ',n=1).str[1].replace({'nCR':'RD'}).eq(observed).all()
z.to_csv(D/'GSE32646_fixed_prediction_crosswalk.csv',index=False)
result['GSE32646_control']={'n':len(z),'archive_labels_match_observed':True,**metrics(observed.eq('pCR').to_numpy(),z.score.to_numpy())}
result['limits']=['Evaluation conditional on fixed saved predictions; bootstrap excludes training uncertainty.', 'GSE25055 received taxane-anthracycline therapy; not isolated paclitaxel efficacy or a treatment recommendation.', 'GSE32646 checks outcome provenance in another cohort, not independent reproduction of the GSE25055 correction.', 'DIPK figure agreement and archive metadata do not recover the complete historical model-training run.']
result['input_hashes']={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in sorted(A.rglob('*.csv'))}
result['runtime']={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
(D/'label_repair_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
pd.DataFrame([{'label':k,**result[k]} for k in ['same_306_stored_DLDA30','same_306_observed_pathology']]).to_csv(D/'label_repair_comparison.csv',index=False)
print(json.dumps(result,indent=2))
