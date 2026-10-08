from pathlib import Path
import json, hashlib, platform, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.optimize import minimize
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

P = Path(__file__).resolve().parent
D = P / 'method_screen/DIPK'
O = D / 'extension'
checks = []

def metrics(y, p):
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                brier=brier_score_loss(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                predicted_mean=float(np.mean(p)), prevalence=float(np.mean(y)))

def save(rows, name):
    pd.DataFrame(rows).to_csv(O / name, index=False)

def bounds(x):
    return np.quantile(np.asarray(x)[np.isfinite(x)], [.025, .975])

def fitpredict(train, test, cols):
    if not cols:
        return np.full(len(test), train.y.mean()), None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs', max_iter=2000, tol=1e-10))
    with warnings.catch_warnings(record=True) as w:
        model.fit(train[cols], train.y)
    assert not w, str(w)
    return model.predict_proba(test[cols])[:, 1], model

base = pd.read_csv(D / 'GSE25055_fixed_prediction_crosswalk.csv')
base = base[base.observed_known].copy()
y = (base.observed_pcr_rd == 'pCR').astype(int).to_numpy()
z = (base.dlda30_prediction == 'pCR').astype(int).to_numpy()
score = base.score.to_numpy()
assert len(y) == 306 and sum(y != z) == 90
transition = []
for a in [0, 1]:
    for b in [0, 1]:
        x = score[(z == a) & (y == b)]
        transition.append(dict(archived=a, observed=b, n=len(x), score_mean=x.mean(), score_median=np.median(x)))
save(transition, 'label_transition.csv')
label_rows = []
for label, yy, ss in [('DIPK_archived', z, score), ('DIPK_observed', y, score), ('DLDA30_binary_observed', y, z)]:
    label_rows.append(dict(evaluation=label, n=len(yy), pcr=int(yy.sum()), auc=roc_auc_score(yy, ss), ap=average_precision_score(yy, ss), prevalence=yy.mean()))
rng = np.random.default_rng(202609291)
draws = []
for b in range(2000):
    ix = rng.integers(len(y), size=len(y))
    yy, zz, ss = y[ix], z[ix], score[ix]
    if len(np.unique(yy)) < 2 or len(np.unique(zz)) < 2:
        draws.append(dict(draw=b, auc_label_delta=np.nan, ap_label_delta=np.nan, auc_DIPK_minus_DLDA30=np.nan))
        continue
    draws.append(dict(draw=b, auc_label_delta=roc_auc_score(yy, ss)-roc_auc_score(zz, ss),
                      ap_label_delta=average_precision_score(yy, ss)-average_precision_score(zz, ss),
                      auc_DIPK_minus_DLDA30=roc_auc_score(yy, ss)-roc_auc_score(yy, zz)))
save(label_rows, 'label_metrics.csv')
save(draws, 'label_bootstrap.csv')
label_contrasts = []
for key, point in [('auc_label_delta',label_rows[1]['auc']-label_rows[0]['auc']),
                   ('ap_label_delta',label_rows[1]['ap']-label_rows[0]['ap']),
                   ('auc_DIPK_minus_DLDA30',label_rows[1]['auc']-label_rows[2]['auc'])]:
    lo, hi = bounds(pd.DataFrame(draws)[key])
    label_contrasts.append(dict(contrast=key, estimate=point, low=lo, high=hi))
save(label_contrasts, 'label_contrasts.csv')

data = pd.read_csv(D / 'sensitivity/analysis_input.csv')
data = data[data.er.isin(['positive', 'negative'])].copy()
data['er_positive'] = (data.er == 'positive').astype(int)
cohorts = {k: v.sort_values('GSM').reset_index(drop=True) for k, v in data.groupby('cohort')}
assert {k:len(v) for k,v in cohorts.items()} == {'GSE25055':301, 'GSE32646':115}
models = {'intercept':[], 'clinical':['age10','er_positive'], 'score':['score'], 'combined':['age10','er_positive','score']}
preds, cv = [], []
for cohort, df in cohorts.items():
    assert df.GSM.is_unique and not df[['y','age10','er_positive','score']].isna().any().any()
    for repeat in range(10):
        split = list(StratifiedKFold(5, shuffle=True, random_state=202609290+repeat).split(df, df.y))
        for name, cols in models.items():
            pp = np.empty(len(df)); ff = np.empty(len(df), dtype=int)
            for fold, (tr, te) in enumerate(split):
                assert not set(tr) & set(te)
                pp[te], _ = fitpredict(df.iloc[tr], df.iloc[te], cols); ff[te] = fold
            cv.append(dict(cohort=cohort, repeat=repeat, model=name, **metrics(df.y, pp)))
            preds.extend(dict(cohort=cohort, repeat=repeat, model=name, GSM=g, fold=int(f), y=int(yy), probability=float(pr)) for g,f,yy,pr in zip(df.GSM, ff, df.y, pp))
save(cv, 'cv_metrics.csv'); save(preds, 'cv_predictions.csv')
cvdf = pd.DataFrame(cv)
summary, contrasts = [], []
for (cohort, name), dd in cvdf.groupby(['cohort','model']):
    for metric in ['auc','ap','brier','logloss']:
        summary.append(dict(cohort=cohort, model=name, metric=metric, mean=dd[metric].mean(), minimum=dd[metric].min(), maximum=dd[metric].max()))
for cohort, dd in cvdf.groupby('cohort'):
    for metric in ['auc','ap','brier','logloss']:
        x=dd.pivot(index='repeat',columns='model',values=metric)
        delta=x.combined-x.clinical
        contrasts.append(dict(cohort=cohort,metric=metric,mean=delta.mean(),minimum=delta.min(),maximum=delta.max()))
save(summary,'cv_summary.csv');save(contrasts,'cv_increment.csv')

transport, tp, td, coeff, shifts = [], [], [], [], []
rng = np.random.default_rng(202609292)
for source, train in cohorts.items():
    target=next(k for k in cohorts if k!=source);test=cohorts[target]
    direction=source+'->'+target; predictions={}
    for name, cols in models.items():
        pr, model = fitpredict(train,test,cols); predictions[name]=pr
        transport.append(dict(source=source,target=target,model=name,**metrics(test.y,pr)))
        tp.extend(dict(source=source,target=target,model=name,GSM=g,y=int(yy),probability=float(v)) for g,yy,v in zip(test.GSM,test.y,pr))
        if model is not None:
            sc, lr = model.steps[0][1], model.steps[1][1]
            x=sc.transform(train[cols]); yy=train.y.to_numpy()
            def obj(beta):
                eta=beta[0]+x@beta[1:]
                return np.logaddexp(0,eta).sum()-yy@eta+.5*(beta[1:]@beta[1:])
            def jac(beta):
                e=expit(beta[0]+x@beta[1:])-yy
                return np.r_[e.sum(),x.T@e+beta[1:]]
            opt=minimize(obj,np.zeros(x.shape[1]+1),jac=jac,method='BFGS',options={'gtol':1e-7,'maxiter':2000})
            reference=np.r_[lr.intercept_,lr.coef_[0]]
            error=float(np.max(np.abs(reference-opt.x)))
            assert error<2e-5,(direction,name,error,opt.message)
            checks.append(dict(check='scipy_logistic',source=source,model=name,max_beta_difference=error))
            coeff.append(dict(source=source,model=name,feature='intercept',coefficient=lr.intercept_[0],mean=0,scale=1))
            for j,c in enumerate(cols):coeff.append(dict(source=source,model=name,feature=c,coefficient=lr.coef_[0,j],mean=sc.mean_[j],scale=sc.scale_[j]))
    for field in ['age10','er_positive','score']:
        shifts.append(dict(source=source,target=target,feature=field,source_mean=train[field].mean(),source_sd=train[field].std(ddof=0),target_mean=test[field].mean(),target_sd=test[field].std(ddof=0)))
    yy=test.y.to_numpy()
    for draw in range(2000):
        ix=rng.integers(len(test),size=len(test)); yb=yy[ix]
        if len(np.unique(yb))<2:continue
        a=metrics(yb,predictions['combined'][ix]);b=metrics(yb,predictions['clinical'][ix])
        td.append(dict(source=source,target=target,draw=draw,**{k:a[k]-b[k] for k in ['auc','ap','brier','logloss']}))
save(transport,'transport_metrics.csv');save(tp,'transport_predictions.csv');save(td,'transport_bootstrap.csv');save(coeff,'transport_coefficients.csv');save(shifts,'cohort_shift.csv')
delta=[]; tt=pd.DataFrame(transport); bd=pd.DataFrame(td)
for (source,target),dd in tt.groupby(['source','target']):
    for metric in ['auc','ap','brier','logloss']:
        point=dd.set_index('model').loc['combined',metric]-dd.set_index('model').loc['clinical',metric]
        bb=bd[(bd.source==source)&(bd.target==target)][metric];lo,hi=bounds(bb)
        delta.append(dict(source=source,target=target,metric=metric,estimate=point,low=lo,high=hi,valid_draws=len(bb)))
save(delta,'transport_increment.csv')

arr=pd.read_csv(D/'third_cohort/GSE20194_verified_array_crosswalk.csv')
groups=[g.index.to_numpy() for _,g in arr.groupby('patient_key',sort=True)]
pat=arr.groupby('patient_key',sort=True).agg(y=('observed_pcr','first'),score=('score','mean'))
ay=arr.observed_pcr.to_numpy();ss=arr.score.to_numpy();py=pat.y.to_numpy();ps=pat.score.to_numpy()
rng=np.random.default_rng(202609293); unit=[]
for draw in range(2000):
    ar=rng.integers(len(arr),size=len(arr));pi=rng.integers(len(pat),size=len(pat));cl=np.concatenate([groups[i] for i in pi])
    unit.append(dict(draw=draw,naive_array=roc_auc_score(ay[ar],ss[ar]),cluster_array=roc_auc_score(ay[cl],ss[cl]),patient_mean=roc_auc_score(py[pi],ps[pi])))
save(unit,'unit_bootstrap.csv');ud=pd.DataFrame(unit);unit_summary=[]
for method in ['naive_array','cluster_array','patient_mean']:
    lo,hi=bounds(ud[method]);unit_summary.append(dict(method=method,n_records=len(arr),n_patients=len(pat),auc=roc_auc_score(py,ps) if method=='patient_mean' else roc_auc_score(ay,ss),low=lo,high=hi,width=hi-lo))
save(unit_summary,'unit_summary.csv')
for yy,pr in [(y,score),(y,z),(py,ps)]:
    a=pr[yy==1][:,None];b=pr[yy==0][None,:];direct=np.mean((a>b)+.5*(a==b))
    assert abs(direct-roc_auc_score(yy,pr))<1e-12
for rr in tp:
    assert 0<rr['probability']<1
for (source,target,name),dd in pd.DataFrame(tp).groupby(['source','target','model']):
    yy=dd.y.to_numpy();pr=dd.probability.to_numpy()
    assert abs(np.mean((yy-pr)**2)-brier_score_loss(yy,pr))<1e-12
    assert abs(-np.mean(yy*np.log(pr)+(1-yy)*np.log(1-pr))-log_loss(yy,pr))<1e-12
assert len(pd.DataFrame(preds).drop_duplicates(['cohort','repeat','model','GSM']))==len(preds)
prov={}
prov.update(passed=True,python=platform.python_version(),packages={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},checks=checks,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_oof_predictions=len(preds))
(O/'verification.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
print('COMPLETE',len(preds),'held-out predictions')
print(pd.DataFrame(label_contrasts).to_string(index=False))
print(pd.DataFrame(contrasts).to_string(index=False))
print(pd.DataFrame(delta).to_string(index=False))
print(pd.DataFrame(unit_summary).to_string(index=False))
