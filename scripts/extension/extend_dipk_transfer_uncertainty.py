from pathlib import Path
import json,hashlib,warnings,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss

P=Path(__file__).resolve().parent
D=P/'method_screen/DIPK'
O=D/'extension'
input_path=D/'sensitivity/analysis_input.csv'
input_hash=hashlib.sha256(input_path.read_bytes()).hexdigest()
data=pd.read_csv(input_path)
data=data[data.er.isin(['positive','negative'])].copy()
data['er_positive']=(data.er=='positive').astype(int)
cohorts={k:g.sort_values('GSM').reset_index(drop=True) for k,g in data.groupby('cohort')}
earlier=pd.read_csv(O/'transport_metrics.csv').set_index(['source','target','model'])
rng=np.random.default_rng(202609294)
draws=[];checks=[];skipped=[]
def fit_apply(x,y,target,n_features,verify=False):
    xx=x[:,:n_features];tt=target[:,:n_features]
    scaler=StandardScaler().fit(xx)
    assert np.allclose(scaler.mean_,xx.mean(axis=0),atol=1e-14,rtol=1e-14)
    z=scaler.transform(xx)
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-10)
    with warnings.catch_warnings(record=True) as w:model.fit(z,y)
    assert not w,str(w)
    if verify:
        def fun(beta):
            eta=beta[0]+z@beta[1:]
            return np.logaddexp(0,eta).sum()-y@eta+.5*(beta[1:]@beta[1:])
        def jac(beta):
            e=expit(beta[0]+z@beta[1:])-y
            return np.r_[e.sum(),z.T@e+beta[1:]]
        ref=minimize(fun,np.zeros(z.shape[1]+1),jac=jac,method='BFGS',options={'gtol':1e-7,'maxiter':2000})
        error=float(np.max(np.abs(np.r_[model.intercept_,model.coef_[0]]-ref.x)))
        assert error<2e-5,(error,ref.message)
        checks.append(dict(features=n_features,max_coefficient_difference=error))
    pr=model.predict_proba(scaler.transform(tt))[:,1]
    assert np.isfinite(pr).all() and ((pr>0)&(pr<1)).all()
    return pr,float(model.coef_[0,-1])
def metrics(y,p):
    return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),brier=brier_score_loss(y,p),logloss=log_loss(y,p),mean_error=float(p.mean()-y.mean()))
started=time.perf_counter()
for source,train in cohorts.items():
    target=next(k for k in cohorts if k!=source);test=cohorts[target]
    x=train[['age10','er_positive','score']].to_numpy();y=train.y.to_numpy()
    xx=test[['age10','er_positive','score']].to_numpy();yy=test.y.to_numpy()
    for name,n in [('clinical',2),('combined',3)]:
        pred,_=fit_apply(x,y,xx,n,True);m=metrics(yy,pred)
        for k in ['auc','ap','brier','logloss']:assert abs(m[k]-earlier.loc[(source,target,name),k])<1e-12,(source,name,k)
    for draw in range(2000):
        si=rng.integers(len(train),size=len(train));ti=rng.integers(len(test),size=len(test))
        sy=y[si];ty=yy[ti]
        if len(np.unique(sy))<2 or len(np.unique(ty))<2:
            skipped.append(dict(source=source,target=target,draw=draw,reason='one_outcome_class'));continue
        check=draw in [0,499,999,1999]
        pc,_=fit_apply(x[si],sy,xx[ti],2,check)
        ps,beta=fit_apply(x[si],sy,xx[ti],3,check)
        mc=metrics(ty,pc);ms=metrics(ty,ps)
        draws.append(dict(source=source,target=target,draw=draw,source_unique=len(np.unique(si)),target_unique=len(np.unique(ti)),source_pcr=int(sy.sum()),target_pcr=int(ty.sum()),score_coefficient=beta,**{'clinical_'+k:v for k,v in mc.items()},**{'combined_'+k:v for k,v in ms.items()},**{'delta_'+k:ms[k]-mc[k] for k in ['auc','ap','brier','logloss']}))
        if (draw+1)%500==0:print(source,'to',target,draw+1,'draws',round(time.perf_counter()-started,1),'seconds',flush=True)
results=pd.DataFrame(draws)
results.to_csv(O/'transport_refit_bootstrap.csv',index=False)
summary=[]
for (source,target),g in results.groupby(['source','target']):
    for metric in ['auc','ap','brier','logloss']:
        values=g['delta_'+metric];lo,hi=np.quantile(values,[.025,.975])
        estimate=earlier.loc[(source,target,'combined'),metric]-earlier.loc[(source,target,'clinical'),metric]
        summary.append(dict(source=source,target=target,metric=metric,estimate=estimate,low=lo,high=hi,valid_draws=len(values),positive_fraction=float((values>0).mean())))
pd.DataFrame(summary).to_csv(O/'transport_refit_increment.csv',index=False)
diagnostics=[]
for (source,target),g in results.groupby(['source','target']):
    for field in ['clinical_mean_error','combined_mean_error','score_coefficient']:
        lo,hi=np.quantile(g[field],[.025,.975]);diagnostics.append(dict(source=source,target=target,field=field,mean=g[field].mean(),low=lo,high=hi))
pd.DataFrame(diagnostics).to_csv(O/'transport_refit_diagnostics.csv',index=False)
assert input_hash==hashlib.sha256(input_path.read_bytes()).hexdigest()
record=dict(passed=True,seed=202609294,draws_per_direction=2000,valid_draws=len(results),skipped=skipped,input_sha256=input_hash,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,elapsed_seconds=time.perf_counter()-started)
(O/'transport_refit_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(diagnostics).to_string(index=False))
