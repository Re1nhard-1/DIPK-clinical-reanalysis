from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss

P=Path(__file__).resolve().parent
D=P/'method_screen/DIPK'
O=D/'extension'

subdata=pd.read_csv(D/'sensitivity/analysis_input.csv')
subdata=subdata[(subdata.cohort=='GSE25055')&subdata.er.isin(['positive','negative'])].sort_values('GSM').reset_index(drop=True)
subdata['er_positive']=(subdata.er=='positive').astype(int)
assert len(subdata)==301 and subdata.GSM.is_unique and subdata.pam50.notna().all()
assert set(subdata.pam50)=={'Basal','Her2','LumA','LumB','Normal'}
submodels={'subtype':['age10','er_positive'],'subtype_score':['age10','er_positive','score']}
submetrics=[];subpreds=[]
def subtype_model(cols):
    transform=ColumnTransformer([('numeric',StandardScaler(),cols),('subtype',OneHotEncoder(drop='first',handle_unknown='error',sparse_output=False),['pam50'])])
    return make_pipeline(transform,LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-10))
for repeat in range(10):
    split=list(StratifiedKFold(5,shuffle=True,random_state=202609290+repeat).split(subdata,subdata.y))
    for name,cols in submodels.items():
        pr=np.empty(len(subdata));folds=np.empty(len(subdata),dtype=int)
        for fold,(train,test) in enumerate(split):
            assert not set(train)&set(test)
            model=subtype_model(cols)
            with warnings.catch_warnings(record=True) as w:model.fit(subdata.iloc[train],subdata.y.iloc[train])
            assert not w,str(w)
            pr[test]=model.predict_proba(subdata.iloc[test])[:,1];folds[test]=fold
        submetrics.append(dict(cohort='GSE25055',repeat=repeat,model=name,auc=roc_auc_score(subdata.y,pr),ap=average_precision_score(subdata.y,pr),brier=brier_score_loss(subdata.y,pr),logloss=log_loss(subdata.y,pr)))
        subpreds.extend(dict(cohort='GSE25055',repeat=repeat,model=name,GSM=g,fold=int(f),y=int(y),probability=float(p)) for g,f,y,p in zip(subdata.GSM,folds,subdata.y,pr))
pd.DataFrame(submetrics).to_csv(O/'subtype_metrics.csv',index=False)
predframe=pd.DataFrame(subpreds)
predframe.to_csv(O/'subtype_predictions.csv',index=False)
prev=pd.read_csv(O/'cv_predictions.csv').query("cohort=='GSE25055' and model=='clinical'")[['repeat','GSM','fold','y']]
merged=predframe.merge(prev,on=['repeat','GSM'],suffixes=('','_reference'),validate='many_to_one')
assert len(merged)==6020 and (merged.fold==merged.fold_reference).all() and (merged.y==merged.y_reference).all()
assert not predframe.duplicated(['repeat','model','GSM']).any()
subcontrasts=[]
for metric in ['auc','ap','brier','logloss']:
    values=pd.DataFrame(submetrics).pivot(index='repeat',columns='model',values=metric)
    difference=values.subtype_score-values.subtype
    subcontrasts.append(dict(cohort='GSE25055',metric=metric,mean=difference.mean(),minimum=difference.min(),maximum=difference.max()))
pd.DataFrame(subcontrasts).to_csv(O/'subtype_increment.csv',index=False)
for name,cols in submodels.items():
    model=subtype_model(cols).fit(subdata,subdata.y)
    design=model.steps[0][1].transform(subdata);target=subdata.y.to_numpy();lr=model.steps[1][1]
    def objective(beta):
        eta=beta[0]+design@beta[1:]
        return np.logaddexp(0,eta).sum()-target@eta+.5*(beta[1:]@beta[1:])
    def gradient(beta):
        residual=expit(beta[0]+design@beta[1:])-target
        return np.r_[residual.sum(),design.T@residual+beta[1:]]
    fit=minimize(objective,np.zeros(design.shape[1]+1),jac=gradient,method='BFGS',options={'gtol':1e-7,'maxiter':2000})
    difference=np.max(np.abs(np.r_[lr.intercept_,lr.coef_[0]]-fit.x))
    assert difference<2e-5
    print('Subtype coefficient comparison',name,float(difference))
print(pd.DataFrame(submetrics).groupby('model')[['auc','ap','brier','logloss']].mean().to_string())
print(pd.DataFrame(subcontrasts).to_string(index=False))
print('COMPLETE',len(predframe),'additional held-out predictions')
