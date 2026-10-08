from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit
P=Path(__file__).resolve().parent;O=P/'method_screen'/'DIPK'/'sensitivity'
r=json.loads((O/'sensitivity_results.json').read_text(encoding='utf-8'))
for f,h in r['input_hashes'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h
c=pd.read_csv(O/'R_coefficients.csv');checks=[]
for model,a in c.groupby('model',sort=True):
    d=pd.read_csv(O/f'{model}_design.csv');X=d.iloc[:,2:].to_numpy();y=d.y.to_numpy()
    assert list(d.columns[2:])==list(a.term)
    beta=np.zeros(X.shape[1])
    for iteration in range(100):
        prob=expit(X@beta);H=X.T@((prob*(1-prob))[:,None]*X)
        step=np.linalg.solve(H,X.T@(y-prob));beta+=step
        if np.max(np.abs(step))<1e-11:break
    assert iteration<99
    prob=expit(X@beta);H=X.T@((prob*(1-prob))[:,None]*X)
    se=np.sqrt(np.diag(np.linalg.inv(H)))
    be=float(np.max(np.abs(beta-a.estimate.to_numpy())))
    se_error=float(np.max(np.abs(se-a.se.to_numpy())))
    assert be<1e-7 and se_error<1e-5
    checks.append(dict(model=model,max_coefficient_difference=be,max_se_difference=se_error,
                       python_iterations=iteration+1,information_condition_number=float(np.linalg.cond(H))))
b=pd.read_csv(O/'bootstrap_draws.csv')
for file,col in [('subgroup_auc.csv','auc'),('within_stratum_auc.csv','estimate')]:
    for a in pd.read_csv(O/file).to_dict('records'):
        target=a.get('target','observed_auc');stratum=a.get('stratum','within_stratum')
        v=b[(b.cohort==a['cohort'])&(b.grouping==a['grouping'])&(b.stratum==stratum)&(b.target==target)].estimate
        assert len(v)==2000 and int(v.notna().sum())==a['valid_draws']
        assert np.allclose(v.dropna().quantile([.025,.975]),[a['ci_low'],a['ci_high']],rtol=0,atol=1e-12)
report=dict(unchanged_input_hashes=True,R_direct_pair_auc_max_error=float(pd.read_csv(O/'R_auc_crosscheck.csv').difference.abs().max()),
    logistic_crosschecks=checks,bootstrap_saved_draw_intervals_verified=True,
    scripts={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'dipk_background_sensitivity.py',P/'dipk_adjusted_association.R',Path(__file__)]})
(O/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
