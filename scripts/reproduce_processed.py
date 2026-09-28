from pathlib import Path
import argparse, json, platform, sys
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_ind

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results'
checks=[]
def check(name, passed):
    if not bool(passed): raise AssertionError(name)
    checks.append(name)
def equal(name,a,b,tol=1e-12):
    check(name,np.allclose(a,b,rtol=tol,atol=tol,equal_nan=True))
def auc(y,s):
    y=np.asarray(y,dtype=bool);s=np.asarray(s)
    a=s[y,None];b=s[~y][None,:]
    return float(((a>b)+0.5*(a==b)).mean())
def within(x,g,y):
    concordant=0.;pairs=0
    for _,q in x.groupby(g):
        count=int(q[y].sum())*(len(q)-int(q[y].sum()))
        if count: concordant+=auc(q[y],q.score)*count; pairs+=count
    return concordant/pairs
def load(name): return pd.read_csv(P/name)

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--output',required=True);args=arg.parse_args()
    out=Path(args.output)
    if out.exists(): raise ValueError('Output already exists; choose a new report path.')
    r=json.loads((P/'label_repair_results.json').read_text())
    x=load('GSE25055_fixed_prediction_crosswalk.csv')
    check('310 unique source records',len(x)==310 and x.GSM.is_unique)
    check('all archived labels match DLDA30',x.archive_group.eq(x.dlda30_prediction).all())
    x=x[x.observed_pcr_rd.isin(['pCR','RD'])]
    check('306 paired records and 90 label disagreements',len(x)==306 and x.archive_group.ne(x.observed_pcr_rd).sum()==90)
    for col,key in [('archive_group','same_306_stored_DLDA30'),('observed_pcr_rd','same_306_observed_pathology')]:
        y=x[col].eq('pCR');equal(key+' AUC',auc(y,x.score),r[key]['auc'])
        equal(key+' mean difference',x.loc[y,'score'].mean()-x.loc[~y,'score'].mean(),r[key]['mean_difference'])
        equal(key+' Welch P',ttest_ind(x.loc[y,'score'],x.loc[~y,'score'],equal_var=False).pvalue,r[key]['welch_p'])
    equal('paired AUC difference',auc(x.observed_pcr_rd.eq('pCR'),x.score)-auc(x.archive_group.eq('pCR'),x.score),r['auc_delta'])
    b=load('paired_label_bootstrap.csv');check('2000 primary saved draws',len(b)==2000)
    for col,ci in r['bootstrap']['percentile_95'].items(): equal('primary interval '+col,b[col].quantile([.025,.975]),ci)
    c=load('GSE32646_fixed_prediction_crosswalk.csv')
    equal('GSE32646 observed AUC',auc(c['pathologic response pcr ncr'].eq('pCR'),c.score),r['GSE32646_control']['auc'])
    a=load('sensitivity/analysis_input.csv');bd=load('sensitivity/bootstrap_draws.csv')
    for z in load('sensitivity/subgroup_auc.csv').to_dict('records'):
        q=a[(a.cohort==z['cohort'])&(a[z['grouping']]==z['stratum'])]
        equal('subgroup AUC '+str((z['cohort'],z['stratum'])),auc(q.y,q.score),z['auc'])
    for z in load('sensitivity/within_stratum_auc.csv').to_dict('records'):
        q=a[a.cohort==z['cohort']].dropna(subset=[z['grouping']])
        if z['grouping']=='er': q=q[q.er.isin(['positive','negative'])]
        if z['target']=='observed': value=within(q,z['grouping'],'y')
        elif z['target']=='DLDA30': value=within(q,z['grouping'],'stored')
        else: value=within(q,z['grouping'],'y')-within(q,z['grouping'],'stored')
        equal('within-stratum '+str((z['cohort'],z['grouping'],z['target'])),value,z['estimate'])
    for name in ['subgroup_auc.csv','within_stratum_auc.csv']:
        for z in load('sensitivity/'+name).to_dict('records'):
            v=bd[(bd.cohort==z['cohort'])&(bd.grouping==z['grouping'])&(bd.stratum==z.get('stratum','within_stratum'))&(bd.target==z.get('target','observed_auc'))].estimate
            check('saved sensitivity draw count',len(v)==2000 and v.notna().sum()==z['valid_draws'])
            equal('sensitivity interval',v.dropna().quantile([.025,.975]),[z['ci_low'],z['ci_high']])
    for z in load('sensitivity/adjusted_associations.csv').to_dict('records'):
        q=load('sensitivity/'+z['model']+'_design.csv');X=q.iloc[:,2:].to_numpy();y=q.y.to_numpy();beta=np.zeros(X.shape[1])
        for i in range(100):
            prob=expit(X@beta);H=X.T@((prob*(1-prob))[:,None]*X)
            step=np.linalg.solve(H,X.T@(y-prob));beta+=step
            if abs(step).max()<1e-11: break
        check('logistic convergence '+z['model'],i<99)
        prob=expit(X@beta);H=X.T@((prob*(1-prob))[:,None]*X);se=np.sqrt(np.diag(np.linalg.inv(H)))
        j=list(q.columns[2:]).index('score_z')
        equal('logistic coefficient '+z['model'],beta[j],z['beta'],1e-7)
        equal('logistic SE '+z['model'],se[j],z['se'],1e-5)
    arrays=load('third_cohort/GSE20194_verified_array_crosswalk.csv');patients=load('third_cohort/GSE20194_patient_means.csv')
    check('278 arrays and 248 patient IDs',len(arrays)==278 and arrays.patient_key.nunique()==248 and len(patients)==248)
    q=arrays.groupby('patient_key').score.mean();equal('patient mean aggregation',q.sort_index(),patients.set_index('patient_key').score.sort_index())
    check('188 linked patients',patients.linked_GSE25055.sum()==188)
    sets=load('third_cohort/patient_analysis_sets.csv');draws=load('third_cohort/patient_bootstrap.csv')
    for z in load('third_cohort/patient_level_results.csv').to_dict('records'):
        q=sets[sets.analysis==z['analysis']];v=draws[draws.analysis==z['analysis']].auc
        check('patient eligibility '+z['analysis'],q.patient_key.is_unique and len(q)==z['n'] and q.observed_pcr.sum()==z['pcr'])
        equal('patient AUC '+z['analysis'],auc(q.observed_pcr,q.score),z['auc'])
        check('2000 patient saved draws',len(v)==2000 and v.notna().all())
        equal('patient interval '+z['analysis'],v.quantile([.025,.975]),[z['ci_low'],z['ci_high']])
    report={'passed':True,'scope':'Processed-data AUC/mean/Welch recalculation, Newton logistic check, patient aggregation and saved-bootstrap interval checks. No fresh bootstrap sampling, upstream audit, training or inference.',
            'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__},'checks':checks}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'report':str(out)}))
if __name__=='__main__': main()
