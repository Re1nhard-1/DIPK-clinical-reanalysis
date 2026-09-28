from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import pandas as pd
import numpy as np
from scipy.stats import rankdata
P=Path(__file__).resolve().parent;D=P/'method_screen'/'DIPK';O=D/'third_cohort'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads((O/'identity_audit.json').read_text(encoding='utf-8'))
for f,h in identity['input_hashes'].items():assert sha(P/f)==h
x=pd.read_csv(O/'GSE20194_verified_array_crosswalk.csv',keep_default_na=False)
p=pd.read_csv(O/'GSE20194_patient_means.csv',keep_default_na=False)
def auc(y,s):
    y=np.asarray(y,dtype=bool);a=int(y.sum());b=int((~y).sum())
    return float((rankdata(s)[y].sum()-a*(a+1)/2)/(a*b)) if a*b else np.nan
old=pd.read_csv(D/'saved_result_recalculation.csv').set_index('accession').loc['GSE20194']
array_auc=auc(x.observed_pcr,x.score)
assert abs(array_auc-old.auc_against_provided_labels)<1e-12
sets={
 'all_patient_means':p,
 'source_MAQC_TV':x[x.maqc_status.isin(['MAQC_T','MAQC_V'])].copy(),
 'patient_means_not_linked_GSE25055':p[~p.linked_GSE25055].copy()}
assert len(sets['all_patient_means'])==248 and len(sets['source_MAQC_TV'])==230
assert len(sets['patient_means_not_linked_GSE25055'])==60
rng=np.random.default_rng(20260928);rows=[];boot=[];unit_tables=[]
for name,d in sets.items():
    assert d.patient_key.is_unique
    y,s=d.observed_pcr.to_numpy(),d.score.to_numpy()
    vals=[]
    for k in range(2000):
        ix=rng.integers(0,len(y),len(y));v=auc(y[ix],s[ix]);vals.append(v);boot.append([name,k,v])
    vals=np.asarray(vals);valid=np.isfinite(vals);ci=np.quantile(vals[valid],[.025,.975])
    rows.append(dict(analysis=name,n=len(d),pcr=int(y.sum()),nonpcr=int((1-y).sum()),auc=auc(y,s),
      ci_low=float(ci[0]),ci_high=float(ci[1]),valid_draws=int(valid.sum()),undefined_draws=int((~valid).sum())))
    q=d[['patient_key','observed_pcr','score']].copy();q['analysis']=name;unit_tables.append(q)
pd.DataFrame(rows).to_csv(O/'patient_level_results.csv',index=False)
pd.concat(unit_tables,ignore_index=True).to_csv(O/'patient_analysis_sets.csv',index=False)
pd.DataFrame(boot,columns=['analysis','draw','auc']).to_csv(O/'patient_bootstrap.csv',index=False)
out=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=20260928,draws=2000,
 original_arrays=dict(n=len(x),pcr=int(x.observed_pcr.sum()),auc=array_auc,interval=None,
   note='Source reproduction of nonindependent arrays; no naive independent-array confidence interval.'),
 analyses=rows,
 input_hashes=identity['input_hashes'],
 derived_input_hashes={f.name:sha(f) for f in [O/'identity_audit.json',O/'GSE20194_verified_array_crosswalk.csv',O/'GSE20194_patient_means.csv']},
 script_sha256=sha(Path(__file__)),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
 limitations=['Patient means change score construction; not a label-only correction.',
  'MAQC subset changes source quality eligibility as well as removing repeats.',
  'Unlinked identifiers do not establish a new independent population.',
  'Descriptive sensitivities with fixed saved scores; no causal difference test, new model or pooled analysis.'])
(O/'unit_sensitivity_results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
