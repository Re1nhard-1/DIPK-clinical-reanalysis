from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;D=P/'method_screen'/'DIPK';O=D/'third_cohort'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
i=json.loads((O/'identity_audit.json').read_text(encoding='utf-8'))
r=json.loads((O/'unit_sensitivity_results.json').read_text(encoding='utf-8'))
for f,h in i['input_hashes'].items():assert sha(P/f)==h
for f,h in r['derived_input_hashes'].items():assert sha(O/f)==h
old=json.loads((D/'label_repair_results.json').read_text(encoding='utf-8'))
for f,h in old['input_hashes'].items():assert sha(P/f)==h
sensitivity=json.loads((D/'sensitivity'/'sensitivity_results.json').read_text(encoding='utf-8'))
for f,h in sensitivity['input_hashes'].items():assert sha(P/f)==h
a=pd.read_csv(O/'GSE20194_verified_array_crosswalk.csv');p=pd.read_csv(O/'GSE20194_patient_means.csv')
assert len(a)==278 and len(p)==248 and int((p.array_count-1).sum())==30
assert int(a.maqc_status.eq('MDA_R').sum())==29
assert a[a.maqc_status.eq('MAQC_Q')].patient_key.nunique()==18
assert len(a[a.maqc_status.isin(['MAQC_T','MAQC_V'])])==230
assert p.linked_GSE25055.sum()==188 and a.linked_GSE25055.sum()==199
cross=pd.read_csv(O/'R_unit_crosscheck.csv')
assert cross.AUC_difference.abs().max()<1e-12 and cross.max_interval_difference.max()<1e-12
raw=json.loads((O/'raw_value_checks.json').read_text(encoding='utf-8'))
equal={z['patient_key']:z['comparison']['intensities']['equal_after_float32_conversion'] for z in raw['pairs']}
assert equal=={'MDACC:M157':True,'MDACC:M806':True,'MDACC:M367':False}
for z in raw['pairs']:assert z['v4_header_validation']['full_file_length_accounted']
report=dict(input_hashes_unchanged=True,old_label_and_background_analysis_inputs_unchanged=True,
 exact_label_identity_checks=True,source_patient_group_counts_verified=True,R_crosscheck_max_auc_error=float(cross.AUC_difference.abs().max()),
 
 scripts={f:sha(P/f) for f in ['fetch_geo20194_metadata.py','audit_geo20194_identity.py','spotcheck_dipk_raw_arrays.py','check_dipk_cel_values.py','recalculate_geo20194_units.py','verify_geo20194_units.R','validate_geo20194_audit.py']})
(O/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('validated')
