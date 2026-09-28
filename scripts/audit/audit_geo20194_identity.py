from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,sys
import pandas as pd
import numpy as np
import xlrd
P=Path(__file__).resolve().parent;D=P/'method_screen'/'DIPK';O=D/'third_cohort'
O.mkdir(exist_ok=True);A=D/'archive_clinical'/'Task5'
sys.stdout.reconfigure(encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[P/f'{g}_metadata.csv' for g in ['GSE20194','GSE25055','GSE32646']]+[
 P/'GSE20194_matrix_header.txt',P/'GSE20194_MDACC_Sample_Info.xls',P/'GSE20194_MDACC_Sample_Info.xls.gz',
 A/'DataPreprocess/GEO/3/GSE20194.csv',A/'test_geo03/predict_pCR.csv',A/'test_geo03/predict_RD.csv']
hashes={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in inputs}
m=pd.read_csv(P/'GSE20194_metadata.csv',keep_default_na=False)
w=pd.read_excel(P/'GSE20194_MDACC_Sample_Info.xls',sheet_name='Sample_Info',keep_default_na=False)
assert len(m)==len(w)==278 and m.GSM.is_unique and w.title.is_unique
assert set(m.title)==set(w.title)
x=m.merge(w[['title','CEL file','source name','description','characteristics: pCR_vs_RD','Treatment Code']],on='title',suffixes=('','_workbook'),validate='one_to_one')
def basename(s):return re.sub(r'^GSM\d+_','',s.rsplit('/',1)[-1])
x['raw_basename']=x.supplementary_file.map(basename)
filename_normalization={'FL786 642.CEL':'FL786_642.CEL','FL1141-801(2).CEL':'FL1141-801_2_.CEL'}
assert x.raw_basename.eq(x['CEL file'].replace(filename_normalization)+'.gz').all()
assert x.source_name_ch1.eq(x['source name']).all()
assert x.pcr_vs_rd.eq(x['characteristics: pCR_vs_RD']).all()
assert x['treatment code'].eq(x['Treatment Code'].astype(str)).all()
x['source_patient_id']=x.source_name_ch1.str.extract(r'^Sample ID -- ([^,]+),',expand=False)
assert x.source_patient_id.notna().all()
x['sample_alias']=x.title.str.replace(r'^BR_FNA_','',regex=True).str.replace('-','_',regex=False)
x['patient_alias']=x.sample_alias.str.replace(r'R1$','',regex=True)
assert x.groupby('source_patient_id').patient_alias.nunique().eq(1).all()
assert x.groupby('patient_alias').source_patient_id.nunique().eq(1).all()
assert x.groupby('source_patient_id').pcr_vs_rd.nunique().eq(1).all()
x['patient_key']='MDACC:'+x.patient_alias
x['maqc_status']=x.description_workbook.str.extract(r'Status: (\w+)',expand=False)
assert x.maqc_status.isin(['MAQC_T','MAQC_V','MDA_R','MAQC_Q']).all()
assert x.loc[x.maqc_status.isin(['MAQC_T','MAQC_V']),'patient_key'].is_unique
raw_labels=pd.read_csv(A/'DataPreprocess/GEO/3/GSE20194.csv',sep='\t',header=None,keep_default_na=False)

label_rows=[];reference=x.set_index('GSM')
for values in raw_labels.itertuples(index=False,name=None):
    gsm=values[0];fields={}
    for cell in values[1:]:
        key,sep,value=cell.partition(': ')
        assert sep and key not in fields
        assert reference.loc[gsm,key]==value,(gsm,key,value)
        fields[key]=value
    assert 'pcr_vs_rd' in fields
    label_rows.append(dict(GSM=gsm,response_archive='pcr_vs_rd: '+fields['pcr_vs_rd'],
                           raw_archive_fields=' | '.join(values[1:])))
labels=pd.DataFrame(label_rows)
assert labels.GSM.is_unique and set(labels.GSM)==set(x.GSM)
x=x.merge(labels,on='GSM',validate='one_to_one')
assert x.pcr_vs_rd.eq(x.response_archive.str.removeprefix('pcr_vs_rd: ')).all()
scores=[]
for name in ['pCR','RD']:
    s=pd.read_csv(A/f'test_geo03/predict_{name}.csv').rename(columns={'Cell':'GSM',name:'ln_ic50'})
    s['archive_group']=name;scores.append(s)
s=pd.concat(scores,ignore_index=True)
assert s.GSM.is_unique and set(s.GSM)==set(x.GSM)
x=x.merge(s,on='GSM',validate='one_to_one')
assert x.archive_group.eq(x.pcr_vs_rd).all()
x['score']=-x.ln_ic50;x['observed_pcr']=x.pcr_vs_rd.eq('pCR').astype(int)
b=pd.read_csv(P/'GSE25055_metadata.csv',keep_default_na=False)
b['sample_alias']=b['sample id'].str.replace('-','_',regex=False)
b['patient_key']=b.source+':'+b.sample_alias
b['raw_basename']=b.supplementary_file.map(basename)
assert b.patient_key.is_unique
z=x.merge(b,on='patient_key',suffixes=('_20194','_25055'),validate='many_to_one')
z['same_raw_basename']=z.raw_basename_20194.eq(z.raw_basename_25055)
z['response_equal']=z.pcr_vs_rd.eq(z.pathologic_response_pcr_rd)
z['age_difference']=pd.to_numeric(z.age)-pd.to_numeric(z.age_years)
z['er_ihc_equal']=z.er_status.eq(z.er_status_ihc)
cols=['patient_key','GSM_20194','GSM_25055','source_patient_id','sample_alias_20194','sample_alias_25055','maqc_status',
 'raw_basename_20194','raw_basename_25055','same_raw_basename','pcr_vs_rd','pathologic_response_pcr_rd','response_equal',
 'age','age_years','age_difference','er_status','er_status_ihc','er_ihc_equal','supplementary_file_20194','supplementary_file_25055']
z[cols].to_csv(O/'GSE20194_GSE25055_array_links.csv',index=False)
raw=x.merge(b,on='raw_basename',suffixes=('_20194','_25055'),validate='one_to_one')
assert raw.patient_key_20194.eq(raw.patient_key_25055).all()
x['linked_GSE25055']=x.patient_key.isin(b.patient_key)
x['GSE25055_GSM']=x.patient_key.map(b.set_index('patient_key').GSM).fillna('')
x.to_csv(O/'GSE20194_verified_array_crosswalk.csv',index=False)
patient=x.groupby('patient_key',sort=True).agg(source_patient_id=('source_patient_id','first'),
 array_count=('GSM','size'),observed_pcr=('observed_pcr','first'),score=('score','mean'),
 linked_GSE25055=('linked_GSE25055','first'),GSE25055_GSM=('GSE25055_GSM','first'),
 GSMs=('GSM',lambda s:';'.join(sorted(s))),maqc_status=('maqc_status',lambda s:';'.join(sorted(s))))
patient.to_csv(O/'GSE20194_patient_means.csv')
c=pd.read_csv(P/'GSE32646_metadata.csv',keep_default_na=False)
c['raw_basename']=c.supplementary_file.map(basename)
same=z[z.same_raw_basename].sort_values('GSM_20194')
different=z[~z.same_raw_basename & ~z.sample_alias_20194.str.endswith('R1')].sort_values('GSM_20194')
selected=pd.concat([same.iloc[[0,-1]],different.iloc[:1]])[cols].drop_duplicates()
selected.to_csv(O/'raw_spotcheck_selection.csv',index=False)
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),
 scope='Official label mapping and declared source-identity audit; not independent clinical adjudication or DIPK training reproduction.',
 input_hashes=hashes,workbook_to_GEO_filename_formatting=filename_normalization,rows=len(x),all_archive_labels_match_observed=True,provided_pcr=int(x.observed_pcr.sum()),
 source_patient_groups=len(patient),patient_pcr=int(patient.observed_pcr.sum()),
 array_multiplicities={str(k):int(v) for k,v in patient.array_count.value_counts().items()},
 maqc_status_counts=x.maqc_status.value_counts().to_dict(),no_within_patient_response_disagreement=True,
 cross_series=dict(GSM_overlap=len(set(x.GSM)&set(b.GSM)),shared_patient_aliases=z.patient_key.nunique(),
 linked_GSE20194_arrays=len(z),shared_raw_basenames=len(raw),
 unmatched_GSE20194_patient_groups=int((~patient.linked_GSE25055).sum()),
 response_disagreements_arrays=int((~z.response_equal).sum()),
 response_disagreements_patient_groups=z.loc[~z.response_equal,'patient_key'].nunique(),
 age_difference_counts=z.age_difference.value_counts().to_dict(),ER_IHC_disagreement_arrays=int((~z.er_ihc_equal).sum()),
 GSE32646_GSM_overlap=len(set(c.GSM)&set(x.GSM)),GSE32646_raw_basename_overlap=len(set(c.raw_basename)&set(x.raw_basename))),
 limitations=['Source-ID evidence is not a genome-wide identity test; absence of a link does not prove independence.',
 'Clinical annotations can differ; neither cohort is silently overwritten.',
 'MAQC training/validation refer to the original MAQC study, not DIPK training.',
 'Do not pool series or call cross-clinical overlap DIPK training/test leakage.'],
 runtime=dict(python=sys.version,pandas=pd.__version__,numpy=np.__version__,xlrd=xlrd.__version__),script_sha256=sha(Path(__file__)))
assert hashes=={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in inputs}
(O/'identity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['input_hashes','runtime']},indent=2))
