from pathlib import Path
import gzip,hashlib,json,struct
from types import SimpleNamespace
import numpy as np
import pandas as pd
import Bio
from Bio.Affy import CelFile
P=Path(__file__).resolve().parent;O=P/'method_screen'/'DIPK'/'third_cohort'
sel=pd.read_csv(O/'raw_spotcheck_selection.csv');pairs=[]
def read_v4(f):


    data=gzip.decompress(f.read_bytes());pos=0
    def unpack(fmt):
        nonlocal pos
        v=struct.unpack_from(fmt,data,pos);pos+=struct.calcsize(fmt);return v
    magic,version,rows,cols,n=unpack('<5i')
    assert magic==64 and version==4 and rows*cols==n==712*712
    strings=[]
    for k in range(3):
        size,=unpack('<i');assert 0<=size<100000
        strings.append(data[pos:pos+size].decode('ascii'));pos+=size
    margin,nout,nmask,subgrids=unpack('<iIIi')
    assert subgrids==0 and 0<=margin<=20
    assert pos+10*n+4*(nout+nmask)==len(data)
    a=np.frombuffer(data,dtype=[('intensities','<f4'),('stdevs','<f4'),('npix','<i2')],count=n,offset=pos)
    return SimpleNamespace(**{field:a[field].reshape(rows,cols) for field in a.dtype.names},
       header_validation=dict(cell_margin=margin,outliers=nout,masks=nmask,subgrids=subgrids,data_offset=pos,full_file_length_accounted=True))
for row in sel.to_dict('records'):
    records=[]
    for cohort in ['20194','25055']:
        f=O/'raw_spotcheck'/row[f'supplementary_file_{cohort}'].rsplit('/',1)[-1]
        if cohort=='20194':
            with gzip.open(f,'rt') as h:rec=CelFile.read(h,version=3)
        else:rec=read_v4(f)
        assert rec.intensities.shape==(712,712) and np.isfinite(rec.intensities).all()
        records.append(rec)
    a,b=records;detail={}
    for field in ['intensities','stdevs','npix']:
        x,y=[np.asarray(getattr(r,field)) for r in records]

        xx,yy=x.astype('<f4'),y.astype('<f4')
        detail[field]=dict(equal_after_float32_conversion=bool(np.array_equal(xx,yy)),
          differing_cells=int(np.count_nonzero(xx!=yy)),max_abs_difference=float(np.max(np.abs(x-y))),
          GSE20194_float32_sha256=hashlib.sha256(xx.tobytes()).hexdigest(),
          GSE25055_float32_sha256=hashlib.sha256(yy.tobytes()).hexdigest())
    pairs.append(dict(patient_key=row['patient_key'],GSM_20194=row['GSM_20194'],GSM_25055=row['GSM_25055'],
      n_cells=712*712,formats=[3,4],v4_header_validation=b.header_validation,comparison=detail))
report=dict(
  biopython_version=Bio.__version__,documentation='https://biopython.org/docs/latest/api/Bio.Affy.CelFile.html',
  parser_source_sha256=hashlib.sha256(Path(CelFile.__file__).read_bytes()).hexdigest(),
  v4_layout_source={'url': 'https://raw.githubusercontent.com/HenrikBengtsson/affxparser/6fd83fcaca7707c1b198482ae11bc8aeb0e2c0b2/src/fusion/file/CELFileData.cpp', 'commit': '6fd83fcaca7707c1b198482ae11bc8aeb0e2c0b2', 'sha256': '9a0ffa003475b7dd6cb8c7cd320273cfe3ce8cb1028f472df88d72a4f4ee2d57'},
  
  pairs=pairs)
(O/'raw_value_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for z in pairs:
    print(z['patient_key'],{k:{q:v for q,v in d.items() if 'sha256' not in q} for k,d in z['comparison'].items()})
