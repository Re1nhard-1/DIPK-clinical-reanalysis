from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip,hashlib,io,json,sys
import pandas as pd
import requests
P=Path(__file__).resolve().parent;O=P/'method_screen'/'DIPK'/'third_cohort'
R=O/'raw_spotcheck';R.mkdir(exist_ok=True)
selection=pd.read_csv(O/'raw_spotcheck_selection.csv')
urls=sorted(set(selection.supplementary_file_20194)|set(selection.supplementary_file_25055))
assert len(urls)<=6
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(ftp):
    url=ftp.replace('ftp://','https://',1);f=R/url.rsplit('/',1)[-1]
    record=dict(url=url,file=f.name)
    try:
        if not f.exists():
            with requests.get(url,stream=True,timeout=40) as r:
                r.raise_for_status();chunks=[];size=0
                for chunk in r.iter_content(262144):
                    size+=len(chunk)
                    if size>10*1024**2:raise ValueError('10 MiB compressed file bound exceeded')
                    chunks.append(chunk)
                f.write_bytes(b''.join(chunks))
            record['retrieved_utc']=datetime.now(timezone.utc).isoformat()
        b=f.read_bytes();assert len(b)<=10*1024**2
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as g:raw=g.read(60*1024**2+1)
        assert len(raw)<=60*1024**2
        record.update(ok=True,compressed_bytes=len(b),compressed_sha256=sha(b),
                      cel_bytes=len(raw),cel_sha256=sha(raw),cel_prefix_hex=raw[:16].hex())
    except Exception as e:record.update(ok=False,error=str(e))
    return ftp,record
with ThreadPoolExecutor(max_workers=3) as pool:files=dict(pool.map(fetch,urls))
pairs=[]
for row in selection.to_dict('records'):
    a,b=files[row['supplementary_file_20194']],files[row['supplementary_file_25055']]
    pairs.append(dict(patient_key=row['patient_key'],GSM_20194=row['GSM_20194'],GSM_25055=row['GSM_25055'],
       same_raw_basename=row['same_raw_basename'],both_downloads_ok=a['ok'] and b['ok'],
       identical_decompressed_CEL=(a['cel_sha256']==b['cel_sha256']) if a['ok'] and b['ok'] else None))
result=dict(scope='Deterministic spot checks, not raw-content validation of every linked patient.',
    selection_sha256=sha((O/'raw_spotcheck_selection.csv').read_bytes()),files=files,pairs=pairs,
    compressed_total_bytes=sum(f.get('compressed_bytes',0) for f in files.values()))
(O/'raw_content_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(pairs=pairs,compressed_total_bytes=result['compressed_total_bytes']),indent=2))
