from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json,sys
import pandas as pd
import requests
sys.stdout.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parent
acc='GSE20194'
url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/{acc}/matrix/{acc}_series_matrix.txt.gz'
f=P/f'{acc}_matrix_header.txt'
manifest=P/f'{acc}_metadata_source.json'
record=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else dict(
 accession=acc,url=url,scope='Decompressed matrix metadata through table-begin only, not expression data')
if not f.exists():
    with requests.get(url,stream=True,timeout=40) as r:
        r.raise_for_status()
        record.update(retrieved_utc=datetime.now(timezone.utc).isoformat(),http_status=r.status_code,last_modified=r.headers.get('Last-Modified'))
        lines=[];size=0
        with gzip.GzipFile(fileobj=r.raw) as g:
            for line in g:
                lines.append(line);size+=len(line)
                if line.startswith(b'!series_matrix_table_begin'):break
                if size>6000000:raise ValueError('Header size bound exceeded')
        assert lines[-1].startswith(b'!series_matrix_table_begin')
        f.write_bytes(b''.join(lines))
b=f.read_bytes()
parsed=[z for z in csv.reader(b.decode('utf-8-sig').splitlines(),delimiter='\t') if z]
gsms=next(z[1:] for z in parsed if z[0]=='!Sample_geo_accession')
assert len(set(gsms))==len(gsms)
rows=[dict(GSM=g,accession=acc) for g in gsms];series={}
for fields in parsed:
    key,values=fields[0],fields[1:]
    if key.startswith('!Series_'):series.setdefault(key,[]).append(values)
    if key.startswith('!Sample_') and len(values)==len(rows):
        for row,value in zip(rows,values):
            if key=='!Sample_characteristics_ch1':
                field,sep,v=value.partition(': ')
                if sep:
                    assert field not in row or row[field]==v,(field,row['GSM'])
                    row[field]=v
            else:row[key.removeprefix('!Sample_')]=value
d=pd.DataFrame(rows)
d.to_csv(P/f'{acc}_metadata.csv',index=False)
(P/f'{acc}_series.json').write_text(json.dumps(series,ensure_ascii=False,indent=2),encoding='utf-8')
record.update(success=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),samples=len(d),fields=list(d.columns))
manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
for c in ['title','source_name_ch1','description','supplementary_file','pcr_vs_rd']:
    if c in d:print(c,':',d[c].head(8).tolist())
print({k:v for k,v in series.items() if any(t in k.lower() for t in ['title','summary','overall_design','pubmed'])})
