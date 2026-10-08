from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import json

import openpyxl
import requests

TIL = 'https://raw.githubusercontent.com/RamanLab/TNBC-single-cell-markers/55cbe09820c7c679995630b65e682b1c98f4b869/code-data-for-submission/'
GEO = 'https://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/matrix/{1}_series_matrix.txt.gz'
SOURCES = [
    ('GSE25055_matrix_header.txt', GEO.format('GSE25', 'GSE25055'), '17bcc1f90587fb35fb383894745872accc75695ccf210e6fb7e27227f8cf43f5'),
    ('GSE32646_matrix_header.txt', GEO.format('GSE32', 'GSE32646'), 'a66431b274cb6e037c7c98be47dd04e8fdd057070ae5a1f80f63808928c81da1'),
    ('GSE41998_matrix_header.txt', GEO.format('GSE41', 'GSE41998'), '00d379ebc705c35948692848f91e5fcbfd05408c69c1848e77b05cc454f22dae'),
    ('GSE20194_matrix_header.txt', GEO.format('GSE20', 'GSE20194'), '998acc6ca9aa469a8944120c04d67bbc5d5cb54ba3c265f2600e286b075af8da'),
    ('method_screen/DIPK/inference_feasibility/GSE25065_matrix_header.txt', GEO.format('GSE25', 'GSE25065'),
     'e6413be43872c052374213f73643b1ff3d60c5024e0d6b99dbc71c3bb733ee7d'),
    ('method_screen/SYLVER/source_figure4.xlsx',
     'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41588-025-02108-2/MediaObjects/41588_2025_2108_MOESM7_ESM.xlsx',
     '63484d3a3e9f0b7f31e9d8183775b4cd7100520804c7711b1228e1b2bc89868f'),
    ('method_screen/DIPK/literature/RPS_2017_supplementary_tables.xlsx', 'https://ndownloader.figshare.com/files/39919940',
     '262f38bee76d73369d6927e328c742aaaf7f5e5a54e1e9a4de44ab2b9532c375'),
    ('method_screen/TNBC_TIL/source/external_validation_11_test_set.ipynb', TIL + 'code/external_validation_11_test_set.ipynb',
     'f2d4ae23131f48f8cc1ed992730fe0771cba12b0978c17041d6ab1ab918438e3'),
    ('method_screen/TNBC_TIL/saved_outputs/y_test_20194.pkl', TIL + 'data/pickled_models/y_test_20194.pkl',
     '19512a0e51c7253a593e107e041e3b6e9eea17f6559b873d26cd71ec79c4a089'),
    ('method_screen/TNBC_TIL/saved_outputs/y_probs_20194.pkl', TIL + 'data/pickled_models/y_probs_20194.pkl',
     'a043745a88a3d47ffcef054b796ef18a259690bbb9ea5d296c60705ab343a78e'),
    ('method_screen/TNBC_TIL/saved_outputs/y_test_41998.pkl', TIL + 'data/pickled_models/y_test_41998.pkl',
     '50db062af78e79eee6b5ab96e441dd7683d848ac8f18e97eebbf2169ea47f9e9'),
    ('method_screen/TNBC_TIL/saved_outputs/y_probs_41998.pkl', TIL + 'data/pickled_models/y_probs_41998.pkl',
     '2482f5d578ab911a6c757fe571618dcdc27b71430e22fce858b905c4a665ae45'),
    ('method_screen/TNBC_TIL/source/train_features_new_only_tnbc.txt', TIL + 'data/train_features_new_only_tnbc.txt',
     '222538f7b74e8a213037d3d1f4297b09b8203b7804cf5f55981cd4611e9fe8ba'),
    ('method_screen/TNBC_TIL/source/test_features_new_only_tnbc_GSE20194.txt', TIL + 'data/test_features_new_only_tnbc_GSE20194.txt',
     '54762b0532931f4380e14de67eb6a3e0a51cafed21e6c582e33e59bbf12e97cb'),
    ('method_screen/TNBC_TIL/source/GSE41998_processed_log2_using_ac_response_label.txt',
     TIL + 'data/GSE41998_processed_log2_using_ac_response_label.txt',
     '1c7fad17505e95d209835a21c96c0aff3dbb5ebb38ad7878a5d594575998dacb'),
    ('vaen_source/Figure/Figure5/GSE25055/03.pCR.R',
     'https://raw.githubusercontent.com/bsml320/VAEN/69b90b80c1c1419679c51c65a681c071fdba2779/Figure/Figure5/GSE25055/03.pCR.R',
     'dbeff6576bc5cc2b61f05f55458f2dd189e93bf235d9faf621d0cee55fc090b5'),
]
DERIVED = {
    'method_screen/SYLVER/hatzis1_released_scores.csv': '024b8e7bdeedfbb318a66efa0636dedec5a1cd9a06e04faf9c5675b3fa5a0e9e',
    'method_screen/TNBC_TIL/train_input_identity.csv': 'f29c9308a6d1337cc17e2bad39d9e2024491cf6ea92abfaaf5713f684824f4e2',
    'method_screen/TNBC_TIL/GSE20194_input_identity.csv': '0c66e5bb7b43a73b988d4397eefa5d89fa1cdabe7e00a8302d31dd4d09a01647',
    'method_screen/TNBC_TIL/GSE41998_input_identity.csv': 'e9f99b5ee0c785d6dd7e17df632f6596750aee84a7ee17f55bf6cc6dbf0f75bd',
    'method_screen/TNBC_TIL/GSE41998_source_outcomes.csv': '2f9da2ab9ab6ceef7988eb97a43368b8a59f67955644b6544815ce79b8295cae',
    'method_screen/TNBC_TIL/GSE20194_links_to_training.csv': '6d4a8266e5494547a15ea0d014c63d800158de25bce980a76eb054261ee53f4b',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(url, geo_header):
    if not geo_header:
        response = requests.get(url, timeout=300, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return response.content
    lines = []
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for line in handle:
                lines.append(line)
                if line.startswith(b'!series_matrix_table_begin'):
                    break
    return b''.join(lines)


def table(rows, header):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def identity(path, labelled):
    rows = []
    with path.open(encoding='utf-8') as handle:
        handle.readline()
        for i, line in enumerate(handle):
            cells = line.rstrip('\n').split('\t')
            if labelled:
                rows.append([i, cells[0], cells[0], cells[-1]])
            else:
                gsm, label = cells[0].split('_', 1)
                rows.append([i, cells[0], gsm, label])
    return rows


def write(target, name, data, expected=None):
    path = target / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if expected is not None and sha(data) != expected:
        raise ValueError(f'{name}: SHA256 differs from the analysed version')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    args = parser.parse_args()
    target = Path(args.analysis).resolve() / 'work/treatment_response'
    if not (target / 'method_screen/DIPK').is_dir():
        raise ValueError('Run prepare_analysis.py first.')
    for name, url, expected in SOURCES:
        path = target / name
        if not path.exists():
            write(target, name, download(url, name.endswith('_matrix_header.txt')), expected)
        elif sha(path.read_bytes()) != expected:
            raise ValueError(f'{name}: SHA256 differs from the analysed version')
    workbook = openpyxl.load_workbook(target / 'method_screen/SYLVER/source_figure4.xlsx', read_only=True, data_only=True)
    rows = [list(r) for r in workbook['g'].iter_rows(min_row=3, max_row=308, min_col=1, max_col=4, values_only=True)]
    name = 'method_screen/SYLVER/hatzis1_released_scores.csv'
    write(target, name, table(rows, ['source_sample_id', 'SLMHRD_score', 'source_response_group', 'source_hormone_receptor_status']), DERIVED[name])
    til = target / 'method_screen/TNBC_TIL'
    header = ['source_row', 'source_index', 'GSM', 'source_label']
    train = identity(til / 'source/train_features_new_only_tnbc.txt', False)
    validation = identity(til / 'source/test_features_new_only_tnbc_GSE20194.txt', False)
    external = identity(til / 'source/GSE41998_processed_log2_using_ac_response_label.txt', True)
    for name, rows in [('train_input_identity.csv', train), ('GSE20194_input_identity.csv', validation),
                       ('GSE41998_input_identity.csv', external)]:
        write(til, name, table(rows, header), DERIVED['method_screen/TNBC_TIL/' + name])
    with (target / 'GSE41998_metadata.csv').open(encoding='utf-8', newline='') as handle:
        meta = {r['GSM']: r for r in csv.DictReader(handle)}
    fields = ['ac response', 'pcr', 'pcrrcb1', 'treatment arm']
    write(til, 'GSE41998_source_outcomes.csv', table([r + [meta[r[2]][f] for f in fields] for r in external], header + fields), DERIVED['method_screen/TNBC_TIL/GSE41998_source_outcomes.csv'])
    with (target / 'method_screen/DIPK/third_cohort/GSE20194_GSE25055_array_links.csv').open(encoding='utf-8', newline='') as handle:
        links = list(csv.DictReader(handle))
    validation_gsm = {r[2] for r in validation}
    training_gsm = {r[2] for r in train}
    kept = [r for r in links if r['GSM_20194'] in validation_gsm and r['GSM_25055'] in training_gsm]
    fieldnames = list(links[0])
    write(til, 'GSE20194_links_to_training.csv', table([[r[k] for k in fieldnames] for r in kept], fieldnames), DERIVED['method_screen/TNBC_TIL/GSE20194_links_to_training.csv'])
    print(json.dumps({'downloaded_or_verified': len(SOURCES), 'derived_tables': 6, 'training_links': len(kept)}))


if __name__ == '__main__':
    main()
