from pathlib import Path
import csv
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "method_screen/DIPK"
OUT = BASE / "literature"
WORKBOOK = OUT / "RPS_2017_supplementary_tables.xlsx"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
assert hashlib.sha256(WORKBOOK.read_bytes()).hexdigest() == "262f38bee76d73369d6927e328c742aaaf7f5e5a54e1e9a4de44ab2b9532c375"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sheet_cells(archive, name):
    strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
    book = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {n.attrib["Id"]: n.attrib["Target"] for n in rels}
    sheet = next(n for n in book.findall("s:sheets/s:sheet", NS) if n.attrib["name"] == name)
    rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    target = targets[rid]
    member = target.lstrip("/") if target.startswith("/") else "xl/" + target
    cells = {}
    for cell in ET.fromstring(archive.read(member)).findall(".//s:sheetData/s:row/s:c", NS):
        value = cell.find("s:v", NS)
        if cell.attrib.get("t") == "inlineStr":
            text = "".join(cell.find("s:is", NS).itertext())
        elif value is None:
            text = ""
        elif cell.attrib.get("t") == "s":
            text = strings[int(value.text)]
        else:
            text = value.text
        cells[cell.attrib["r"]] = text
    return cells


with zipfile.ZipFile(WORKBOOK) as archive:
    s3 = sheet_cells(archive, "TableS3")
    s1 = sheet_cells(archive, "TableS1")
assert s3["A1"] == "GEO ID"
listed = [v for k, v in s3.items() if k.startswith("A") and v.startswith("GSM")]
assert len(listed) == len(set(listed)) == 308
assert s1["A4"] == "GSE25055" and s1["F4"] == "188"

clinical_path = BASE / "GSE25055_fixed_prediction_crosswalk.csv"
array_path = BASE / "third_cohort/GSE20194_verified_array_crosswalk.csv"
metadata_path = ROOT / "GSE25055_metadata.csv"
clinical = read_rows(clinical_path)
arrays = read_rows(array_path)
metadata = {row["GSM"]: row for row in read_rows(metadata_path)}
cohort = {row["GSM"] for row in clinical}
assert len(cohort) == len(clinical) == 310
linked = {row["GSE25055_GSM"] for row in arrays if row["linked_GSE25055"] == "True"}
assert linked <= cohort and len(linked) == 188
reference = set(listed) & cohort
assert len(reference) == int(s1["F4"])

records = [
    {
        "GSE25055_GSM": gsm,
        "source_patient_title": metadata[gsm]["title"],
        "linked_to_GSE20194_here": gsm in linked,
        "listed_as_duplicate_in_RPS_TableS3": gsm in reference,
        "same_classification": (gsm in linked) == (gsm in reference),
    }
    for gsm in sorted(cohort)
]
with (OUT / "RPS_identity_reference_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

try:
    import openpyxl
    book = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    independent = {r[0] for r in book["TableS3"].iter_rows(values_only=True) if r[0] and str(r[0]).startswith("GSM")}
    assert independent == set(listed)
    book.close()
    extractor_check = {"reader": "openpyxl read-only", "version": openpyxl.__version__, "exact_identifier_agreement": True}
except ImportError:
    extractor_check = {"reader": "openpyxl read-only", "available": False}

evidence = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    
    "source_doi": "10.1158/1078-0432.CCR-16-2845",
    "supplement_doi": "10.1158/1078-0432.22468487.v1",
    "publisher_file_sha256_verified": True,
    "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORKBOOK, clinical_path, array_path, metadata_path]},
    "all_table_S3_unique_ids": len(listed),
    "GSE25055_listed_ids": len(reference),
    "current_GSE20194_linked_ids": len(linked),
    "intersection": len(reference & linked),
    "reference_only": sorted(reference - linked),
    "current_only": sorted(linked - reference),
    "neither": len(cohort - (reference | linked)),
    "separate_extractor_check": extractor_check,
    
}
(OUT / "RPS_identity_reference_verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: evidence[k] for k in ["all_table_S3_unique_ids", "GSE25055_listed_ids", "current_GSE20194_linked_ids", "intersection", "reference_only", "current_only", "neither", "separate_extractor_check"]}, indent=2))
