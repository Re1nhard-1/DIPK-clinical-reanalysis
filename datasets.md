# Data sources

| Source | Access | Use |
|---|---|---|
| GSE25055 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25055) | Observed outcomes, DLDA30 labels and clinical metadata |
| GSE32646 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32646) | Observed outcomes and clinical metadata |
| GSE20194 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20194) | Outcomes, patient identifiers and array filenames |
| GSE25065 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25065) | Outcomes and DLDA30 labels for the SLM-HRD second cohort |
| GSE41998 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41998) | AC response, final pathological response and specimen names |
| GSE20194 workbook | [Sample information](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/suppl/GSE20194_MDACC_Sample_Info.xls.gz) | Sample_Info and CodeBook |
| DIPK | [Repository, commit 0edfc15](https://github.com/user15632/DIPK/tree/0edfc151fbbdb7864d577865f9541a689d706c5b) | Saved predictions and figure values; original archive linked from its README |
| SLM-HRD | [Source Data Figure 4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41588-025-02108-2/MediaObjects/41588_2025_2108_MOESM7_ESM.xlsx), Haider et al. 2025, doi:10.1038/s41588-025-02108-2 | Released scores, sheet g |
| RPS duplicate list | [Supplementary tables](https://doi.org/10.1158/1078-0432.22468487.v1), Pitroda et al. 2017, CC BY 4.0 | Table S3 exclusion list |
| TIL-30 | [Repository, commit 55cbe09](https://github.com/RamanLab/TNBC-single-cell-markers/tree/55cbe09820c7c679995630b65e682b1c98f4b869), GPL-3.0 | Saved labels and probabilities, input row identities and cached notebook output |
| VAEN | [Repository, commit 69b90b8](https://github.com/bsml320/VAEN/tree/69b90b80c1c1419679c51c65a681c071fdba2779) | GSE25055 label-selection code |

`scripts/fetch_inputs.py` downloads the GEO headers and the SLM-HRD, RPS, TIL-30 and VAEN files and checks their SHA256.

The original audit also needs these files under `INPUTS/`:

| Relative path | Files |
|---|---|
| Root | `GSE20194_matrix_header.txt`, `GSE20194_MDACC_Sample_Info.xls` and `.xls.gz` |
| `method_screen/DIPK/Figure/Task5/` | `result_geo01.csv`, `result_geo02.csv`, `result_geo03.csv` |
| `method_screen/DIPK/archive_clinical/Task5/DataPreprocess/GEO/` | `1/GSE25055.csv`, `2/GSE32646.csv`, `3/GSE20194.csv` |
| `method_screen/DIPK/archive_clinical/Task5/test_geo01/`, `test_geo02/`, `test_geo03/` | `predict_pCR.csv` and `predict_RD.csv` in each directory |
| `method_screen/DIPK/third_cohort/` | `CELFileData.cpp` from the [versioned parser source](https://raw.githubusercontent.com/HenrikBengtsson/affxparser/6fd83fcaca7707c1b198482ae11bc8aeb0e2c0b2/src/fusion/file/CELFileData.cpp) |
| `method_screen/DIPK/third_cohort/raw_spotcheck/` | Six CEL archives listed in [raw_spotcheck_selection.csv](results/third_cohort/raw_spotcheck_selection.csv) |
