# Data sources

| Source | Access | Use |
|---|---|---|
| GSE25055 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25055) | Observed outcomes, DLDA30 labels and clinical metadata |
| GSE32646 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32646) | Observed outcomes and clinical metadata |
| GSE20194 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20194) | Outcomes, patient identifiers and array filenames |
| GSE20194 workbook | [Sample information](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/suppl/GSE20194_MDACC_Sample_Info.xls.gz) | Sample_Info and CodeBook |
| DIPK | [Fixed repository version](https://github.com/user15632/DIPK/tree/0edfc151fbbdb7864d577865f9541a689d706c5b) | Saved predictions and figure values; original archive linked from its README |

Processed tables are in `results/`. Original archives, raw CEL files and the source workbook are obtained from their providers.

For the full analysis, arrange the original files under `INPUTS/`:

| Relative path | Files |
|---|---|
| Root | `GSE20194_matrix_header.txt`, `GSE20194_MDACC_Sample_Info.xls` and `.xls.gz` |
| `method_screen/DIPK/Figure/Task5/` | `result_geo01.csv`, `result_geo02.csv`, `result_geo03.csv` |
| `method_screen/DIPK/archive_clinical/Task5/DataPreprocess/GEO/` | `1/GSE25055.csv`, `2/GSE32646.csv`, `3/GSE20194.csv` |
| `method_screen/DIPK/archive_clinical/Task5/test_geo01/`, `test_geo02/`, `test_geo03/` | `predict_pCR.csv` and `predict_RD.csv` in each directory |
| `method_screen/DIPK/third_cohort/` | `CELFileData.cpp` from the [versioned parser source](https://raw.githubusercontent.com/HenrikBengtsson/affxparser/6fd83fcaca7707c1b198482ae11bc8aeb0e2c0b2/src/fusion/file/CELFileData.cpp) |
| `method_screen/DIPK/third_cohort/raw_spotcheck/` | Six CEL archives listed in [raw_spotcheck_selection.csv](results/third_cohort/raw_spotcheck_selection.csv) |

The matrix header ends at `!series_matrix_table_begin`; expression rows are not needed. GEO submitter contact columns are omitted from distributed tables. Saved execution records retain original data and script hashes.
