# DIPK clinical-evaluation reanalysis

Code and processed data accompanying *Outcome provenance and patient identity in clinical validation: a reanalysis of saved DIPK predictions* by Yunhao Jiang.

This study re-examines published DIPK predictions from three breast-cancer datasets. It checks whether evaluation labels represent observed pathological response and whether array records represent distinct patients. The analyses use fixed saved predictions; they do not retrain DIPK or run model inference.

## Main findings

In GSE25055, archived evaluation labels match the existing DLDA30 classifier output. Among the same 306 patients with observed outcomes, replacing those labels with pathological response reduces AUC from 0.852 to 0.748 while retaining response discrimination.

GSE20194 contains 278 arrays representing 248 source patient identifiers; 188 identifiers link to GSE25055. Patient-level and clinical-background sensitivity analyses assess how these features affect interpretation.

## Data and contents

| Dataset | Records used |
|---|---|
| GSE25055 | 310 archived records; 306 with observed outcomes |
| GSE32646 | 115 records with observed outcomes |
| GSE20194 | 278 arrays; 248 source patient identifiers |

- `results/`: outcome and patient mappings, statistical estimates, saved bootstrap draws and five analysis-result JSON files.
- `scripts/`: analysis, plotting and numerical-checking code.
- `figures/`: main and supplementary figures in PNG and PDF formats.

Data accessions, source versions and upstream input locations are listed in [datasets.md](datasets.md). Original archives, raw CEL files and the source workbook are obtained from their providers.

## Use

With Python 3.12, run from the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_processed.py --output processed_check.json
```

This checks the supplied estimates and saved bootstrap intervals. It does not rerun the upstream audit or generate new bootstrap samples.

For the complete saved-prediction analysis and figure generation, follow [run_commands.md](run_commands.md). This also requires the listed upstream files and base R (tested with R 4.5.2).

## License

Original code is licensed under the [MIT License](LICENSE). Original results and figures are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party data and materials retain their original terms.
