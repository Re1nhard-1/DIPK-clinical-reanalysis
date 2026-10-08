# Run commands

Python 3.12 and R 4.5.2.

## Check supplied results

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_processed.py --output processed_check.json
```

## Regenerate results/extension and results/til30

```sh
python scripts/prepare_analysis.py --output analysis
python scripts/fetch_inputs.py --analysis analysis
cd analysis
python work/treatment_response/extend_dipk.py
python work/treatment_response/extend_dipk_subtype.py
python work/treatment_response/extend_dipk_transfer_uncertainty.py
python work/treatment_response/extend_dipk_clinical.py
python work/treatment_response/check_dipk_source_conditioning.py
python work/treatment_response/check_dipk_published_identity.py
python work/treatment_response/check_dipk_proxy_alignment.py
python work/treatment_response/check_dipk_proxy_background.py
python work/treatment_response/check_dipk_sylver_comparison.py
python work/treatment_response/extract_sylver_second_cohort.py
python work/treatment_response/check_sylver_second_cohort.py
python work/treatment_response/method_screen/TNBC_TIL/analyse_saved_outputs.py
python work/treatment_response/method_screen/TNBC_TIL/analyse_specimen_groups.py
python work/treatment_response/method_screen/TNBC_TIL/check_training_labels.py
Rscript work/treatment_response/verify_vaen_label_snippet.R
cd ..
python scripts/compare_results.py --analysis analysis
```

## Original audit

Place the files listed in [datasets.md](datasets.md) under `INPUTS/`, then:

```sh
python scripts/prepare_analysis.py --inputs INPUTS --output audit
cd audit
python work/treatment_response/recalculate_dipk_labels.py
Rscript work/treatment_response/verify_dipk_recalculation.R
python work/treatment_response/dipk_background_sensitivity.py
Rscript work/treatment_response/dipk_adjusted_association.R
python work/treatment_response/check_dipk_sensitivity.py
python work/treatment_response/audit_geo20194_identity.py
python work/treatment_response/recalculate_geo20194_units.py
Rscript work/treatment_response/verify_geo20194_units.R
python work/treatment_response/check_dipk_cel_values.py
python work/treatment_response/validate_geo20194_audit.py
```
