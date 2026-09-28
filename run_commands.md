# Run commands

Use Python 3.12 and base R (tested with R 4.5.2).

## Supplied results

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_processed.py --output processed_check.json
```

This checks estimates and confidence limits using the supplied tables and saved bootstrap draws.

## Analysis and figures

Arrange upstream files as described in [datasets.md](datasets.md). Use a new output directory and run these commands in order:

```sh
python scripts/prepare_analysis.py --inputs INPUTS --output analysis
cd analysis
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
python work/treatment_response/plot_dipk_repair.py
python work/treatment_response/plot_dipk_sensitivity.py
python work/treatment_response/plot_geo20194_audit.py
```

Results are written under `analysis/work/treatment_response/method_screen/DIPK/`. These commands analyze saved predictions; they do not train DIPK. A fresh-environment installation has not been tested.
