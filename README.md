# DIPK and TIL-30 clinical-evaluation reanalysis

Code and processed data for *Outcome labels and patient reuse in clinical validation: reanalyses of the DIPK and TIL-30 breast-cancer evaluations* (Yunhao Jiang).

- `results/`: estimates, processed inputs and bootstrap/permutation draws.
- `scripts/`: analysis code. `scripts/fetch_inputs.py` downloads third-party inputs and checks their SHA256.

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_processed.py --output processed_check.json
```

Full steps: [run_commands.md](run_commands.md). Sources: [datasets.md](datasets.md).

Code: [MIT](LICENSE). Results: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party materials keep their own terms.
