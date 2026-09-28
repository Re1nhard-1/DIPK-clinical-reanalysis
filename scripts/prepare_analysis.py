from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = Path(args.inputs).resolve()
    output = Path(args.output).resolve()
    if not inputs.is_dir():
        raise ValueError('Input directory does not exist.')
    if output.exists() or output.is_relative_to(inputs) or inputs.is_relative_to(output):
        raise ValueError('Choose a new output directory separate from the inputs.')
    target = output / 'work/treatment_response'
    shutil.copytree(inputs, target)
    results = target / 'method_screen/DIPK'
    shutil.copytree(root / 'results', results, dirs_exist_ok=True, ignore=shutil.ignore_patterns('inputs'))
    for path in (root / 'results/inputs').glob('*.csv'):
        shutil.copyfile(path, (results if path.name == 'saved_result_recalculation.csv' else target) / path.name)
    for path in (root / 'scripts/audit').iterdir():
        if path.is_file():
            shutil.copyfile(path, target / path.name)
    print(str(target))


if __name__ == '__main__':
    main()
