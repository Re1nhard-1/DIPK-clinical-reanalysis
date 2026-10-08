from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory.')
    target = output / 'work/treatment_response'
    if args.inputs:
        inputs = Path(args.inputs).resolve()
        if not inputs.is_dir() or output.is_relative_to(inputs) or inputs.is_relative_to(output):
            raise ValueError('Use an existing input directory separate from the output.')
        shutil.copytree(inputs, target)
    target.mkdir(parents=True, exist_ok=True)
    dipk = target / 'method_screen/DIPK'
    shutil.copytree(root / 'results', dipk, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('inputs', 'extension', 'til30'))
    for path in (root / 'results/inputs').glob('*.csv'):
        if path.name == 'saved_result_recalculation.csv':
            destination = dipk
        elif path.name == 'GSE25065_candidate_metadata.csv':
            destination = dipk / 'inference_feasibility'
        else:
            destination = target
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination / path.name)
    for folder, destination in [('audit', target), ('extension', target), ('til30', target / 'method_screen/TNBC_TIL')]:
        destination.mkdir(parents=True, exist_ok=True)
        for path in (root / 'scripts' / folder).iterdir():
            if path.is_file():
                shutil.copyfile(path, destination / path.name)
    (dipk / 'extension').mkdir(parents=True, exist_ok=True)
    print(str(target))


if __name__ == '__main__':
    main()
