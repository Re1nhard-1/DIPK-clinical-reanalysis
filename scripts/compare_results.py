from pathlib import Path
import argparse
import json
import math

import pandas as pd

SKIP = ('sha256', 'utc', 'date', 'environment', 'python', 'reader', 'hash', 'script', 'platform', 'package', 'runtime', 'seconds', 'path', 'verification')


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def compare_json(a, b, where, problems):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a:
            if any(word in key.lower() for word in SKIP):
                continue
            if key not in b:
                problems.append(f'{where}/{key}: missing')
            else:
                compare_json(a[key], b[key], f'{where}/{key}', problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f'{where}: length {len(a)} != {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            compare_json(x, y, f'{where}[{i}]', problems)
    elif not close(a, b):
        problems.append(f'{where}: {a!r} != {b!r}')


def compare_csv(a, b, where, problems):
    x = pd.read_csv(a, keep_default_na=False, float_precision='round_trip')
    y = pd.read_csv(b, keep_default_na=False, float_precision='round_trip')
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        problems.append(f'{where}: shape or columns differ')
        return
    for column in x.columns:
        u, v = x[column], y[column]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            for i, (p, q) in enumerate(zip(u.tolist(), v.tolist())):
                if not close(p, q):
                    problems.append(f'{where}:{column}[{i}]: {p!r} != {q!r}')
                    break
        elif not u.astype(str).equals(v.astype(str)):
            problems.append(f'{where}:{column}: values differ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(args.analysis).resolve() / 'work/treatment_response/method_screen'
    pairs = [(root / 'results/extension', target / 'DIPK/extension'), (root / 'results/til30', target / 'TNBC_TIL')]
    problems, checked = [], 0
    for supplied, regenerated in pairs:
        for path in sorted(supplied.iterdir()):
            other = regenerated / path.name
            if not other.exists():
                problems.append(f'{path.name}: not regenerated')
                continue
            if path.suffix == '.json':
                compare_json(json.loads(path.read_text(encoding='utf-8')), json.loads(other.read_text(encoding='utf-8')), path.name, problems)
            else:
                compare_csv(path, other, path.name, problems)
            checked += 1
    print(json.dumps({'files_compared': checked, 'differences': problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
