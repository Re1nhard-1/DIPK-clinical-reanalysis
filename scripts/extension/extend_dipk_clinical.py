from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

P = Path(__file__).resolve().parent
D = P / 'method_screen/DIPK'
O = D / 'extension'
COLS = ['age10', 'er_positive', 'pr_positive', 'her2_positive', 't_stage', 'node_positive', 'grade', 'score']
METRICS = ['auc', 'ap', 'brier', 'logloss']
CATEGORIES = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']
CHECK = {'reference_fits': 0, 'max_coefficient_difference': 0.0, 'max_probability_difference': 0.0,
         'max_preprocessing_difference': 0.0, 'max_gradient': 0.0, 'fits': 0, 'invalid_draws': []}



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fields(cohort):
    path = P / f'{cohort}_matrix_header.txt'
    lines = [r for r in csv.reader(path.read_text(encoding='utf-8-sig').splitlines(), delimiter='\t') if r]
    ids = next(r[1:] for r in lines if r[0] == '!Sample_geo_accession')
    rows = {g: {} for g in ids}
    for r in lines:
        if r[0] == '!Sample_characteristics_ch1':
            assert len(r) == len(ids) + 1
            for g, text in zip(ids, r[1:]):
                if not text:
                    continue
                field, sep, value = text.partition(': ')
                assert sep and field not in rows[g]
                rows[g][field] = value
    return pd.DataFrame.from_dict(rows, orient='index')


def inputs():
    prior = pd.read_csv(D / 'sensitivity/analysis_input.csv')
    prior = prior[prior.er.isin(['positive', 'negative'])].copy()
    audit, frames = [], {}
    for cohort, current in prior.groupby('cohort'):
        current = current.sort_values('GSM').reset_index(drop=True)
        meta = pd.read_csv(P / f'{cohort}_metadata.csv', keep_default_na=False, dtype=str).set_index('GSM')
        raw = source_fields(cohort)
        mapping = ({'age10': 'age_years', 'er_positive': 'er_status_ihc', 'pr_positive': 'pr_status_ihc',
                    'her2_positive': 'her2_status', 't_stage': 'clinical_t_stage',
                    'node_positive': 'clinical_nodal_status', 'grade': 'grade'} if cohort == 'GSE25055' else
                   {'age10': 'age', 'er_positive': 'er status ihc', 'pr_positive': 'pr status ihc',
                    'her2_positive': 'her2 status fish', 't_stage': 'clinical t stage',
                    'node_positive': 'lymph node status', 'grade': 'histological grade'})
        for output, field in mapping.items():
            values = meta.loc[current.GSM, field].reset_index(drop=True)
            assert values.tolist() == raw.loc[current.GSM, field].tolist(), (cohort, field)
            if output == 'age10':
                transformed = pd.to_numeric(values) / 10
                assert np.allclose(transformed, current.age10, rtol=0, atol=1e-12)
            elif output in ['er_positive', 'pr_positive', 'her2_positive']:
                assert set(values) <= {'P', 'N', 'I', 'NA', 'positive', 'negative'}
                transformed = values.map({'P': 1.0, 'N': 0.0, 'positive': 1.0, 'negative': 0.0})
            elif output == 't_stage':
                transformed = pd.to_numeric(values.str.removeprefix('T'))
                assert transformed.isin([0, 1, 2, 3, 4]).all()
            elif output == 'node_positive':
                transformed = values.map({'N0': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'negative': 0, 'positive': 1})
                assert transformed.notna().all()
            else:
                assert set(values) <= {'1', '2', '3', '4=Indeterminate', 'NA'}
                transformed = values.map({'1': 1.0, '2': 2.0, '3': 3.0})
            current[output] = transformed
            for value, count in values.value_counts().items():
                audit.append({'cohort': cohort, 'variable': output, 'source_field': field,
                              'source_value': value, 'n': int(count), 'missing_after_mapping': int(transformed.isna().sum())})
        assert np.array_equal(current.er_positive, current.er.eq('positive').astype(float))
        assert len(current) == (301 if cohort == 'GSE25055' else 115)
        assert int(current.y.sum()) == (56 if cohort == 'GSE25055' else 27)
        assert current.GSM.is_unique
        frames[cohort] = current[['cohort', 'GSM', 'y', 'pam50'] + COLS].copy()
    pd.concat(frames.values()).to_csv(O / 'clinical_expanded_input.csv', index=False)
    pd.DataFrame(audit).to_csv(O / 'clinical_covariate_audit.csv', index=False)
    return frames


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    counts = np.cumsum(weights)
    total = int(counts[-1])
    positions = [(total - 1) // 2, total // 2]
    return float(np.mean(values[np.searchsorted(counts, positions, side='right')]))


def design(x, weights, numeric, dummies):
    z = x[:, :numeric].copy()
    for j in range(numeric):
        missing = np.isnan(z[:, j])
        if missing.any():
            observed = (~missing) & (weights > 0)
            assert observed.any(), ('all_missing_training', j)
            if j in (2, 3):
                counts = np.bincount(z[observed, j].astype(int), weights=weights[observed], minlength=2)
                fill = float(np.argmax(counts))
            else:
                assert j == 6
                fill = weighted_median(z[observed, j], weights[observed])
            z[missing, j] = fill
    mean = np.average(z, axis=0, weights=weights)
    var = np.average((z - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(var)
    scale[scale < 1e-12] = 1
    z = (z - mean) / scale
    return np.column_stack([np.ones(len(x)), z, dummies])


def newton(z, y, weights):
    beta = np.zeros(z.shape[1])
    prevalence = np.average(y, weights=weights)
    assert 0 < prevalence < 1
    beta[0] = np.log(prevalence / (1 - prevalence))
    penalty = np.ones(len(beta))
    penalty[0] = 0
    for iteration in range(80):
        eta = z @ beta
        pr = expit(eta)
        gradient = z.T @ (weights * (pr - y)) + penalty * beta
        if np.max(np.abs(gradient)) < 1e-8:
            break
        hessian = z.T @ ((weights * pr * (1 - pr))[:, None] * z) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        objective = np.sum(weights * (np.logaddexp(0, eta) - y * eta)) + 0.5 * np.sum(penalty * beta ** 2)
        rate = 1.0
        while rate > 1e-9:
            candidate = beta - rate * step
            eta_new = z @ candidate
            value = np.sum(weights * (np.logaddexp(0, eta_new) - y * eta_new)) + 0.5 * np.sum(penalty * candidate ** 2)
            if value <= objective - 1e-4 * rate * gradient @ step + 1e-12:
                break
            rate *= 0.5
        beta = candidate
    final_gradient = z.T @ (weights * (expit(z @ beta) - y)) + penalty * beta
    error = float(np.max(np.abs(final_gradient)))
    assert error < 1e-6, (iteration, error)
    CHECK['max_gradient'] = max(CHECK['max_gradient'], error)
    CHECK['fits'] += 1
    return beta


def reference(x, y, train_weights, numeric, dummies, z, beta):
    ix = np.repeat(np.arange(len(x)), train_weights.astype(int))
    x0 = x[:, :numeric].copy()
    for j in range(numeric):
        if np.isnan(x0[:, j]).any():
            imputer = SimpleImputer(strategy='most_frequent' if j in (2, 3) else 'median')
            imputer.fit(x0[ix, j:j+1])
            x0[:, j:j+1] = imputer.transform(x0[:, j:j+1])
    scaler = StandardScaler().fit(x0[ix])
    reference_z = np.column_stack([scaler.transform(x0), dummies])
    preparation_error = float(np.max(np.abs(z[:, 1:] - reference_z)))
    assert preparation_error < 1e-10, preparation_error
    lr = LogisticRegression(C=1, solver='lbfgs', max_iter=2000, tol=1e-10)
    with warnings.catch_warnings(record=True) as found:
        lr.fit(reference_z[ix], y[ix])
    assert not found, str(found)
    ref_beta = np.r_[lr.intercept_, lr.coef_[0]]
    coefficient_error = float(np.max(np.abs(ref_beta - beta)))
    prediction_error = float(np.max(np.abs(lr.predict_proba(reference_z)[:, 1] - expit(z @ beta))))
    assert coefficient_error < 2e-5 and prediction_error < 2e-6, (coefficient_error, prediction_error)
    CHECK['reference_fits'] += 1
    CHECK['max_preprocessing_difference'] = max(CHECK['max_preprocessing_difference'], preparation_error)
    CHECK['max_coefficient_difference'] = max(CHECK['max_coefficient_difference'], coefficient_error)
    CHECK['max_probability_difference'] = max(CHECK['max_probability_difference'], prediction_error)


def metrics(y, pr, w):
    order = np.argsort(pr, kind='stable')
    s, a, b = pr[order], w[order] * y[order], w[order] * (1 - y[order])
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    ac, bc = np.cumsum(a)[ends], np.cumsum(b)[ends]
    positive, negative = ac[-1], bc[-1]
    if positive == 0 or negative == 0:
        return np.full(4, np.nan)
    ag, bg = np.diff(np.r_[0, ac]), np.diff(np.r_[0, bc])
    auc = np.sum(ag * (bc - bg / 2)) / (positive * negative)
    tp, total = np.cumsum(ag[::-1]), np.cumsum((ag + bg)[::-1])
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    ap = np.sum(ag[::-1] * precision) / positive
    clip = np.clip(pr, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return np.array([auc, ap, np.average((y - pr) ** 2, weights=w),
                     np.average(-y * np.log(clip) - (1 - y) * np.log1p(-clip), weights=w)])


def evaluate(frame, folds, weights, verify=False, keep_predictions=False):
    x, y = frame[COLS].to_numpy(float), frame.y.to_numpy(float)
    pam = np.column_stack([frame.pam50.eq(c).to_numpy(float) for c in CATEGORIES[1:]])
    specifications = [('expanded', 7, False), ('expanded_score', 8, False)]
    if frame.cohort.iloc[0] == 'GSE25055':
        specifications += [('expanded_pam50', 7, True), ('expanded_pam50_score', 8, True)]
    records, predictions = [], []
    for repeat, assignments in enumerate(folds):
        for name, numeric, subtype in specifications:
            dummies = pam if subtype else np.empty((len(frame), 0))
            probabilities = np.full(len(frame), np.nan)
            for fold in range(5):
                test = assignments == fold
                train_weights = weights * (~test)
                assert not np.any(train_weights[test])
                if train_weights.sum() == 0 or len(np.unique(y[train_weights > 0])) != 2:
                    raise ValueError('Empty or single-class training sample')
                z = design(x, train_weights, numeric, dummies)
                beta = newton(z, y, train_weights)
                probabilities[test] = expit(z[test] @ beta)
                if verify and (keep_predictions or (repeat == 0 and fold == 0)):
                    reference(x, y, train_weights, numeric, dummies, z, beta)
            assert np.isfinite(probabilities).all()
            values = metrics(y, probabilities, weights)
            if verify:
                physical = np.repeat(np.arange(len(y)), weights.astype(int))
                yy, pp = y[physical], probabilities[physical]
                expected = [roc_auc_score(yy, pp), average_precision_score(yy, pp),
                            brier_score_loss(yy, pp), log_loss(yy, pp)]
                assert np.allclose(values, expected, atol=1e-12, rtol=0)
                if repeat == 0:
                    pos, neg = pp[yy == 1], pp[yy == 0]
                    direct = ((pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean())
                    assert abs(direct - values[0]) < 1e-12
            records.append({'repeat': repeat, 'model': name, **dict(zip(METRICS, values))})
            if keep_predictions:
                predictions.extend(dict(cohort=frame.cohort.iloc[0], GSM=g, repeat=repeat, fold=int(f),
                                        model=name, y=int(yy), probability=float(pp))
                                   for g, f, yy, pp in zip(frame.GSM, assignments, y, probabilities))
    return pd.DataFrame(records), predictions


def contrasts(result):
    rows = []
    for name in ['expanded', 'expanded_pam50']:
        if name not in set(result.model):
            continue
        for metric in METRICS:
            paired = result.pivot(index='repeat', columns='model', values=metric)
            delta = paired[name + '_score'] - paired[name]
            rows.append(dict(comparator=name, metric=metric, estimate=float(delta.mean()),
                             minimum=float(delta.min()), maximum=float(delta.max())))
    return rows


def main():
    start = time.monotonic()
    paths = [Path(__file__), D / 'sensitivity/analysis_input.csv', O / 'cv_predictions.csv']
    paths += [P / (c + s) for c in ['GSE25055', 'GSE32646'] for s in ['_metadata.csv', '_matrix_header.txt']]
    hashes = {str(p.relative_to(P)).replace('\\', '/'): sha(p) for p in paths}
    frames = inputs()
    rng = np.random.default_rng(202609295)
    records, predicted, point_rows, draw_rows = [], [], [], []
    for cohort, frame in frames.items():
        folds = []
        for repeat in range(10):
            assignment = np.full(len(frame), -1)
            for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=202609290 + repeat).split(frame, frame.y)):
                assert not set(train) & set(test)
                assignment[test] = fold
            folds.append(assignment)
        old = pd.read_csv(O / 'cv_predictions.csv').query("cohort == @cohort and model == 'clinical'")
        for repeat in range(10):
            rows = old[old.repeat == repeat].set_index('GSM').loc[frame.GSM]
            assert np.array_equal(rows.fold, folds[repeat]) and np.array_equal(rows.y, frame.y)
        observed, predictions = evaluate(frame, folds, np.ones(len(frame), dtype=int), verify=True, keep_predictions=True)
        observed.insert(0, 'cohort', cohort)
        records.append(observed)
        predicted.extend(predictions)
        for row in contrasts(observed):
            point_rows.append(dict(cohort=cohort, **row))
        pd.concat(records).to_csv(O / 'clinical_expanded_metrics.csv', index=False)
        pd.DataFrame(predicted).to_csv(O / 'clinical_expanded_predictions.csv', index=False)
        pd.DataFrame(point_rows).to_csv(O / 'clinical_expanded_increment.csv', index=False)
        print(cohort, observed.groupby('model')[METRICS].mean().to_string(), flush=True)
        for draw in range(1000):
            weights = np.bincount(rng.integers(len(frame), size=len(frame)), minlength=len(frame))
            try:
                sampled, _ = evaluate(frame, folds, weights, verify=draw in [0, 1, 2, 499, 999])
                for row in contrasts(sampled):
                    draw_rows.append(dict(cohort=cohort, draw=draw, comparator=row['comparator'],
                                          metric=row['metric'], difference=row['estimate'],
                                          unique_patients=int((weights > 0).sum())))
            except ValueError as error:
                CHECK['invalid_draws'].append(dict(cohort=cohort, draw=draw, reason=str(error)))
            if draw % 25 == 24:
                pd.DataFrame(draw_rows).to_csv(O / 'clinical_expanded_bootstrap.csv', index=False)
                print(cohort, 'draw', draw + 1, 'seconds', round(time.monotonic() - start, 1), flush=True)
    boot = pd.DataFrame(draw_rows)
    summaries = []
    for row in point_rows:
        selected = boot[(boot.cohort == row['cohort']) & (boot.comparator == row['comparator']) & (boot.metric == row['metric'])]
        low, high = selected.difference.quantile([0.025, 0.975])
        summaries.append(dict(**row, low=float(low), high=float(high), valid_draws=len(selected),
                              bootstrap_mean=float(selected.difference.mean())))
    pd.DataFrame(summaries).to_csv(O / 'clinical_expanded_uncertainty.csv', index=False)
    assert all(sha(P / p) == h for p, h in hashes.items())
    assert not pd.DataFrame(predicted).duplicated(['cohort', 'GSM', 'repeat', 'model']).any()
    assert len(predicted) == 14340
    CHECK.update(completed_utc=datetime.now(timezone.utc).isoformat(), input_hashes=hashes, seed=202609295,
                 draws_per_cohort=1000, bootstrap_rows=len(boot), prediction_records=len(predicted),
                 seconds=time.monotonic() - start,
                 
                 versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    (O / 'clinical_expanded_verification.json').write_text(json.dumps(CHECK, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print('COMPLETE', json.dumps({k: v for k, v in CHECK.items() if k not in ['input_hashes', 'versions']}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
