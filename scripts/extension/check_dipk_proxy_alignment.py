from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations, product
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.metrics import roc_auc_score

P = Path(__file__).resolve().parent
D = P / 'method_screen/DIPK'
O = D / 'extension'
INPUT = O / 'source_label_input.csv'
FROZEN = D / 'GSE25055_fixed_prediction_crosswalk.csv'
SEED = 202609300
DRAWS = 10000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_auc(y, score):
    delta = score[y == 1, None] - score[None, y == 0]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def rank_auc(y, ranks):
    m = int(y.sum())
    return float((y @ ranks - m * (m + 1) / 2) / (m * (len(y) - m)))


def expected_auc(z, score, groups):
    probabilities = np.empty(len(z))
    for ix in groups:
        probabilities[ix] = z[ix].mean()
    m = int(z.sum())
    by_rank = (probabilities @ rankdata(score) - m * (m + 1) / 2) / (m * (len(z) - m))
    membership = probabilities[:, None] * (1 - probabilities[None, :])
    for ix in groups:
        n = len(ix)
        k = int(z[ix].sum())
        membership[np.ix_(ix, ix)] = k * (n - k) / (n * (n - 1)) if n > 1 else 0
    np.fill_diagonal(membership, 0)
    delta = score[:, None] - score[None, :]
    concordance = (delta > 0) + .5 * (delta == 0)
    by_pair = float((membership * concordance).sum() / (m * (len(z) - m)))
    assert abs(by_rank - by_pair) < 1e-12
    return float(by_rank)


def exhaustive_check():
    score = np.array([.2, .2, .7, .1, .7, .9])
    z = np.array([0, 1, 0, 1, 1, 0])
    groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    values = []
    for choices in product(*(list(combinations(ix, int(z[ix].sum()))) for ix in groups)):
        perm = np.zeros(len(z), dtype=int)
        for choice in choices:
            perm[list(choice)] = 1
        values.append(pair_auc(perm, score))
    exact = expected_auc(z, score, groups)
    assert abs(exact - np.mean(values)) < 1e-12
    return {'enumerated_assignments': len(values), 'analytic': exact, 'enumerated_mean': float(np.mean(values))}


assert digest(FROZEN) == 'd769210ea5d2be25d840ceea5b667abf4e0fb4a37712cee7c012146245b58c10'
input_hashes = {str(p.relative_to(P)): digest(p) for p in [INPUT, FROZEN]}
frame = pd.read_csv(INPUT)
original = pd.read_csv(FROZEN)
original = original[original.observed_known].set_index('GSM').loc[frame.GSM]
assert len(frame) == 306 and frame.GSM.is_unique
assert np.array_equal(frame.y, original.observed_pcr_rd.eq('pCR').astype(int))
assert np.array_equal(frame.z, original.dlda30_prediction.eq('pCR').astype(int))
assert np.array_equal(frame.score, original.score)
y = frame.y.to_numpy(dtype=int)
z = frame.z.to_numpy(dtype=int)
score = frame.score.to_numpy()
ranks = rankdata(score)
points = {}
for name, labels in [('observed_outcome', y), ('archived_proxy', z)]:
    value = pair_auc(labels, score)
    assert abs(value - rank_auc(labels, ranks)) < 1e-12
    assert abs(value - roc_auc_score(labels, score)) < 1e-12
    points[name] = value
assert int(y.sum()) == 57 and int(z.sum()) == 121 and int((y != z).sum()) == 90

rng = np.random.default_rng(SEED)
permutation_rows = []
summaries = []
for scheme, columns in [('outcome', ['y']), ('outcome_source', ['y', 'source'])]:
    groups = [np.asarray(ix) for ix in frame.groupby(columns, sort=True).indices.values()]
    exact = expected_auc(z, score, groups)
    values = np.empty(DRAWS)
    for b in range(DRAWS):
        perm = z.copy()
        for ix in groups:
            perm[ix] = rng.permutation(z[ix])
            assert int(perm[ix].sum()) == int(z[ix].sum())
        assert int((perm != y).sum()) == 90
        values[b] = rank_auc(perm, ranks)
        if b in [0, 1, 2, DRAWS - 1]:
            assert abs(values[b] - pair_auc(perm, score)) < 1e-12
        permutation_rows.append({'scheme': scheme, 'draw': b, 'auc': values[b]})
    mc_se = float(values.std(ddof=1) / np.sqrt(DRAWS))
    assert abs(values.mean() - exact) < 6 * mc_se
    exceedances = int((values >= points['archived_proxy']).sum())
    summaries.append({'scheme': scheme, 'n_strata': len(groups), 'draws': DRAWS, 'expected_auc': exact,
                      'permutation_mean': float(values.mean()), 'monte_carlo_mean_se': mc_se,
                      'reference_low': float(np.quantile(values, .025)), 'reference_high': float(np.quantile(values, .975)),
                      'observed_proxy_auc': points['archived_proxy'], 'observed_minus_reference_mean': points['archived_proxy'] - exact,
                      'exceedances': exceedances, 'plus_one_upper_tail_fraction': (1 + exceedances) / (DRAWS + 1)})

ppv = float(y[z == 1].mean())
false_omission = float(y[z == 0].mean())
mixture = .5 + (ppv - false_omission) * (points['observed_outcome'] - .5)
assert abs(mixture - summaries[0]['expected_auc']) < 1e-12

conditionals = []
for fixed, target in [('y', 'z'), ('z', 'y')]:
    for value in [0, 1]:
        sub = frame[frame[fixed] == value]
        labels = sub[target].to_numpy(dtype=int)
        conditionals.append({'fixed_field': fixed, 'fixed_value': value, 'target': target, 'n': len(sub),
                             'positive': int(labels.sum()), 'negative': int(len(sub) - labels.sum()),
                             'auc': pair_auc(labels, sub.score.to_numpy())})
    total = 0
    numerator = 0.
    strata = []
    for key, sub in frame.groupby([fixed, 'source'], sort=True):
        labels = sub[target].to_numpy(dtype=int)
        pairs = int(labels.sum() * (len(sub) - labels.sum()))
        auc = pair_auc(labels, sub.score.to_numpy()) if pairs else None
        strata.append({'fixed_value': int(key[0]), 'source': key[1], 'n': len(sub), 'pairs': pairs, 'auc': auc})
        if pairs:
            total += pairs
            numerator += pairs * auc
    conditionals.append({'fixed_field': fixed + '_and_source', 'target': target, 'n': len(frame),
                         'eligible_pairs': total, 'auc': numerator / total, 'strata': strata})

blocks = []
for target, label in [('y', 'observed_outcome'), ('z', 'archived_proxy')]:
    denominator = int(frame[target].sum() * (len(frame) - frame[target].sum()))
    for pos, negative in product(frame.groupby(['y', 'z'], sort=True), repeat=2):
        pos_key, positive = pos
        neg_key, neg = negative
        target_index = 0 if target == 'y' else 1
        if pos_key[target_index] != 1 or neg_key[target_index] != 0:
            continue
        delta = positive.score.to_numpy()[:, None] - neg.score.to_numpy()[None, :]
        c = float(((delta > 0) + .5 * (delta == 0)).sum())
        pairs = len(positive) * len(neg)
        blocks.append({'target': label, 'positive_y': int(pos_key[0]), 'positive_z': int(pos_key[1]),
                       'negative_y': int(neg_key[0]), 'negative_z': int(neg_key[1]), 'pairs': pairs,
                       'pair_weight': pairs / denominator, 'block_auc': c / pairs, 'weighted_contribution': c / denominator})
    assert abs(sum(b['weighted_contribution'] for b in blocks if b['target'] == label) - points[label]) < 1e-12

for p in [INPUT, FROZEN]:
    assert digest(p) == input_hashes[str(p.relative_to(P))]
result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED, 'draws_per_scheme': DRAWS,
          'plan_recorded_before_effect_calculation': True, 'input_hashes': input_hashes, 'script_sha256': digest(Path(__file__)),
          'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                       'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
          'points': points, 'class_conditional_identity': {'ppv': ppv, 'false_omission_rate': false_omission, 'expected_auc': mixture},
          'permutation_summaries': summaries, 'conditional_aucs': conditionals, 'pair_decomposition': blocks,
          'verification': {'direct_pair_rank_sklearn_agreement': True, 'all_permutation_counts_preserved': True,
                           'original_inputs_unchanged': True, 'exact_enumeration_with_ties': exhaustive_check()}}
pd.DataFrame(permutation_rows).to_csv(O / 'proxy_alignment_permutations.csv', index=False)
(O / 'proxy_alignment_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'points': points, 'permutation_summaries': summaries, 'conditional_aucs': conditionals,
                  'verification': result['verification']}, ensure_ascii=False, indent=2))
