"""Leave one terrain out to select between two signed, settled team commands."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('parent', 'student')
POLICIES = (*ARMS, 'intercept_only', 'learned')
METRICS = ('margin', 'deposit8', 'deposit24', 'own_cash', 'enemy_cash')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(rows):
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate root id')
    return by_id


def examples(roots, labels):
    inputs, outcomes = index(roots['roots']), index(labels['rows'])
    if inputs.keys() != outcomes.keys():
        raise ValueError('Signed roots and verified outcomes do not match')
    rows = []
    for key, root in inputs.items():
        label = outcomes[key]
        if any(root[field] != label[field] for field in ('seed', 'seat', 'terrain', 't', 'features')):
            raise ValueError(f'Root metadata mismatch for {key}')
        features = np.asarray(root['features'], dtype=float)
        if features.shape != (2,) or not np.isfinite(features).all() or np.any(np.abs(features) > 1):
            raise ValueError(f'Invalid two-feature public contrast for {key}')
        arms = {}
        for arm in ARMS:
            source = label['arms'][arm]
            data = {k: float(source[k]) for k in METRICS}
            if not all(math.isfinite(v) for v in data.values()):
                raise ValueError(f'Nonfinite outcome in {key} {arm}')
            margin_cents = round(100 * data['margin'])
            wdl = (margin_cents > 0) - (margin_cents < 0)
            if wdl != source['wdl']:
                raise ValueError(f'Official result disagrees with margin in {key} {arm}')
            arms[arm] = {**data, 'wdl': wdl}
        difference = round(100 * arms['student']['margin']) - round(100 * arms['parent']['margin'])
        target = None if difference == 0 else int(difference > 0)
        rows.append(dict(id=key, seed=root['seed'], terrain=root['terrain'], seat=root['seat'], t=root['t'],
                         features=features.tolist(), target=target, arms=arms))
    return rows


def design(row, intercept_only):
    return [1.] if intercept_only else [1., *row['features']]


def train(rows, intercept_only):
    x = np.array([design(r, intercept_only) for r in rows])
    y = np.array([r['target'] for r in rows], dtype=float)
    terrain_counts = Counter(r['terrain'] for r in rows)
    weights = np.array([1. / (len(terrain_counts) * terrain_counts[r['terrain']]) for r in rows])
    coef = np.zeros(x.shape[1])
    for _ in range(24):
        logits = x @ coef
        probability = 1. / (1. + np.exp(-logits))
        gradient = x.T @ (weights * (probability - y)) + coef
        curvature = weights * probability * (1. - probability)
        hessian = (x.T * curvature) @ x + np.eye(x.shape[1])
        coef -= np.linalg.solve(hessian, gradient)
    logits = x @ coef
    loss = weights @ (np.logaddexp(0., logits) - y * logits) + (coef @ coef) * .5
    if not np.isfinite(coef).all() or not math.isfinite(loss):
        raise ValueError('Nonfinite training result')
    return dict(coefficients=coef.tolist(), penalized_train_loss=float(loss))


def fit_fold(training, held):
    labeled = [r for r in training if r['target'] is not None]
    counts = Counter(r['target'] for r in labeled)
    info = dict(train_ids=[r['id'] for r in labeled], excluded_ties=[r['id'] for r in training if r['target'] is None],
                train_terrains=sorted({r['terrain'] for r in labeled}), parent_preferred=counts[0], student_preferred=counts[1])
    supported = len(counts) == 2
    models = {name: train(labeled, name == 'intercept_only') for name in ('intercept_only', 'learned')} if supported else {}
    predictions = []
    for row in held:
        choices = {arm: arm for arm in ARMS}
        probabilities = {}
        for name, model in models.items():
            z = np.dot(design(row, name == 'intercept_only'), model['coefficients'])
            p = float(1. / (1. + np.exp(-z)))
            probabilities[name] = p
            choices[name] = 'student' if p > .5 else 'parent'
        predictions.append({**row, 'choices': choices, 'probability_student': probabilities})
    return {**info, 'supported': supported, 'models': models, 'held_predictions': predictions}


def summarize(rows, policy):
    selected = [(r, r['arms'][r['choices'][policy]]) for r in rows if policy in r['choices']]
    if not selected:
        return dict(scored=0, unscored=len(rows))
    wdl = Counter(a['wdl'] for _, a in selected)
    mean = {key: sum(a[key] for _, a in selected) / len(selected) for key in METRICS}
    gain = {key: sum(a[key] - r['arms']['parent'][key] for r, a in selected) / len(selected) for key in METRICS}
    return dict(scored=len(selected), unscored=len(rows)-len(selected), student_choices=sum(r['choices'][policy]=='student' for r,_ in selected),
                wins=wdl[1], draws=wdl[0], losses=wdl[-1], mean=mean, mean_gain_vs_parent=gain)


def evaluate(roots, labels):
    rows = examples(roots, labels)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['terrain']].append(row)
    folds = {}
    pooled = []
    for terrain, held in sorted(grouped.items()):
        fold = fit_fold([r for r in rows if r['terrain'] != terrain], held)
        predictions = fold['held_predictions']
        pooled.extend(predictions)
        fold['outcomes'] = {name: summarize(predictions, name) for name in POLICIES}
        folds[terrain] = fold
    pooled_outcomes = {name: summarize(pooled, name) for name in POLICIES}
    by_terrain = {}
    for name in POLICIES:
        available = [f['outcomes'][name] for f in folds.values() if f['outcomes'][name]['scored']]
        by_terrain[name] = dict(scored_terrains=len(available), unscored_terrains=len(folds)-len(available))
        if available:
            for key in ('mean', 'mean_gain_vs_parent'):
                by_terrain[name][key] = {m: sum(a[key][m] for a in available) / len(available) for m in METRICS}
    improvements = []
    for terrain, fold in folds.items():
        members = fold['held_predictions']
        if fold['supported']:
            cents = {k: sum(round(100*r['arms'][r['choices'][k]]['margin']) for r in members) for k in POLICIES}
            if cents['learned'] > max(cents['parent'], cents['intercept_only']):
                improvements.append(terrain)
    complete = len(grouped) > 0 and all(f['supported'] for f in folds.values())
    tests = dict(all_folds_supported=complete, positive_vs_both_on_three_terrains=len(improvements) >= 3,
                 positive_terrain_equal_margin_gain_vs_parent=complete and by_terrain['learned']['mean_gain_vs_parent']['margin'] > 0,
                 win_count_at_least_parent=complete and pooled_outcomes['learned']['wins'] >= pooled_outcomes['parent']['wins'])
    return dict(roots=len(rows), terrains=len(folds), folds=folds, pooled=pooled_outcomes, terrain_equal=by_terrain,
                strict_improvements_vs_both=improvements, tests=tests, larger_independent_collection=all(tests.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, default=HERE/'roots.json')
    parser.add_argument('--labels', type=Path, default=HERE/'labels.json')
    parser.add_argument('--out', type=Path, default=HERE/'fit.json')
    args = parser.parse_args()
    result = evaluate(json.loads(args.roots.read_bytes()), json.loads(args.labels.read_bytes()))
    result['inputs_sha256'] = {k: sha(p) for k,p in [('roots', args.roots), ('verified_labels', args.labels), ('program', Path(__file__))]}
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.out.exists() and args.out.read_text() != data:
        raise ValueError('Existing fit output has a different identity')
    args.out.write_text(data)
