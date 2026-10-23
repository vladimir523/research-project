"""Run from any directory: python code/experiment.py. See protocol.json."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from methods import explain, pca_scores, correlation_scores

ROOT = Path(__file__).resolve().parent


def generate(seed, n=600, d=10, rho=.8):
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1], n//2)
    x = rng.normal(size=(n, d))
    take = label == 1
    x[take, 1] = rho*x[take, 0] + np.sqrt(1-rho**2)*x[take, 1]
    return x, label


def shifted(x, labels, delta):
    result = x.copy()
    result[:, 2] += (labels-.5)*delta
    return result


def run(reps=20):
    for directory in ('data', 'results', 'figures'):
        (ROOT/directory).mkdir(exist_ok=True)
    rows, feature_rows, controls = [], [], []
    start = time.perf_counter()
    deltas = [0., .5, 1., 2., 3., 4.]
    for rep in range(reps):
        seed = 100000+rep
        train0, ytrain = generate(seed)
        test0, ytest = generate(seed+10000)
        np.savez_compressed(ROOT/'data'/f'base_{rep:02d}.npz', train=train0,
            test=test0, train_labels=ytrain, test_labels=ytest)
        tie = np.random.default_rng(seed+20000).random(10)
        for delta in deltas:
            train, test = shifted(train0, ytrain, delta), shifted(test0, ytest, delta)
            model = RandomForestClassifier(n_estimators=128, min_samples_leaf=2,
                max_features='sqrt', random_state=seed, n_jobs=1).fit(train, ytrain)
            probs = model.predict_proba(test)[:, 1]
            acc = accuracy_score(ytest, probs >= .5)
            loss = log_loss(ytest, probs)
            # Exact known-population log likelihood ratio for class A vs B.
            rho = .8
            ll = -.5*np.log(1-rho**2) - .5*((test[:, 0]**2 - 2*rho*test[:, 0]*test[:, 1]
                + test[:, 1]**2)/(1-rho**2) - test[:, 0]**2 - test[:, 1]**2) + delta*test[:, 2]
            oracle = accuracy_score(ytest, ll >= 0)
            scores, fidelity = explain(model, train, test, ytest, seed)
            scores['PCA_pooled'] = pca_scores(train, ytrain)
            scores['PCA_within'] = pca_scores(train, ytrain, within=True)
            corr, selected, pairs = correlation_scores(train, ytrain)
            scores['Correlation'] = corr
            true_pair = (pairs[:, 0] == 0) & (pairs[:, 1] == 1)
            for method, score in scores.items():
                order = np.lexsort((tie, -score))
                ranks = np.argsort(order)+1
                row = dict(rep=rep, seed=seed, delta=delta, method=method,
                    rank_X=int(ranks[0]), rank_Y=int(ranks[1]), rank_Z=int(ranks[2]),
                    accuracy=acc, logloss=loss, oracle_accuracy=oracle, lime_fidelity=fidelity,
                    pair_significant=bool(selected[true_pair][0]),
                    false_significant_pairs=int(selected[~true_pair].sum()))
                for k in (2, 3, 5):
                    row[f'both_top{k}'] = int(ranks[0] <= k and ranks[1] <= k)
                    row[f'recall_top{k}'] = (int(ranks[0] <= k)+int(ranks[1] <= k))/2
                rows.append(row)
                for j in range(10):
                    feature_rows.append(dict(rep=rep, delta=delta, method=method,
                        feature=j, score=float(score[j]), rank=int(ranks[j])))
            if rep == 0 and delta in (0., 4.):
                for split, x, y in [('train', train, ytrain), ('test', test, ytest)]:
                    frame = pd.DataFrame(x, columns=['X', 'Y', 'Z']+[f'noise{i}' for i in range(7)])
                    frame['group'] = y
                    frame.to_csv(ROOT/'data'/f'example_delta{int(delta)}_{split}.csv', index=False)
            pd.DataFrame(rows).to_csv(ROOT/'results'/'runs.csv', index=False)
            print(f'rep {rep+1}/{reps} delta={delta:g} accuracy={acc:.3f}', flush=True)
        null0, null_labels = generate(200000+rep, rho=0.)
        for delta in (0., 4.):
            _, selected, pairs = correlation_scores(shifted(null0, null_labels, delta), null_labels)
            controls.append(dict(rep=rep, delta=delta, any_rejection=int(selected.any()),
                                 rejected_pairs=int(selected.sum())))
    pd.DataFrame(feature_rows).to_csv(ROOT/'results'/'feature_scores.csv', index=False)
    pd.DataFrame(controls).to_csv(ROOT/'results'/'negative_controls.csv', index=False)
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'lime']}
    manifest = dict(replicates=reps, seconds=time.perf_counter()-start,
        python=platform.python_version(), versions=versions,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).resolve().parent.glob('*.py')})
    (ROOT/'results'/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--output', type=Path, help='Separate directory for generated artifacts')
    args = parser.parse_args()
    if args.reps < 1:
        parser.error('--reps must be positive')
    if args.output is not None:
        ROOT = args.output.resolve()
        ROOT.mkdir(parents=True, exist_ok=True)
    from visualization import make_report
    make_report(run(args.reps), ROOT)
