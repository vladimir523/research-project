"""Explicit benchmark definitions; not a reproduction of the source paper's code."""
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer


def pca_scores(x, labels, within=False):
    groups = [x[labels == g] for g in (0, 1)] if within else [x]
    # Each group gets its own centering and scaling in the within-group control.
    return np.mean([np.abs(PCA(n_components=1, svd_solver='full').fit(
        StandardScaler().fit_transform(z)).components_[0]) for z in groups], axis=0)


def correlation_scores(x, labels):
    a, b = [x[labels == g] for g in (0, 1)]
    ca, cb = [np.corrcoef(z, rowvar=False) for z in (a, b)]
    pairs = np.array(np.triu_indices(x.shape[1], 1)).T
    stat = np.abs(np.arctanh(np.clip(ca[pairs[:, 0], pairs[:, 1]], -.999999, .999999))
        - np.arctanh(np.clip(cb[pairs[:, 0], pairs[:, 1]], -.999999, .999999)))
    stat /= np.sqrt(1/(len(a)-3) + 1/(len(b)-3))
    scores = np.zeros(x.shape[1])
    for (i, j), val in zip(pairs, stat):
        scores[i] = max(scores[i], val)
        scores[j] = max(scores[j], val)
    p = 2 * norm.sf(stat)
    order = np.argsort(p)
    selected = np.zeros(len(p), dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] > .05/(len(p)-rank):
            break
        selected[idx] = True
    return scores, selected, pairs


def explain(model, train, test, labels, seed, anchors=8, perturbations=1500):
    perm = permutation_importance(model, test, labels, n_repeats=5, random_state=seed,
        scoring={'accuracy': 'accuracy', 'logloss': 'neg_log_loss'}, n_jobs=1)
    scores = {'RF_accuracy': perm['accuracy'].importances_mean,
              'RF_logloss': perm['logloss'].importances_mean}
    explainer = LimeTabularExplainer(train, mode='classification',
        feature_names=['X', 'Y', 'Z'] + [f'noise{i}' for i in range(train.shape[1]-3)],
        class_names=['B', 'A'], discretize_continuous=True,
        feature_selection='none', random_state=seed)
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.choice(np.flatnonzero(labels == g), anchors//2, replace=False)
                          for g in (0, 1)])
    weights, fidelity = [], []
    for idx in ids:
        e = explainer.explain_instance(test[idx], model.predict_proba, labels=(1,),
            num_features=train.shape[1], num_samples=perturbations)
        w = np.zeros(train.shape[1])
        for feature, weight in e.as_map()[1]:
            w[feature] = abs(weight)
        weights.append(w)
        fidelity.append(e.score)
    scores['LIME'] = np.mean(weights, axis=0)
    return scores, float(np.mean(fidelity))
