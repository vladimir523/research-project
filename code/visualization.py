import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wilson(successes, n):
    p = successes/n
    z = 1.95996398454
    mid = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return mid-half, mid+half


def make_slide_figure(summary, root):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    series = [('RF_accuracy', 'RF: accuracy', '#2563eb'),
              ('RF_logloss', 'RF: log loss', '#db6c27'),
              ('LIME', 'LIME', '#727272'),
              ('Correlation', 'PCA / сравнение корреляций', '#218859')]
    for method, label, color in series:
        g = summary[summary.method == method].sort_values('delta')
        ax.plot(g.delta, 100*g.both_top3, marker='o', color=color, label=label)
    ax.set(xlabel='Разность средних Z (в стандартных отклонениях)',
           ylabel='X и Y в top-3, %', ylim=(-5, 106),
           title='Корреляция X–Y фиксирована: группа A — 0,8; группа B — 0')
    ax.legend(loc='center left', fontsize=10, framealpha=.95)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(root/'figures'/f'slide_figure.{ext}', dpi=180)
    plt.close(fig)


def make_report(frame, root):
    summary = []
    for (method, delta), group in frame.groupby(['method', 'delta']):
        row = dict(method=method, delta=delta, n=len(group), accuracy=group.accuracy.mean(),
                   oracle_accuracy=group.oracle_accuracy.mean(), logloss=group.logloss.mean(),
                   mean_rank_X=group.rank_X.mean(), mean_rank_Y=group.rank_Y.mean(),
                   lime_fidelity=group.lime_fidelity.mean())
        for k in (2, 3, 5):
            successes = int(group[f'both_top{k}'].sum())
            lo, hi = wilson(successes, len(group))
            row.update({f'both_top{k}': successes/len(group), f'lo_top{k}': lo, f'hi_top{k}': hi})
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(root/'results'/'summary.csv', index=False)
    make_slide_figure(summary, root)
    methods = ['RF_accuracy', 'RF_logloss', 'LIME', 'PCA_pooled', 'PCA_within', 'Correlation']
    labels = ['RF: accuracy', 'RF: log-loss', 'LIME: mean |weight|', 'PCA: pooled',
              'PCA: within groups', 'Direct correlations']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for method, label in zip(methods, labels):
        g = summary[summary.method == method].sort_values('delta')
        line, = axes[0].plot(g.delta, g.both_top3, marker='o', label=label)
        axes[0].fill_between(g.delta, g.lo_top3, g.hi_top3, alpha=.09, color=line.get_color())
        axes[1].plot(g.delta, g.both_top2, marker='o', label=label)
    axes[0].axhline(3*2/(10*9), ls=':', color='gray', label='Random top-3: 6.7%')
    axes[1].axhline(2/(10*9), ls=':', color='gray')
    for ax, k in zip(axes[:2], [3, 2]):
        ax.set(title=f'Both X and Y in top-{k}', xlabel='Mean difference in Z (within-group SD)',
               ylabel='Fraction of independent datasets', ylim=(-.04, 1.04))
    one = frame[frame.method == 'RF_accuracy']
    for col, label in [('accuracy', 'RF accuracy'), ('oracle_accuracy', 'Known-distribution oracle')]:
        g = one.groupby('delta')[col].agg(['mean', 'std', 'count'])
        axes[2].errorbar(g.index, g['mean'], yerr=1.96*g['std']/np.sqrt(g['count']), marker='o', label=label)
    axes[2].set(title='Group prediction on independent test data', xlabel='Mean difference in Z',
                ylabel='Accuracy', ylim=(.45, 1.04))
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc='lower left')
    axes[2].legend(fontsize=8)
    fig.suptitle('Fixed correlation pattern: rho(A)=0.8, rho(B)=0; 10 features; 600 train + 600 test')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(root/'figures'/f'main_figure.{ext}', dpi=180)
    plt.close(fig)
    # Structural and paired-design checks on the actual generated artifacts.
    assert len(frame) == frame.rep.nunique()*6*6
    assert not frame.duplicated(['rep', 'delta', 'method']).any()
    for rep in frame.rep.unique():
        g = frame[(frame.rep == rep) & (frame.method == 'PCA_within')]
        assert g.rank_X.nunique() == 1 and g.rank_Y.nunique() == 1
    diagnostics = {'checked_rows': len(frame), 'within_group_pca_rank_invariance': True,
                   'independent_replicates': int(frame.rep.nunique())}
    (root/'results'/'validation.json').write_text(json.dumps(diagnostics, indent=2))
    print(summary[summary.delta.isin([0., 4.])].to_string(index=False), flush=True)
