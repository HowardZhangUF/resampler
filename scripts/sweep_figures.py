"""Dose-response of resampling frequency: E_2 < tau N for tau in {0.1, 0.25, 0.5, 0.75, 1.0}.

Reads results/sweep/ess_<tau>/parts/*/filter.csv (scripts/run_triggers.py --triggers ess:<tau> ...) and writes
results/sweep/report/:
- sweep_summary.csv: mean and 95% CI (clustered by observation sequence) per tau, method, model, N.
- sweep_paired_vs_0.5.csv: per-trial difference from tau = 0.5 on the same trials, clustered by observation sequence.
- figures/<model>/<model>-<metric>.png: metric against mean resampling steps (of 50), one panel per N, one line
  per method through its five tau values.
- figures/overview-<metric>.png: paired change from tau = 0.5 against tau, one panel per model, at N = 512.
- REPORT.md: verdict counts and the overview numbers.

    .venv/bin/python scripts/sweep_figures.py --input results/sweep
"""
import argparse
import glob
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import plot_figures as pf  # noqa: E402
import dsr_table_figures as dtf  # noqa: E402
from dsr_latex_tables import NAMES  # noqa: E402
from report import summary  # noqa: E402

TAUS = [.1, .25, .5, .75, 1.]
NS = [32, 128, 512, 2048]
METHODS = ['dsr/dsr', 'dsr/dsr_unsorted', 'dsr/dsr_tau0.5', 'dsr/dsr_scaled',
           'particles/systematic', 'particles/multinomial', 'pydpf/systematic_stop', 'pydpf/soft', 'torchfilter/soft',
           'pydpf/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel', 'variational/kl']
MODELS = [('linear', 'Linear Gaussian SSM'), ('ungm', 'UNGM'), ('volatility', 'Stochastic volatility'),
          ('bearing', 'Bearing-only tracking'), ('lorenz63', 'Stochastic Lorenz-63'), ('lorenz96', 'Stochastic Lorenz-96')]
# metric, rule ('lower' or 'target95'), axis label
METRICS = [('rmse', 'lower', 'State RMSE'), ('predictive_nll', 'lower', 'Predictive NLL'), ('coverage95', 'target95', '95% coverage')]
EXTRA = {'linear': [('posterior_mean_rmse', 'lower', 'Posterior-mean RMSE vs. Kalman')],
         'bearing': [('post_occlusion_rmse', 'lower', 'RMSE after the observation gap')]}
NAMES = {**NAMES, 'torchfilter/soft': r'Soft ($\alpha{=}0.5$, torchfilter)'}


def names(m): return NAMES[m]


def markdown(df):
    """Markdown table of a frame, index included (without the tabulate dependency)."""
    df = df.reset_index()
    fmt = lambda v: f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str, df.columns))+' |', '|'+'---|'*len(df.columns)]
                     + ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False)])


def load(root):
    files = glob.glob(str(root/'ess_*/parts/*/filter.csv'))
    df = pd.concat([pd.read_csv(f).assign(tau=float(Path(f).parts[-4].split('_')[1])) for f in files], ignore_index=True)
    df['method_id'] = df.provider+'/'+df.method
    dup = df.duplicated(['tau', 'method_id', 'model', 'n', 'seed'])
    if dup.any(): raise SystemExit(f'{dup.sum()} duplicate trials')
    return df


def paired(df):
    """Per-trial difference from tau = 0.5; coverage as |coverage - 0.95|, so + is always worse."""
    keys = ['method_id', 'model', 'n', 'seed']
    ref = df[df.tau == .5].set_index(keys)
    rows = []
    for tau in [t for t in TAUS if t != .5]:
        both = df[df.tau == tau].set_index(keys).join(ref, rsuffix='_ref', how='inner')
        for metric, rule, _ in METRICS:
            d = both[metric]-both[metric+'_ref'] if rule == 'lower' else (both[metric]-.95).abs()-(both[metric+'_ref']-.95).abs()
            part = pd.DataFrame(dict(diff=d, ref=both[metric+'_ref'], data_seed=both.data_seed)).reset_index()
            rows.append(summary(part, ['method_id', 'model', 'n'], ['diff', 'ref'], cluster='data_seed').assign(tau=tau, metric=metric))
    p = pd.concat(rows, ignore_index=True)
    lo, hi = p.diff_mean-p.diff_ci95_half, p.diff_mean+p.diff_ci95_half
    p['verdict'] = np.where(hi < 0, 'better', np.where(lo > 0, 'worse', 'tie'))
    return p


def dose_figure(s, model, task, metric, rule, ylabel, out):
    part = s[s.model == model]
    rows = [m for m in METHODS if m in set(part.method_id)]
    sty = dtf.styles(rows)
    fig, axes = plt.subplots(1, len(NS), figsize=(9.6, 3.5), sharey=True, layout='constrained')
    for ax, n in zip(axes, NS):
        for m in rows:
            r = part[(part.method_id == m) & (part.n == n)].sort_values('tau')
            if r.empty: continue
            c, mk, ls = sty[m]
            x, y, h = r.resampling_steps_mean.values, r[metric+'_mean'].values, r[metric+'_ci95_half'].values
            ax.errorbar(x, y, yerr=h, fmt='none', ecolor=c, elinewidth=.8, alpha=.3, capsize=0, zorder=2)
            dsr = dtf.family(m) == 'dsr'
            ax.plot(x, y, color=c, marker=mk, ls=ls, ms=3.6 if dsr else 3, lw=1.1 if dsr else .8, mec=pf.SURFACE, mew=.5,
                    alpha=1 if dsr else .8, zorder=4 if dsr else 3)
        if rule == 'target95': ax.axhline(.95, color=pf.INK2, lw=.8, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f'N = {n}', loc='center', fontweight='bold')
        ax.set_xlim(0, 52); ax.set_xlabel('Mean resampling steps (of 50)')
    v = part[metric+'_mean'].dropna()
    if dtf.is_log(v.values):
        for ax in axes: ax.set_yscale('log'); pf.plain_log(ax.yaxis)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f'{task}: {ylabel} vs. resampling frequency (E$_2$ < τN, τ = 0.1, 0.25, 0.5, 0.75, 1.0 from left to right)',
                 x=.01, ha='left', fontsize=7.5, fontweight='bold')
    fig.legend(handles=dtf.row_legend(fig, rows, names, sty, 1, False), loc='outside right center', fontsize=5.8, frameon=False)
    folder = out/model; folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/f'{model}-{metric}.png', dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def overview_figure(p, metric, ylabel, out, n=512):
    """Paired change from tau = 0.5 against tau; N = 512 (128 for methods that stop there)."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.3), layout='constrained')
    rows = [m for m in METHODS if m in set(p.method_id)]
    sty = dtf.styles(rows)
    for ax, (model, task) in zip(axes, MODELS):
        for m in rows:
            r = p[(p.method_id == m) & (p.model == model) & (p.metric == metric)]
            nn = n if (r.n == n).any() else r.n.max()
            r = r[r.n == nn].set_index('tau').reindex(TAUS)
            if r.empty: continue
            y = r.diff_mean.fillna(0.).values  # tau = 0.5 is the reference: 0 by definition
            c, mk, ls = sty[m]; dsr = dtf.family(m) == 'dsr'
            ax.plot(TAUS, y, color=c, marker=mk, ls=ls, ms=3.4 if dsr else 2.6, lw=1.1 if dsr else .7, mec=pf.SURFACE, mew=.4,
                    alpha=1 if dsr else .7, zorder=4 if dsr else 3)
            sig = (r.verdict != 'tie').fillna(False).values
            ax.plot(np.array(TAUS)[sig], y[sig], ls='none', marker=mk, ms=4.2, color=c, mec=pf.SURFACE, mew=.6, zorder=5)
        ax.axhline(0, color=pf.INK2, lw=.8, zorder=1); ax.axvline(.5, color=pf.INK2, lw=.6, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(TAUS, ['0.1', '.25', '0.5', '.75', '1']); ax.set_xlabel('τ in E$_2$ < τN')
        ax.set_title(task, loc='center')
    axes[0].set_ylabel(f'{ylabel}: change from τ = 0.5\n(+ = worse; same trials)')
    fig.legend(handles=dtf.row_legend(fig, rows, names, sty, 1, False), loc='outside right center', fontsize=5.8, frameon=False)
    fig.suptitle(f'Paired change from τ = 0.5 at N = {n} (N = 128 for OT and Gumbel); large markers: 95% CI excludes 0',
                 x=.01, ha='left', fontsize=7.5, fontweight='bold')
    fig.savefig(out/f'overview-{metric}.png', dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--input', default='results/sweep'); a = a.parse_args()
    root = ROOT/a.input; rep = root/'report'; out = rep/'figures'; out.mkdir(parents=True, exist_ok=True)
    df = load(root)
    metrics = ['rmse', 'coverage95', 'predictive_nll', 'resampling_steps', 'mean_ess_fraction', 'filter_ms', 'posterior_mean_rmse', 'post_occlusion_rmse']
    s = summary(df, ['tau', 'method_id', 'model', 'n'], metrics, cluster='data_seed'); s.to_csv(rep/'sweep_summary.csv', index=False)
    p = paired(df); p.to_csv(rep/'sweep_paired_vs_0.5.csv', index=False)
    dtf.setup()
    for model, task in MODELS:
        for metric, rule, ylabel in METRICS+EXTRA.get(model, []):
            dose_figure(s, model, task, metric, rule, ylabel, out)
    for metric, _, ylabel in METRICS: overview_figure(p, metric, ylabel, out)

    # Markdown summary.
    md = ['# Resampling-frequency sweep (E_2 < tau N)', '',
          f'{len(df)} filter runs; 50 steps; tau in {TAUS}; 100 runs per cell over 10 observation sequences (10 for OT and Gumbel, N <= 128). '
          'Paired differences are against tau = 0.5 on the same trials (same data, propagation noise and resampler seeds); '
          '95% CIs cluster by observation sequence. Coverage is compared as |coverage - 0.95|.', '']
    rate = s.groupby(['model', 'tau']).resampling_steps_mean.mean().unstack('tau')
    md += ['## Resampling steps (of 50), mean over methods and N', '', markdown(rate.round(1)), '']
    p['dsr'] = np.where(p.method_id.str.startswith('dsr/'), 'DSR', 'standard')
    v = p.groupby(['dsr', 'tau', 'metric']).verdict.value_counts(normalize=True).unstack(fill_value=0)
    md += ['## Share of cells better / worse than tau = 0.5', '', markdown((100*v).round(1)), '']
    for metric, _, label in METRICS:
        t = p[(p.metric == metric) & ((p.n == 512) | (p.method_id.isin(['pydpf/ot', 'custom/gumbel']) & (p.n == 128)))]
        piv = t.assign(cell=t.diff_mean.map('{:+.3g}'.format)+np.where(t.verdict == 'tie', '', '*')).pivot_table(
            index=['model', 'method_id'], columns='tau', values='cell', aggfunc='first')
        md += [f'## {label}: change from tau = 0.5 at N = 512 (128 for OT/Gumbel); * = 95% CI excludes 0, + = worse', '', markdown(piv), '']
    (rep/'REPORT.md').write_text('\n'.join(md)+'\n')
    print('Wrote', rep)


if __name__ == '__main__':
    main()
