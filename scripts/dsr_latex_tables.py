"""LaTeX tables of the DSR benchmark results: one table per task and metric.

Rows are methods (DSR variants and representative baselines); columns are particle
counts N. Values are means, with a 95% confidence half-width where the summary has
one; the best value per column is bold. Reads results/dsr/report/dsr_*_summary.csv
(written by dsr_compare.py) and the completed results/standard/report summaries.
Requires booktabs, amsmath, amssymb and adjustbox.

    .venv/bin/python scripts/dsr_latex_tables.py --input results/dsr
"""
import argparse
import glob
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from resampler_bench.adapters import METHODS  # noqa: E402
BASE = ROOT / 'results/standard/report'

NAMES = {
    'dsr/dsr': r'DSR ($\tau{=}0.1$, paper)', 'dsr/dsr_noclip': r'DSR ($\tau{=}0.1$, no clamp)',
    'dsr/dsr_unsorted': r'DSR ($\tau{=}0.1$, unsorted)', 'dsr/dsr_tau0.5': r'DSR ($\tau{=}0.5$)',
    'dsr/dsr_scaled': r'DSR ($\tau{=}2.5/N$)$^\dagger$',
    'particles/systematic': 'Systematic', 'particles/multinomial': 'Multinomial',
    'pydpf/systematic': 'Systematic (no gradient)', 'pydpf/systematic_stop': 'Systematic, stop-gradient (PyDPF)',
    'cuthbert/systematic_stop': 'Systematic, stop-gradient (Cuthbert)',
    'pydpf/soft': r'Soft ($\alpha{=}0.5$)', 'pydpf/ot': 'OT (PyDPF)', 'pfjax/ot': 'OT (PFJAX)', 'filterflow/ot': 'OT (FilterFlow)',
    'custom/eigen': 'Eigen moment matching', 'custom/gaussian': 'Gaussian moment matching',
    'custom/gumbel': 'Gumbel-softmax', 'variational/kl': 'Variational (KL)',
}
DSR = ['dsr/dsr', 'dsr/dsr_noclip', 'dsr/dsr_unsorted', 'dsr/dsr_tau0.5', 'dsr/dsr_scaled']
FORWARD = DSR + ['particles/systematic', 'particles/multinomial', 'pydpf/systematic_stop', 'pydpf/soft', 'pydpf/ot', 'pfjax/ot',
                 'filterflow/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel', 'variational/kl']
DIFFERENTIABLE = DSR + ['pydpf/systematic', 'pydpf/systematic_stop', 'cuthbert/systematic_stop', 'pydpf/soft', 'pydpf/ot', 'pfjax/ot',
                        'filterflow/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel']
PROVIDER_NAMES = {'particles': 'particles', 'stonesoup': 'Stone Soup', 'cuthbert': 'Cuthbert', 'pfjax': 'PFJAX', 'pydpf': 'PyDPF',
                  'torchfilter': 'torchfilter', 'filterflow': 'FilterFlow', 'julia': 'Resamplings.jl'}
METHOD_NAMES = {'multinomial': 'Multinomial', 'systematic': 'Systematic', 'stratified': 'Stratified', 'residual': 'Residual', 'ssp': 'SSP',
                'killing': 'Killing', 'systematic_stop': 'Systematic, stop-gradient', 'soft': r'Soft ($\alpha{=}0.5$)', 'ot': 'OT',
                'corrected_ot': 'Covariance-corrected OT', 'mvn': 'Gaussian moment matching, MVN', 'eigen': 'Eigen moment matching',
                'gaussian': 'Gaussian moment matching', 'gumbel': 'Gumbel-softmax', 'kl': 'Variational (KL)', 'tv': 'Variational (TV)'}


def full_name(m):
    """Display name with the source repository, for tables that list every method."""
    provider, method = m.split('/')
    if provider in ('dsr', 'custom', 'variational'): return NAMES.get(m) or METHOD_NAMES[method]
    return f'{METHOD_NAMES[method]} ({PROVIDER_NAMES[provider]})'


NOTE = (r"$^\dagger$Local diagnostic, not from the paper: $\tau$ rescaled so that $\tau N$ equals the paper's value at $N{=}25$. "
        r'DSR variants use Morton sorting unless marked unsorted. A dash means not run (dense methods stop at $N{=}128$). Bold marks the best mean per column, not a statistically significant difference.')


def num(x):
    """Three significant figures; scientific notation outside [1e-3, 1e4)."""
    if not np.isfinite(x): return '--'
    if x == 0: return '0'
    if 1e-3 <= abs(x) < 1e4: return f'{x:.3g}'
    m, e = f'{x:.2e}'.split('e')
    return rf'{float(m):.3g}\times10^{{{int(e)}}}'


def cell(mean, half, bold):
    # Omit intervals that are only floating-point noise or exactly zero (deterministic methods).
    noise = half is None or not np.isfinite(half) or abs(half) <= 1e-6 * abs(mean)
    s = num(mean) if noise else rf'{num(mean)}\pm{num(half)}'
    if s == '--': return s
    return rf'$\mathbf{{{s}}}$' if bold else f'${s}$'


def tex_table(df, methods, metric, ci, rule, caption, label, ns, names=NAMES.get):
    """df: one task's summary with method_id, n, <metric> columns."""
    pivot = df.pivot_table(index='method_id', columns='n', values=metric)
    half = df.pivot_table(index='method_id', columns='n', values=ci) if ci and ci in df else None
    rows = [m for m in methods if m in pivot.index]
    pivot = pivot.reindex(index=rows, columns=ns)
    if half is not None: half = half.reindex(index=rows, columns=ns)
    score = {'lower': pivot, 'higher': -pivot, 'abs': pivot.abs(), 'target95': (pivot - .95).abs()}[rule]
    best = score.eq(score.min(axis=0), axis=1)
    lines = [r'\begin{table}[ht]', r'\centering', r'\small', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{l' + 'c' * len(ns) + '}', r'\toprule', 'Method & ' + ' & '.join(f'$N{{=}}{n}$' for n in ns) + r' \\', r'\midrule']
    for i, m in enumerate(rows):
        if i == len([r for r in rows if r in DSR]) and i: lines.append(r'\midrule')
        vals = [cell(pivot.at[m, n], None if half is None else half.at[m, n], bool(best.at[m, n])) for n in ns]
        lines.append(names(m) + ' & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\par\smallskip\parbox{0.95\linewidth}{\footnotesize ' + NOTE + '}', r'\end{table}', '']
    return '\n'.join(lines)


LEARNING = ['dsr/dsr', 'particles/systematic', 'stonesoup/systematic', 'julia/systematic', 'cuthbert/systematic_stop', 'torchfilter/soft',
            'pydpf/ot', 'pfjax/ot', 'filterflow/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel', 'variational/kl']
LEARNING_NAMES = {**NAMES, 'particles/systematic': 'Systematic (particles)', 'stonesoup/systematic': 'Systematic (Stone Soup)',
                  'julia/systematic': 'Systematic (Resamplings.jl)', 'torchfilter/soft': r'Soft ($\alpha{=}0.5$, torchfilter)'}


def learning_table(df):
    """Stochastic-volatility scale fitting: one N, so the columns are its metrics."""
    df = df.set_index('method_id').reindex([m for m in LEARNING if m in set(df.method_id)])
    columns = [('parameter_absolute_error', 'lower', r'$|\hat s-s^\star|$'), ('mle_reference_error', 'lower', r'$|\hat s-\hat s_{\mathrm{grid}}|$'),
               ('score_error', 'abs', r'Score error'), ('score_rmse', 'lower', r'Score RMSE')]
    cells = {}
    for metric, rule, _ in columns:
        mean = df[metric + '_mean'] if metric + '_mean' in df else df[metric]
        half = df.get(metric + '_ci95_half')
        score = mean.abs() if rule == 'abs' else mean
        best = score.eq(score.min())
        cells[metric] = [cell(mean[m], None if half is None else half[m], bool(best[m])) for m in df.index]
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\caption{Stochastic volatility, parameter learning: observation-scale fitting and likelihood score. '
             r'$N{=}128$; $y_t=s\exp(x_t/2)\eta_t$ with true scale $s^\star{=}1$; 50 steps, resampling when ESS $<0.5N$. '
             r'$\hat s$ maximizes the PF log-likelihood over the grid $\{0.6,0.8,1.0,1.2,1.4\}$ with common random numbers; '
             r'$\hat s_{\mathrm{grid}}$ is the maximizer for a 1001-point numerical grid filter. The score is a central difference in $\log s$ '
             r'($\delta{=}0.01$) at $s{=}1$, compared with the grid-filter score. Mean $\pm$ 95\% CI over 10 observation sequences.}',
             r'\label{tab:volatility-learning}', r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{lcccc}', r'\toprule',
             'Method & ' + ' & '.join(name for _, _, name in columns) + r' \\',
             r'\midrule']
    for i, m in enumerate(df.index):
        if i == 1: lines.append(r'\midrule')
        lines.append(LEARNING_NAMES[m] + ' & ' + ' & '.join(cells[metric][i] for metric, _, _ in columns) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}',
              r'\par\smallskip\parbox{0.95\linewidth}{\footnotesize Lower is better for the two parameter errors and the score RMSE; '
              r'closer to 0 is better for the mean score error. Parameter estimates lie on a 0.2-spaced grid, so errors are multiples of 0.2 per dataset. '
              r'Only the paper-default DSR ($\tau{=}0.1$, Morton sorting) was run in this stage; baselines are the representative method of each other repository. '
              r'Bold marks the best mean per column, not a statistically significant difference.}', r'\end{table}', '']
    return '\n'.join(lines)


def label(df):
    df = df.copy(); df['method_id'] = df.provider + '/' + df.method; return df


DIRECTION = {'lower': 'lower is better', 'higher': 'higher is better', 'abs': 'closer to 0 is better', 'target95': 'closer to 0.95 is better'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', default='results/dsr'); a = p.parse_args()
    rep = ROOT / a.input / 'report'
    both = lambda name, base: label(pd.concat([pd.read_csv(rep / f'dsr_{name}_summary.csv'), pd.read_csv(BASE / f'{base}_summary.csv')], ignore_index=True))
    fs, ss, gs, ns_ = both('filter', 'filter'), both('static', 'static'), both('gradient', 'gradient'), both('torch_native', 'torch_native')
    N4, N3 = [32, 128, 512, 2048], [32, 128, 512]
    out = []

    def section(title): out.append(f'% {"=" * 70}\n% {title}\n% {"=" * 70}\n\\clearpage\n\\section*{{{title}}}\n')

    def add(df, methods, metric, ci, rule, task, metric_name, setting, label_key, ns=N4, names=NAMES.get):
        caption = f'{task}: {metric_name} ({DIRECTION[rule]}). {setting}'
        out.append(tex_table(df, methods, metric, ci, rule, caption, f'tab:{label_key}', ns, names))

    filt = '50 steps, resampling when ESS $<0.5N$; mean $\\pm$ 95\\% CI over 10 observation sequences (100 filter runs; 10 for OT and Gumbel).'
    section('Linear Gaussian state-space model')
    lin = fs[fs.model == 'linear']
    for metric, rule, name in [('rmse', 'lower', 'state RMSE'), ('posterior_mean_rmse', 'lower', 'posterior-mean RMSE vs.\\ exact Kalman filter'),
                               ('posterior_covariance_error', 'lower', 'posterior-variance error vs.\\ Kalman filter'),
                               ('loglik_error', 'abs', 'log-likelihood error vs.\\ Kalman filter'), ('coverage95', 'target95', '95\\% coverage'),
                               ('conditional_loglik_variance', 'lower', 'log-likelihood variance given the data')]:
        ci = None if metric == 'conditional_loglik_variance' else metric + '_ci95_half'
        add(lin, FORWARD, metric + ('' if metric == 'conditional_loglik_variance' else '_mean'), ci, rule, 'Linear Gaussian SSM', name,
            '$x_t=0.8x_{t-1}+0.5\\varepsilon_t$, $y_t=x_t+0.8\\eta_t$. ' + filt, f'linear-{metric}')
    native = '$x_t=x_{t-1}+0.4\\varepsilon_t$, $y_t=x_t+0.8\\eta_t$, 50 steps; gradient of the PF log-likelihood w.r.t.\\ $\\theta=\\log(q,r)$ against the exact Kalman gradient, 100 seeds (10 for OT and Gumbel).'
    for metric, rule, name in [('gradient_bias', 'lower', 'likelihood-gradient bias $\\|\\mathbb{E}\\hat g-\\nabla\\ell\\|$'),
                               ('gradient_variance', 'lower', 'likelihood-gradient variance'),
                               ('gradient_mse', 'lower', 'likelihood-gradient MSE'), ('loglik_bias', 'abs', 'log-likelihood bias'),
                               ('loglik_variance', 'lower', 'log-likelihood variance')]:
        add(ns_, DIFFERENTIABLE, metric, None, rule, 'Linear Gaussian random walk (differentiable PF)', name, native, f'linear-native-{metric}')

    section('Nonlinear state-space models')
    tasks = [('ungm', 'UNGM', '$x_t=0.5x_{t-1}+25x_{t-1}/(1+x_{t-1}^2)+8\\cos(1.2t)+\\sqrt{10}\\varepsilon_t$, $y_t=x_t^2/20+\\eta_t$. '),
             ('volatility', 'Stochastic volatility', '$x_t=0.95x_{t-1}+0.2\\varepsilon_t$, $y_t=\\exp(x_t/2)\\eta_t$. '),
             ('bearing', 'Bearing-only tracking', '2-D constant velocity, bearing noise 0.05 rad, observations hidden at steps 21--30. ')]
    for model, task, desc in tasks:
        part = fs[fs.model == model]
        metrics = [('rmse', 'lower', 'position RMSE' if model == 'bearing' else 'state RMSE'), ('coverage95', 'target95', '95\\% coverage'),
                   ('predictive_nll', 'lower', 'predictive NLL'), ('conditional_loglik_variance', 'lower', 'log-likelihood variance given the data')]
        if model == 'bearing': metrics.insert(1, ('post_occlusion_rmse', 'lower', 'RMSE after the observation gap'))
        for metric, rule, name in metrics:
            ci = None if metric == 'conditional_loglik_variance' else metric + '_ci95_half'
            add(part, FORWARD, metric + ('' if metric == 'conditional_loglik_variance' else '_mean'), ci, rule, task, name, desc + filt, f'{model}-{metric}')

    # Stochastic Lorenz models: every method of every repository (results/lorenz, including DSR).
    lorenz_root = ROOT / 'results/lorenz'
    if any(lorenz_root.glob('*/filter.csv')):
        from report import summary
        lf = pd.concat([pd.read_csv(x) for x in sorted(lorenz_root.glob('*/filter.csv'))], ignore_index=True)
        ls = summary(lf, ['provider', 'method', 'model', 'n'], ['rmse', 'coverage95', 'predictive_nll', 'mean_ess_fraction'], cluster='data_seed')
        lv = lf.groupby(['provider', 'method', 'model', 'n', 'data_seed']).loglik.var().groupby(['provider', 'method', 'model', 'n']).mean()
        ls = label(ls.merge(lv.rename('conditional_loglik_variance').reset_index(), on=['provider', 'method', 'model', 'n']))
        ls.to_csv(rep / 'lorenz_filter_summary.csv', index=False)
        order = list(METHODS)
        everything = DSR + sorted(set(ls.method_id) - set(DSR), key=lambda m: (order.index(m.split('/')[0]), METHODS[m.split('/')[0]].index(m.split('/')[1])))
        section('Stochastic Lorenz models (all methods)')
        lorenz = [('lorenz63', 'Stochastic Lorenz-63', '$x_{t+1}=F(x_t)x_t+e_t$, $F$ the 5th-order Taylor series of $\\exp(A(x_t)\\,0.02)$, $e_t\\sim\\mathcal N(0,0.1I_3)$; '
                   '8$\\times$8 camera observation (64 pixels) at SMNR 10\\,dB, as in the DSR paper; $x_0\\sim\\mathcal N(0,I_3)$. '),
                  ('lorenz96', 'Stochastic Lorenz-96', '$d{=}10$, $F{=}8$, RK4 over $\\Delta t{=}0.05$ per step plus $\\mathcal N(0,0.1I)$; $y_t=x_t+\\mathcal N(0,I)$; '
                   '$x_0$ from the climatological Gaussian. ')]
        for model, task, desc in lorenz:
            part = ls[ls.model == model]
            failed = part[part.failure_rate > 0]
            fail = (' Numerical failures, excluded from the means: ' + '; '.join(f'{full_name(r.method_id)} {100*r.failure_rate:.0f}\\% at $N{{=}}{r.n}$' for r in failed.itertuples()) + '.') if len(failed) else ''
            for metric, rule, name in [('rmse', 'lower', 'state RMSE'), ('coverage95', 'target95', '95\\% coverage'), ('predictive_nll', 'lower', 'predictive NLL'),
                                       ('conditional_loglik_variance', 'lower', 'log-likelihood variance given the data'), ('mean_ess_fraction', 'higher', 'mean ESS$/N$')]:
                ci = None if metric == 'conditional_loglik_variance' else metric + '_ci95_half'
                add(part, everything, metric + ('' if metric == 'conditional_loglik_variance' else '_mean'), ci, rule, task, name, desc + filt + fail,
                    f'{model}-{metric}', names=full_name)

    section('Parameter learning (stochastic volatility)')
    out.append(learning_table(both('learning', 'learning')))

    section('Fixed weighted clouds (resampling only)')
    static = '$D{=}2$; mean $\\pm$ 95\\% CI over 100 resampling draws (10 for OT and Gumbel at $N\\le128$, 3 above).'
    clouds = [('gaussian', 'Gaussian cloud'), ('mixture', 'Bimodal mixture cloud'), ('skewed', 'Skewed-weight cloud'), ('fixed', 'Fixed-grid cloud')]
    for cloud, task in clouds:
        part = ss[(ss.cloud == cloud) & (ss.dim == 2)]
        metrics = [('sliced_w1', 'lower', 'sliced Wasserstein-1'), ('rff_mmd2', 'lower', 'Gaussian-kernel MMD$^2$ (random features)'),
                   ('covariance_error', 'lower', 'covariance error (Frobenius)'), ('estimator_mse_x2', 'lower', 'MSE of $\\mathbb{E}[x^2]$')]
        if cloud == 'mixture':
            metrics += [('mode_mass_error', 'lower', 'mode-mass error'),
                        ('mode_coverage', 'higher', 'fraction of draws keeping both modes (each mode $>1\\%$ of the mass)')]
        for metric, rule, name in metrics:
            add(part, FORWARD, metric + '_mean', metric + '_ci95_half', rule, task, name, static, f'static-{cloud}-{metric}')

    section('Local gradients of a resampled expectation')
    for cloud, cname in [('gaussian', 'Gaussian'), ('mixture', 'Bimodal mixture')]:
        for fn, fname in [('x2', '$f(x)=\\|x\\|^2$'), ('sin', '$f(x)=\\sum_k\\sin x_k$')]:
            part = gs[(gs.cloud == cloud) & (gs.function == fn)]
            setting = f'{cname} cloud, $D{{=}}2$, {fname}; derivative of $\\sum_j v_j f(y_j)$ w.r.t.\\ a weight tilt, against the exact weighted-measure derivative; 100 seeds (10 for OT and Gumbel).'
            for metric, ci, name in [('gradient_bias', None, 'gradient bias'), ('gradient_variance', None, 'gradient variance'),
                                     ('squared_error_mean', 'squared_error_ci95_half', 'gradient MSE')]:
                add(part, DIFFERENTIABLE, metric, ci, 'lower', f'Local gradient, {cname.lower()} cloud, {fname}', name, setting, f'grad-{cloud}-{fn}-{metric}', N3)

    section('Runtime')
    t = pd.concat([pd.read_csv(ROOT / a.input / 'performance/dsr/timing.csv')] + [pd.read_csv(x) for x in glob.glob(str(ROOT / 'results/performance/*/timing.csv'))], ignore_index=True)
    for cloud, cname, key in [('gaussian', 'Gaussian cloud', 'runtime-forward'), ('mixture', 'Bimodal mixture cloud', 'runtime-forward-mixture')]:
        tc = label(t[(t.cloud == cloud) & (t.dim == 2)]).groupby(['method_id', 'n']).forward_ms.median().rename('forward_ms').reset_index()
        timed = [m for m in ['dsr/dsr', 'dsr/dsr_scaled'] + FORWARD[5:] if m in set(tc.method_id)]
        add(tc, timed, 'forward_ms', None, 'lower', f'{cname}, $D{{=}}2$', 'forward time per resampling call in ms',
            'Median over 100 calls (3--10 for OT and Gumbel), serial pass on one CPU thread (i7-6700HQ), fresh process per method; includes adapter conversions.', key)

    # Hand-written discussion of the trends; kept in its own file so regeneration does not overwrite it.
    trends = rep / 'trends.tex'
    if trends.exists(): out.append(trends.read_text())
    path = rep / 'tables.tex'; path.write_text('\n'.join(out))
    (rep / 'tables_preview.tex').write_text('\\documentclass{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{booktabs,amsmath,amssymb,adjustbox}\n'
                                            '\\begin{document}\n\\input{tables.tex}\n\\end{document}\n')
    print('Wrote', path, '-', sum(s.count('\\begin{table}') for s in out), 'tables')


if __name__ == '__main__':
    main()
