"""Summarize the DSR provider and place it next to the completed benchmark summaries.

DSR trials live in their own folder (default results/dsr). Summaries use the same
`summary` and `native_summary` functions as scripts/report.py; the other methods
are read from the completed results/standard/report CSVs, which are not modified.

    .venv/bin/python scripts/dsr_compare.py --input results/dsr
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from report import summary, native_summary  # noqa: E402


def table(df, columns):
    """Markdown table with 4 significant figures."""
    lines = ['| ' + ' | '.join(c.replace('_mean', '').replace('_', ' ') for c in columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(f'{row[c]:.4g}' if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)

BASE = ROOT / 'results/standard/report'
# Representative baselines: classical, soft, OT, moment-matching, relaxations.
BASELINES = ['particles/systematic', 'particles/multinomial', 'pydpf/systematic_stop', 'pydpf/soft', 'torchfilter/soft',
             'pydpf/ot', 'pfjax/ot', 'filterflow/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel', 'variational/kl']


def label(df):
    df = df.copy(); df.insert(0, 'method_id', df.provider + '/' + df.method); return df


def pick(df, ids):
    df = df[df.method_id.isin(ids)].copy()
    df['_order'] = df.method_id.map({m: i for i, m in enumerate(ids)})
    return df.sort_values('_order').drop(columns='_order')


def rank_among_all(base, dsr, keys, metric, lower=True):
    """Rank of each DSR row among every benchmark method in the same cell (1 = best)."""
    rows = []
    for _, r in dsr.iterrows():
        cell = base
        for k in keys: cell = cell[cell[k] == r[k]]
        values = cell[metric].dropna().to_numpy()
        better = np.sum(values < r[metric]) if lower else np.sum(values > r[metric])
        rows.append(dict(**{k: r[k] for k in keys}, method_id=r.method_id, value=r[metric], rank=int(better) + 1, of=len(values) + 1))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', default='results/dsr'); a = p.parse_args()
    root = ROOT / a.input; trials = root / 'dsr'; out = root / 'report'; out.mkdir(parents=True, exist_ok=True)
    s, f, g = [pd.read_csv(trials / f'{n}.csv') for n in ['static', 'filter', 'gradient']]
    ss = summary(s, ['provider', 'method', 'cloud', 'n', 'dim'], ['sliced_w1', 'rff_mmd2', 'mean_error', 'covariance_error', 'eigenvalue_error', 'estimator_mse_x', 'estimator_mse_x2', 'estimator_mse_sin', 'mode_coverage', 'mode_mass_error', 'forward_ms', 'process_peak_rss_mb'])
    fs = summary(f, ['provider', 'method', 'model', 'n'], ['rmse', 'coverage95', 'predictive_nll', 'posterior_mean_rmse', 'posterior_covariance_error', 'loglik_error', 'mean_ess_fraction', 'filter_ms', 'recovery_steps', 'recovered', 'post_occlusion_rmse'], cluster='data_seed')
    lv = f.groupby(['provider', 'method', 'model', 'n', 'data_seed']).loglik.var().groupby(['provider', 'method', 'model', 'n']).mean().rename('conditional_loglik_variance').reset_index()
    fs = fs.merge(lv, on=['provider', 'method', 'model', 'n'])
    gs = summary(g, ['provider', 'method', 'cloud', 'function', 'n'], ['gradient', 'error', 'squared_error', 'forward_ms', 'backward_ms'])
    gs = gs.merge(g.groupby(['provider', 'method', 'cloud', 'function', 'n']).gradient.var().rename('gradient_variance').reset_index(), on=['provider', 'method', 'cloud', 'function', 'n'])
    gs['gradient_bias'] = gs.error_mean.abs()
    for name, df in [('static', ss), ('filter', fs), ('gradient', gs)]: df.to_csv(out / f'dsr_{name}_summary.csv', index=False)
    ss, fs, gs = label(ss), label(fs), label(gs)
    bs, bf, bg = [label(pd.read_csv(BASE / f'{n}_summary.csv')) for n in ['static', 'filter', 'gradient']]
    dsr_ids = list(dict.fromkeys(ss.method_id))
    md = ['# DSR on the particle-resampler benchmark', '',
          'Differentiable systematic resampling (Cumlin & Chatterjee, NeurIPS 2026; [code](https://github.com/fcumlin/differentiable-systematic-resampling), [arXiv:2610.12094](https://arxiv.org/abs/2610.12094)) run through the shared harness. '
          'The adapter calls the unmodified upstream `resampling.py`. Variants: `dsr` = paper default (Morton sort, tau=0.1, transport gradient clamped to [-1,1]); `dsr_unsorted` = no Morton sort; `dsr_noclip` = default without the gradient clamp; `dsr_tau0.5` = tau=0.5; `dsr_scaled` = tau=2.5/N, a local diagnostic (not from the paper) that keeps tau*N at the paper\'s N=25 operating point. '
          'Baselines come from the completed `results/standard` summaries (same data, seeds and metrics). Lower is better unless noted.', '']
    audit = [__import__('json').loads(x) for x in (trials / 'status.jsonl').read_text().splitlines()]
    nf = trials / 'numerical_failures.jsonl'
    md += [f"Cells: {sum(r['status'] == 'complete' for r in audit)} complete, {sum(r['status'] == 'failed' for r in audit)} failed. Numerical trial failures: {len(nf.read_text().splitlines()) if nf.exists() else 0}.", '']

    # Static distribution preservation.
    md += ['## Fixed weighted clouds (N=512, D=2)', '']
    for cloud in ['gaussian', 'mixture', 'skewed']:
        cols = ['method_id', 'sliced_w1_mean', 'rff_mmd2_mean', 'mean_error_mean', 'covariance_error_mean', 'estimator_mse_x2_mean'] + (['mode_mass_error_mean'] if cloud == 'mixture' else [])
        sel = lambda d: d[(d.cloud == cloud) & (d.n == 512) & (d.dim == 2)]
        md += [f'### {cloud}', '', table(pd.concat([pick(sel(ss), dsr_ids), pick(sel(bs), BASELINES)]), cols), '']
    rk = rank_among_all(bs, ss[ss.cloud.isin(['gaussian', 'mixture', 'skewed', 'fixed'])], ['cloud', 'n', 'dim'], 'sliced_w1_mean')
    rk.to_csv(out / 'dsr_static_ranks.csv', index=False)
    md += ['Median sliced-W1 rank of each DSR variant among all benchmark methods, across every static cell:', '',
           table(rk.groupby('method_id').agg(median_rank=('rank', 'median'), methods_per_cell=('of', 'max')).reset_index().astype({'median_rank': float}), ['method_id', 'median_rank', 'methods_per_cell']), '']

    # Filtering.
    md += ['## Particle filters (50 steps, ESS < 0.5N; 95% CIs cluster by observation sequence)', '']
    for model in ['linear', 'ungm', 'volatility', 'bearing']:
        extra = ['posterior_mean_rmse_mean', 'loglik_error_mean'] if model == 'linear' else (['post_occlusion_rmse_mean'] if model == 'bearing' else ['predictive_nll_mean'])
        cols = ['method_id', 'n', 'rmse_mean', 'rmse_ci95_half', 'coverage95_mean'] + extra + ['conditional_loglik_variance']
        for n in [128, 512]:
            sel = lambda d: d[(d.model == model) & (d.n == n)]
            md += [f'### {model}, N={n}', '', table(pd.concat([pick(sel(fs), dsr_ids), pick(sel(bf), BASELINES)]), cols), '']
    rk = rank_among_all(bf, fs, ['model', 'n'], 'rmse_mean'); rk.to_csv(out / 'dsr_filter_ranks.csv', index=False)
    pivot = rk.pivot_table(index='method_id', columns=['model', 'n'], values='rank')
    pivot.columns = [f'{m}/N{n}' for m, n in pivot.columns]; pivot = pivot.reset_index()
    counts = ', '.join(f'N={n}: {c}' for n, c in rk.groupby('n').of.max().items())
    md += [f'RMSE rank of each DSR variant among the benchmark methods in the same cell, itself included (1 = lowest RMSE; methods per cell {counts}, since dense methods stop at N=128):', '',
           table(pivot.astype({c: float for c in pivot.columns[1:]}), list(pivot.columns)), '']

    # Local gradients.
    grad_ids = [m for m in BASELINES if m in set(bg.method_id)] + ['cuthbert/systematic_stop', 'pydpf/systematic']
    md += ['## Local gradients of a resampled expectation vs the exact weighted-distribution derivative (D=2, N=128)', '',
           'Bias = |mean gradient - exact|; variance over 100 seeds (10 for dense methods).', '']
    for cloud in ['gaussian', 'mixture']:
        for fn in ['x2', 'sin']:
            sel = lambda d: d[(d.cloud == cloud) & (d.function == fn) & (d.n == 128)]
            md += [f'### {cloud}, f={fn}', '', table(pd.concat([pick(sel(gs), dsr_ids), pick(sel(bg), list(dict.fromkeys(grad_ids)))]), ['method_id', 'gradient_mean', 'gradient_bias', 'gradient_variance', 'squared_error_mean']), '']

    # Native PF likelihood gradient vs exact Kalman gradient.
    native = root / 'torch_native/trials.csv'
    if native.exists():
        nt = native_summary(pd.read_csv(native), ['provider', 'method', 'n']); nt.to_csv(out / 'dsr_torch_native_summary.csv', index=False)
        bn = pd.read_csv(BASE / 'torch_native_summary.csv')
        both = label(pd.concat([nt, bn])); both = pick(both, dsr_ids + ['pydpf/systematic', 'pydpf/systematic_stop', 'pydpf/soft', 'torchfilter/soft', 'pydpf/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel'])
        md += ['## PF log-likelihood gradient vs exact Kalman gradient (random-walk model, theta = log(q, r))', '']
        for n in [128, 512]:
            md += [f'### N={n}', '', table(both[both.n == n], ['method_id', 'trials', 'gradient_bias', 'gradient_variance', 'gradient_mse', 'loglik_bias', 'loglik_variance', 'forward_ms', 'backward_ms']), '']

    # Parameter learning.
    learn = root / 'dsr/learning.csv'
    if learn.exists():
        l = pd.read_csv(learn); l['score_squared_error'] = l.score_error ** 2
        ls = summary(l, ['provider', 'method', 'n'], ['parameter_absolute_error', 'mle_reference_error', 'score_error', 'score_squared_error', 'learning_ms']); ls['score_rmse'] = np.sqrt(ls.score_squared_error_mean)
        ls.to_csv(out / 'dsr_learning_summary.csv', index=False)
        both = label(pd.concat([ls, pd.read_csv(BASE / 'learning_summary.csv')]))
        md += ['## Stochastic-volatility scale fitting (N=128, 10 datasets)', '', table(both, ['method_id', 'parameter_absolute_error_mean', 'mle_reference_error_mean', 'score_error_mean', 'score_rmse']), '']

    # Serial timing pass (fresh interpreter per method, idle machine), compared with results/performance.
    perf = root / 'performance/dsr/timing.csv'
    if perf.exists():
        import glob
        t = pd.concat([pd.read_csv(perf)] + [pd.read_csv(x) for x in glob.glob(str(ROOT / 'results/performance/*/timing.csv'))], ignore_index=True)
        t['method_id'] = t.provider + '/' + t.method
        ids = dsr_ids + ['particles/systematic', 'pydpf/systematic', 'pydpf/soft', 'custom/eigen', 'custom/gumbel', 'pfjax/ot', 'pydpf/ot']
        pv = t[t.method_id.isin(ids) & (t.cloud == 'gaussian') & (t.dim == 2)].pivot_table(index='method_id', columns='n', values='forward_ms', aggfunc='median')
        pv = pv.reindex([m for m in ids if m in pv.index]); pv.columns = [f'N={c}' for c in pv.columns]; pv = pv.reset_index()
        md += ['## Forward time per resampling call (serial pass, CPU, 1 thread; median ms, Gaussian cloud, D=2)', '', table(pv, list(pv.columns)), '']

    (out / 'REPORT.md').write_text('\n'.join(md) + '\n')
    print('Wrote', out / 'REPORT.md')


if __name__ == '__main__':
    main()
