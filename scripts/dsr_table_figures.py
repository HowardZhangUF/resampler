"""One figure per table of scripts/dsr_latex_tables.py, placed directly after its table.

Panel (a) draws the table itself in the style of results/standard/report/figures/08_filter_rmse_vs_n:
the metric against N on a log axis, one line per table row, coloured by family, whiskers = the table's 95% CI.
Metrics that span orders of magnitude get a log y axis and a dashed Monte Carlo reference slope, so error
floors (a line that stops falling) are visible.

Panel (b), for filtering metrics, shows what the table hides. The table's +/- is a 95% CI over 10
observation sequences, and most of its width is the sequences' own difficulty, which every method shares.
Panel (b) differences each method from systematic resampling on the same sequence first (paired t interval),
which removes that shared part.

Panel (c), only for the linear-Gaussian state RMSE, splits the mean squared error into the exact Kalman
filter's error and the part the particle filter adds. Used by dsr_latex_tables.py; not a standalone script.
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import t as student_t
import plot_figures as pf

REFERENCE = 'particles/systematic'
FAMILY_COLOUR = {'dsr': pf.INK, 'classical': pf.MUTED, 'moment': pf.BLUE, 'soft': pf.ORANGE, 'ot': pf.AQUA, 'variational': pf.PLUM}
FAMILY_NAME = {'dsr': 'DSR', 'classical': 'Classical', 'moment': 'Moment matching', 'soft': 'Soft / Gumbel', 'ot': 'Optimal transport',
               'variational': 'Variational'}
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', 'h', '<', '>', 'p', '*', 'd', 'H', '8']
DASHES = ['-', '--', ':', '-.']
# Monte Carlo rate of each metric (exponent of N), drawn as a dashed reference on log axes.
SLOPE = {'posterior_mean_rmse': -.5, 'posterior_covariance_error': -.5, 'sliced_w1': -.5, 'rff_mmd2': -1, 'covariance_error': -.5,
         'estimator_mse_x2': -1, 'mode_mass_error': -.5, 'conditional_loglik_variance': -1, 'gradient_variance': -1,
         'loglik_variance': -1, 'gradient_mse': -1, 'squared_error_mean': -1, 'forward_ms': 1}
PAIRED_DIRECTION = {'lower': '+ = worse than systematic', 'higher': '+ = better than systematic',
                    'abs': 'signed difference', 'target95': 'signed difference (target 0.95)'}
CAPTION_FAMILIES = (r'Colour is the method family: black DSR, gray classical, blue moment matching, orange soft/Gumbel, green optimal '
                    r'transport, purple variational; line style and marker separate rows within a family.')


def slope_of(metric):
    return SLOPE.get(metric, SLOPE.get(metric.removesuffix('_mean')))


def family(method_id):
    provider, method = method_id.split('/')
    return 'dsr' if provider == 'dsr' else pf.family(method)


def plain(name):
    """LaTeX row or metric name -> matplotlib mathtext."""
    name = name.replace(r'$^\dagger$', '†').replace('{=}', '=').replace(r'\ ', ' ').replace('vs.\\', 'vs.')
    name = re.sub(r'\$([^$\\]*)\\%([^$\\]*)\$', r'\1%\2', name)  # math-mode percent, e.g. $>1\%$
    return name.replace('\\%', '%')


_STYLE, _PER_FAMILY = {}, {}


def styles(rows):
    """Colour by family; marker and dash by order of first appearance within the family. The assignment is
    remembered across calls, so a method keeps the same look in every figure of a run."""
    for m in rows:
        if m in _STYLE: continue
        f = family(m); k = _PER_FAMILY.get(f, 0); _PER_FAMILY[f] = k + 1
        _STYLE[m] = (FAMILY_COLOUR[f], MARKERS[k % len(MARKERS)], DASHES[k % len(DASHES)])
    return {m: _STYLE[m] for m in rows}


def setup():
    pf.style()
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'legend.fontsize': 6.3, 'ytick.major.size': 2, 'lines.linewidth': 1.1})


def is_log(values):
    v = values[np.isfinite(values)]
    return len(v) > 0 and (v > 0).all() and v.max() / v.min() > 8


def lines_panel(ax, pivot, half, rows, sty, metric, ylabel, rule):
    """Panel (a): the table, value against N. Returns whether the y axis is logarithmic."""
    ns = list(pivot.columns)
    log = is_log(pivot.values.astype(float))
    k = len(rows); spread = .24 if k > 20 else .22
    for i, m in enumerate(rows):
        y = pivot.loc[m].astype(float); ok = y.notna().values
        if not ok.any(): continue
        off = 2 ** (spread * (2 * i / max(k - 1, 1) - 1))
        xs = np.array(ns, float)[ok] * off; ys = y.values[ok]
        c, mk, ls = sty[m]
        thin = k > 20 and family(m) != 'dsr'
        ax.plot(xs, ys, color=c, marker=mk, ls=ls, ms=3 if thin else 4, mec=pf.SURFACE, mew=.5, lw=.7 if thin else 1.1,
                alpha=.75 if thin else 1, zorder=3 + (family(m) == 'dsr'))
        if half is not None:
            h = half.loc[m].astype(float).values[ok]
            if np.isfinite(h).any():
                lo = np.minimum(h, .95 * ys) if log else h
                ax.errorbar(xs, ys, yerr=[lo, h], fmt='none', ecolor=c, elinewidth=.8, alpha=.3, capsize=0, zorder=2)
    ax.set_xscale('log', base=2); ax.set_xticks(ns, [str(n) for n in ns]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ns[0] / 1.6, ns[-1] * 1.6); ax.set_xlabel('Particles N (log scale)')
    if log:
        ax.set_yscale('log')
        v = pivot.values.astype(float); v = v[np.isfinite(v)]
        if np.log10(v.max() / v.min()) <= 2.5: ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        pf.plain_log(ax.yaxis)
        slope = slope_of(metric)
        if slope is not None:
            v0 = np.nanmedian(pivot[ns[0]].astype(float).values)
            xs = np.array(ns, float); ref = v0 * (xs / ns[0]) ** slope
            ax.plot(xs, ref, color=pf.INK2, ls=(0, (4, 3)), lw=.9, zorder=1)
            txt = {-.5: r'$\propto N^{-1/2}$', -1: r'$\propto N^{-1}$', 1: r'$\propto N$'}[slope]
            ax.annotate(txt, (xs[-1], ref[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=6.5, color=pf.INK2)
    elif rule == 'abs':
        ax.axhline(0, color=pf.INK2, lw=.8, zorder=1)
    if rule == 'target95':
        ax.axhline(.95, color=pf.INK2, lw=.8, ls=(0, (4, 3)), zorder=1)
    ax.set_ylabel(ylabel)
    return log


def paired(raw, metric, rows, ns):
    """Per observation sequence: method mean minus systematic mean. Returns {(m, n): (mean, half, k)}."""
    per = raw.groupby(['method_id', 'n', 'data_seed'])[metric].mean()
    out = {}
    for n in ns:
        if (REFERENCE, n) not in per.index.droplevel(2): continue
        ref = per.loc[(REFERENCE, n)]
        for m in rows:
            if (m, n) not in per.index.droplevel(2): continue
            d = (per.loc[(m, n)] - ref).dropna()
            if len(d) < 2: continue
            half = student_t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            out[(m, n)] = (d.mean(), half, len(d))
    return out


def forest(axes, diffs, rows, names, sty, ns, table_half, rule):
    """Panel (b): one column per N, rows in table order. Filled marker = paired CI excludes 0."""
    for j, (ax, n) in enumerate(zip(axes, ns)):
        lim = 0
        for i, m in enumerate(rows):
            if m == REFERENCE:
                ax.annotate('reference', (0, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=6, color=pf.MUTED)
                continue
            if (m, n) not in diffs: continue
            d, h, _ = diffs[(m, n)]; c, mk, _ = sty[m]
            sig = abs(d) > h
            ax.plot([d - h, d + h], [i, i], color=c, lw=1.1, alpha=.45, solid_capstyle='butt', zorder=2)
            ax.plot(d, i, marker=mk, ls='none', ms=4, color=c, mfc=c if sig else pf.SURFACE, mec=c if not sig else pf.SURFACE,
                    mew=.8, zorder=3)
            lim = max(lim, abs(d) + h)
        ax.axvline(0, color=pf.INK2, lw=.8, zorder=1)
        lim = lim * 1.12 or 1
        ax.set_xlim(-lim, lim)
        tick = float(f'{lim * .7:.1g}')  # three symmetric ticks fit the narrow columns
        ax.set_xticks([-tick, 0, tick], [f'{-tick:g}', '0', f'{tick:g}'])
        ax.set_ylim(len(rows) - .5, -.5); ax.grid(axis='y', color=pf.ROWGRID)
        if j == 0: ax.set_yticks(range(len(rows)), [plain(names(m)) for m in rows])
        else: ax.tick_params(axis='y', labelleft=False)
        th = table_half.get(n, np.nan)
        sub = f'\ntable ±{th:.2g}' if np.isfinite(th) else ''
        ax.set_title(f'N = {n}{sub}', loc='center', fontsize=7, fontweight='bold', linespacing=1.3)
        ax.tick_params(axis='x', labelsize=6)
    axes[len(axes) // 2].set_xlabel(f'Paired difference from systematic ({PAIRED_DIRECTION[rule]})', loc='center')


def decomposition(axes, raw, rows, names, sty, ns):
    """Panel (c), linear Gaussian only: MSE = Kalman MSE + particle-filter excess (mean over all runs)."""
    shares = []
    for ax, n in zip(axes, ns):
        part = raw[raw.n == n]
        g = part.assign(mse=part.rmse ** 2, pm2=part.posterior_mean_rmse ** 2).groupby('method_id')[['mse', 'pm2']].mean()
        for i, m in enumerate(rows):
            if m not in g.index: continue
            total, extra = g.at[m, 'mse'], g.at[m, 'pm2']
            ax.barh(i, total - extra, height=.7, color=pf.GRID, edgecolor=pf.SURFACE, lw=1)
            ax.barh(i, extra, left=total - extra, height=.7, color=sty[m][0], edgecolor=pf.SURFACE, lw=1)
            ax.annotate(f'{100 * extra / total:.0f}%', (total, i), xytext=(3, 0), textcoords='offset points', va='center', fontsize=5.8,
                        color=pf.INK2)
            shares.append(extra / total)
        ax.set_ylim(len(rows) - .5, -.5); ax.grid(axis='y', visible=False)
        ax.set_xlim(0, g.mse.max() * 1.18)
        ax.set_title(f'N = {n}', loc='center', fontsize=7, fontweight='bold')
        if ax is axes[0]: ax.set_yticks(range(len(rows)), [plain(names(m)) for m in rows])
        else: ax.tick_params(axis='y', labelleft=False)
    axes[0].set_xlabel('Mean squared error: gray = exact Kalman filter, colour = added by the particle filter', loc='left')
    return min(shares), max(shares)


def row_legend(target, rows, names, sty, ncol, many):
    if many:  # one entry per family, DSR variants individually
        handles = [Line2D([], [], color=sty[m][0], marker=sty[m][1], ls=sty[m][2], ms=4, mec=pf.SURFACE, label=plain(names(m)))
                   for m in rows if family(m) == 'dsr']
        handles += [Line2D([], [], color=FAMILY_COLOUR[f], marker='o', ls='-', ms=4, mec=pf.SURFACE, label=FAMILY_NAME[f] + ' (all rows)')
                    for f in FAMILY_COLOUR if f != 'dsr' and any(family(m) == f for m in rows)]
    else:
        handles = [Line2D([], [], color=sty[m][0], marker=sty[m][1], ls=sty[m][2], ms=4, mec=pf.SURFACE, label=plain(names(m))) for m in rows]
    return handles


def table_figure(df, methods, metric, ci, rule, ylabel, label_key, ns, names, out_dir, raw=None, raw_metric=None):
    """Draws the figure for one table; returns the LaTeX figure block."""
    pivot = df.pivot_table(index='method_id', columns='n', values=metric)
    half = df.pivot_table(index='method_id', columns='n', values=ci) if ci and ci in df else None
    rows = [m for m in methods if m in pivot.index]
    pivot = pivot.reindex(index=rows, columns=ns)
    if half is not None: half = half.reindex(index=rows, columns=ns)
    sty = styles(rows); many = len(rows) > 20
    diffs = paired(raw, raw_metric, rows, ns) if raw is not None and raw_metric else {}
    decompose = label_key == 'linear-rmse' and raw is not None
    ylabel = plain(ylabel)

    if diffs:
        if many:
            fig = plt.figure(figsize=(7, 7.9), layout='constrained')
            top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.4], hspace=.02)
            ax = top.subplots(); log = lines_panel(ax, pivot, half, rows, sty, metric, ylabel, rule)
            ax.legend(handles=row_legend(top, rows, names, sty, 1, True), loc='center left', bbox_to_anchor=(1.01, .5), ncol=1)
            right = bottom
        else:
            fig = plt.figure(figsize=(7.8, 6.9 if decompose else 4.4), layout='constrained')
            if decompose:
                upper, lower = fig.subfigures(2, 1, height_ratios=[4.3, 2.3], hspace=.03)
            else:
                upper = fig
            left, right = upper.subfigures(1, 2, width_ratios=[1.15, 1.75], wspace=.02)
            ax = left.subplots(); log = lines_panel(ax, pivot, half, rows, sty, metric, ylabel, rule)
            left.legend(handles=row_legend(left, rows, names, sty, 2, False), loc='outside lower center', ncol=2, handlelength=2,
                        columnspacing=.6, handletextpad=.3, fontsize=5.8)
        ax.set_title('(a) Table values')
        baxes = right.subplots(1, len(ns), sharey=True)
        ref_half = half.loc[REFERENCE].to_dict() if half is not None and REFERENCE in half.index else {}
        forest(baxes, diffs, rows, names, sty, ns, ref_half, rule)
        right.suptitle('(b) Paired difference from systematic on the same observation sequences', x=0.02, ha='left', fontsize=7.5,
                       fontweight='bold')
        share = None
        if decompose:
            caxes = lower.subplots(1, 2, sharey=True)
            share = decomposition(caxes, raw, rows, names, sty, ns[:2])
            lower.suptitle('(c) Where the state MSE comes from', x=0.02, ha='left', fontsize=7.5, fontweight='bold')
    else:
        many_legend = many
        fig, ax = plt.subplots(figsize=(7, 3.4 if not many else 4.2), layout='constrained')
        log = lines_panel(ax, pivot, half, rows, sty, metric, ylabel, rule)
        ax.legend(handles=row_legend(fig, rows, names, sty, 1, many_legend), loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
                  handlelength=2.2)
        share = None

    path = out_dir / f'{label_key}.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=.04); fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    return latex(label_key, rule, log, metric, bool(half is not None), bool(diffs), share)


def latex(label_key, rule, log, metric, has_ci, has_paired, share):
    a = [r'\textbf{(a)} The values of Table~\ref{tab:' + label_key + r'} against $N$ (log scale), one line per row.']
    a.append(CAPTION_FAMILIES)
    if has_ci: a.append(r"Whiskers are the table's 95\% CIs; points are shifted sideways so whiskers do not overlap.")
    if log:
        a.append(r'Log $y$ axis.')
        if slope_of(metric) is not None:
            a.append(r'The dashed line is the Monte Carlo reference slope; a method whose line flattens has an error floor.')
    if rule == 'target95': a.append(r'Dashed line: nominal 0.95.')
    if has_paired:
        a.append(r'\textbf{(b)} Each method minus systematic resampling, computed per observation sequence and then averaged '
                 r'(mean $\pm$ paired 95\% $t$ interval over the 10 sequences). Filled markers: the interval excludes 0. '
                 r"The table's $\pm$ for systematic, printed under each $N$, is dominated by how hard each sequence is, which every method "
                 r'shares; pairing removes that, so differences invisible in (a) can be resolved here.')
    if share is not None:
        a.append(r"\textbf{(c)} Mean squared state error, split into the exact Kalman filter's error (gray; it depends only on the data) "
                 r'and the extra error from the particle filter (colour; the squared posterior-mean RMSE of '
                 r'Table~\ref{tab:linear-posterior_mean_rmse}). The particle filter contributes only '
                 rf'{100 * share[0]:.0f}--{100 * share[1]:.0f}\% of the MSE, and RMSE adds the two in quadrature, so the state RMSE '
                 r'cannot separate resamplers; compare methods with (b) or with Table~\ref{tab:linear-posterior_mean_rmse}.')
    return '\n'.join([r'\begin{figure}[H]', r'\centering', rf'\includegraphics[width=\linewidth]{{figures/{label_key}.pdf}}',
                      r'\caption{' + ' '.join(a) + '}', rf'\label{{fig:{label_key}}}', r'\end{figure}', ''])


def learning_figure(df, rows, names, out_dir):
    """Parameter-learning table: one N, so one dot-and-whisker panel per column."""
    df = df.set_index('method_id').reindex(rows)
    cols = [('parameter_absolute_error', r'$|\hat s-s^\star|$'), ('mle_reference_error', r'$|\hat s-\hat s_{\rm grid}|$'),
            ('score_error', 'Score error'), ('score_rmse', 'Score RMSE')]
    sty = styles(rows)
    fig, axes = plt.subplots(1, 4, figsize=(7, 3), sharey=True, layout='constrained')
    for ax, (metric, title) in zip(axes, cols):
        mean = df[metric + '_mean'] if metric + '_mean' in df else df[metric]
        h = df.get(metric + '_ci95_half', pd.Series(np.nan, index=df.index))
        for i, m in enumerate(rows):
            if not np.isfinite(mean[m]): continue
            c, mk, _ = sty[m]
            if np.isfinite(h[m]): ax.plot([mean[m] - h[m], mean[m] + h[m]], [i, i], color=c, lw=1.1, alpha=.45, solid_capstyle='butt')
            ax.plot(mean[m], i, marker=mk, ms=4, color=c, mec=pf.SURFACE, mew=.6, ls='none')
        if metric == 'score_error': ax.axvline(0, color=pf.INK2, lw=.8)
        ax.set_title(title, loc='center'); ax.grid(axis='y', color=pf.ROWGRID); ax.locator_params(axis='x', nbins=4)
    axes[0].set_ylim(len(rows) - .5, -.5); axes[0].set_yticks(range(len(rows)), [plain(names[m]) for m in rows])
    path = out_dir / 'volatility-learning.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=.04); fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    cap = (r'Plot of Table~\ref{tab:volatility-learning}: mean $\pm$ 95\% CI over 10 observation sequences, one panel per column. '
           r'Rows are in table order. ' + CAPTION_FAMILIES)
    return '\n'.join([r'\begin{figure}[H]', r'\centering', r'\includegraphics[width=\linewidth]{figures/volatility-learning.pdf}',
                      r'\caption{' + cap + '}', r'\label{fig:volatility-learning}', r'\end{figure}', ''])
