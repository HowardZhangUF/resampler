"""PNG figures of the resampling-trigger comparison, in the style of results/dsr/report/figures.

One figure per table of scripts/trigger_latex_tables.py, plus two overviews:
- <set>-<model>/<set>-<model>-<metric>.png: panel (a) draws the table, the metric against N with one small panel per
  trigger (same y axis); panel (b), for RMSE, predictive NLL and coverage, shows each method's paired
  difference from E_2 on the same trials (mean +/- 95% CI over observation sequences), one marker per trigger.
- overview-rates.png: resampling steps per trigger, standard methods and DSR separately.
- overview-verdicts.png: share of (model, N, metric) cells where a trigger is significantly better / worse than E_2.
Reads the CSVs written by scripts/trigger_compare.py.

    .venv/bin/python scripts/trigger_figures.py --input results/triggers
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import plot_figures as pf  # noqa: E402
import dsr_table_figures as dtf  # noqa: E402
from dsr_latex_tables import NAMES  # noqa: E402
from trigger_latex_tables import METHODS, NS, SETS, TASKS, COMMON  # noqa: E402

TITLE = {'ess': '$E_2$ (current)', 'entropy': '$E_H$', 'maxweight': r'$E_\infty$',
         'entropy_matched': '$E_H$ (matched rate)', 'maxweight_matched': r'$E_\infty$ (matched rate)'}
# Panel (b) and overviews: trigger by marker / colour.
TRIGGER_MARKER = {'entropy': 'o', 'maxweight': '^', 'entropy_matched': 'o', 'maxweight_matched': '^'}
TRIGGER_COLOUR = {'ess': pf.INK, 'entropy': pf.BLUE, 'maxweight': pf.ORANGE, 'entropy_matched': pf.BLUE, 'maxweight_matched': pf.ORANGE}
DIRECTION = {'lower': '+ = worse than $E_2$', 'target95': '+ = further from 0.95 than $E_2$'}


def names(m): return NAMES[m]


def forest(axes, pr, model, metric, rows, sty, others, rule):
    """Panel (b): one column per N; per method one marker per trigger, filled when the CI excludes 0."""
    part = pr[(pr.model == model) & (pr.metric == metric)].set_index(['trigger', 'method_id', 'n'])
    offsets = dict(zip(others, [-.17, .17]))
    for j, (ax, n) in enumerate(zip(axes, NS)):
        lim = 0
        for i, m in enumerate(rows):
            c = sty[m][0]
            for t in others:
                if (t, m, n) not in part.index: continue
                r = part.loc[(t, m, n)]; d, h = r.diff_mean, r.diff_ci95_half
                if not np.isfinite(d): continue
                sig = np.isfinite(h) and abs(d) > h
                y = i + offsets[t]
                if np.isfinite(h): ax.plot([d - h, d + h], [y, y], color=c, lw=1.1, alpha=.45, solid_capstyle='butt', zorder=2)
                ax.plot(d, y, marker=TRIGGER_MARKER[t], ls='none', ms=3.8, color=c, mfc=c if sig else pf.SURFACE,
                        mec=pf.SURFACE if sig else c, mew=.8, zorder=3)
                lim = max(lim, abs(d) + (h if np.isfinite(h) else 0))
        ax.axvline(0, color=pf.INK2, lw=.8, zorder=1)
        lim = lim * 1.12 or 1
        ax.set_xlim(-lim, lim)
        tick = float(f'{lim * .7:.1g}')
        ax.set_xticks([-tick, 0, tick], [f'{-tick:g}', '0', f'{tick:g}'])
        ax.set_ylim(len(rows) - .5, -.5); ax.grid(axis='y', color=pf.ROWGRID)
        if j == 0: ax.set_yticks(range(len(rows)), [dtf.plain(names(m)) for m in rows])
        else: ax.tick_params(axis='y', labelleft=False)
        ax.set_title(f'N = {n}', loc='center', fontsize=7, fontweight='bold')
        ax.tick_params(axis='x', labelsize=6)
    axes[len(axes) // 2].set_xlabel(f'Paired difference from $E_2$ ({DIRECTION[rule]})', loc='center')


def table_figure(s, pr, model, task, metric, rule, name, key, triggers, out):
    part = s[s.model == model]
    rows = [m for m in METHODS if m in set(part.method_id)]
    sty = dtf.styles(rows)
    others = triggers[1:]
    paired = rule in DIRECTION and metric in set(pr.metric)
    fig = plt.figure(figsize=(7.8, 7.6 if paired else 4.2), layout='constrained')
    if paired: top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.35], hspace=.03)
    else: top = fig
    left, right = top.subfigures(1, 2, width_ratios=[3, .95], wspace=.01)
    axes = left.subplots(1, len(triggers), sharey=True)
    logs = []
    for ax, t in zip(axes, triggers):
        sel = part[part.trigger == t]
        pivot = sel.pivot_table(index='method_id', columns='n', values=metric+'_mean').reindex(index=rows, columns=NS)
        half = sel.pivot_table(index='method_id', columns='n', values=metric+'_ci95_half').reindex(index=rows, columns=NS)
        logs.append(dtf.lines_panel(ax, pivot, half, rows, sty, metric, dtf.plain(name), rule or 'lower'))
        ax.set_title(TITLE[t], loc='center')
        if ax is not axes[0]: ax.set_ylabel('')
    if any(logs) and not all(logs):  # one panel decided log, the others linear: make them agree
        for ax in axes: ax.set_yscale('linear')
    left.suptitle(f'(a) Table values: {task}, {dtf.plain(name)}', x=0.02, ha='left', fontsize=7.5, fontweight='bold')
    right.legend(handles=dtf.row_legend(right, rows, names, sty, 1, False), loc='center left', ncol=1, handlelength=2, fontsize=5.8, frameon=False)
    if paired:
        baxes = bottom.subplots(1, len(NS), sharey=True)
        forest(baxes, pr, model, metric, rows, sty, others, rule)
        bottom.suptitle('(b) Paired difference from $E_2$ on the same trials (filled: 95% CI excludes 0)', x=0.02, ha='left',
                        fontsize=7.5, fontweight='bold')
        bottom.legend(handles=[Line2D([], [], ls='none', marker=TRIGGER_MARKER[t], ms=4.5, color=pf.INK2, mec=pf.SURFACE, label=TITLE[t])
                               for t in others], loc='outside lower right', ncol=2, frameon=False)
    # One folder per task and trigger set, e.g. figures/fixed-ungm/fixed-ungm-rmse.png.
    folder = out/key.rsplit('-', 1)[0]; folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'{key}.png', dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def rates_figure(s, out):
    models = [m for m, _, _ in TASKS]
    titles = dict((m, t) for m, t, _ in TASKS)
    fig, axes = plt.subplots(2, len(models), figsize=(9.5, 4.2), sharex=True, layout='constrained')
    for row, (label, mask) in enumerate([('Standard resamplers', ~s.method_id.str.startswith('dsr/')), ('DSR variants', s.method_id.str.startswith('dsr/'))]):
        g = s[mask].groupby(['trigger', 'model', 'n']).resampling_steps_mean.mean()
        for ax, model in zip(axes[row], models):
            for t in ['ess', 'entropy', 'maxweight', 'entropy_matched', 'maxweight_matched']:
                if (t, model) not in g.index.droplevel(2): continue
                y = g.loc[(t, model)].reindex(NS)
                ax.plot(NS, y, color=TRIGGER_COLOUR[t], ls='--' if t.endswith('matched') else '-', marker='o' if t == 'ess' else TRIGGER_MARKER[t],
                        ms=3.5, mec=pf.SURFACE, mew=.5, lw=1.4 if t == 'ess' else 1.1, zorder=4 if t == 'ess' else 3)
            ax.set_xscale('log', base=2); ax.set_xticks(NS, [str(n) for n in NS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(0, 52)
            if row == 0: ax.set_title(titles[model], loc='center')
            if row == 1: ax.set_xlabel('N')
        axes[row][0].set_ylabel(f'{label}\nresampling steps (of 50)')
    handles = [Line2D([], [], color=TRIGGER_COLOUR[t], ls='--' if t.endswith('matched') else '-', marker='o' if t == 'ess' else TRIGGER_MARKER[t],
                      ms=4, mec=pf.SURFACE, label=TITLE[t]) for t in ['ess', 'entropy', 'maxweight', 'entropy_matched', 'maxweight_matched']]
    fig.legend(handles=handles, loc='outside upper center', ncol=5, frameon=False)
    fig.savefig(out/'overview-rates.png', dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def verdicts_figure(pr, out):
    rows = [m for m in METHODS if m in set(pr.method_id)]
    sets = [('Common threshold 0.5N', ['entropy', 'maxweight']), ('Matched resampling rate', ['entropy_matched', 'maxweight_matched'])]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.6), sharey=True, layout='constrained')
    for ax, (title, triggers) in zip(axes, sets):
        for k, t in enumerate(triggers):
            share = pr[pr.trigger == t].groupby('method_id').verdict.value_counts(normalize=True).unstack(fill_value=0)
            for i, m in enumerate(rows):
                if m not in share.index: continue
                y = i + (k - .5) * .38
                b, w = share.loc[m].get('better', 0), share.loc[m].get('worse', 0)
                ax.barh(y, -100*b, height=.36, color=pf.AQUA, edgecolor=pf.SURFACE, lw=.5)
                ax.barh(y, 100*w, height=.36, color=pf.ORANGE, edgecolor=pf.SURFACE, lw=.5)
                ax.plot(-100*b - 1.5 if b else -1.5, y, marker=TRIGGER_MARKER[t], ms=3, color=pf.INK2, ls='none')
        ax.axvline(0, color=pf.INK2, lw=.8)
        for x in (-5, 5): ax.axvline(x, color=pf.INK2, lw=.7, ls=(0, (3, 3)))
        ax.set_xlim(-40, 40); ax.set_xticks([-40, -20, -5, 0, 5, 20, 40], ['40', '20', '5', '0', '5', '20', '40'])
        ax.set_title(title, loc='center'); ax.grid(axis='y', color=pf.ROWGRID)
        ax.set_xlabel('% of cells: better than $E_2$ (left) / worse (right)')
        ax.axhline(3.5, color=pf.AXIS, lw=.8)
    axes[0].set_ylim(len(rows) - .5, -.5); axes[0].set_yticks(range(len(rows)), [dtf.plain(names(m)) for m in rows])
    handles = [Patch(color=pf.AQUA, label='significantly better'), Patch(color=pf.ORANGE, label='significantly worse'),
               Line2D([], [], ls='none', marker='o', ms=4, color=pf.INK2, label='$E_H$ (upper bar)'),
               Line2D([], [], ls='none', marker='^', ms=4, color=pf.INK2, label=r'$E_\infty$ (lower bar)'),
               Line2D([], [], color=pf.INK2, lw=.7, ls=(0, (3, 3)), label='5%: expected by chance')]
    fig.legend(handles=handles, loc='outside upper center', ncol=5, frameon=False)
    fig.savefig(out/'overview-verdicts.png', dpi=160, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', default='results/triggers'); a = p.parse_args()
    rep = ROOT/a.input/'report'; out = rep/'figures'; out.mkdir(exist_ok=True)
    s = pd.read_csv(rep/'trigger_filter_summary.csv')
    pr = pd.read_csv(rep/'trigger_paired_vs_ess.csv')
    dtf.setup()
    count = 0
    for key, _, triggers in SETS:
        if not set(triggers) <= set(s.trigger): continue
        for model, task, extra in TASKS:
            for metric, rule, name in COMMON[:1]+extra+COMMON[1:]:
                table_figure(s, pr, model, task, metric, rule, name, f'{key}-{model}-{metric}', triggers, out); count += 1
    rates_figure(s, out); verdicts_figure(pr, out)
    print('Wrote', count+2, 'figures to', out)


if __name__ == '__main__':
    main()
