"""LaTeX tables of the resampling-trigger comparison: one table per filtering task and metric.

Rows are methods; column groups are triggers, each with N = 32 ... 2048. Two sets: triggers at the
common threshold 0.5N, and triggers at the matched resampling rate (scripts/calibrate_triggers.py).
Cells are means; a superscript +/- marks a paired difference from 'ess' whose 95% CI excludes 0
(better/worse), and bold marks the best trigger for that method and N. Reads the CSVs written by
scripts/trigger_compare.py. Requires booktabs, amsmath, amssymb and adjustbox.

    .venv/bin/python scripts/trigger_latex_tables.py --input results/triggers
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from dsr_latex_tables import NAMES, DIRECTION, num  # noqa: E402

NS = [32, 128, 512, 2048]
METHODS = ['dsr/dsr', 'dsr/dsr_unsorted', 'dsr/dsr_tau0.5', 'dsr/dsr_scaled',
           'particles/systematic', 'particles/multinomial', 'pydpf/systematic_stop', 'pydpf/soft', 'pydpf/ot', 'pfjax/ot',
           'filterflow/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel', 'variational/kl']
TRIGGER_NAMES = {'ess': r'$E_2$', 'entropy': r'$E_H$', 'maxweight': r'$E_\infty$',
                 'entropy_matched': r'$E_H$ (matched)', 'maxweight_matched': r'$E_\infty$ (matched)'}
SETS = [('fixed', 'common threshold $0.5N$', ['ess', 'entropy', 'maxweight']),
        ('matched', 'matched resampling rate', ['ess', 'entropy_matched', 'maxweight_matched'])]
TASKS = [('linear', 'Linear Gaussian SSM', [('posterior_mean_rmse', 'lower', 'posterior-mean RMSE vs.\\ Kalman'),
                                            ('loglik_error', 'abs', 'log-likelihood error vs.\\ Kalman')]),
         ('ungm', 'UNGM', []), ('volatility', 'Stochastic volatility', []),
         ('bearing', 'Bearing-only tracking', [('post_occlusion_rmse', 'lower', 'RMSE after the observation gap')]),
         ('lorenz63', 'Stochastic Lorenz-63', []), ('lorenz96', 'Stochastic Lorenz-96', [])]
COMMON = [('rmse', 'lower', 'state RMSE'), ('coverage95', 'target95', '95\\% coverage'), ('predictive_nll', 'lower', 'predictive NLL'),
          ('resampling_steps', None, 'resampling steps out of 50')]


def tex_table(s, verdicts, model, metric, rule, triggers, caption, label):
    part = s[s.model == model]
    mean = part.pivot_table(index='method_id', columns=['trigger', 'n'], values=metric+'_mean')
    rows = [m for m in METHODS if m in mean.index]
    head = ' & '.join(rf'\multicolumn{{{len(NS)}}}{{c}}{{{TRIGGER_NAMES[t]}}}' for t in triggers)
    rules = ' '.join(rf'\cmidrule(lr){{{2+i*len(NS)}-{1+(i+1)*len(NS)}}}' for i in range(len(triggers)))
    lines = [r'\begin{table}[ht]', r'\centering', r'\small', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{l'+('c'*len(NS)+'|')*(len(triggers)-1)+'c'*len(NS)+'}', r'\toprule',
             ' & '+head+r' \\', rules, 'Method & '+' & '.join(f'${n}$' for _ in triggers for n in NS)+r' \\', r'\midrule']
    for i, m in enumerate(rows):
        if i == 4: lines.append(r'\midrule')
        cells = []
        for t in triggers:
            for n in NS:
                v = mean.get((t, n), pd.Series(dtype=float)).get(m, np.nan)
                if not np.isfinite(v): cells.append('--'); continue
                s_ = num(v)
                if rule:
                    vals = {u: mean.get((u, n), pd.Series(dtype=float)).get(m, np.nan) for u in triggers}
                    score = {u: {'lower': x, 'abs': abs(x), 'target95': abs(x-.95)}[rule] for u, x in vals.items() if np.isfinite(x)}
                    if len(score) > 1 and np.isclose(score[t], min(score.values())): s_ = rf'\mathbf{{{s_}}}'
                verdict = verdicts.get((t, metric, m, model, n))
                if verdict in ('better', 'worse'): s_ += '^{+}' if verdict == 'better' else '^{-}'
                cells.append(f'${s_}$')
        lines.append(NAMES[m]+' & '+' & '.join(cells)+r' \\')
    note = (r"$^\dagger$Local diagnostic, not from the paper: $\tau$ rescaled so that $\tau N$ equals the paper's value at $N{=}25$. "
            r'Cells are means; $^{+}$/$^{-}$: paired difference from $E_2$ on the same trials significantly better/worse (95\% CI over observation sequences excludes 0). '
            r'Bold marks the best trigger for that method and $N$. A dash means not run (dense methods stop at $N{=}128$).')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\par\smallskip\parbox{0.95\linewidth}{\footnotesize '+note+'}', r'\end{table}', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', default='results/triggers'); a = p.parse_args()
    rep = ROOT/a.input/'report'
    s = pd.read_csv(rep/'trigger_filter_summary.csv')
    pr = pd.read_csv(rep/'trigger_paired_vs_ess.csv')
    verdicts = {(r.trigger, r.metric, r.method_id, r.model, r.n): r.verdict for r in pr.itertuples()}
    cal = ROOT/a.input/'matched_thresholds.json'
    tau = json.loads(cal.read_text())['thresholds'] if cal.exists() else {}
    out = []
    for key, set_name, triggers in SETS:
        if not set(triggers) <= set(s.trigger): continue
        title = f'Resampling triggers, {set_name}'
        out.append(f'% {"=" * 70}\n% {title}\n% {"=" * 70}\n\\clearpage\n\\section*{{{title}}}\n')
        if key == 'fixed':
            rule = (r'Resample when $E_2=1/\sum_iw_i^2$, $E_H=\exp(-\sum_iw_i\log w_i)$ or $E_\infty=1/\max_iw_i$ falls below $0.5N$. '
                    r'$E_\infty\le E_2\le E_H$, so $E_\infty$ resamples most often and $E_H$ least. $CV^2>1$ is the same rule as $E_2<0.5N$.')
        else:
            th = '; '.join(f'{m}: $\\tau_H{{=}}{tau["entropy"][m]:.3g}$, $\\tau_\\infty{{=}}{tau["maxweight"][m]:.3g}$' for m in tau['entropy'])
            rule = (r'Resample when $E_H<\tau_HN$ or $E_\infty<\tau_\infty N$, with per-model thresholds calibrated so that systematic resampling '
                    rf'resamples as often as under $E_2<0.5N$ ({th}).')
        for model, task, extra in TASKS:
            for metric, rule_, name in COMMON[:1]+extra+COMMON[1:]:
                direction = f' ({DIRECTION[rule_]})' if rule_ else ''
                caption = (f'{task}: {name}{direction}, {set_name}. {rule} 50 steps; 100 filter runs over 10 observation sequences (10 for OT and Gumbel).')
                out.append(tex_table(s, verdicts, model, metric, rule_, triggers, caption, f'tab:trigger-{key}-{model}-{metric}'))
    # Hand-written conclusions; kept in their own file so regeneration does not overwrite them.
    if (rep/'conclusions.tex').exists(): out.append((rep/'conclusions.tex').read_text())
    path = rep/'tables.tex'; path.write_text('\n'.join(out))
    (rep/'tables_preview.tex').write_text('\\documentclass{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{booktabs,amsmath,amssymb,adjustbox}\n'
                                          '\\begin{document}\n\\input{tables.tex}\n\\end{document}\n')
    print('Wrote', path, '-', sum(x.count('\\begin{table}') for x in out), 'tables')
    if shutil.which('pdflatex'):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'tables_preview.tex'], cwd=rep, capture_output=True, text=True)
        print('pdflatex exit', r.returncode)


if __name__ == '__main__':
    main()
