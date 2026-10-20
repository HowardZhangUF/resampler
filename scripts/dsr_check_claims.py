"""Compare reproduced DSR appendix results with the published values (arXiv:2610.12094).

Reads the outputs written to results/dsr_upstream by the upstream scripts and the
chunked drivers, and writes results/dsr_upstream/claims.csv with one row per number.

    .venv/bin/python scripts/dsr_check_claims.py
"""
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dsr_upstream'
VENDOR = ROOT / 'vendor/differentiable-systematic-resampling'

# Table 6: (bias, var, MSE) per method for f(x)=x and f(x)=x^2.
TABLE6 = {
    'f(x)=x': {'hard SR': (2.3e-4, 1.2e-3, 1.2e-3), 'multinomial': (4.9e-4, 5.5e-3, 5.5e-3),
               'DSR tau=0.01': (2.4e-3, 2.9e-5, 3.4e-5), 'DSR tau=0.05': (1.2e-2, 8.6e-6, 1.6e-4),
               'DSR tau=0.1': (2.0e-2, 4.7e-6, 3.9e-4), 'DSR tau=0.5': (1.2e-2, 1.6e-7, 1.4e-4),
               'DSR tau=1.0': (3.6e-3, 1.3e-8, 1.3e-5)},
    'f(x)=x^2': {'hard SR': (1.2e-4, 7.0e-4, 7.0e-4), 'multinomial': (2.8e-4, 2.6e-3, 2.6e-3),
                 'DSR tau=0.01': (1.2e-1, 3.8e-7, 1.4e-2), 'DSR tau=0.05': (2.3e-1, 3.1e-7, 5.3e-2),
                 'DSR tau=0.1': (2.7e-1, 2.0e-7, 7.3e-2), 'DSR tau=0.5': (3.1e-1, 2.2e-9, 9.9e-2),
                 'DSR tau=1.0': (3.2e-1, 2.0e-10, 1.0e-1)}}
# Table 13: inter-modal mass mean per dimension.
TABLE13 = {'hard': [0, 0, 0, 0, 0], 'soft': [0, 0, 0, 0, 0], 'erot': [0, 0, .747, 1, 1],
           'dsr_sorted': [.287, .283, .282, .282, .282], 'dsr_unsorted': [1, 1, 1, 1, 1]}
# Table 15: (bias, variance, MSE).
TABLE15 = {2: {'Hard SR': (2.324, .308, 5.707), 'DSR (tau=1.0)': (1.256, .241, 1.818), 'DSR (tau=0.5)': (1.257, .245, 1.825),
               'DSR (tau=0.1)': (.764, .389, .973), 'DSR (tau=0.05)': (.370, .723, .860), 'DSR (tau=0.01)': (.343, 13.409, 13.526)},
           4: {'Hard SR': (3.627, 2.092, 15.248), 'DSR (tau=1.0)': (2.869, 2.073, 10.304), 'DSR (tau=0.5)': (2.914, 2.127, 10.615),
               'DSR (tau=0.1)': (2.131, 4.895, 9.437), 'DSR (tau=0.05)': (1.720, 16.382, 19.340), 'DSR (tau=0.01)': (1.412, 384.805, 386.799)}}


def sig2(x):
    return float(f'{x:.1e}')


def main():
    rows = []
    bv = OUT / 'bias_variance.json'
    if bv.exists():
        got = json.loads(bv.read_text())
        for f, methods in TABLE6.items():
            for m, vals in methods.items():
                for stat, paper in zip(['bias', 'var', 'mse'], vals):
                    ours = got[f][m][stat]
                    # Paper prints 2 significant figures; bias of unbiased schemes is pure MC noise.
                    rows.append(dict(table='6', setting=f, method=m, stat=stat, paper=paper, ours=ours,
                                     match=sig2(ours) == paper, note='chunked rerun; not bit-exact'))
    im = OUT / 'intermodal_mass.json'
    if im.exists():
        got = json.loads(im.read_text())
        for m, vals in TABLE13.items():
            for d, paper in zip([1, 2, 3, 5, 10], vals):
                ours = got[str(d)][m]['mass_mean']
                rows.append(dict(table='13', setting=f'd={d}', method=m, stat='mass_mean', paper=paper, ours=ours,
                                 match=round(ours, 3) == paper, note='chunked rerun; not bit-exact'))
    vg = OUT / 'vsmc_gradient_bias.log'
    if vg.exists():
        d = None
        for line in vg.read_text().splitlines():
            h = re.match(r'### d=(\d+)', line)
            if h: d = int(h.group(1)); continue
            r = re.match(r'(Hard SR|DSR \(tau=[\d.]+\))\s+([\d.]+) \+- [\d.]+\s+([\d.]+) \+- [\d.]+\s+([\d.]+) \+-', line)
            if r and d in TABLE15:
                for stat, paper, ours in zip(['bias', 'variance', 'mse'], TABLE15[d][r.group(1)], map(float, r.group(2, 3, 4))):
                    rows.append(dict(table='15', setting=f'd={d}', method=r.group(1), stat=stat, paper=paper, ours=ours,
                                     match=round(ours, 3) == paper, note='upstream script, unmodified'))
    rm = OUT / 'relaxation_metrics.json'
    if rm.exists():
        got = json.loads(rm.read_text()); pub = json.loads((VENDOR / 'appendix/empirical_analysis/published_relaxation_metrics.json').read_text())
        for n, taus in pub.items():
            for tau, methods in taus.items():
                for m, metrics in methods.items():
                    for metric, v in metrics.items():
                        if n in got and tau in got[n] and m in got[n][tau] and metric in got[n][tau][m]:
                            ours = got[n][tau][m][metric]['mean']
                            rows.append(dict(table='relaxation (Fig. 6/Table 8 L2)', setting=f'N={n} tau={tau}', method=m, stat=metric,
                                             paper=v['mean'], ours=ours, match=abs(ours - v['mean']) <= 1e-3 * max(1, abs(v['mean'])), note='upstream script, unmodified'))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'claims.csv', index=False)
    for t, part in df.groupby('table'):
        print(f'Table {t}: {part.match.sum()}/{len(part)} numbers match the paper')
        miss = part[~part.match]
        if len(miss): print(miss[['setting', 'method', 'stat', 'paper', 'ours']].to_string(index=False))


if __name__ == '__main__':
    main()
