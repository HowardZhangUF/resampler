"""Compare resampling triggers on the filtering tasks.

The 'ess' rows are the published benchmark trials (results/standard, results/lorenz, results/dsr);
the other triggers come from scripts/run_triggers.py. A trial under every trigger uses the same
observation sequence, propagation noise and resampler seeds, so differences are paired by trial.
CIs cluster by observation sequence, as in scripts/report.py.

    .venv/bin/python scripts/trigger_compare.py --input results/triggers
"""
import argparse
import glob
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from report import summary  # noqa: E402
from run_triggers import METHODS  # noqa: E402
from resampler_bench.triggers import DESCRIPTIONS  # noqa: E402

KEYS = ['trigger', 'method_id', 'model', 'n']
METRICS = ['rmse', 'coverage95', 'predictive_nll', 'resampling_steps', 'mean_ess_fraction', 'filter_ms',
           'posterior_mean_rmse', 'loglik_error', 'post_occlusion_rmse']
# Metric, direction: 'lower' or 'nominal' (closest to 0.95).
PAIRED = [('rmse', 'lower'), ('predictive_nll', 'lower'), ('coverage95', 'nominal')]
# '_matched': per-model thresholds from scripts/calibrate_triggers.py (same resampling rate as 'ess').
TRIGGER_ORDER = ['entropy', 'maxweight', 'entropy_matched', 'maxweight_matched']
MODELS = ['linear', 'ungm', 'volatility', 'bearing', 'lorenz63', 'lorenz96']


def load(a):
    base = [pd.read_csv(f) for f in glob.glob(str(ROOT/'results/standard/*/filter.csv')) + glob.glob(str(ROOT/'results/lorenz/*/filter.csv'))
            + [str(ROOT/'results/dsr/dsr/filter.csv')]]
    base = pd.concat(base, ignore_index=True).assign(trigger='ess')
    # Folder layout <input>/<trigger>/parts/<cell>/filter.csv.
    new = pd.concat([pd.read_csv(f).assign(trigger=Path(f).parts[-4]) for f in glob.glob(str(ROOT/a.input/'*/parts/*/filter.csv'))], ignore_index=True)
    df = pd.concat([base, new], ignore_index=True)
    df['method_id'] = df.provider+'/'+df.method
    df = df[df.method_id.isin(METHODS)]
    dup = df.duplicated(['trigger', 'method_id', 'model', 'n', 'seed'])
    if dup.any(): raise SystemExit(f'{dup.sum()} duplicate trials')
    return df


def table(df, columns, fmt=None):
    fmt = fmt or {}
    lines = ['| '+' | '.join(columns)+' |', '|'+'---|'*len(columns)]
    for _, r in df.iterrows():
        lines.append('| '+' | '.join(fmt.get(c, lambda v: f'{v:.4g}' if isinstance(v, (float, np.floating)) else str(v))(r[c]) for c in columns)+' |')
    return '\n'.join(lines)


def paired(df, triggers):
    """Per-trial difference trigger - ess, clustered by observation sequence."""
    ess = df[df.trigger == 'ess'].set_index(['method_id', 'model', 'n', 'seed'])
    rows = []
    for trigger in triggers:
        t = df[df.trigger == trigger].set_index(['method_id', 'model', 'n', 'seed'])
        both = t.join(ess, rsuffix='_ess', how='inner')
        for metric, rule in PAIRED:
            d = both[metric]-both[metric+'_ess'] if rule == 'lower' else (both[metric]-.95).abs()-(both[metric+'_ess']-.95).abs()
            part = pd.DataFrame(dict(diff=d, data_seed=both.data_seed, ess=both[metric+'_ess'])).reset_index()
            s = summary(part, ['method_id', 'model', 'n'], ['diff', 'ess'], cluster='data_seed').assign(trigger=trigger, metric=metric)
            rows.append(s)
    p = pd.concat(rows, ignore_index=True)
    lo, hi = p.diff_mean-p.diff_ci95_half, p.diff_mean+p.diff_ci95_half
    p['verdict'] = np.where(hi < 0, 'better', np.where(lo > 0, 'worse', 'tie'))
    return p


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', default='results/triggers'); a = p.parse_args()
    out = ROOT/a.input/'report'; out.mkdir(parents=True, exist_ok=True)
    df = load(a)
    triggers = [t for t in TRIGGER_ORDER if t in set(df.trigger)]
    s = summary(df, KEYS, METRICS, cluster='data_seed')
    s.to_csv(out/'trigger_filter_summary.csv', index=False)
    pr = paired(df, triggers); pr.to_csv(out/'trigger_paired_vs_ess.csv', index=False)

    md = ['# Resampling triggers on the filtering benchmark', '',
          'Same harness, data, seeds and resamplers as the published runs; only the rule that decides *when* to resample changes. '
          '50 steps; 100 filter runs over 10 observation sequences (10 runs for OT and Gumbel, which stop at N=128). CIs cluster by observation sequence.', '',
          '| Trigger | Rule |', '|---|---|'] + [f'| `{k}` | {v} |' for k, v in DESCRIPTIONS.items()] + ['',
          '`cv2` is not run separately: CV^2 = N/E_2 - 1, so CV^2 > 1 is exactly E_2 < N/2 (checked trial by trial against stored results). '
          'Since E_inf <= E_2 <= E_H, at the same 0.5N threshold `maxweight` resamples most often and `entropy` least often. '
          '`dsr_noclip` is omitted: the clamp acts only on gradients, so its filter trials equal `dsr`.', '']
    cal = ROOT/a.input/'matched_thresholds.json'
    if cal.exists():
        c = json.loads(cal.read_text())
        rows = pd.DataFrame([dict(model=m, ess_rate=c['target'][m], **{f'{t} tau': c['thresholds'][t][m] for t in c['thresholds']},
                                  **{f'{t} rate': c['achieved'][t][m] for t in c['achieved']}) for m in c['target']])
        md += ['`*_matched` triggers use per-model thresholds tau (resample when score < tau N), calibrated by bisection so that '
               f"{c['reference']} resamples as often as under `ess` (N={c['ns']}, {c['seeds']} seeds each). Rates are resampling steps out of 50:", '',
               table(rows, list(rows.columns)), '']
    audit = [l for f in glob.glob(str(ROOT/a.input/'*/parts/*/status.jsonl')) for l in Path(f).read_text().splitlines()]
    nf = sum(len(Path(f).read_text().splitlines()) for f in glob.glob(str(ROOT/a.input/'*/parts/*/numerical_failures.jsonl')))
    done = sum(json.loads(l)['status'] == 'complete' for l in audit)
    md += [f'New cells: {done} complete, {len(audit)-done} failed; numerical trial failures: {nf}.', '']

    # How often each trigger fires.
    freq = s.groupby(['trigger', 'model', 'n']).resampling_steps_mean.mean().unstack('trigger').reset_index()
    md += ['## Resampling frequency', '', 'Mean number of resampling steps out of 50, averaged over methods:', '',
           table(freq, ['model', 'n']+[t for t in ['ess']+triggers if t in freq], {'n': str}), '']

    # Verdict counts per method and trigger.
    md += ['## Paired verdicts against `ess`', '',
           'Per (model, N) cell: *better* / *worse* when the 95% CI of the paired difference excludes 0; coverage uses |coverage - 0.95|. '
           'Counts are better / tie / worse over all cells the method ran.', '']
    for metric, _ in PAIRED:
        v = pr[pr.metric == metric].groupby(['method_id', 'trigger']).verdict.value_counts().unstack(fill_value=0)
        v = v.reindex(columns=['better', 'tie', 'worse'], fill_value=0)
        v['bt/w'] = v.better.astype(str)+' / '+v.tie.astype(str)+' / '+v.worse.astype(str)
        w = v['bt/w'].unstack('trigger').reindex([m for m in METHODS if m in v.index.get_level_values(0)]).reset_index()
        md += [f'### {metric}', '', table(w, ['method_id']+triggers), '']

    # Per-model detail: RMSE and coverage under each trigger.
    for model in MODELS:
        md += [f'## {model}', '']
        for n in [128, 2048]:
            sel = s[(s.model == model) & (s.n == n)]
            if sel.empty: continue
            rows = []
            for m in METHODS:
                r = sel[sel.method_id == m].set_index('trigger')
                if r.empty: continue
                row = dict(method_id=m)
                for t in ['ess']+triggers:
                    if t in r.index:
                        row[f'RMSE {t}'] = f"{r.loc[t, 'rmse_mean']:.4g} ± {r.loc[t, 'rmse_ci95_half']:.2g}"
                        row[f'cov {t}'] = f"{r.loc[t, 'coverage95_mean']:.3f}"
                        row[f'#rs {t}'] = f"{r.loc[t, 'resampling_steps_mean']:.1f}"
                rows.append(row)
            rows = pd.DataFrame(rows)
            md += [f'### {model}, N={n}', '', table(rows, list(rows.columns)), '']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n')
    print('Wrote', out/'REPORT.md')


if __name__ == '__main__':
    main()
