"""Filtering stage under alternative resampling triggers (resampler_bench/triggers.py).

One process per (trigger, method, model) so that the slow DSR cells spread across workers;
each writes its own folder results/triggers/<trigger>/parts/<provider>__<method>__<model>.
The 'ess' baseline is not rerun: it is the published benchmark (results/standard, results/lorenz, results/dsr).

    .venv/bin/python scripts/run_triggers.py --triggers entropy maxweight --jobs 4
    # Matched resampling rate: per-model thresholds from scripts/calibrate_triggers.py, written to <trigger>_matched.
    .venv/bin/python scripts/run_triggers.py --thresholds results/triggers/matched_thresholds.json --jobs 4
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resampler_bench.run import MODELS  # noqa: E402
from resampler_bench.triggers import TRIGGERS, DESCRIPTIONS  # noqa: E402

# Methods of the DSR report tables. dsr_noclip is omitted: the clamp only acts on gradients,
# so its filter trials are identical to dsr.
METHODS = ['dsr/dsr', 'dsr/dsr_unsorted', 'dsr/dsr_tau0.5', 'dsr/dsr_scaled',
           'particles/systematic', 'particles/multinomial', 'pydpf/systematic_stop', 'pydpf/soft', 'torchfilter/soft',
           'pydpf/ot', 'pfjax/ot', 'filterflow/ot', 'custom/eigen', 'custom/gaussian', 'custom/gumbel', 'variational/kl']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triggers', nargs='+', choices=list(TRIGGERS), default=['entropy', 'maxweight'])
    p.add_argument('--methods', nargs='+', default=METHODS)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--profile', choices=['smoke', 'standard'], default='standard')
    p.add_argument('--output', default='results/triggers')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--thresholds', help='JSON from calibrate_triggers.py; per-model thresholds, folders <trigger>_matched')
    a = p.parse_args()
    thresholds = json.loads((ROOT/a.thresholds).read_text())['thresholds'] if a.thresholds else None
    label = lambda trigger: trigger+'_matched' if thresholds else trigger
    spec = lambda trigger, model: f'{trigger}:{thresholds[trigger][model]}' if thresholds else trigger
    out = ROOT/a.output
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='3', CUDA_VISIBLE_DEVICES='')
    for trigger in a.triggers:
        (out/label(trigger)).mkdir(parents=True, exist_ok=True)
        rule = DESCRIPTIONS[trigger] if not thresholds else DESCRIPTIONS[trigger].replace('0.5 N', 'tau N')+', tau per model: '+json.dumps(thresholds[trigger])
        (out/label(trigger)/'manifest.json').write_text(json.dumps(dict(trigger=trigger, rule=rule, profile=a.profile,
                                                                 start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
    # DSR first: its cells are the longest, so they should not be left for the end.
    jobs = [(t, m, k) for m in a.methods for t in a.triggers for k in a.models]
    jobs.sort(key=lambda j: not j[1].startswith('dsr/'))

    def run(job):
        trigger, method_id, model = job
        provider, method = method_id.split('/')
        folder = out/label(trigger)/'parts'/f'{provider}__{method}__{model}'; folder.mkdir(parents=True, exist_ok=True)
        python = ROOT/('.venv-tf' if provider == 'filterflow' else '.venv')/'bin/python'
        command = [str(python), '-m', 'resampler_bench.run', '--provider', provider, '--methods', method, '--profile', a.profile,
                   '--stages', 'filter', '--models', model, '--trigger', spec(trigger, model), '--output', str(folder), '--resume']
        with (folder/'run.log').open('a') as f:
            r = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime('%H:%M:%S'), spec(trigger, model), method_id, model, 'exit', r.returncode, flush=True)

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run, jobs))


if __name__ == '__main__':
    main()
