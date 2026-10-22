"""Per-model thresholds at which each trigger resamples as often as ESS < 0.5N.

Calibrated on particles/systematic (N=128 and 512, filter seeds 0-29) by bisection on the
threshold; the target is the mean number of resampling steps under 'ess' on the same trials.
The thresholds are then used for every method by scripts/run_triggers.py --thresholds.

    .venv/bin/python scripts/calibrate_triggers.py --output results/triggers/matched_thresholds.json
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resampler_bench.adapters import Adapter  # noqa: E402
from resampler_bench.run import MODELS, filter_once  # noqa: E402

NS, SEEDS, STEPS = [128, 512], range(30), 50


def rate(adapter, model, spec):
    return np.mean([filter_once(adapter, model, n, seed, STEPS, spec)['resampling_steps'] for n in NS for seed in SEEDS])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triggers', nargs='+', default=['entropy', 'maxweight'])
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--iterations', type=int, default=12)
    p.add_argument('--output', default='results/triggers/matched_thresholds.json')
    a = p.parse_args()
    adapter = Adapter('particles', 'systematic'); adapter.record_ancestors = False
    result = dict(reference='particles/systematic', ns=NS, seeds=len(SEEDS), target={}, thresholds={t: {} for t in a.triggers}, achieved={t: {} for t in a.triggers})
    for model in a.models:
        target = rate(adapter, model, 'ess'); result['target'][model] = target
        for trigger in a.triggers:
            # More resampling as the threshold rises, so bisect on the rate.
            lo, hi = 1e-3, 1.
            for _ in range(a.iterations):
                mid = (lo+hi)/2
                if rate(adapter, model, f'{trigger}:{mid}') < target: lo = mid
                else: hi = mid
            tau = round((lo+hi)/2, 4)
            result['thresholds'][trigger][model] = tau
            result['achieved'][trigger][model] = rate(adapter, model, f'{trigger}:{tau}')
            print(model, trigger, 'target', target, 'tau', tau, 'achieved', result['achieved'][trigger][model], flush=True)
    out = ROOT/a.output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
