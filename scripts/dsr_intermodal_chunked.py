"""Memory-bounded rerun of DSR's appendix Table 13 (inter-modal mass).

Upstream `appendix.intermodal_mass` resamples 10^4 particle sets in one batch,
which needs far more than the free memory here (its Slurm script asks for 64 GB).
Particle sets are independent, so this driver calls the unchanged upstream
`intermodal_mass` on chunks with distinct seeds and pools mean and standard
deviation exactly. Statistically equivalent, not bit-exact.

    .venv-dsr/bin/python scripts/dsr_intermodal_chunked.py --output results/dsr_upstream/intermodal_mass.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/differentiable-systematic-resampling'))
from appendix import intermodal_mass as upstream  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dims', type=int, nargs='+', default=[1, 2, 3, 5, 10])
    p.add_argument('--reps', type=int, default=10000)
    p.add_argument('--chunk', type=int, default=1000)
    p.add_argument('--output')
    a = p.parse_args()
    results = {}
    for d in a.dims:
        sums = {}
        for k in range(a.reps // a.chunk):
            r = upstream.intermodal_mass(dim=d, n_reps=a.chunk, seed=k)
            for m, v in r.items():
                s = sums.setdefault(m, [0., 0.])
                # Recover per-chunk sum and sum of squares from mean and unbiased std.
                s[0] += v['mass_mean'] * a.chunk
                s[1] += v['mass_std'] ** 2 * (a.chunk - 1) + v['mass_mean'] ** 2 * a.chunk
        n = a.chunk * (a.reps // a.chunk)
        results[d] = {m: {'mass_mean': s[0] / n, 'mass_std': math.sqrt(max(s[1] - s[0] ** 2 / n, 0.) / (n - 1))}
                      for m, s in sums.items()}
        print(f'd={d}: ' + '  '.join(f"{m} {v['mass_mean']:.3f}±{v['mass_std']:.3f}" for m, v in results[d].items()), flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
