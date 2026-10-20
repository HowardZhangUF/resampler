"""Memory-bounded rerun of DSR's appendix Table 6 (bias/variance of resampled estimates).

Upstream `appendix.dsr_bias_variance` resamples all draws in one batch, which
needs tens of GB at the published 20000 draws (its Slurm script asks for 64 GB).
This driver uses the upstream particle set, resamplers and estimator definitions
unchanged, and only accumulates the statistics over chunks of draws. The result
is statistically equivalent, not bit-exact, because RNG consumption differs.

    .venv-dsr/bin/python scripts/dsr_bias_variance_chunked.py --output results/dsr_upstream/bias_variance.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/differentiable-systematic-resampling'))
from appendix import dsr_bias_variance as upstream  # noqa: E402


def chunked_error(x1, w1, resample, f, draws, chunk):
    """Same bias/var/MSE as upstream.estimator_error, accumulated over chunks."""
    target = (w1.unsqueeze(-1) * f(x1)).sum(1)  # (1, d)
    total, total_sq, total_err_sq, seen = 0., 0., 0., 0
    while seen < draws:
        b = min(chunk, draws - seen)
        x, w = x1.expand(b, *x1.shape[1:]).contiguous(), w1.expand(b, w1.shape[1]).contiguous()
        with torch.no_grad():
            x_new, w_new = resample(x, w)
        w_new = w_new / w_new.sum(1, keepdim=True)
        estimate = (w_new.unsqueeze(-1) * f(x_new)).sum(1)
        total = total + estimate.sum(0)
        total_sq = total_sq + (estimate ** 2).sum()
        total_err_sq = total_err_sq + ((estimate - target) ** 2).sum()
        seen += b
    mean = total / draws
    return {'bias': math.sqrt(float(((mean - target[0]) ** 2).sum())),
            'var': float(total_sq / draws - (mean ** 2).sum()),
            'mse': float(total_err_sq / draws)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=100)
    p.add_argument('--dim', type=int, default=3)
    p.add_argument('--draws', type=int, default=20000)
    p.add_argument('--chunk', type=int, default=1000)
    p.add_argument('--eps', type=float, default=0.5)
    p.add_argument('--output')
    a = p.parse_args()
    torch.set_default_dtype(upstream.DTYPE)
    x1, w1 = upstream.make_particle_set(1, a.N, a.dim, seed=0)
    methods = upstream.build_methods(upstream.TAUS, a.eps)
    results = {}
    for f_name, f in {'f(x)=x': lambda z: z, 'f(x)=x^2': lambda z: z ** 2}.items():
        print(f'\n=== {f_name} ===  N={a.N}, d={a.dim}, draws={a.draws} (chunk {a.chunk})')
        print(f"{'method':<16}{'bias':>12}{'variance':>14}{'MSE':>14}")
        for name, resample in methods.items():
            r = chunked_error(x1, w1, resample, f, a.draws, a.chunk)
            results.setdefault(f_name, {})[name] = r
            print(f"{name:<16}{r['bias']:>12.3e}{r['var']:>14.3e}{r['mse']:>14.3e}", flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
