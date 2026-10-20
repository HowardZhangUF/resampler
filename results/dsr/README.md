# DSR on the particle-resampler benchmark

The `dsr` provider calls the unmodified upstream `resampling.py` from [fcumlin/differentiable-systematic-resampling](https://github.com/fcumlin/differentiable-systematic-resampling) (`vendor/differentiable-systematic-resampling`, commit `39a86e5`). It runs through the same harness, data, seeds and metrics as every other method. Full tables are in [report/REPORT.md](report/REPORT.md) and the summary CSVs are in [report/](report/).

| Variant | Setting |
|---|---|
| `dsr` | Paper default: Morton sort, τ=0.1, transport gradient clamped to [−1, 1] |
| `dsr_noclip` | Same as `dsr` without the gradient clamp |
| `dsr_unsorted` | Same as `dsr` without Morton sorting |
| `dsr_tau0.5` | Same as `dsr` with τ=0.5 |
| `dsr_scaled` | **Local diagnostic, not from the paper:** τ = 2.5/N, which keeps τ·N at the paper's N=25 operating point |

Execution: 380/380 cells complete, 0 failures, 0 numerical trial failures. That is 24 000 static trials, 8 000 filtering trials and 6 000 local-gradient trials. In addition there are 2 000 native PF-gradient trials and 10 SV-learning datasets. The serial timing pass covers `dsr` and `dsr_scaled`.

## Findings

**1. At the paper's fixed τ=0.1, DSR is not consistent in N.** τ is measured in CDF units, while the comb spacing is 1/N. A fixed τ therefore averages over a number of ancestors that grows with N. The paper's own `published_relaxation_metrics.json` shows N_eff = 6.5, 12.3, 23.1 and 48.3 at N = 25, 50, 100 and 200, about 0.24·N. On the benchmark:
- On the Gaussian cloud (D=2), sliced-W1 stays at about 0.35 from N=32 to N=2048. Systematic resampling falls from 0.095 to 0.011.
- Covariance error stays at about 1 for every N, because DSR shrinks the spread of the particles.
- Across all static cells, the median sliced-W1 rank is 37 of 39.

**2. That shrinkage harms filtering where calibration matters.**
- Bearing-only tracking: 95% coverage is 0.06–0.09 at N ≥ 128, compared with 0.64–0.90 for systematic. RMSE ranks last at N ≥ 512.
- Linear and UNGM: RMSE is close to the other methods in absolute terms but ranks near last at N ≥ 512.
- Stochastic volatility: `dsr` ranks first on RMSE at N ≥ 128, with coverage 0.95.

**3. Likelihood gradients are low-variance but biased, and the bias does not shrink with N.** These are PF score errors against exact Kalman gradients:

| | N=128 | N=512 | N=2048 |
|---|---|---|---|
| `dsr` gradient MSE | 88.0 | 52.5 | 51.2 |
| `dsr` gradient bias | 8.8 | 7.1 | 7.1 |
| stop-gradient systematic, MSE | 152 | 27.0 | 8.9 |
| eigen baseline, MSE | 5.4 | 1.06 | 0.27 |

This matches the paper's bias–variance description (its Tables 6 and 15), but at fixed τ the bias is an O(1) floor.

**4. Ablations.**
- Morton sorting matters: `dsr_unsorted` is much worse almost everywhere.
- τ=0.5 is worse than τ=0.1.
- The gradient clamp changes nothing in the forward pass or the local-gradient cells. In the PF-score experiment it raises MSE at small N (433 vs 124 at N=32) and is neutral from N=512 upward.

**5. Rescaling τ with N (`dsr_scaled`) removes most of the problem.** This is not the paper's method; it is a diagnostic.
- Gaussian sliced-W1 falls from 0.33 to 0.031 across N=32–2048.
- On the bimodal cloud at N=2048, sliced-W1 is 0.030, the best of the representative methods.
- PF-gradient MSE at N=2048 drops from 51.2 to 0.31, close to the eigen baseline and below stop-gradient systematic.
- Bearing coverage at N=2048 rises from 0.06 to 0.68, still below systematic's 0.90.
- The median static rank improves from 37 to 28.

**6. Cost.** Serial pass, CPU, 1 thread, median per call:
- At N ≤ 128, DSR takes about 1 ms, comparable to soft resampling.
- At N=2048 it takes 201 ms: about 630× slower than systematic (0.32 ms), but about 11× faster than PFJAX OT and about 117× faster than PyDPF OT.

So the paper's "no iterative solver, much cheaper than EROT" holds. "Close to a discrete systematic step" holds only at small N, because the dense N×N transport dominates at larger N.

**7. Stochastic-volatility scale fitting** (N=128). Parameter error is 0.24 and score RMSE is 6.99, both mid-pack.

## Reproduce

```bash
.venv/bin/python -m resampler_bench.run --provider dsr --stages static filter gradient --output results/dsr     # or via scripts/run_all.py --providers dsr
.venv/bin/python scripts/torch_gradient.py --methods dsr/dsr dsr/dsr_unsorted dsr/dsr_noclip dsr/dsr_tau0.5 dsr/dsr_scaled --output results/dsr/torch_native
.venv/bin/python -m resampler_bench.learning --provider dsr --output results/dsr
.venv/bin/python -m resampler_bench.run --provider dsr --methods dsr --stages timing --output results/dsr/performance/dsr   # one process per method, idle machine
.venv/bin/python scripts/dsr_compare.py --input results/dsr
```

This run was split across worker folders under `results/dsr-parts/`, which were merged into `results/dsr/dsr/`.
