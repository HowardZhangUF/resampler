# Particle-resampler benchmarks

This folder implements and runs the benchmark problems in [the supplied text](docs/benchmark-request.txt). It contains checked-out source repositories, adapters, an eigenvalue-based resampler, seeded experiments, raw results, and reports.

Open [the completed report](results/standard/report/REPORT.md), [interactive plots](results/standard/report/report.html), or the supplemental [native task and upstream-test results](results/upstream_tasks/README.md). The HTML includes Plotly and works offline.

## Interactive learning guide

Open **[Resampler Lab](https://particle-resampler-lab.navy-prawn-3424.chatgpt.site)**, the private online tutorial. Its 12 lessons move from weighted particles to the eigen construction, differentiability, benchmark metrics, and measured results. Eight live science labs let you change inputs and inspect the outcome. The results explorer filters the actual summary tables and exports CSV selections.

See [the learning structure](docs/WEBSITE-STRUCTURE.md), [website source and local preview instructions](website/README.md), and [website verification](results/website-qa/README.md). The website checks passed: 18 scientific tests and 32 browser check groups, including all lessons on desktop and mobile. Teaching simulations, completed measurements, mathematical guarantees, and open research questions are labeled separately.

## Included methods

| Source | Methods used |
|---|---|
| [nchopin/particles](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/resampling.py) | Multinomial, systematic, stratified, residual, SSP |
| [dstl/Stone-Soup](https://github.com/dstl/Stone-Soup/blob/d6c2d5713b5fdc0bc0c5c0226898bc1c598a882a/stonesoup/resampler/particle.py) | Multinomial, systematic, stratified, residual |
| [state-space-models/cuthbert](https://github.com/state-space-models/cuthbert/blob/c737efe3098b2b5ecaa7d8b976e7cea246998b17/docs/examples/diff_pf_resampling.md) | Multinomial, systematic, systematic with stop-gradient correction; native PF-versus-Kalman gradient tutorial |
| [mlysy/pfjax](https://github.com/mlysy/pfjax/blob/97652aa1bdff73a92c0286549b010e99cc6f7264/src/pfjax/particle_resamplers.py) | Multinomial, multivariate Gaussian moment matching, entropy-regularized OT |
| [John-JoB/pydpf](https://github.com/John-JoB/pydpf/blob/a7487ce1c344c70e1a244e75725b950deb15e146/pydpf/pydpf/resampling.py) | Multinomial, systematic, stop-gradient systematic, soft resampling, OT |
| [stanford-iprl-lab/torchfilter](https://github.com/stanford-iprl-lab/torchfilter/blob/4a8a085f6406ec07ce6a701e7d8cb4c62c97a390/torchfilter/filters/_particle_filter.py) | Multinomial and soft resampling |
| [JTT94/filterflow](https://github.com/JTT94/filterflow/tree/5d8300ba247c4c17e1a301a22560c24fd0670bfe/filterflow/resampling) | Multinomial, systematic, stratified, OT, covariance-corrected OT |
| [skarppinen/Resamplings.jl](https://github.com/skarppinen/Resamplings.jl/tree/5cc4dd050c568abd8aedc45c6dbe318b0bc6ec7f/src) | Multinomial, systematic, stratified, residual, SSP, killing |
| Local PyTorch baselines | Eigen affine moment matching, Gaussian moment matching, Gumbel-softmax relaxation |

The adapters call upstream implementations. The shared models and experiment harness are local so each method receives the same inputs. The eigen method is an explicit baseline designed for this suite; it is not a reproduction of an unspecified research algorithm.

## Benchmarks and metrics

1. **Fixed weighted cloud:** distribution error, expectation MSE, offspring variance and bias.
2. **Gaussian / mixture:** mean, population covariance and eigenvalue errors; sliced W1; approximate Gaussian-kernel MMD²; mode survival and mass error.
3. **Linear Gaussian state-space model:** filtering RMSE, covariance calibration, posterior and likelihood errors against exact Kalman filtering; likelihood-gradient bias and variance against exact Kalman differentiation.
4. **UNGM:** RMSE, predictive negative log likelihood, Gaussian-ellipsoid coverage, conditional likelihood variance, ESS and runtime.
5. **Stochastic volatility:** the same filtering metrics, plus observation-scale parameter fitting and likelihood-score comparison to a numerical grid-filter reference.
6. **Bearing-only tracking:** position RMSE, calibration, and recovery after a ten-step observation gap.

Forward time, backward time or combined forward/backward time, process peak RSS, and scaling are recorded. Exact metric definitions and measurement limits are in the report. Full multivariate Wasserstein distance, exact kernel MMD, and isolated per-call native allocation are not claimed.

## Open-source task sources and what was tested

The six items below are common task families, rather than one official benchmark with a fixed dataset and leaderboard. Each row links to an exact checked-out source revision. All six were tested in the main shared suite. That suite calls upstream resamplers inside the local models in [data.py](resampler_bench/data.py), allowing comparisons on identical inputs. The additional native checks execute upstream models, filters, examples, or tests directly; they are stored separately because their settings differ.

| Task | Open-source reference | Tests completed here |
|---|---|---|
| Fixed weighted cloud / classical resampling | [Stone Soup resampling tutorial](https://github.com/dstl/Stone-Soup/blob/d6c2d5713b5fdc0bc0c5c0226898bc1c598a882a/docs/tutorials/sampling/ResamplingTutorial.py); [particles book comparison](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/book/resampling/compare_tv_distance_resampling.py) | Main cloud tests; native Stone Soup tutorial and 32 upstream assertions; native book TV comparison with a bounded grid. |
| Weighted Gaussian / Gaussian mixture | [particles weighted-Gaussian experiment](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/book/resampling/compare_tv_distance_resampling.py); [PFJAX Gaussian and OT resamplers](https://github.com/mlysy/pfjax/blob/97652aa1bdff73a92c0286549b010e99cc6f7264/src/pfjax/particle_resamplers.py) | Main Gaussian and bimodal-mixture tests; native weighted-Gaussian experiment and five PFJAX resampler tests. The bimodal cloud is a local test design in `data.py`, not a downloaded reference dataset. |
| Linear Gaussian / exact Kalman reference | [particles model and Kalman filter](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/kalman.py); [Cuthbert gradient experiment](https://github.com/state-space-models/cuthbert/blob/c737efe3098b2b5ecaa7d8b976e7cea246998b17/docs/examples/diff_pf_resampling.md); [FilterFlow likelihood comparison](https://github.com/JTT94/filterflow/blob/5d8300ba247c4c17e1a301a22560c24fd0670bfe/scripts/simple_linear_comparison.py) | Main filtering and gradient comparisons; 800 native Cuthbert trials and 84 profile evaluations; native `particles` filtering; native FilterFlow model/filter checks; PyDPF bounded replication and five Torchfilter PF tests. The FilterFlow checks use its model/SMC functions rather than reproducing its full likelihood-profile script. |
| UNGM / nonlinear growth | [particles `Gordon_etal`](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/state_space_models.py#L546) | Main UNGM variant; additional native `Gordon_etal` model with upstream defaults, five classical resamplers, and the eigen resampler. |
| Stochastic volatility / parameter learning | [particles `StochVol`](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/state_space_models.py#L446); [PyDPF replication tasks](https://github.com/John-JoB/pydpf/blob/a7487ce1c344c70e1a244e75725b950deb15e146/replication_materials/README.md); [FilterFlow model](https://github.com/JTT94/filterflow/blob/5d8300ba247c4c17e1a301a22560c24fd0670bfe/filterflow/models/stochastic_volatility.py) | Main volatility filtering and scale fitting; native `particles` model; bounded native PyDPF volatility filtering; native FilterFlow filtering, including one recorded covariance-corrected OT failure. |
| Bearing-only target tracking | [particles `BearingsOnly`](https://github.com/nchopin/particles/blob/3388c4bd30ab9acc5a27c970089230f93610b627/particles/state_space_models.py#L580); [Stone Soup moving-sensor example](https://github.com/dstl/Stone-Soup/blob/d6c2d5713b5fdc0bc0c5c0226898bc1c598a882a/docs/examples/filters/bearing_only_example.py) | Main constant-velocity task with an observation gap; additional native `particles` model; unmodified 100-step Stone Soup example. The Stone Soup example uses an EKF and serves as a task-execution check. |

The native `particles` experiments contain **2,880 filtering trials**: four models × six methods × four particle counts × ten datasets × three PF seeds, with 50 steps. Their [summary](results/upstream_tasks/particles_native_summary.csv) includes native exact-Kalman errors for the linear model. The native book experiment adds **8,400 resampling trials**, using its original TV metric, N=1024, 21 tau values, and 100 repetitions; the original book grid has N=10000 and 500 tau values. These extra measurements supplement the earlier 223,211 main/native-gradient trials.

### Native model differences

The main UNGM variant starts from variance 5, propagates before the first observation, and uses `cos(1.2t)`. Native `Gordon_etal` starts from variance 4, observes the initial state, uses `cos(1.2(t-1))`, and sets process SD to the rounded value 3.162278. Native `StochVol` defaults to mu=-1.02, rho=0.9702, sigma=0.178; the main suite uses mu=0, rho=0.95, sigma=0.2. Native linear defaults also differ. Native `BearingsOnly` uses state order [vx,vy,px,py], a different transition/noise model, and no observation gap. The main tracking task uses [px,vx,py,vy] and the gap described below. Results from these variants are kept in separate CSVs.

### Supplemental test outcomes

Selected upstream pytest checks produced **55 passed and 4 failed**: Stone Soup 32 passed, Torchfilter 5 passed, PFJAX 5 passed, FilterFlow 13 passed / 4 failed. These are pytest-reported counts, including upstream TensorFlow helper cases. Full upstream test suites were not run.

FilterFlow's failures are retained: two use an obsolete `State` constructor; one compares against POT with different scaling/regularization conventions; one checks a gradient affected by the upstream transport-gradient clipping. The independent diagnostic using matching native cost/epsilon agrees with POT to a maximum plan-entry error of 2.63e-5. Scaling the gradient objective to avoid clipping and using a finite-difference step of 1e-3 passes the upstream tolerances, while the original test remains failed. See [the diagnostic measurements](results/upstream_tasks/filterflow_diagnostics.json).

Native FilterFlow filtering attempted 30 trials: **29 completed and one covariance-corrected OT volatility trial failed** because its Riccati solver encountered a singular matrix. The failed row retains the error and missing accuracy metrics. PyDPF's seven requested aggregate result rows passed their finite-value checks. Its upstream smoke handling leaves several full-size dimensions unchanged, so the wrapper explicitly bounds generated data to 50 trajectories / 21 states, batch size to 16, filter time_extent to 10, and the reference PF to 512 particles. These are bounded pipeline checks, not published training or performance reproductions. [Full outcomes, commands, and source revisions](results/upstream_tasks/README.md) are available alongside the raw CSVs and JUnit XML files.

Shared filtering models use independent standard normal process/observation noises:

| Model | State transition | Observation and initial state |
|---|---|---|
| Linear | xₜ = 0.8xₜ₋₁ + 0.5εₜ | yₜ = xₜ + 0.8ηₜ; x₀ ∼ Normal(0,1) |
| UNGM | xₜ = 0.5xₜ₋₁ + 25xₜ₋₁/(1+xₜ₋₁²) + 8cos(1.2t) + √10 εₜ | yₜ = xₜ²/20 + ηₜ; x₀ ∼ Normal(0,5) (variance 5) |
| Volatility | xₜ = 0.95xₜ₋₁ + 0.2εₜ | yₜ = s exp(xₜ/2)ηₜ, s=1; stationary Gaussian x₀ |
| Bearing | Constant velocity in 2D with independent acceleration SD 0.03 per axis; state order [px,vx,py,vy] | atan2(py,px) + 0.05ηₜ radians; initial mean [10,0.35,−6,0.25], covariance diag(4,0.04,4,0.04) |

Time starts at t=1. Bearing observations at zero-based indices 20–29 are hidden. The separate native likelihood-gradient experiment uses the Cuthbert random-walk model, xₜ=xₜ₋₁+0.4εₜ and yₜ=xₜ+0.8ηₜ.

## Eigen method

`resampler_bench/eigen.py` implements:

```text
μw = Σ wᵢ xᵢ
Cw = Σ wᵢ (xᵢ − μw)(xᵢ − μw)ᵀ
Z  = X − mean(X)
C0 = Zᵀ Z / N
Y  = μw + Z C0^(-1/2) Cw^(1/2)
```

Symmetric matrix roots use eigendecomposition. At full rank, away from the eigenvalue floor, the output matches the weighted input's mean and population covariance. It requires O(Nd²+d³) arithmetic and O(Nd+d²) working storage. It moves particles and can change mode shapes. Moment preservation does not establish preservation of a full distribution or accuracy of a PF likelihood gradient.

The backward pass uses the matrix function's Fréchet derivative instead of eigenvector derivatives. First-order gradients remain finite at repeated positive eigenvalues. Small covariance eigenvalues can still cause intrinsic ill-conditioning. The custom backward supports first-order automatic differentiation; second-order derivatives are explicitly unsupported.

For positive eigenvalues λᵢ, λⱼ, the derivative multipliers in the eigenvector basis are `1/(√λᵢ+√λⱼ)` for the square root and `−1/(√λᵢ√λⱼ(√λᵢ+√λⱼ))` for the inverse square root. These expressions stay defined when the two eigenvalues coincide. The square-root derivative solves a Sylvester equation; see [Higham's derivation](https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/). PyTorch documents the limitations of direct eigenvector gradients in [torch.linalg.eigh](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html).

```python
import torch
from resampler_bench.eigen import eigen_resample

x = torch.randn(128, 4, dtype=torch.float64, requires_grad=True)
log_w = torch.randn(128, dtype=torch.float64, requires_grad=True)
y = eigen_resample(x, log_w)
y.sin().mean().backward()
```

## Environments

The existing `.venv` contains Python 3.12, CPU PyTorch, JAX, OTT-JAX, and classical dependencies. `.venv-tf` contains Python 3.11, TensorFlow 2.15.1, and TensorFlow Probability 0.23. Julia 1.10.10 is under `.tools`.

Repository commits are pinned in `sources.lock.json`. Package versions and tool checksums are pinned in `requirements-*.lock` and `tools.lock.json`. Vendor files retain their upstream licenses and are not edited by the adapters.

To recreate on Linux x86_64:

```bash
python -m pip install uv
python scripts/setup.py
```

## Run

Use a new output folder for repeated runs. The orchestrator rejects nonempty provider outputs to prevent duplicate observations. Direct calls to the lower-level `resampler_bench.run` writer append trials. For a complete clean rerun, move the existing `results/standard`, `results/performance`, and `results/smoke` directories aside first.

```bash
# Mathematical checks and adapter smoke tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_all.py --profile smoke --output results/smoke

# Source-defined tasks and selected upstream tests; use a fresh output folder.
# Runs every check even when an earlier check fails. The current pinned FilterFlow
# failures are reported, and the command returns nonzero for those outcomes.
.venv/bin/python scripts/check_upstream_tasks.py --output results/upstream_tasks_new

# Rebuild the supplemental summary from existing measurements, without rerunning.
.venv/bin/python scripts/check_upstream_tasks.py --summarize-only

# Main accuracy experiments (two CPU worker processes)
.venv/bin/python scripts/run_all.py --profile standard --jobs 2 --output results/standard

# Native likelihood gradients, using the same saved dataset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/cuthbert_gradient.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/torch_gradient.py

# Parameter learning; includes an independent numerical reference
.venv/bin/python scripts/run_learning.py

# Serial timing pass, after other workers finish
.venv/bin/python scripts/run_all.py --profile standard --jobs 1 --isolate-methods \
  --stages timing gradient --output results/performance

# Summaries, confidence intervals, PNGs, and interactive HTML
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/report.py
```

Or use `bash scripts/run_experiments.sh` in a clean results tree. It sets thread limits and clears inherited ROS Python paths.

To run one provider or method:

```bash
.venv/bin/python scripts/run_all.py --providers particles custom \
  --output results/my-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m resampler_bench.run --provider custom --methods eigen \
  --stages static filter gradient --output results/eigen-only
```

## Experimental settings

- Fast methods: 100 repeats; N = 32, 128, 512, 2048; static d = 1, 2, 4.
- OT and dense Gumbel methods: 10 repeats at N ≤ 128; three static scaling repeats at larger N. Full-filter and gradient runs stop at N=128.
- Covariance-corrected FilterFlow OT: N ≤ 128 because its dense correction solver has cubic cost.
- Dense PyDPF/FilterFlow OT at N=2048 uses the Gaussian d=2 scaling case; the complete distribution grid stops at N=512. PyDPF takes about 25 seconds per N=2048 call on this CPU. Any additional large-N cells completed before this limit was set are retained as extra measurements.
- Local gradients: N ≤ 512 for fast methods. Two test functions, x² and sin, are compared with exact weighted-distribution derivatives.
- Full filters: 50 steps, ESS-triggered resampling below 0.5N. Cuthbert-style likelihood gradients use 0.7N.
- Fast full filters: ten observation datasets × ten PF seeds. Expensive full filters: ten datasets × one seed. Reported filter confidence intervals cluster by dataset.
- Native Cuthbert gradients: 100 seeds per N, plus 21-point profile slices with N=3000.
- Parameter fitting: 10 observation datasets, N=128, five-point observation-scale grid; true scale 1.0.
- CPU only. The available GTX 960M is a legacy 2 GB Maxwell device; a compatible GPU environment has not been established.

## Limits and compatibility

The completed run passed the result-integrity audit (`results/standard/validation.json`), 11 mathematical tests, and 1,803 selected upstream Julia assertions. All 216 shared input arrays were bitwise identical between the NumPy 2.1.3 and 1.26.4 environments; see `results/tests-shared-inputs.log`. The much larger upstream Julia statistical stress suite was interrupted and is not included in the passing test count.

Soft resamplers keep importance-correction weights. Torchfilter's upstream soft path leaves those weights in input order; its adapter gathers the matching output weights explicitly. The upstream numerical resampling code remains unchanged. `overrides==7.7.0` avoids an old decorator's incompatibility with Python 3.12.

FilterFlow requires shape-specific TensorFlow traces and uses float32 inputs; other providers use float64 inputs. Accuracy workers may run concurrently. The report uses a separate serial pass for forward and local-gradient timing when available. Wrapper timings include conversions and seed handling; Julia additionally includes IPC. RSS is the whole-process high-water mark. The serial Julia pass records worker peak RSS separately and provides a combined upper bound in the summary. These numbers cannot establish isolated kernel allocation or GPU performance.

The serial pass uses a fresh interpreter per method, so RSS from a previously evaluated method cannot inflate the next method's measurement. Each process still includes framework imports and accumulated JIT caches across particle counts.

This suite compares the methods under shared local models. It does not reproduce every paper's architecture, training schedule, experiment, or published figure. Failures and excluded grid cells are explicit in the report audit.

Numerical failure is a benchmark outcome. In particular, PFJAX's native MVN resampler uses a corrected weighted covariance whose denominator can vanish when the weights collapse onto one particle. Each failed trial is recorded in `numerical_failures.jsonl`; its accuracy metrics are missing rather than replaced with a different algorithm. Summaries report the number of successful trials and failure rate. Accuracy averages condition on success and must be read with those counts.

`scripts/retry_failed_cells.py` finishes any cells interrupted by an earlier whole-cell numerical error. It keeps already completed cells and uses the current per-trial failure reporting. It does not stabilize or replace the upstream resampler.

Julia uses binary IPC for weights and ancestor indices. The adapter seeds both the passed `MersenneTwister` and the global Julia RNG: upstream multinomial/residual sampling makes one `randexp(T)` draw from the global RNG. Protocol equivalence and repeated-seed reproducibility are checked by `scripts/check_julia_bridge.py`. Measurements from the earlier, incompletely seeded multinomial run are archived outside the final results under `results/julia-before-global-seed-fix`.
