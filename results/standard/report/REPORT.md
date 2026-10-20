# Particle-resampler benchmark report

Completed 2591 benchmark cells across 10 upstream repositories plus custom baselines. Raw trials: 159,861 static, 53,200 filtering, 19,600 local gradients, 120 parameter-learning trials.

Execution failures: **0**. Recorded numerical trial failures: **28**. Explicit grid exclusions: **33**. See `audit.json` for the full list. Numerical failures are benchmark outcomes, not silently repaired results. Metric means condition on successful trials; compare them together with the failure rate.

## Scope and method

This is a common benchmark suite that calls the checked-out repositories. It is not a claim to reproduce every experiment or headline number in their papers. The Cuthbert likelihood-gradient experiment additionally runs the native tutorial functions and native Kalman reference.

All six requested families are covered: fixed weighted clouds; Gaussian and Gaussian-mixture clouds; a linear Gaussian state-space model; UNGM; stochastic volatility; and bearing-only target tracking with a ten-step observation gap.

Shared models: linear x_t=0.8x_(t−1)+0.5ε, y_t=x_t+0.8η, initial variance 1; UNGM x_t=0.5x+25x/(1+x²)+8cos(1.2t)+sqrt(10)ε, y_t=x_t²/20+η, initial variance 5; volatility x_t=0.95x+0.2ε, y_t=exp(x_t/2)η, stationary initial state. Tracking has [px,vx,py,vy] constant-velocity dynamics, acceleration SD 0.03, and atan2(py,px) observations with noise SD 0.05 radians. Initial tracking mean is [10,0.35,−6,0.25] with diagonal covariance [4,0.04,4,0.04]. All noises are independent standard normals before scaling. Static input measures, not ideal analytic distributions, are the distribution-metric references.

The standard grid uses N = 32, 128, 512, 2048 and static dimensions 1, 2, 4. Fast methods use 100 resampling/filter repeats. Expensive OT/Gumbel methods use 10 repeats at N ≤ 128 and 3 static scaling repeats above that. Expensive full-filter and gradient runs stop at N=128; covariance-corrected OT also stops at N=128. Other local gradients stop at N=512. Full filters have 50 time steps and resample when ESS < 0.5N. Native likelihood gradients use the tutorial threshold 0.7N.

Dense PyDPF and FilterFlow OT use the Gaussian d=2 case for scaling at N=2048, while retaining the complete static-distribution grid through N=512. PyDPF took about 25 seconds per N=2048 call in initial accuracy measurements. Extra large-N accuracy cells completed before this limit was introduced are retained; serial timing for those extra cells is unavailable. They are not used in the N=2048 common scaling comparison.

Static trials share one fixed weighted cloud per grid cell and vary only the resampling seed. Filtering uses ten observation sequences, with ten particle-filter random seeds per sequence for fast methods. Expensive filters use one seed per sequence. Static confidence intervals condition on the supplied cloud. Filter confidence intervals use successful sequence-level means (up to ten independent clusters), not 100 falsely independent observations. Each metric records its actual interval sample count in the summary CSV. Intervals are Student-t intervals for the mean.

Accuracy workers run concurrently. Separate serial timing pass used for forward/local-gradient timing and process RSS summaries: **True**. Raw accuracy-run times are retained as diagnostics; they should not be used for controlled cross-framework speed claims.

## Eigen baseline

Let μw and Cw be the weighted population mean and covariance. Let Z = X − mean(X), and C0 = ZᵀZ/N. The baseline returns Y = μw + Z C0^(-1/2) Cw^(1/2), with symmetric matrix roots computed by eigendecomposition. When C0 has full rank and no eigenvalue floor is active, mean(Y)=μw and Cov(Y)=Cw. Complexity is O(Nd²+d³), and the algorithm does not construct an N×N transport matrix.

The eigenvalue floor is 1e-10. Rank-deficient inputs remain finite but need not have exact covariance matching. The final implementation uses analytic divided differences for the root functions’ first-order Fréchet derivatives, avoiding unstable eigenvector derivatives at repeated positive eigenvalues. Small covariance eigenvalues can still cause intrinsic ill-conditioning; second-order derivatives are explicitly unsupported. This is an affine moment-matching baseline, not an established distribution-preserving resampler or a new-method claim. It can distort modes and nonlinear expectations. Matching the x² gradient is an algebraic consequence of matching moments; the sin gradient and full PF likelihood gradient are separate tests.

Largest observed eigen-baseline mean error: 4e-15; covariance Frobenius error: 3.21e-14.

The root derivative follows the matrix-function Sylvester-equation relation described in [Higham’s derivation](https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/). The implementation avoids the direct eigenvector-gradient limitation documented by [PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html).

## Variational resampling

The `variational` provider calls `kl` and `tv` from [okviman/Variational-Resampling](https://github.com/okviman/Variational-Resampling) (Kviman et al., *Variational Resampling*, AISTATS 2024). `kl` greedily assigns offspring counts B_s with a max-heap so that Σ B_s(log w_s − log B_s), the resampling ELBO up to constants, is maximal; `tv` rounds N·w_s and gives the remainders to the largest fractional parts. Both are deterministic, so repeated calls on one cloud return identical offspring (zero offspring variance). `tv` keeps |B_s − N·w_s| < 1; `kl` can exceed 1 (largest observed deviation 1.09 in 400 random weight vectors), because it optimizes the ELBO rather than the count error.

The adapter passes normalized log importance weights, which is the upstream `kl-iw` setting. Upstream `main.py` also runs `kl` on the path log joint log p(x_1:t, y_1:t); that input needs model densities and ancestry, which a generic resampler interface does not receive, so that variant is executed only by `scripts/run_variational_upstream.py`. Both methods return indices and have no gradient path, so they have no local-gradient cells. Upstream `kl` recomputes Σ B_s at every heap step, so its cost grows as O(N²).

The tables below compare `kl` and `tv` with three classical resamplers from `particles` on the mixture cloud (N=128, d=2) and in the four filters (N=128). Every provider/method appears in the full tables later in this report.

### Variational vs classical · mixture cloud (N=128, d=2) · sliced W1

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | sliced W1 |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **ssp** | 100/100 | **0.1208 ± 0.01435** |
| 2 | particles | multinomial | 100/100 | 0.1953 ± 0.01885 |
| 3 | variational | kl | 100/100 | 0.652 ± 0 |
| 3 | variational | tv | 100/100 | 0.652 ± 0 |
| 5 | particles | systematic | 100/100 | 0.8868 ± 0.07697 |

### Variational vs classical · mixture cloud (N=128, d=2) · trials keeping both modes

Best = higher is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | trials keeping both modes |
| ---: | --- | --- | ---: | ---: |
| 1 | variational | **kl** | 100/100 | **1 ± 0** |
| 1 | variational | **tv** | 100/100 | **1 ± 0** |
| 1 | particles | **ssp** | 100/100 | **1 ± 0** |
| 1 | particles | **multinomial** | 100/100 | **1 ± 0** |
| 5 | particles | systematic | 100/100 | 0.865 ± 0.04427 |

### Variational vs classical · mixture cloud (N=128, d=2) · forward time (ms)

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | forward time (ms) |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **ssp** | 100/100 | **0.1907 ± 0.005689** |
| 2 | particles | systematic | 100/100 | 0.2093 ± 0.009762 |
| 3 | particles | multinomial | 100/100 | 0.3192 ± 0.02899 |
| 4 | variational | tv | 100/100 | 1.109 ± 0.1258 |
| 5 | variational | kl | 100/100 | 2.227 ± 0.1671 |

On this mixture cloud every N·w_s is 0.7 or 1.3, so `kl` and `tv` both give each particle exactly one offspring: the output is the input positions with uniform weights, and the 35% mode receives 50% of the mass on every call. Random schemes are unbiased in expectation; a deterministic allocation keeps its rounding error on every call, which is why the static errors here are higher than SSP or multinomial. Offspring-count variance is exactly zero for both methods.

### Variational vs classical · Linear Gaussian filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **systematic** | 100/100 | **0.5061 ± 0.04637** |
| 2 | particles | multinomial | 100/100 | 0.5065 ± 0.04512 |
| 3 | particles | ssp | 100/100 | 0.5068 ± 0.04587 |
| 4 | variational | kl | 100/100 | 0.507 ± 0.04685 |
| 5 | variational | tv | 100/100 | 0.5081 ± 0.04693 |

### Variational vs classical · UNGM filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | variational | **kl** | 100/100 | **4.831 ± 0.8575** |
| 2 | particles | ssp | 100/100 | 4.854 ± 0.792 |
| 3 | particles | systematic | 100/100 | 4.906 ± 0.9289 |
| 4 | variational | tv | 100/100 | 4.906 ± 0.794 |
| 5 | particles | multinomial | 100/100 | 4.916 ± 0.8803 |

### Variational vs classical · Stochastic volatility filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **ssp** | 100/100 | **0.4164 ± 0.06348** |
| 2 | particles | systematic | 100/100 | 0.4173 ± 0.0634 |
| 3 | particles | multinomial | 100/100 | 0.4174 ± 0.06486 |
| 4 | variational | kl | 100/100 | 0.4174 ± 0.06379 |
| 5 | variational | tv | 100/100 | 0.4181 ± 0.06438 |

### Variational vs classical · Bearing-only tracking filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **systematic** | 100/100 | **4.548 ± 1.657** |
| 2 | particles | ssp | 100/100 | 4.649 ± 1.688 |
| 3 | variational | kl | 100/100 | 4.662 ± 1.626 |
| 4 | particles | multinomial | 100/100 | 4.787 ± 1.635 |
| 5 | variational | tv | 100/100 | 4.869 ± 1.788 |

### Variational vs classical · Linear Gaussian filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | variational | **tv** | 100/100 | **0.3037** |
| 2 | particles | ssp | 100/100 | 0.3507 |
| 3 | particles | systematic | 100/100 | 0.3526 |
| 4 | variational | kl | 100/100 | 0.358 |
| 5 | particles | multinomial | 100/100 | 0.4062 |

### Variational vs classical · UNGM filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **ssp** | 100/100 | **633.7** |
| 2 | particles | systematic | 100/100 | 653.3 |
| 3 | variational | kl | 100/100 | 756 |
| 4 | variational | tv | 100/100 | 2923 |
| 5 | particles | multinomial | 100/100 | 4174 |

### Variational vs classical · Stochastic volatility filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **systematic** | 100/100 | **0.08285** |
| 2 | particles | ssp | 100/100 | 0.08356 |
| 3 | variational | tv | 100/100 | 0.08647 |
| 4 | variational | kl | 100/100 | 0.09313 |
| 5 | particles | multinomial | 100/100 | 0.107 |

### Variational vs classical · Bearing-only tracking filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **ssp** | 100/100 | **2.019** |
| 2 | particles | systematic | 100/100 | 4.739 |
| 3 | variational | tv | 100/100 | 6.066 |
| 4 | variational | kl | 100/100 | 7.522 |
| 5 | particles | multinomial | 100/100 | 19.1 |

## Mixture distribution results (N=128, d=2)

### Mixture cloud (N=128, d=2) · sliced W1

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | sliced W1 |
| ---: | --- | --- | ---: | ---: |
| 1 | stonesoup | **systematic** | 100/100 | **0.04703 ± 0.001707** |
| 2 | julia | systematic | 100/100 | 0.05308 ± 0.001836 |
| 3 | stonesoup | stratified | 100/100 | 0.05856 ± 0.002222 |
| 4 | julia | stratified | 100/100 | 0.1159 ± 0.01168 |
| 5 | filterflow | stratified | 100/100 | 0.1171 ± 0.01285 |
| 6 | particles | ssp | 100/100 | 0.1208 ± 0.01435 |
| 7 | julia | killing | 100/100 | 0.1209 ± 0.01291 |
| 8 | particles | stratified | 100/100 | 0.1236 ± 0.01296 |
| 9 | particles | residual | 100/100 | 0.1285 ± 0.01141 |
| 10 | julia | ssp | 100/100 | 0.1303 ± 0.01494 |
| 11 | julia | residual | 100/100 | 0.1364 ± 0.0113 |
| 12 | stonesoup | residual | 100/100 | 0.1405 ± 0.01307 |
| 13 | cuthbert | multinomial | 100/100 | 0.1669 ± 0.01554 |
| 14 | julia | multinomial | 100/100 | 0.1841 ± 0.01849 |
| 15 | pydpf | soft | 100/100 | 0.186 ± 0.01745 |
| 15 | torchfilter | soft | 100/100 | 0.186 ± 0.01745 |
| 17 | pfjax | multinomial | 100/100 | 0.1892 ± 0.01716 |
| 18 | pydpf | multinomial | 100/100 | 0.1935 ± 0.01816 |
| 18 | torchfilter | multinomial | 100/100 | 0.1935 ± 0.01816 |
| 20 | particles | multinomial | 100/100 | 0.1953 ± 0.01885 |
| 21 | filterflow | multinomial | 100/100 | 0.2042 ± 0.01813 |
| 22 | stonesoup | multinomial | 100/100 | 0.2066 ± 0.02016 |
| 23 | pfjax | ot | 10/10 | 0.4299 ± 0 |
| 24 | pydpf | ot | 10/10 | 0.4356 ± 0 |
| 25 | filterflow | corrected_ot | 10/10 | 0.6283 ± 0 |
| 25 | filterflow | ot | 10/10 | 0.6283 ± 0 |
| 27 | variational | kl | 100/100 | 0.652 ± 0 |
| 27 | variational | tv | 100/100 | 0.652 ± 0 |
| 29 | custom | eigen | 100/100 | 0.7163 ± 0 |
| 30 | custom | gumbel | 10/10 | 0.7719 ± 0.038 |
| 31 | pfjax | mvn | 100/100 | 0.8582 ± 0.01277 |
| 32 | custom | gaussian | 100/100 | 0.8599 ± 0.0134 |
| 33 | particles | systematic | 100/100 | 0.8868 ± 0.07697 |
| 34 | filterflow | systematic | 100/100 | 0.9303 ± 0.08088 |
| 35 | pydpf | systematic | 100/100 | 0.9303 ± 0.08088 |
| 35 | pydpf | systematic_stop | 100/100 | 0.9303 ± 0.08088 |
| 35 | cuthbert | systematic | 100/100 | 0.9303 ± 0.08088 |
| 35 | cuthbert | systematic_stop | 100/100 | 0.9303 ± 0.08088 |

### Mixture cloud (N=128, d=2) · RFF MMD²

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RFF MMD² |
| ---: | --- | --- | ---: | ---: |
| 1 | stonesoup | **systematic** | 100/100 | **0.0001244 ± 1.445e-05** |
| 2 | julia | systematic | 100/100 | 0.000196 ± 2.053e-05 |
| 3 | stonesoup | stratified | 100/100 | 0.000223 ± 2.897e-05 |
| 4 | julia | stratified | 100/100 | 0.001388 ± 0.0003045 |
| 5 | filterflow | stratified | 100/100 | 0.001477 ± 0.0003767 |
| 6 | julia | killing | 100/100 | 0.00148 ± 0.0003833 |
| 7 | particles | residual | 100/100 | 0.001523 ± 0.0003433 |
| 8 | particles | ssp | 100/100 | 0.001649 ± 0.0005011 |
| 9 | particles | stratified | 100/100 | 0.001665 ± 0.0003661 |
| 10 | julia | residual | 100/100 | 0.001711 ± 0.0003182 |
| 11 | stonesoup | residual | 100/100 | 0.001896 ± 0.0003947 |
| 12 | julia | ssp | 100/100 | 0.001898 ± 0.0004742 |
| 13 | cuthbert | multinomial | 100/100 | 0.00263 ± 0.000644 |
| 14 | pydpf | soft | 100/100 | 0.003264 ± 0.0007189 |
| 14 | torchfilter | soft | 100/100 | 0.003264 ± 0.0007189 |
| 16 | julia | multinomial | 100/100 | 0.003346 ± 0.0007996 |
| 17 | pfjax | multinomial | 100/100 | 0.003463 ± 0.0007332 |
| 18 | pydpf | multinomial | 100/100 | 0.003563 ± 0.0007708 |
| 18 | torchfilter | multinomial | 100/100 | 0.003563 ± 0.0007708 |
| 20 | particles | multinomial | 100/100 | 0.003629 ± 0.0007993 |
| 21 | filterflow | multinomial | 100/100 | 0.003941 ± 0.0007707 |
| 22 | stonesoup | multinomial | 100/100 | 0.004211 ± 0.0008681 |
| 23 | pydpf | ot | 10/10 | 0.01226 ± 0 |
| 24 | pfjax | ot | 10/10 | 0.0126 ± 0 |
| 25 | custom | eigen | 100/100 | 0.02422 ± 0 |
| 26 | filterflow | corrected_ot | 10/10 | 0.02492 ± 0 |
| 26 | filterflow | ot | 10/10 | 0.02492 ± 0 |
| 28 | pfjax | mvn | 100/100 | 0.0315 ± 0.002173 |
| 29 | custom | gaussian | 100/100 | 0.0325 ± 0.001953 |
| 30 | variational | kl | 100/100 | 0.03978 ± 0 |
| 30 | variational | tv | 100/100 | 0.03978 ± 0 |
| 32 | custom | gumbel | 10/10 | 0.05546 ± 0.00518 |
| 33 | particles | systematic | 100/100 | 0.08753 ± 0.01565 |
| 34 | filterflow | systematic | 100/100 | 0.09637 ± 0.01645 |
| 35 | pydpf | systematic | 100/100 | 0.09637 ± 0.01645 |
| 35 | pydpf | systematic_stop | 100/100 | 0.09637 ± 0.01645 |
| 35 | cuthbert | systematic | 100/100 | 0.09637 ± 0.01645 |
| 35 | cuthbert | systematic_stop | 100/100 | 0.09637 ± 0.01645 |

### Mixture cloud (N=128, d=2) · covariance error (Frobenius)

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | covariance error (Frobenius) |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100/100 | **1.311e-15 ± 0** |
| 2 | stonesoup | systematic | 100/100 | 0.1629 ± 0.01208 |
| 3 | julia | systematic | 100/100 | 0.2301 ± 0.01716 |
| 4 | stonesoup | stratified | 100/100 | 0.2409 ± 0.02174 |
| 5 | filterflow | stratified | 100/100 | 0.3263 ± 0.03138 |
| 6 | julia | stratified | 100/100 | 0.3264 ± 0.03411 |
| 7 | particles | stratified | 100/100 | 0.3337 ± 0.04024 |
| 8 | particles | ssp | 100/100 | 0.3442 ± 0.04689 |
| 9 | julia | killing | 100/100 | 0.3478 ± 0.04049 |
| 10 | julia | ssp | 100/100 | 0.3662 ± 0.04016 |
| 11 | particles | residual | 100/100 | 0.405 ± 0.03817 |
| 12 | julia | residual | 100/100 | 0.4303 ± 0.04155 |
| 13 | cuthbert | multinomial | 100/100 | 0.4797 ± 0.0496 |
| 14 | stonesoup | residual | 100/100 | 0.4878 ± 0.04743 |
| 15 | pfjax | multinomial | 100/100 | 0.5114 ± 0.05508 |
| 16 | pydpf | soft | 100/100 | 0.5216 ± 0.05562 |
| 16 | torchfilter | soft | 100/100 | 0.5216 ± 0.05562 |
| 18 | julia | multinomial | 100/100 | 0.5409 ± 0.05384 |
| 19 | particles | multinomial | 100/100 | 0.5588 ± 0.058 |
| 20 | pydpf | multinomial | 100/100 | 0.5698 ± 0.06163 |
| 20 | torchfilter | multinomial | 100/100 | 0.5698 ± 0.06163 |
| 22 | stonesoup | multinomial | 100/100 | 0.5729 ± 0.06508 |
| 23 | filterflow | multinomial | 100/100 | 0.5978 ± 0.0549 |
| 24 | variational | kl | 100/100 | 0.833 ± 0 |
| 24 | variational | tv | 100/100 | 0.833 ± 0 |
| 26 | pfjax | mvn | 100/100 | 0.9077 ± 0.1145 |
| 27 | custom | gaussian | 100/100 | 0.9295 ± 0.1319 |
| 28 | pydpf | ot | 10/10 | 1.925 ± 0 |
| 29 | pfjax | ot | 10/10 | 2.134 ± 0 |
| 30 | filterflow | corrected_ot | 10/10 | 2.857 ± 0 |
| 30 | filterflow | ot | 10/10 | 2.857 ± 0 |
| 32 | particles | systematic | 100/100 | 2.906 ± 0.6798 |
| 33 | filterflow | systematic | 100/100 | 3.29 ± 0.7143 |
| 34 | pydpf | systematic | 100/100 | 3.29 ± 0.7143 |
| 34 | pydpf | systematic_stop | 100/100 | 3.29 ± 0.7143 |
| 34 | cuthbert | systematic | 100/100 | 3.29 ± 0.7143 |
| 34 | cuthbert | systematic_stop | 100/100 | 3.29 ± 0.7143 |
| 38 | custom | gumbel | 10/10 | 4.487 ± 0.3302 |

### Mixture cloud (N=128, d=2) · trials keeping both modes

Best = higher is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | trials keeping both modes |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100/100 | **1 ± 0** |
| 1 | custom | **gaussian** | 100/100 | **1 ± 0** |
| 1 | custom | **gumbel** | 10/10 | **1 ± 0** |
| 1 | cuthbert | **multinomial** | 100/100 | **1 ± 0** |
| 1 | filterflow | **corrected_ot** | 10/10 | **1 ± 0** |
| 1 | filterflow | **multinomial** | 100/100 | **1 ± 0** |
| 1 | filterflow | **ot** | 10/10 | **1 ± 0** |
| 1 | filterflow | **stratified** | 100/100 | **1 ± 0** |
| 1 | julia | **killing** | 100/100 | **1 ± 0** |
| 1 | julia | **multinomial** | 100/100 | **1 ± 0** |
| 1 | julia | **residual** | 100/100 | **1 ± 0** |
| 1 | julia | **ssp** | 100/100 | **1 ± 0** |
| 1 | julia | **stratified** | 100/100 | **1 ± 0** |
| 1 | julia | **systematic** | 100/100 | **1 ± 0** |
| 1 | particles | **multinomial** | 100/100 | **1 ± 0** |
| 1 | particles | **residual** | 100/100 | **1 ± 0** |
| 1 | particles | **ssp** | 100/100 | **1 ± 0** |
| 1 | particles | **stratified** | 100/100 | **1 ± 0** |
| 1 | pfjax | **multinomial** | 100/100 | **1 ± 0** |
| 1 | pfjax | **mvn** | 100/100 | **1 ± 0** |
| 1 | pfjax | **ot** | 10/10 | **1 ± 0** |
| 1 | pydpf | **multinomial** | 100/100 | **1 ± 0** |
| 1 | pydpf | **ot** | 10/10 | **1 ± 0** |
| 1 | pydpf | **soft** | 100/100 | **1 ± 0** |
| 1 | stonesoup | **multinomial** | 100/100 | **1 ± 0** |
| 1 | stonesoup | **residual** | 100/100 | **1 ± 0** |
| 1 | stonesoup | **stratified** | 100/100 | **1 ± 0** |
| 1 | stonesoup | **systematic** | 100/100 | **1 ± 0** |
| 1 | torchfilter | **multinomial** | 100/100 | **1 ± 0** |
| 1 | torchfilter | **soft** | 100/100 | **1 ± 0** |
| 1 | variational | **kl** | 100/100 | **1 ± 0** |
| 1 | variational | **tv** | 100/100 | **1 ± 0** |
| 33 | particles | systematic | 100/100 | 0.865 ± 0.04427 |
| 34 | cuthbert | systematic | 100/100 | 0.84 ± 0.04651 |
| 34 | cuthbert | systematic_stop | 100/100 | 0.84 ± 0.04651 |
| 34 | filterflow | systematic | 100/100 | 0.84 ± 0.04651 |
| 34 | pydpf | systematic | 100/100 | 0.84 ± 0.04651 |
| 34 | pydpf | systematic_stop | 100/100 | 0.84 ± 0.04651 |

### Mixture cloud (N=128, d=2) · forward time (ms)

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | forward time (ms) |
| ---: | --- | --- | ---: | ---: |
| 1 | particles | **ssp** | 100/100 | **0.1907 ± 0.005689** |
| 2 | particles | systematic | 100/100 | 0.2093 ± 0.009762 |
| 3 | particles | stratified | 100/100 | 0.2263 ± 0.01447 |
| 4 | particles | residual | 100/100 | 0.2528 ± 0.01066 |
| 5 | particles | multinomial | 100/100 | 0.3192 ± 0.02899 |
| 6 | julia | stratified | 100/100 | 0.343 ± 0.01239 |
| 7 | julia | ssp | 100/100 | 0.3633 ± 0.01138 |
| 8 | julia | multinomial | 100/100 | 0.3702 ± 0.01667 |
| 9 | julia | residual | 100/100 | 0.3709 ± 0.01241 |
| 10 | julia | systematic | 100/100 | 0.3779 ± 0.01573 |
| 11 | stonesoup | systematic | 100/100 | 0.3949 ± 0.01314 |
| 12 | stonesoup | stratified | 100/100 | 0.4062 ± 0.01413 |
| 13 | julia | killing | 100/100 | 0.4145 ± 0.01357 |
| 14 | stonesoup | multinomial | 100/100 | 0.415 ± 0.02028 |
| 15 | stonesoup | residual | 100/100 | 0.5988 ± 0.0498 |
| 16 | pydpf | multinomial | 100/100 | 0.6702 ± 0.02421 |
| 17 | custom | gaussian | 100/100 | 0.7702 ± 0.02496 |
| 18 | pydpf | soft | 100/100 | 0.8031 ± 0.03183 |
| 19 | torchfilter | multinomial | 100/100 | 0.8181 ± 0.02647 |
| 20 | pydpf | systematic | 100/100 | 0.8183 ± 0.03524 |
| 21 | pydpf | systematic_stop | 100/100 | 0.827 ± 0.02605 |
| 22 | pfjax | multinomial | 100/100 | 0.9381 ± 0.0229 |
| 23 | filterflow | multinomial | 100/100 | 0.947 ± 0.02772 |
| 24 | filterflow | stratified | 100/100 | 0.9598 ± 0.03241 |
| 25 | filterflow | systematic | 100/100 | 0.9602 ± 0.03042 |
| 26 | custom | gumbel | 10/10 | 0.9708 ± 0.08369 |
| 27 | custom | eigen | 100/100 | 1.013 ± 0.06631 |
| 28 | pfjax | mvn | 100/100 | 1.027 ± 0.02351 |
| 29 | variational | tv | 100/100 | 1.109 ± 0.1258 |
| 30 | torchfilter | soft | 100/100 | 1.152 ± 0.03697 |
| 31 | cuthbert | systematic_stop | 100/100 | 1.362 ± 0.03292 |
| 32 | cuthbert | systematic | 100/100 | 1.391 ± 0.04303 |
| 33 | cuthbert | multinomial | 100/100 | 1.926 ± 0.05821 |
| 34 | variational | kl | 100/100 | 2.227 ± 0.1671 |
| 35 | filterflow | ot | 10/10 | 11.67 ± 0.5836 |
| 36 | filterflow | corrected_ot | 10/10 | 26.17 ± 1.347 |
| 37 | pfjax | ot | 10/10 | 27.64 ± 2.206 |
| 38 | pydpf | ot | 10/10 | 33.95 ± 2.185 |

## Filtering results (N=128)

Trials are successful/total; means condition on successful trials. PFJAX MVN has numerical failures on UNGM.

### Linear Gaussian filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | julia | **multinomial** | 100/100 | **0.505 ± 0.04559** |
| 2 | particles | residual | 100/100 | 0.5054 ± 0.04552 |
| 3 | stonesoup | residual | 100/100 | 0.5056 ± 0.0462 |
| 4 | pfjax | multinomial | 100/100 | 0.5056 ± 0.04498 |
| 5 | pfjax | mvn | 100/100 | 0.5058 ± 0.04612 |
| 6 | pydpf | multinomial | 100/100 | 0.5058 ± 0.04658 |
| 6 | torchfilter | multinomial | 100/100 | 0.5058 ± 0.04658 |
| 8 | stonesoup | multinomial | 100/100 | 0.5058 ± 0.04639 |
| 9 | cuthbert | multinomial | 100/100 | 0.5058 ± 0.04598 |
| 10 | filterflow | multinomial | 100/100 | 0.506 ± 0.0471 |
| 11 | particles | systematic | 100/100 | 0.5061 ± 0.04637 |
| 12 | julia | residual | 100/100 | 0.5061 ± 0.0458 |
| 13 | pydpf | systematic | 100/100 | 0.5064 ± 0.0469 |
| 13 | pydpf | systematic_stop | 100/100 | 0.5064 ± 0.0469 |
| 15 | stonesoup | systematic | 100/100 | 0.5064 ± 0.04698 |
| 16 | cuthbert | systematic | 100/100 | 0.5065 ± 0.04593 |
| 16 | cuthbert | systematic_stop | 100/100 | 0.5065 ± 0.04593 |
| 18 | particles | multinomial | 100/100 | 0.5065 ± 0.04512 |
| 19 | julia | stratified | 100/100 | 0.5066 ± 0.04604 |
| 20 | julia | killing | 100/100 | 0.5066 ± 0.04645 |
| 21 | particles | stratified | 100/100 | 0.5066 ± 0.04543 |
| 22 | filterflow | stratified | 100/100 | 0.5066 ± 0.04594 |
| 23 | filterflow | systematic | 100/100 | 0.5067 ± 0.04571 |
| 24 | julia | systematic | 100/100 | 0.5067 ± 0.04616 |
| 25 | particles | ssp | 100/100 | 0.5068 ± 0.04587 |
| 26 | custom | eigen | 100/100 | 0.5069 ± 0.0455 |
| 27 | julia | ssp | 100/100 | 0.507 ± 0.04575 |
| 28 | variational | kl | 100/100 | 0.507 ± 0.04685 |
| 29 | stonesoup | stratified | 100/100 | 0.5072 ± 0.04628 |
| 30 | custom | gaussian | 100/100 | 0.5072 ± 0.04556 |
| 31 | variational | tv | 100/100 | 0.5081 ± 0.04693 |
| 32 | pydpf | soft | 100/100 | 0.5082 ± 0.04603 |
| 32 | torchfilter | soft | 100/100 | 0.5082 ± 0.04603 |
| 34 | filterflow | corrected_ot | 10/10 | 0.5144 ± 0.04365 |
| 35 | custom | gumbel | 10/10 | 0.5144 ± 0.04802 |
| 36 | filterflow | ot | 10/10 | 0.5145 ± 0.04393 |
| 37 | pydpf | ot | 10/10 | 0.5145 ± 0.04344 |
| 38 | pfjax | ot | 10/10 | 0.5146 ± 0.04344 |

### Linear Gaussian filter (N=128) · 95% coverage

Best = closest to 95% is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | 95% coverage |
| ---: | --- | --- | ---: | ---: |
| 1 | filterflow | **corrected_ot** | 10/10 | **0.932 ± 0.02154** |
| 2 | stonesoup | residual | 100/100 | 0.9316 ± 0.02163 |
| 3 | julia | residual | 100/100 | 0.9308 ± 0.02094 |
| 3 | pydpf | systematic | 100/100 | 0.9308 ± 0.02122 |
| 3 | pydpf | systematic_stop | 100/100 | 0.9308 ± 0.02122 |
| 6 | particles | stratified | 100/100 | 0.9306 ± 0.02209 |
| 6 | pydpf | multinomial | 100/100 | 0.9306 ± 0.02266 |
| 6 | torchfilter | multinomial | 100/100 | 0.9306 ± 0.02266 |
| 9 | cuthbert | multinomial | 100/100 | 0.9304 ± 0.01987 |
| 9 | particles | residual | 100/100 | 0.9304 ± 0.01918 |
| 11 | pfjax | ot | 10/10 | 0.93 ± 0.02051 |
| 12 | julia | killing | 100/100 | 0.9296 ± 0.02241 |
| 12 | julia | stratified | 100/100 | 0.9296 ± 0.0213 |
| 14 | julia | ssp | 100/100 | 0.9294 ± 0.02137 |
| 14 | variational | kl | 100/100 | 0.9294 ± 0.01994 |
| 16 | custom | gaussian | 100/100 | 0.9292 ± 0.02216 |
| 16 | particles | systematic | 100/100 | 0.9292 ± 0.02184 |
| 18 | particles | ssp | 100/100 | 0.929 ± 0.02118 |
| 19 | stonesoup | multinomial | 100/100 | 0.9288 ± 0.0217 |
| 19 | custom | eigen | 100/100 | 0.9288 ± 0.02219 |
| 21 | filterflow | multinomial | 100/100 | 0.9286 ± 0.02167 |
| 21 | pfjax | multinomial | 100/100 | 0.9286 ± 0.02243 |
| 21 | pfjax | mvn | 100/100 | 0.9286 ± 0.02183 |
| 24 | julia | multinomial | 100/100 | 0.9284 ± 0.0214 |
| 24 | variational | tv | 100/100 | 0.9284 ± 0.02214 |
| 26 | custom | gumbel | 10/10 | 0.928 ± 0.02154 |
| 26 | pydpf | ot | 10/10 | 0.928 ± 0.02154 |
| 28 | filterflow | stratified | 100/100 | 0.9278 ± 0.02192 |
| 28 | stonesoup | stratified | 100/100 | 0.9278 ± 0.02262 |
| 30 | cuthbert | systematic | 100/100 | 0.9276 ± 0.02216 |
| 30 | cuthbert | systematic_stop | 100/100 | 0.9276 ± 0.02216 |
| 30 | particles | multinomial | 100/100 | 0.9276 ± 0.02232 |
| 33 | filterflow | systematic | 100/100 | 0.9274 ± 0.0219 |
| 33 | pydpf | soft | 100/100 | 0.9274 ± 0.0228 |
| 33 | torchfilter | soft | 100/100 | 0.9274 ± 0.0228 |
| 36 | julia | systematic | 100/100 | 0.9264 ± 0.02278 |
| 37 | stonesoup | systematic | 100/100 | 0.9262 ± 0.02292 |
| 38 | filterflow | ot | 10/10 | 0.924 ± 0.02001 |

### Linear Gaussian filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | julia | **killing** | 100/100 | **0.2993** |
| 2 | variational | tv | 100/100 | 0.3037 |
| 3 | stonesoup | systematic | 100/100 | 0.3047 |
| 4 | custom | eigen | 100/100 | 0.3199 |
| 5 | cuthbert | systematic | 100/100 | 0.321 |
| 5 | cuthbert | systematic_stop | 100/100 | 0.321 |
| 7 | stonesoup | stratified | 100/100 | 0.3259 |
| 8 | pydpf | systematic | 100/100 | 0.327 |
| 8 | pydpf | systematic_stop | 100/100 | 0.327 |
| 10 | particles | stratified | 100/100 | 0.348 |
| 11 | julia | ssp | 100/100 | 0.3498 |
| 12 | particles | ssp | 100/100 | 0.3507 |
| 13 | filterflow | stratified | 100/100 | 0.3512 |
| 14 | particles | systematic | 100/100 | 0.3526 |
| 15 | variational | kl | 100/100 | 0.358 |
| 16 | filterflow | multinomial | 100/100 | 0.3587 |
| 17 | stonesoup | multinomial | 100/100 | 0.3589 |
| 18 | pfjax | multinomial | 100/100 | 0.3605 |
| 19 | stonesoup | residual | 100/100 | 0.3605 |
| 20 | julia | systematic | 100/100 | 0.3609 |
| 21 | filterflow | systematic | 100/100 | 0.3659 |
| 22 | particles | residual | 100/100 | 0.3669 |
| 23 | pydpf | multinomial | 100/100 | 0.374 |
| 23 | torchfilter | multinomial | 100/100 | 0.374 |
| 25 | cuthbert | multinomial | 100/100 | 0.3784 |
| 26 | julia | multinomial | 100/100 | 0.3786 |
| 27 | particles | multinomial | 100/100 | 0.4062 |
| 28 | pfjax | mvn | 100/100 | 0.4128 |
| 29 | julia | stratified | 100/100 | 0.4132 |
| 30 | custom | gaussian | 100/100 | 0.4173 |
| 31 | pydpf | soft | 100/100 | 0.4185 |
| 31 | torchfilter | soft | 100/100 | 0.4185 |
| 33 | julia | residual | 100/100 | 0.435 |
| – | custom | gumbel | 10/10 | n/a |
| – | filterflow | corrected_ot | 10/10 | n/a |
| – | filterflow | ot | 10/10 | n/a |
| – | pfjax | ot | 10/10 | n/a |
| – | pydpf | ot | 10/10 | n/a |

### UNGM filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | julia | **killing** | 100/100 | **4.762 ± 0.8073** |
| 2 | filterflow | multinomial | 100/100 | 4.79 ± 0.8229 |
| 3 | filterflow | systematic | 100/100 | 4.792 ± 0.8378 |
| 4 | particles | stratified | 100/100 | 4.824 ± 0.8784 |
| 5 | pydpf | soft | 100/100 | 4.825 ± 0.7488 |
| 5 | torchfilter | soft | 100/100 | 4.825 ± 0.7488 |
| 7 | pydpf | multinomial | 100/100 | 4.827 ± 0.8482 |
| 7 | torchfilter | multinomial | 100/100 | 4.827 ± 0.8482 |
| 9 | variational | kl | 100/100 | 4.831 ± 0.8575 |
| 10 | stonesoup | multinomial | 100/100 | 4.843 ± 0.8865 |
| 11 | filterflow | stratified | 100/100 | 4.853 ± 0.8544 |
| 12 | particles | ssp | 100/100 | 4.854 ± 0.792 |
| 13 | julia | ssp | 100/100 | 4.86 ± 0.9073 |
| 14 | cuthbert | systematic | 100/100 | 4.87 ± 0.9592 |
| 14 | cuthbert | systematic_stop | 100/100 | 4.87 ± 0.9592 |
| 16 | julia | multinomial | 100/100 | 4.871 ± 0.7691 |
| 17 | pydpf | systematic | 100/100 | 4.881 ± 0.8893 |
| 17 | pydpf | systematic_stop | 100/100 | 4.881 ± 0.8893 |
| 19 | julia | residual | 100/100 | 4.888 ± 0.9887 |
| 20 | particles | systematic | 100/100 | 4.906 ± 0.9289 |
| 21 | variational | tv | 100/100 | 4.906 ± 0.794 |
| 22 | julia | stratified | 100/100 | 4.906 ± 0.9287 |
| 23 | particles | multinomial | 100/100 | 4.916 ± 0.8803 |
| 24 | julia | systematic | 100/100 | 4.921 ± 0.9203 |
| 25 | stonesoup | residual | 100/100 | 4.935 ± 0.9515 |
| 26 | stonesoup | systematic | 100/100 | 4.935 ± 0.8928 |
| 27 | custom | gaussian | 100/100 | 4.936 ± 0.8551 |
| 28 | particles | residual | 100/100 | 4.942 ± 1.015 |
| 29 | cuthbert | multinomial | 100/100 | 4.972 ± 0.9957 |
| 30 | pfjax | mvn | 93/100 | 5.027 ± 0.9052 |
| 31 | pfjax | multinomial | 100/100 | 5.038 ± 0.8834 |
| 32 | stonesoup | stratified | 100/100 | 5.039 ± 0.9313 |
| 33 | custom | eigen | 100/100 | 5.17 ± 1.039 |
| 34 | custom | gumbel | 10/10 | 5.271 ± 1.41 |
| 35 | pfjax | ot | 10/10 | 5.328 ± 1.421 |
| 36 | filterflow | corrected_ot | 10/10 | 5.335 ± 1.44 |
| 37 | pydpf | ot | 10/10 | 5.416 ± 1.463 |
| 38 | filterflow | ot | 10/10 | 5.48 ± 1.493 |

### UNGM filter (N=128) · 95% coverage

Best = closest to 95% is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | 95% coverage |
| ---: | --- | --- | ---: | ---: |
| 1 | pfjax | **mvn** | 93/100 | **0.9372 ± 0.02179** |
| 2 | cuthbert | systematic | 100/100 | 0.9348 ± 0.02619 |
| 2 | cuthbert | systematic_stop | 100/100 | 0.9348 ± 0.02619 |
| 2 | julia | multinomial | 100/100 | 0.9348 ± 0.02262 |
| 5 | julia | ssp | 100/100 | 0.9334 ± 0.02652 |
| 5 | particles | ssp | 100/100 | 0.9334 ± 0.02391 |
| 7 | custom | gaussian | 100/100 | 0.933 ± 0.02723 |
| 8 | filterflow | multinomial | 100/100 | 0.9326 ± 0.02342 |
| 9 | particles | stratified | 100/100 | 0.9322 ± 0.026 |
| 9 | stonesoup | multinomial | 100/100 | 0.9322 ± 0.02599 |
| 11 | pydpf | systematic | 100/100 | 0.932 ± 0.02609 |
| 11 | pydpf | systematic_stop | 100/100 | 0.932 ± 0.02609 |
| 13 | julia | stratified | 100/100 | 0.9318 ± 0.02484 |
| 14 | pydpf | multinomial | 100/100 | 0.931 ± 0.02108 |
| 14 | torchfilter | multinomial | 100/100 | 0.931 ± 0.02108 |
| 16 | stonesoup | systematic | 100/100 | 0.9308 ± 0.02419 |
| 17 | filterflow | systematic | 100/100 | 0.9304 ± 0.02363 |
| 17 | particles | systematic | 100/100 | 0.9304 ± 0.02754 |
| 19 | filterflow | stratified | 100/100 | 0.9302 ± 0.02318 |
| 19 | stonesoup | residual | 100/100 | 0.9302 ± 0.02652 |
| 21 | julia | killing | 100/100 | 0.93 ± 0.02177 |
| 21 | custom | eigen | 100/100 | 0.93 ± 0.02605 |
| 21 | variational | kl | 100/100 | 0.93 ± 0.02457 |
| 24 | pfjax | multinomial | 100/100 | 0.9292 ± 0.02579 |
| 25 | julia | systematic | 100/100 | 0.9288 ± 0.025 |
| 26 | particles | multinomial | 100/100 | 0.9284 ± 0.02483 |
| 26 | variational | tv | 100/100 | 0.9284 ± 0.02409 |
| 26 | particles | residual | 100/100 | 0.9284 ± 0.02786 |
| 29 | pydpf | soft | 100/100 | 0.928 ± 0.02138 |
| 29 | torchfilter | soft | 100/100 | 0.928 ± 0.02138 |
| 31 | julia | residual | 100/100 | 0.9274 ± 0.0278 |
| 32 | cuthbert | multinomial | 100/100 | 0.9272 ± 0.02584 |
| 33 | stonesoup | stratified | 100/100 | 0.925 ± 0.02727 |
| 34 | custom | gumbel | 10/10 | 0.918 ± 0.04446 |
| 34 | pfjax | ot | 10/10 | 0.918 ± 0.04182 |
| 36 | pydpf | ot | 10/10 | 0.914 ± 0.04577 |
| 36 | filterflow | corrected_ot | 10/10 | 0.914 ± 0.04268 |
| 38 | filterflow | ot | 10/10 | 0.908 ± 0.04484 |

### UNGM filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | pydpf | **soft** | 100/100 | **438.4** |
| 1 | torchfilter | **soft** | 100/100 | **438.4** |
| 3 | pfjax | mvn | 93/100 | 529.8 |
| 4 | particles | stratified | 100/100 | 617.1 |
| 5 | particles | ssp | 100/100 | 633.7 |
| 6 | filterflow | systematic | 100/100 | 635.8 |
| 7 | particles | systematic | 100/100 | 653.3 |
| 8 | particles | residual | 100/100 | 662.5 |
| 9 | pydpf | systematic | 100/100 | 669.5 |
| 9 | pydpf | systematic_stop | 100/100 | 669.5 |
| 11 | filterflow | stratified | 100/100 | 672.7 |
| 12 | julia | ssp | 100/100 | 685.2 |
| 13 | julia | stratified | 100/100 | 719.2 |
| 14 | filterflow | multinomial | 100/100 | 729.7 |
| 15 | variational | kl | 100/100 | 756 |
| 16 | julia | killing | 100/100 | 760.2 |
| 17 | julia | multinomial | 100/100 | 761.1 |
| 18 | cuthbert | systematic | 100/100 | 769.7 |
| 18 | cuthbert | systematic_stop | 100/100 | 769.7 |
| 20 | custom | eigen | 100/100 | 809.7 |
| 21 | julia | residual | 100/100 | 836.5 |
| 22 | stonesoup | residual | 100/100 | 843.9 |
| 23 | custom | gaussian | 100/100 | 845.1 |
| 24 | stonesoup | systematic | 100/100 | 953.5 |
| 25 | stonesoup | multinomial | 100/100 | 969.2 |
| 26 | pydpf | multinomial | 100/100 | 974.5 |
| 26 | torchfilter | multinomial | 100/100 | 974.5 |
| 28 | julia | systematic | 100/100 | 1081 |
| 29 | cuthbert | multinomial | 100/100 | 1181 |
| 30 | variational | tv | 100/100 | 2923 |
| 31 | stonesoup | stratified | 100/100 | 3469 |
| 32 | particles | multinomial | 100/100 | 4174 |
| 33 | pfjax | multinomial | 100/100 | 6550 |
| – | custom | gumbel | 10/10 | n/a |
| – | filterflow | corrected_ot | 10/10 | n/a |
| – | filterflow | ot | 10/10 | n/a |
| – | pfjax | ot | 10/10 | n/a |
| – | pydpf | ot | 10/10 | n/a |

### Stochastic volatility filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **gumbel** | 10/10 | **0.4148 ± 0.06775** |
| 2 | stonesoup | stratified | 100/100 | 0.4151 ± 0.06275 |
| 3 | stonesoup | systematic | 100/100 | 0.4153 ± 0.06295 |
| 4 | filterflow | multinomial | 100/100 | 0.4158 ± 0.06228 |
| 5 | custom | gaussian | 100/100 | 0.4158 ± 0.06298 |
| 6 | filterflow | stratified | 100/100 | 0.4163 ± 0.06335 |
| 7 | pfjax | mvn | 100/100 | 0.4163 ± 0.06235 |
| 8 | julia | multinomial | 100/100 | 0.4163 ± 0.06232 |
| 9 | custom | eigen | 100/100 | 0.4164 ± 0.06297 |
| 10 | particles | ssp | 100/100 | 0.4164 ± 0.06348 |
| 11 | pydpf | systematic | 100/100 | 0.4165 ± 0.06394 |
| 11 | pydpf | systematic_stop | 100/100 | 0.4165 ± 0.06394 |
| 13 | julia | ssp | 100/100 | 0.4165 ± 0.06392 |
| 14 | pydpf | multinomial | 100/100 | 0.4167 ± 0.06288 |
| 14 | torchfilter | multinomial | 100/100 | 0.4167 ± 0.06288 |
| 16 | filterflow | systematic | 100/100 | 0.4168 ± 0.06332 |
| 17 | cuthbert | systematic | 100/100 | 0.4171 ± 0.06367 |
| 17 | cuthbert | systematic_stop | 100/100 | 0.4171 ± 0.06367 |
| 19 | stonesoup | residual | 100/100 | 0.4171 ± 0.06352 |
| 20 | julia | stratified | 100/100 | 0.4173 ± 0.06348 |
| 21 | particles | stratified | 100/100 | 0.4173 ± 0.06379 |
| 22 | particles | systematic | 100/100 | 0.4173 ± 0.0634 |
| 23 | filterflow | ot | 10/10 | 0.4173 ± 0.065 |
| 24 | particles | multinomial | 100/100 | 0.4174 ± 0.06486 |
| 25 | variational | kl | 100/100 | 0.4174 ± 0.06379 |
| 26 | julia | systematic | 100/100 | 0.4174 ± 0.0627 |
| 27 | julia | residual | 100/100 | 0.4174 ± 0.06273 |
| 28 | filterflow | corrected_ot | 10/10 | 0.4176 ± 0.06451 |
| 29 | cuthbert | multinomial | 100/100 | 0.4176 ± 0.06238 |
| 30 | pydpf | ot | 10/10 | 0.4181 ± 0.06481 |
| 31 | variational | tv | 100/100 | 0.4181 ± 0.06438 |
| 32 | pfjax | ot | 10/10 | 0.4183 ± 0.06485 |
| 33 | particles | residual | 100/100 | 0.4183 ± 0.06283 |
| 34 | pfjax | multinomial | 100/100 | 0.4186 ± 0.06349 |
| 35 | stonesoup | multinomial | 100/100 | 0.4189 ± 0.06357 |
| 36 | pydpf | soft | 100/100 | 0.419 ± 0.06346 |
| 36 | torchfilter | soft | 100/100 | 0.419 ± 0.06346 |
| 38 | julia | killing | 100/100 | 0.4191 ± 0.06314 |

### Stochastic volatility filter (N=128) · 95% coverage

Best = closest to 95% is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | 95% coverage |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **gumbel** | 10/10 | **0.946 ± 0.03697** |
| 2 | variational | tv | 100/100 | 0.9562 ± 0.02419 |
| 3 | particles | residual | 100/100 | 0.9592 ± 0.02316 |
| 4 | pydpf | soft | 100/100 | 0.9594 ± 0.02177 |
| 4 | torchfilter | soft | 100/100 | 0.9594 ± 0.02177 |
| 6 | stonesoup | multinomial | 100/100 | 0.9598 ± 0.02265 |
| 7 | variational | kl | 100/100 | 0.9604 ± 0.02065 |
| 8 | pfjax | multinomial | 100/100 | 0.961 ± 0.02148 |
| 9 | stonesoup | residual | 100/100 | 0.9612 ± 0.02046 |
| 10 | particles | multinomial | 100/100 | 0.9614 ± 0.0219 |
| 11 | cuthbert | systematic | 100/100 | 0.9616 ± 0.02213 |
| 11 | cuthbert | systematic_stop | 100/100 | 0.9616 ± 0.02213 |
| 13 | cuthbert | multinomial | 100/100 | 0.9622 ± 0.02241 |
| 14 | julia | killing | 100/100 | 0.9624 ± 0.02087 |
| 15 | julia | stratified | 100/100 | 0.9626 ± 0.02047 |
| 16 | filterflow | stratified | 100/100 | 0.9628 ± 0.01923 |
| 17 | pydpf | multinomial | 100/100 | 0.9632 ± 0.01949 |
| 17 | torchfilter | multinomial | 100/100 | 0.9632 ± 0.01949 |
| 19 | julia | multinomial | 100/100 | 0.9634 ± 0.02093 |
| 20 | filterflow | systematic | 100/100 | 0.9636 ± 0.0214 |
| 21 | julia | ssp | 100/100 | 0.9644 ± 0.01968 |
| 21 | stonesoup | stratified | 100/100 | 0.9644 ± 0.0207 |
| 23 | custom | gaussian | 100/100 | 0.9646 ± 0.0197 |
| 24 | filterflow | multinomial | 100/100 | 0.9648 ± 0.01861 |
| 24 | particles | stratified | 100/100 | 0.9648 ± 0.02066 |
| 26 | julia | residual | 100/100 | 0.965 ± 0.01826 |
| 26 | stonesoup | systematic | 100/100 | 0.965 ± 0.02031 |
| 28 | particles | ssp | 100/100 | 0.9656 ± 0.02095 |
| 29 | particles | systematic | 100/100 | 0.9658 ± 0.01838 |
| 30 | filterflow | corrected_ot | 10/10 | 0.966 ± 0.02138 |
| 31 | custom | eigen | 100/100 | 0.9662 ± 0.02032 |
| 32 | pfjax | mvn | 100/100 | 0.9664 ± 0.01985 |
| 33 | julia | systematic | 100/100 | 0.9668 ± 0.01836 |
| 34 | pydpf | systematic | 100/100 | 0.967 ± 0.01967 |
| 34 | pydpf | systematic_stop | 100/100 | 0.967 ± 0.01967 |
| 36 | filterflow | ot | 10/10 | 0.968 ± 0.02046 |
| 37 | pfjax | ot | 10/10 | 0.97 ± 0.01937 |
| 37 | pydpf | ot | 10/10 | 0.97 ± 0.01937 |

### Stochastic volatility filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | stonesoup | **stratified** | 100/100 | **0.07439** |
| 2 | custom | eigen | 100/100 | 0.07457 |
| 3 | julia | multinomial | 100/100 | 0.07481 |
| 4 | cuthbert | systematic | 100/100 | 0.07696 |
| 4 | cuthbert | systematic_stop | 100/100 | 0.07696 |
| 6 | custom | gaussian | 100/100 | 0.07859 |
| 7 | stonesoup | systematic | 100/100 | 0.07898 |
| 8 | julia | ssp | 100/100 | 0.07956 |
| 9 | pydpf | systematic | 100/100 | 0.08059 |
| 9 | pydpf | systematic_stop | 100/100 | 0.08059 |
| 11 | stonesoup | residual | 100/100 | 0.08088 |
| 12 | particles | systematic | 100/100 | 0.08285 |
| 13 | julia | systematic | 100/100 | 0.08305 |
| 14 | particles | ssp | 100/100 | 0.08356 |
| 15 | pydpf | multinomial | 100/100 | 0.08358 |
| 15 | torchfilter | multinomial | 100/100 | 0.08358 |
| 17 | variational | tv | 100/100 | 0.08647 |
| 18 | julia | stratified | 100/100 | 0.08726 |
| 19 | pfjax | mvn | 100/100 | 0.08799 |
| 20 | julia | residual | 100/100 | 0.08983 |
| 21 | particles | residual | 100/100 | 0.09107 |
| 22 | filterflow | stratified | 100/100 | 0.09186 |
| 23 | pfjax | multinomial | 100/100 | 0.09211 |
| 24 | variational | kl | 100/100 | 0.09313 |
| 25 | particles | stratified | 100/100 | 0.0933 |
| 26 | julia | killing | 100/100 | 0.09429 |
| 27 | filterflow | multinomial | 100/100 | 0.09982 |
| 28 | filterflow | systematic | 100/100 | 0.1003 |
| 29 | stonesoup | multinomial | 100/100 | 0.1031 |
| 30 | cuthbert | multinomial | 100/100 | 0.1066 |
| 31 | particles | multinomial | 100/100 | 0.107 |
| 32 | pydpf | soft | 100/100 | 0.1155 |
| 32 | torchfilter | soft | 100/100 | 0.1155 |
| – | custom | gumbel | 10/10 | n/a |
| – | filterflow | corrected_ot | 10/10 | n/a |
| – | filterflow | ot | 10/10 | n/a |
| – | pfjax | ot | 10/10 | n/a |
| – | pydpf | ot | 10/10 | n/a |

### Bearing-only tracking filter (N=128) · RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | filterflow | **corrected_ot** | 10/10 | **3.25 ± 1.011** |
| 2 | julia | killing | 100/100 | 4.442 ± 1.488 |
| 3 | julia | ssp | 100/100 | 4.478 ± 1.728 |
| 4 | filterflow | systematic | 100/100 | 4.487 ± 1.609 |
| 5 | custom | gaussian | 100/100 | 4.508 ± 1.737 |
| 6 | custom | eigen | 100/100 | 4.51 ± 1.71 |
| 7 | julia | systematic | 100/100 | 4.529 ± 1.732 |
| 8 | particles | systematic | 100/100 | 4.548 ± 1.657 |
| 9 | cuthbert | systematic | 100/100 | 4.553 ± 1.764 |
| 9 | cuthbert | systematic_stop | 100/100 | 4.553 ± 1.764 |
| 11 | julia | stratified | 100/100 | 4.561 ± 1.735 |
| 12 | stonesoup | systematic | 100/100 | 4.6 ± 1.616 |
| 13 | particles | ssp | 100/100 | 4.649 ± 1.688 |
| 14 | stonesoup | multinomial | 100/100 | 4.649 ± 1.548 |
| 15 | variational | kl | 100/100 | 4.662 ± 1.626 |
| 16 | stonesoup | stratified | 100/100 | 4.698 ± 1.65 |
| 17 | filterflow | stratified | 100/100 | 4.705 ± 1.677 |
| 18 | pfjax | mvn | 100/100 | 4.721 ± 1.71 |
| 19 | stonesoup | residual | 100/100 | 4.733 ± 1.612 |
| 20 | particles | residual | 100/100 | 4.746 ± 1.683 |
| 21 | pydpf | systematic | 100/100 | 4.763 ± 1.746 |
| 21 | pydpf | systematic_stop | 100/100 | 4.763 ± 1.746 |
| 23 | julia | residual | 100/100 | 4.763 ± 1.596 |
| 24 | cuthbert | multinomial | 100/100 | 4.773 ± 1.757 |
| 25 | particles | stratified | 100/100 | 4.775 ± 1.675 |
| 26 | pydpf | multinomial | 100/100 | 4.776 ± 1.538 |
| 26 | torchfilter | multinomial | 100/100 | 4.776 ± 1.538 |
| 28 | pfjax | multinomial | 100/100 | 4.777 ± 1.758 |
| 29 | particles | multinomial | 100/100 | 4.787 ± 1.635 |
| 30 | filterflow | multinomial | 100/100 | 4.791 ± 1.673 |
| 31 | torchfilter | soft | 100/100 | 4.799 ± 1.711 |
| 31 | pydpf | soft | 100/100 | 4.799 ± 1.711 |
| 33 | julia | multinomial | 100/100 | 4.807 ± 1.856 |
| 34 | variational | tv | 100/100 | 4.869 ± 1.788 |
| 35 | custom | gumbel | 10/10 | 5.188 ± 2.52 |
| 36 | filterflow | ot | 10/10 | 5.296 ± 1.728 |
| 37 | pfjax | ot | 10/10 | 5.559 ± 2.289 |
| 38 | pydpf | ot | 10/10 | 5.648 ± 2.284 |

### Bearing-only tracking filter (N=128) · 95% coverage

Best = closest to 95% is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | 95% coverage |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100/100 | **0.7776 ± 0.1978** |
| 2 | pfjax | mvn | 100/100 | 0.7414 ± 0.1762 |
| 3 | custom | gaussian | 100/100 | 0.724 ± 0.1915 |
| 4 | filterflow | corrected_ot | 10/10 | 0.72 ± 0.1564 |
| 5 | julia | ssp | 100/100 | 0.6614 ± 0.2063 |
| 6 | julia | stratified | 100/100 | 0.6586 ± 0.1988 |
| 7 | julia | systematic | 100/100 | 0.6522 ± 0.2128 |
| 8 | filterflow | systematic | 100/100 | 0.6508 ± 0.2106 |
| 9 | pydpf | systematic | 100/100 | 0.6492 ± 0.1938 |
| 9 | pydpf | systematic_stop | 100/100 | 0.6492 ± 0.1938 |
| 11 | cuthbert | systematic | 100/100 | 0.6478 ± 0.2098 |
| 11 | cuthbert | systematic_stop | 100/100 | 0.6478 ± 0.2098 |
| 13 | particles | systematic | 100/100 | 0.6364 ± 0.2056 |
| 14 | particles | ssp | 100/100 | 0.633 ± 0.2233 |
| 15 | variational | kl | 100/100 | 0.6326 ± 0.1913 |
| 16 | stonesoup | systematic | 100/100 | 0.6286 ± 0.2024 |
| 17 | julia | killing | 100/100 | 0.6212 ± 0.1858 |
| 18 | particles | residual | 100/100 | 0.62 ± 0.2207 |
| 19 | filterflow | stratified | 100/100 | 0.6176 ± 0.2069 |
| 20 | particles | stratified | 100/100 | 0.6112 ± 0.1851 |
| 21 | stonesoup | residual | 100/100 | 0.6078 ± 0.2107 |
| 22 | stonesoup | stratified | 100/100 | 0.6044 ± 0.1925 |
| 23 | julia | residual | 100/100 | 0.6006 ± 0.2023 |
| 24 | variational | tv | 100/100 | 0.5896 ± 0.1898 |
| 25 | cuthbert | multinomial | 100/100 | 0.5798 ± 0.2044 |
| 26 | pfjax | multinomial | 100/100 | 0.5742 ± 0.2113 |
| 27 | pydpf | multinomial | 100/100 | 0.5696 ± 0.2071 |
| 27 | torchfilter | multinomial | 100/100 | 0.5696 ± 0.2071 |
| 29 | stonesoup | multinomial | 100/100 | 0.5648 ± 0.1863 |
| 30 | filterflow | multinomial | 100/100 | 0.56 ± 0.1744 |
| 31 | julia | multinomial | 100/100 | 0.5588 ± 0.2031 |
| 32 | particles | multinomial | 100/100 | 0.5396 ± 0.2098 |
| 33 | pydpf | soft | 100/100 | 0.5192 ± 0.1963 |
| 33 | torchfilter | soft | 100/100 | 0.5192 ± 0.1963 |
| 35 | custom | gumbel | 10/10 | 0.216 ± 0.1761 |
| 36 | filterflow | ot | 10/10 | 0.052 ± 0.06546 |
| 37 | pfjax | ot | 10/10 | 0.042 ± 0.02564 |
| 38 | pydpf | ot | 10/10 | 0.04 ± 0.02612 |

### Bearing-only tracking filter (N=128) · conditional log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | conditional log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100/100 | **0.586** |
| 2 | pfjax | mvn | 100/100 | 1.197 |
| 3 | custom | gaussian | 100/100 | 1.992 |
| 4 | particles | ssp | 100/100 | 2.019 |
| 5 | julia | multinomial | 100/100 | 2.921 |
| 6 | stonesoup | stratified | 100/100 | 2.93 |
| 7 | particles | stratified | 100/100 | 3.091 |
| 8 | cuthbert | systematic | 100/100 | 3.326 |
| 8 | cuthbert | systematic_stop | 100/100 | 3.326 |
| 10 | stonesoup | systematic | 100/100 | 3.771 |
| 11 | julia | stratified | 100/100 | 3.849 |
| 12 | filterflow | multinomial | 100/100 | 3.972 |
| 13 | pfjax | multinomial | 100/100 | 4.521 |
| 14 | pydpf | multinomial | 100/100 | 4.556 |
| 14 | torchfilter | multinomial | 100/100 | 4.556 |
| 16 | julia | ssp | 100/100 | 4.573 |
| 17 | pydpf | systematic | 100/100 | 4.651 |
| 17 | pydpf | systematic_stop | 100/100 | 4.651 |
| 19 | julia | killing | 100/100 | 4.723 |
| 20 | particles | systematic | 100/100 | 4.739 |
| 21 | stonesoup | multinomial | 100/100 | 4.879 |
| 22 | julia | systematic | 100/100 | 5.275 |
| 23 | variational | tv | 100/100 | 6.066 |
| 24 | cuthbert | multinomial | 100/100 | 6.607 |
| 25 | filterflow | stratified | 100/100 | 7.085 |
| 26 | filterflow | systematic | 100/100 | 7.311 |
| 27 | variational | kl | 100/100 | 7.522 |
| 28 | particles | residual | 100/100 | 7.577 |
| 29 | julia | residual | 100/100 | 7.695 |
| 30 | stonesoup | residual | 100/100 | 7.747 |
| 31 | pydpf | soft | 100/100 | 9.433 |
| 31 | torchfilter | soft | 100/100 | 9.433 |
| 33 | particles | multinomial | 100/100 | 19.1 |
| – | custom | gumbel | 10/10 | n/a |
| – | filterflow | corrected_ot | 10/10 | n/a |
| – | filterflow | ot | 10/10 | n/a |
| – | pfjax | ot | 10/10 | n/a |
| – | pydpf | ot | 10/10 | n/a |

RMSE measures error against the simulated latent truth. Linear-model posterior error and likelihood error use an exact Kalman reference. Coverage is joint 95% Gaussian-ellipsoid coverage from the particle mean/covariance; it is a diagnostic and is not exact posterior credible-set coverage for multimodal targets. Bearing RMSE uses position only; the coverage ellipsoid uses all four state coordinates. Recovery requires position error below 2 units for three consecutive observed steps; unrecovered runs are right-censored at the end of the sequence.

Bearing RMSE averages squared error across both position coordinates before taking the square root. Post-occlusion RMSE instead uses Euclidean position error, so its squared error is the sum of both coordinate errors. Recovery-step means cap unrecovered runs at 20 steps and are not estimates of unrestricted recovery time.

Likelihood variance is calculated across filter seeds conditional on each observation sequence and then averaged. It is unavailable for expensive filters with only one seed per sequence. Log likelihood bias is meaningful against the exact reference only in the linear Gaussian case. Do not treat transport/Gaussian/eigen likelihood estimates as unbiased likelihood estimates.

## Likelihood-gradient results

Gradient bias is ||mean(g) − g_exact||₂; variance is the trace of the sample gradient covariance; MSE is mean(||g−g_exact||₂²). The Cuthbert experiment uses its native tutorial data (T=50, q=0.4, r=0.8), 100 seeds at each N, and 21-point likelihood/gradient profile slices at N=3000. The PyTorch filter uses that same saved dataset and exact Kalman gradients. Stop-gradient estimators are assessed empirically; finite-particle unbiasedness is not assumed.

Both experiments use the Cuthbert tutorial dataset at N=128, so their rows share each table.

### PF likelihood gradient vs exact Kalman (N=128) · gradient bias

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | gradient bias |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100 | **0.5272** |
| 2 | custom | gaussian | 100 | 0.8231 |
| 3 | pydpf | systematic_stop | 100 | 1.748 |
| 4 | cuthbert | systematic_stop | 100 | 2.285 |
| 5 | pydpf | soft | 100 | 6.675 |
| 5 | torchfilter | soft | 100 | 6.675 |
| 7 | custom | gumbel | 10 | 8.148 |
| 8 | pydpf | systematic | 100 | 11.27 |
| 9 | cuthbert | systematic | 100 | 12.98 |
| 10 | pydpf | ot | 10 | 14.21 |

### PF likelihood gradient vs exact Kalman (N=128) · gradient variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | gradient variance |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100 | **5.207** |
| 2 | custom | gaussian | 100 | 5.602 |
| 3 | custom | gumbel | 10 | 20.28 |
| 4 | pydpf | systematic | 100 | 41.82 |
| 5 | cuthbert | systematic | 100 | 44.41 |
| 6 | pydpf | soft | 100 | 51.45 |
| 6 | torchfilter | soft | 100 | 51.45 |
| 8 | pydpf | systematic_stop | 100 | 150.6 |
| 9 | cuthbert | systematic_stop | 100 | 175.9 |
| 10 | pydpf | ot | 10 | 714.6 |

### PF likelihood gradient vs exact Kalman (N=128) · gradient MSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | gradient MSE |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100 | **5.433** |
| 2 | custom | gaussian | 100 | 6.223 |
| 3 | custom | gumbel | 10 | 84.64 |
| 4 | pydpf | soft | 100 | 95.5 |
| 4 | torchfilter | soft | 100 | 95.5 |
| 6 | pydpf | systematic_stop | 100 | 152.2 |
| 7 | pydpf | systematic | 100 | 168.4 |
| 8 | cuthbert | systematic_stop | 100 | 179.4 |
| 9 | cuthbert | systematic | 100 | 212.5 |
| 10 | pydpf | ot | 10 | 844.9 |

### PF likelihood gradient vs exact Kalman (N=128) · log-likelihood variance

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | log-likelihood variance |
| ---: | --- | --- | ---: | ---: |
| 1 | pydpf | **ot** | 10 | **0.3098** |
| 2 | custom | eigen | 100 | 0.5643 |
| 3 | cuthbert | systematic | 100 | 0.621 |
| 3 | cuthbert | systematic_stop | 100 | 0.621 |
| 5 | pydpf | systematic_stop | 100 | 0.6556 |
| 5 | pydpf | systematic | 100 | 0.6556 |
| 7 | custom | gumbel | 10 | 0.7297 |
| 8 | custom | gaussian | 100 | 0.86 |
| 9 | pydpf | soft | 100 | 1.081 |
| 9 | torchfilter | soft | 100 | 1.081 |

### PF likelihood gradient vs exact Kalman (N=128) · log-likelihood bias

Best = closest to 0 is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | log-likelihood bias |
| ---: | --- | --- | ---: | ---: |
| 1 | custom | **eigen** | 100 | **-0.1536** |
| 2 | cuthbert | systematic | 100 | -0.2671 |
| 2 | cuthbert | systematic_stop | 100 | -0.2671 |
| 4 | custom | gaussian | 100 | -0.2781 |
| 5 | pydpf | systematic | 100 | -0.2953 |
| 5 | pydpf | systematic_stop | 100 | -0.2953 |
| 7 | pydpf | soft | 100 | -0.3859 |
| 7 | torchfilter | soft | 100 | -0.3859 |
| 9 | pydpf | ot | 10 | -0.4472 |
| 10 | custom | gumbel | 10 | -2.662 |

## Stochastic-volatility parameter learning

The known state model is x_t=0.95 x_(t−1)+0.2 ε_t; y_t=s exp(x_t/2) η_t, with true s=1. Fit s by maximizing the particle likelihood on {0.6,0.8,1.0,1.2,1.4}. Ten independent datasets and common random numbers are used. These are coarse-grid estimates, not continuous optimizer convergence claims. The likelihood-score metric uses central differences in log(s), step 0.01. The independent reference is a numerical grid filter on [−4,4] with 1001 cells; a 2001-cell likelihood cross-check is stored per dataset. This reference is approximate, unlike the exact Kalman reference.

### SV observation-scale learning (N=128) · |ŝ − 1|

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | \|ŝ − 1\| |
| ---: | --- | --- | ---: | ---: |
| 1 | torchfilter | **soft** | 10/10 | **0.2 ± 0.09538** |
| 2 | custom | gaussian | 10/10 | 0.22 ± 0.1056 |
| 2 | cuthbert | systematic_stop | 10/10 | 0.22 ± 0.1253 |
| 2 | particles | systematic | 10/10 | 0.22 ± 0.1253 |
| 5 | stonesoup | systematic | 10/10 | 0.24 ± 0.1129 |
| 5 | variational | kl | 10/10 | 0.24 ± 0.1129 |
| 7 | custom | eigen | 10/10 | 0.26 ± 0.09657 |
| 7 | custom | gumbel | 10/10 | 0.26 ± 0.09657 |
| 7 | pfjax | ot | 10/10 | 0.26 ± 0.09657 |
| 7 | pydpf | ot | 10/10 | 0.26 ± 0.09657 |
| 11 | filterflow | ot | 10/10 | 0.28 ± 0.1 |
| 11 | julia | systematic | 10/10 | 0.28 ± 0.07388 |

### SV observation-scale learning (N=128) · |ŝ − ŝ_ref|

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | \|ŝ − ŝ_ref\| |
| ---: | --- | --- | ---: | ---: |
| 1 | cuthbert | **systematic_stop** | 10/10 | **0.02 ± 0.04524** |
| 1 | particles | **systematic** | 10/10 | **0.02 ± 0.04524** |
| 3 | variational | kl | 10/10 | 0.04 ± 0.06032 |
| 4 | custom | eigen | 10/10 | 0.06 ± 0.06911 |
| 4 | custom | gaussian | 10/10 | 0.06 ± 0.06911 |
| 4 | pfjax | ot | 10/10 | 0.06 ± 0.06911 |
| 4 | pydpf | ot | 10/10 | 0.06 ± 0.06911 |
| 8 | filterflow | ot | 10/10 | 0.08 ± 0.07388 |
| 8 | stonesoup | systematic | 10/10 | 0.08 ± 0.07388 |
| 8 | torchfilter | soft | 10/10 | 0.08 ± 0.07388 |
| 11 | custom | gumbel | 10/10 | 0.1 ± 0.07541 |
| 12 | julia | systematic | 10/10 | 0.12 ± 0.07388 |

### SV observation-scale learning (N=128) · likelihood-score RMSE

Best = lower is better. Best method in bold; ± is the 95% interval half-width where available.

| rank | provider | method | trials | likelihood-score RMSE |
| ---: | --- | --- | ---: | ---: |
| 1 | pydpf | **ot** | 10/10 | **1.146** |
| 2 | pfjax | ot | 10/10 | 1.179 |
| 3 | filterflow | ot | 10/10 | 1.307 |
| 4 | custom | eigen | 10/10 | 2.112 |
| 5 | custom | gumbel | 10/10 | 4.606 |
| 6 | custom | gaussian | 10/10 | 5.252 |
| 7 | variational | kl | 10/10 | 6.566 |
| 8 | julia | systematic | 10/10 | 7.394 |
| 9 | cuthbert | systematic_stop | 10/10 | 7.468 |
| 10 | particles | systematic | 10/10 | 8.793 |
| 11 | torchfilter | soft | 10/10 | 10.14 |
| 12 | stonesoup | systematic | 10/10 | 10.19 |

## Metric definitions and measurement limits

- Sliced W1 averages exact one-dimensional Wasserstein-1 distances along 16 fixed random unit directions. In 1D it equals W1. It is not full multivariate W1 or W2.
- RFF MMD² approximates Gaussian-kernel squared maximum mean discrepancy using 128 fixed random Fourier features; bandwidth is sqrt(trace(Cw)). It is not an exact kernel MMD calculation.
- Covariances use population moments (no Bessel correction). PFJAX MVN uses its upstream corrected weighted sample covariance, so a small mismatch is expected.
- Mode coverage counts the two mixture neighborhoods centered at ±3 with radius 1.35 if each has output mass greater than 1%. Mode-mass error additionally compares the negative-halfspace mass. Coverage alone cannot show that mode shapes are preserved.
- Offspring-count variance and empirical count bias use repeated draws from an identical cloud. Soft-resampling offspring are compared with the actual mixture proposal 0.5w+0.5/N. Their output measures retain importance-correction weights; they are not falsely treated as equally weighted.
- Forward timing excludes initial compilation/warmup and includes adapter conversion and seed handling; the serial timing pass disables optional ancestor recovery used solely for offspring metrics. Torchfilter soft-weight gathering remains necessary. Raw static accuracy timings also include ancestor recovery. Julia timing includes Python-to-Julia IPC. These are integration timings, not isolated language-kernel speed rankings. JAX results synchronize with block_until_ready; TensorFlow results synchronize by converting to NumPy.
- Shared filtering normalizes the output correction weights of soft methods, as it does all output measures. This is a self-normalized particle approximation; unbiased likelihood estimates are not assumed. Input order differs across repository implementations (for example, Stone Soup sorts weights), which can change systematic and stratified resampling variance.
- PyTorch and TensorFlow local backward timings measure backward only. JAX local backward timings measure the compiled forward-and-backward gradient call and are explicitly labeled in raw CSV. Native Cuthbert likewise reports forward-and-backward together. Subtracting those times is not an exact backward-time measurement.
- Peak RSS is the process high-water mark, in MiB despite the legacy _mb field suffix, including imports and JIT caches; the separate timing pass uses a fresh interpreter for each method and does not calculate distribution metrics. Caches can accumulate across particle counts within that method. Raw accuracy RSS also includes metric calculations. RSS is not isolated per-call native allocation. Julia worker peak RSS is recorded separately from Linux VmHWM; total_peak_rss_mb_bound adds Python and worker high-water marks and is an upper bound because their peaks need not coincide. Other providers have no separate worker. Use these values for process budgeting, not a claim of algorithm-only peak memory.
- Results are CPU only. The installed NVIDIA GTX 960M has 2 GB VRAM and legacy Maxwell compute capability 5.0. No compatible CUDA benchmark environment was established. GPU speed and GPU memory claims remain unmeasured.
- Torchfilter upstream soft resampling leaves correction weights in input order. The adapter gathers the corresponding correction weights for the sampled ancestors; this deliberate fix is in adapters.py and vendor code is unchanged. Modern overrides 7.7.0 is used for Python 3.12 compatibility.
- The Julia bridge sends binary Float64 weights and receives Int64 ancestor indices. Both the explicitly passed RNG and Julia global RNG are seeded, because upstream multinomial/residual sampling includes a global randexp(T) draw. Protocol equivalence and repeated-seed reproducibility were checked for all six methods. Earlier incompletely seeded multinomial measurements were superseded and archived outside this report.
- FilterFlow runs in a separate Python 3.11 / TensorFlow 2.15.1 environment with float32 input states/weights; the other providers use Python 3.12 and float64 inputs (Julia uses Float64). Upstream internal random numbers and intermediates can use their own defaults. JIT compilation is excluded from forward timing. Cross-framework precision and wrapper differences must be considered.

## Source revisions

- [nchopin/particles](https://github.com/nchopin/particles/tree/3388c4bd30ab9acc5a27c970089230f93610b627): `3388c4bd30ab9acc5a27c970089230f93610b627`
- [dstl/Stone-Soup](https://github.com/dstl/Stone-Soup/tree/d6c2d5713b5fdc0bc0c5c0226898bc1c598a882a): `d6c2d5713b5fdc0bc0c5c0226898bc1c598a882a`
- [JTT94/filterflow](https://github.com/JTT94/filterflow/tree/5d8300ba247c4c17e1a301a22560c24fd0670bfe): `5d8300ba247c4c17e1a301a22560c24fd0670bfe`
- [state-space-models/cuthbert](https://github.com/state-space-models/cuthbert/tree/c737efe3098b2b5ecaa7d8b976e7cea246998b17): `c737efe3098b2b5ecaa7d8b976e7cea246998b17`
- [mlysy/pfjax](https://github.com/mlysy/pfjax/tree/97652aa1bdff73a92c0286549b010e99cc6f7264): `97652aa1bdff73a92c0286549b010e99cc6f7264`
- [John-JoB/pydpf](https://github.com/John-JoB/pydpf/tree/a7487ce1c344c70e1a244e75725b950deb15e146): `a7487ce1c344c70e1a244e75725b950deb15e146`
- [stanford-iprl-lab/torchfilter](https://github.com/stanford-iprl-lab/torchfilter/tree/4a8a085f6406ec07ce6a701e7d8cb4c62c97a390): `4a8a085f6406ec07ce6a701e7d8cb4c62c97a390`
- [skarppinen/Resamplings.jl](https://github.com/skarppinen/Resamplings.jl/tree/5cc4dd050c568abd8aedc45c6dbe318b0bc6ec7f): `5cc4dd050c568abd8aedc45c6dbe318b0bc6ec7f`
- [okviman/Variational-Resampling](https://github.com/okviman/Variational-Resampling/tree/929e5a818ea270327876dea4ed7cdec0678d3cd0): `929e5a818ea270327876dea4ed7cdec0678d3cd0`

## Artifacts

- `report.html`: self-contained interactive plots; open in a browser.
- `distribution_vs_runtime.png` and `filter_rmse.png`: exportable static plots.
- `latex/`: every results table as LaTeX (booktabs). `tables/` has one file per table, `all_tables.tex` has all of them, and `benchmark_tables.tex` is a standalone document that compiles them.
- `figures/`: 13 static PNG/PDF figures covering every section of this report; see `figures/README.md`. Rebuild with `scripts/plot_figures.py`.
- `*_summary.csv`: means, 95% intervals, gradient bias/variance, and conditional likelihood variance.
- `../<provider>/*.csv`: raw trial measurements; `status.jsonl` records successful and failed cells.
- `../../environment-main.txt`, `../../environment-tf.txt`, `../../hardware.txt`: dependency and hardware records.
- `../../../README.md`: setup and rerun commands.
