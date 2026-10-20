# Particle-resampler benchmark report

Completed 2463 benchmark cells across 8 upstream repositories plus custom baselines. Raw trials: 150,261 static, 50,000 filtering, 19,600 local gradients, 110 parameter-learning trials.

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

## Mixture distribution results (N=128, d=2)

| provider | method | trials | sliced w1 | rff mmd2 | covariance error | mode coverage | forward ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stonesoup | systematic | 100 | 0.04703 | 0.0001244 | 0.1629 | 1 | 0.3949 |
| julia | systematic | 100 | 0.05308 | 0.000196 | 0.2301 | 1 | 0.3779 |
| stonesoup | stratified | 100 | 0.05856 | 0.000223 | 0.2409 | 1 | 0.4062 |
| julia | stratified | 100 | 0.1159 | 0.001388 | 0.3264 | 1 | 0.343 |
| filterflow | stratified | 100 | 0.1171 | 0.001477 | 0.3263 | 1 | 0.9598 |
| particles | ssp | 100 | 0.1208 | 0.001649 | 0.3442 | 1 | 0.1907 |
| julia | killing | 100 | 0.1209 | 0.00148 | 0.3478 | 1 | 0.4145 |
| particles | stratified | 100 | 0.1236 | 0.001665 | 0.3337 | 1 | 0.2263 |
| particles | residual | 100 | 0.1285 | 0.001523 | 0.405 | 1 | 0.2528 |
| julia | ssp | 100 | 0.1303 | 0.001898 | 0.3662 | 1 | 0.3633 |
| julia | residual | 100 | 0.1364 | 0.001711 | 0.4303 | 1 | 0.3709 |
| stonesoup | residual | 100 | 0.1405 | 0.001896 | 0.4878 | 1 | 0.5988 |
| cuthbert | multinomial | 100 | 0.1669 | 0.00263 | 0.4797 | 1 | 1.926 |
| julia | multinomial | 100 | 0.1841 | 0.003346 | 0.5409 | 1 | 0.3702 |
| pydpf | soft | 100 | 0.186 | 0.003264 | 0.5216 | 1 | 0.8031 |
| torchfilter | soft | 100 | 0.186 | 0.003264 | 0.5216 | 1 | 1.152 |
| pfjax | multinomial | 100 | 0.1892 | 0.003463 | 0.5114 | 1 | 0.9381 |
| torchfilter | multinomial | 100 | 0.1935 | 0.003563 | 0.5698 | 1 | 0.8181 |
| pydpf | multinomial | 100 | 0.1935 | 0.003563 | 0.5698 | 1 | 0.6702 |
| particles | multinomial | 100 | 0.1953 | 0.003629 | 0.5588 | 1 | 0.3192 |
| filterflow | multinomial | 100 | 0.2042 | 0.003941 | 0.5978 | 1 | 0.947 |
| stonesoup | multinomial | 100 | 0.2066 | 0.004211 | 0.5729 | 1 | 0.415 |
| pfjax | ot | 10 | 0.4299 | 0.0126 | 2.134 | 1 | 27.64 |
| pydpf | ot | 10 | 0.4356 | 0.01226 | 1.925 | 1 | 33.95 |
| filterflow | corrected_ot | 10 | 0.6283 | 0.02492 | 2.857 | 1 | 26.17 |
| filterflow | ot | 10 | 0.6283 | 0.02492 | 2.857 | 1 | 11.67 |
| custom | eigen | 100 | 0.7163 | 0.02422 | 1.311e-15 | 1 | 1.013 |
| custom | gumbel | 10 | 0.7719 | 0.05546 | 4.487 | 1 | 0.9708 |
| pfjax | mvn | 100 | 0.8582 | 0.0315 | 0.9077 | 1 | 1.027 |
| custom | gaussian | 100 | 0.8599 | 0.0325 | 0.9295 | 1 | 0.7702 |
| particles | systematic | 100 | 0.8868 | 0.08753 | 2.906 | 0.865 | 0.2093 |
| filterflow | systematic | 100 | 0.9303 | 0.09637 | 3.29 | 0.84 | 0.9602 |
| pydpf | systematic | 100 | 0.9303 | 0.09637 | 3.29 | 0.84 | 0.8183 |
| pydpf | systematic_stop | 100 | 0.9303 | 0.09637 | 3.29 | 0.84 | 0.827 |
| cuthbert | systematic | 100 | 0.9303 | 0.09637 | 3.29 | 0.84 | 1.391 |
| cuthbert | systematic_stop | 100 | 0.9303 | 0.09637 | 3.29 | 0.84 | 1.362 |

## Filtering results (N=128)

| model | provider | method | trials | successful trials | failure rate | rmse | coverage95 | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | filterflow | corrected_ot | 10 | 10 | 0 | 3.25 | 0.72 | nan |
| bearing | julia | killing | 100 | 100 | 0 | 4.442 | 0.6212 | 4.723 |
| bearing | julia | ssp | 100 | 100 | 0 | 4.478 | 0.6614 | 4.573 |
| bearing | filterflow | systematic | 100 | 100 | 0 | 4.487 | 0.6508 | 7.311 |
| bearing | custom | gaussian | 100 | 100 | 0 | 4.508 | 0.724 | 1.992 |
| bearing | custom | eigen | 100 | 100 | 0 | 4.51 | 0.7776 | 0.586 |
| bearing | julia | systematic | 100 | 100 | 0 | 4.529 | 0.6522 | 5.275 |
| bearing | particles | systematic | 100 | 100 | 0 | 4.548 | 0.6364 | 4.739 |
| bearing | cuthbert | systematic | 100 | 100 | 0 | 4.553 | 0.6478 | 3.326 |
| bearing | cuthbert | systematic_stop | 100 | 100 | 0 | 4.553 | 0.6478 | 3.326 |
| bearing | julia | stratified | 100 | 100 | 0 | 4.561 | 0.6586 | 3.849 |
| bearing | stonesoup | systematic | 100 | 100 | 0 | 4.6 | 0.6286 | 3.771 |
| bearing | particles | ssp | 100 | 100 | 0 | 4.649 | 0.633 | 2.019 |
| bearing | stonesoup | multinomial | 100 | 100 | 0 | 4.649 | 0.5648 | 4.879 |
| bearing | stonesoup | stratified | 100 | 100 | 0 | 4.698 | 0.6044 | 2.93 |
| bearing | filterflow | stratified | 100 | 100 | 0 | 4.705 | 0.6176 | 7.085 |
| bearing | pfjax | mvn | 100 | 100 | 0 | 4.721 | 0.7414 | 1.197 |
| bearing | stonesoup | residual | 100 | 100 | 0 | 4.733 | 0.6078 | 7.747 |
| bearing | particles | residual | 100 | 100 | 0 | 4.746 | 0.62 | 7.577 |
| bearing | pydpf | systematic | 100 | 100 | 0 | 4.763 | 0.6492 | 4.651 |
| bearing | pydpf | systematic_stop | 100 | 100 | 0 | 4.763 | 0.6492 | 4.651 |
| bearing | julia | residual | 100 | 100 | 0 | 4.763 | 0.6006 | 7.695 |
| bearing | cuthbert | multinomial | 100 | 100 | 0 | 4.773 | 0.5798 | 6.607 |
| bearing | particles | stratified | 100 | 100 | 0 | 4.775 | 0.6112 | 3.091 |
| bearing | pydpf | multinomial | 100 | 100 | 0 | 4.776 | 0.5696 | 4.556 |
| bearing | torchfilter | multinomial | 100 | 100 | 0 | 4.776 | 0.5696 | 4.556 |
| bearing | pfjax | multinomial | 100 | 100 | 0 | 4.777 | 0.5742 | 4.521 |
| bearing | particles | multinomial | 100 | 100 | 0 | 4.787 | 0.5396 | 19.1 |
| bearing | filterflow | multinomial | 100 | 100 | 0 | 4.791 | 0.56 | 3.972 |
| bearing | torchfilter | soft | 100 | 100 | 0 | 4.799 | 0.5192 | 9.433 |
| bearing | pydpf | soft | 100 | 100 | 0 | 4.799 | 0.5192 | 9.433 |
| bearing | julia | multinomial | 100 | 100 | 0 | 4.807 | 0.5588 | 2.921 |
| bearing | custom | gumbel | 10 | 10 | 0 | 5.188 | 0.216 | nan |
| bearing | filterflow | ot | 10 | 10 | 0 | 5.296 | 0.052 | nan |
| bearing | pfjax | ot | 10 | 10 | 0 | 5.559 | 0.042 | nan |
| bearing | pydpf | ot | 10 | 10 | 0 | 5.648 | 0.04 | nan |
| linear | julia | multinomial | 100 | 100 | 0 | 0.505 | 0.9284 | 0.3786 |
| linear | particles | residual | 100 | 100 | 0 | 0.5054 | 0.9304 | 0.3669 |
| linear | stonesoup | residual | 100 | 100 | 0 | 0.5056 | 0.9316 | 0.3605 |
| linear | pfjax | multinomial | 100 | 100 | 0 | 0.5056 | 0.9286 | 0.3605 |
| linear | pfjax | mvn | 100 | 100 | 0 | 0.5058 | 0.9286 | 0.4128 |
| linear | pydpf | multinomial | 100 | 100 | 0 | 0.5058 | 0.9306 | 0.374 |
| linear | torchfilter | multinomial | 100 | 100 | 0 | 0.5058 | 0.9306 | 0.374 |
| linear | stonesoup | multinomial | 100 | 100 | 0 | 0.5058 | 0.9288 | 0.3589 |
| linear | cuthbert | multinomial | 100 | 100 | 0 | 0.5058 | 0.9304 | 0.3784 |
| linear | filterflow | multinomial | 100 | 100 | 0 | 0.506 | 0.9286 | 0.3587 |
| linear | particles | systematic | 100 | 100 | 0 | 0.5061 | 0.9292 | 0.3526 |
| linear | julia | residual | 100 | 100 | 0 | 0.5061 | 0.9308 | 0.435 |
| linear | pydpf | systematic | 100 | 100 | 0 | 0.5064 | 0.9308 | 0.327 |
| linear | pydpf | systematic_stop | 100 | 100 | 0 | 0.5064 | 0.9308 | 0.327 |
| linear | stonesoup | systematic | 100 | 100 | 0 | 0.5064 | 0.9262 | 0.3047 |
| linear | cuthbert | systematic | 100 | 100 | 0 | 0.5065 | 0.9276 | 0.321 |
| linear | cuthbert | systematic_stop | 100 | 100 | 0 | 0.5065 | 0.9276 | 0.321 |
| linear | particles | multinomial | 100 | 100 | 0 | 0.5065 | 0.9276 | 0.4062 |
| linear | julia | stratified | 100 | 100 | 0 | 0.5066 | 0.9296 | 0.4132 |
| linear | julia | killing | 100 | 100 | 0 | 0.5066 | 0.9296 | 0.2993 |
| linear | particles | stratified | 100 | 100 | 0 | 0.5066 | 0.9306 | 0.348 |
| linear | filterflow | stratified | 100 | 100 | 0 | 0.5066 | 0.9278 | 0.3512 |
| linear | filterflow | systematic | 100 | 100 | 0 | 0.5067 | 0.9274 | 0.3659 |
| linear | julia | systematic | 100 | 100 | 0 | 0.5067 | 0.9264 | 0.3609 |
| linear | particles | ssp | 100 | 100 | 0 | 0.5068 | 0.929 | 0.3507 |
| linear | custom | eigen | 100 | 100 | 0 | 0.5069 | 0.9288 | 0.3199 |
| linear | julia | ssp | 100 | 100 | 0 | 0.507 | 0.9294 | 0.3498 |
| linear | stonesoup | stratified | 100 | 100 | 0 | 0.5072 | 0.9278 | 0.3259 |
| linear | custom | gaussian | 100 | 100 | 0 | 0.5072 | 0.9292 | 0.4173 |
| linear | pydpf | soft | 100 | 100 | 0 | 0.5082 | 0.9274 | 0.4185 |
| linear | torchfilter | soft | 100 | 100 | 0 | 0.5082 | 0.9274 | 0.4185 |
| linear | filterflow | corrected_ot | 10 | 10 | 0 | 0.5144 | 0.932 | nan |
| linear | custom | gumbel | 10 | 10 | 0 | 0.5144 | 0.928 | nan |
| linear | filterflow | ot | 10 | 10 | 0 | 0.5145 | 0.924 | nan |
| linear | pydpf | ot | 10 | 10 | 0 | 0.5145 | 0.928 | nan |
| linear | pfjax | ot | 10 | 10 | 0 | 0.5146 | 0.93 | nan |
| ungm | julia | killing | 100 | 100 | 0 | 4.762 | 0.93 | 760.2 |
| ungm | filterflow | multinomial | 100 | 100 | 0 | 4.79 | 0.9326 | 729.7 |
| ungm | filterflow | systematic | 100 | 100 | 0 | 4.792 | 0.9304 | 635.8 |
| ungm | particles | stratified | 100 | 100 | 0 | 4.824 | 0.9322 | 617.1 |
| ungm | pydpf | soft | 100 | 100 | 0 | 4.825 | 0.928 | 438.4 |
| ungm | torchfilter | soft | 100 | 100 | 0 | 4.825 | 0.928 | 438.4 |
| ungm | pydpf | multinomial | 100 | 100 | 0 | 4.827 | 0.931 | 974.5 |
| ungm | torchfilter | multinomial | 100 | 100 | 0 | 4.827 | 0.931 | 974.5 |
| ungm | stonesoup | multinomial | 100 | 100 | 0 | 4.843 | 0.9322 | 969.2 |
| ungm | filterflow | stratified | 100 | 100 | 0 | 4.853 | 0.9302 | 672.7 |
| ungm | particles | ssp | 100 | 100 | 0 | 4.854 | 0.9334 | 633.7 |
| ungm | julia | ssp | 100 | 100 | 0 | 4.86 | 0.9334 | 685.2 |
| ungm | cuthbert | systematic | 100 | 100 | 0 | 4.87 | 0.9348 | 769.7 |
| ungm | cuthbert | systematic_stop | 100 | 100 | 0 | 4.87 | 0.9348 | 769.7 |
| ungm | julia | multinomial | 100 | 100 | 0 | 4.871 | 0.9348 | 761.1 |
| ungm | pydpf | systematic | 100 | 100 | 0 | 4.881 | 0.932 | 669.5 |
| ungm | pydpf | systematic_stop | 100 | 100 | 0 | 4.881 | 0.932 | 669.5 |
| ungm | julia | residual | 100 | 100 | 0 | 4.888 | 0.9274 | 836.5 |
| ungm | particles | systematic | 100 | 100 | 0 | 4.906 | 0.9304 | 653.3 |
| ungm | julia | stratified | 100 | 100 | 0 | 4.906 | 0.9318 | 719.2 |
| ungm | particles | multinomial | 100 | 100 | 0 | 4.916 | 0.9284 | 4174 |
| ungm | julia | systematic | 100 | 100 | 0 | 4.921 | 0.9288 | 1081 |
| ungm | stonesoup | residual | 100 | 100 | 0 | 4.935 | 0.9302 | 843.9 |
| ungm | stonesoup | systematic | 100 | 100 | 0 | 4.935 | 0.9308 | 953.5 |
| ungm | custom | gaussian | 100 | 100 | 0 | 4.936 | 0.933 | 845.1 |
| ungm | particles | residual | 100 | 100 | 0 | 4.942 | 0.9284 | 662.5 |
| ungm | cuthbert | multinomial | 100 | 100 | 0 | 4.972 | 0.9272 | 1181 |
| ungm | pfjax | mvn | 100 | 93 | 0.07 | 5.027 | 0.9372 | 529.8 |
| ungm | pfjax | multinomial | 100 | 100 | 0 | 5.038 | 0.9292 | 6550 |
| ungm | stonesoup | stratified | 100 | 100 | 0 | 5.039 | 0.925 | 3469 |
| ungm | custom | eigen | 100 | 100 | 0 | 5.17 | 0.93 | 809.7 |
| ungm | custom | gumbel | 10 | 10 | 0 | 5.271 | 0.918 | nan |
| ungm | pfjax | ot | 10 | 10 | 0 | 5.328 | 0.918 | nan |
| ungm | filterflow | corrected_ot | 10 | 10 | 0 | 5.335 | 0.914 | nan |
| ungm | pydpf | ot | 10 | 10 | 0 | 5.416 | 0.914 | nan |
| ungm | filterflow | ot | 10 | 10 | 0 | 5.48 | 0.908 | nan |
| volatility | custom | gumbel | 10 | 10 | 0 | 0.4148 | 0.946 | nan |
| volatility | stonesoup | stratified | 100 | 100 | 0 | 0.4151 | 0.9644 | 0.07439 |
| volatility | stonesoup | systematic | 100 | 100 | 0 | 0.4153 | 0.965 | 0.07898 |
| volatility | filterflow | multinomial | 100 | 100 | 0 | 0.4158 | 0.9648 | 0.09982 |
| volatility | custom | gaussian | 100 | 100 | 0 | 0.4158 | 0.9646 | 0.07859 |
| volatility | filterflow | stratified | 100 | 100 | 0 | 0.4163 | 0.9628 | 0.09186 |
| volatility | pfjax | mvn | 100 | 100 | 0 | 0.4163 | 0.9664 | 0.08799 |
| volatility | julia | multinomial | 100 | 100 | 0 | 0.4163 | 0.9634 | 0.07481 |
| volatility | custom | eigen | 100 | 100 | 0 | 0.4164 | 0.9662 | 0.07457 |
| volatility | particles | ssp | 100 | 100 | 0 | 0.4164 | 0.9656 | 0.08356 |
| volatility | pydpf | systematic | 100 | 100 | 0 | 0.4165 | 0.967 | 0.08059 |
| volatility | pydpf | systematic_stop | 100 | 100 | 0 | 0.4165 | 0.967 | 0.08059 |
| volatility | julia | ssp | 100 | 100 | 0 | 0.4165 | 0.9644 | 0.07956 |
| volatility | pydpf | multinomial | 100 | 100 | 0 | 0.4167 | 0.9632 | 0.08358 |
| volatility | torchfilter | multinomial | 100 | 100 | 0 | 0.4167 | 0.9632 | 0.08358 |
| volatility | filterflow | systematic | 100 | 100 | 0 | 0.4168 | 0.9636 | 0.1003 |
| volatility | cuthbert | systematic | 100 | 100 | 0 | 0.4171 | 0.9616 | 0.07696 |
| volatility | cuthbert | systematic_stop | 100 | 100 | 0 | 0.4171 | 0.9616 | 0.07696 |
| volatility | stonesoup | residual | 100 | 100 | 0 | 0.4171 | 0.9612 | 0.08088 |
| volatility | julia | stratified | 100 | 100 | 0 | 0.4173 | 0.9626 | 0.08726 |
| volatility | particles | stratified | 100 | 100 | 0 | 0.4173 | 0.9648 | 0.0933 |
| volatility | particles | systematic | 100 | 100 | 0 | 0.4173 | 0.9658 | 0.08285 |
| volatility | filterflow | ot | 10 | 10 | 0 | 0.4173 | 0.968 | nan |
| volatility | particles | multinomial | 100 | 100 | 0 | 0.4174 | 0.9614 | 0.107 |
| volatility | julia | systematic | 100 | 100 | 0 | 0.4174 | 0.9668 | 0.08305 |
| volatility | julia | residual | 100 | 100 | 0 | 0.4174 | 0.965 | 0.08983 |
| volatility | filterflow | corrected_ot | 10 | 10 | 0 | 0.4176 | 0.966 | nan |
| volatility | cuthbert | multinomial | 100 | 100 | 0 | 0.4176 | 0.9622 | 0.1066 |
| volatility | pydpf | ot | 10 | 10 | 0 | 0.4181 | 0.97 | nan |
| volatility | pfjax | ot | 10 | 10 | 0 | 0.4183 | 0.97 | nan |
| volatility | particles | residual | 100 | 100 | 0 | 0.4183 | 0.9592 | 0.09107 |
| volatility | pfjax | multinomial | 100 | 100 | 0 | 0.4186 | 0.961 | 0.09211 |
| volatility | stonesoup | multinomial | 100 | 100 | 0 | 0.4189 | 0.9598 | 0.1031 |
| volatility | pydpf | soft | 100 | 100 | 0 | 0.419 | 0.9594 | 0.1155 |
| volatility | torchfilter | soft | 100 | 100 | 0 | 0.419 | 0.9594 | 0.1155 |
| volatility | julia | killing | 100 | 100 | 0 | 0.4191 | 0.9624 | 0.09429 |

RMSE measures error against the simulated latent truth. Linear-model posterior error and likelihood error use an exact Kalman reference. Coverage is joint 95% Gaussian-ellipsoid coverage from the particle mean/covariance; it is a diagnostic and is not exact posterior credible-set coverage for multimodal targets. Bearing RMSE uses position only; the coverage ellipsoid uses all four state coordinates. Recovery requires position error below 2 units for three consecutive observed steps; unrecovered runs are right-censored at the end of the sequence.

Bearing RMSE averages squared error across both position coordinates before taking the square root. Post-occlusion RMSE instead uses Euclidean position error, so its squared error is the sum of both coordinate errors. Recovery-step means cap unrecovered runs at 20 steps and are not estimates of unrestricted recovery time.

Likelihood variance is calculated across filter seeds conditional on each observation sequence and then averaged. It is unavailable for expensive filters with only one seed per sequence. Log likelihood bias is meaningful against the exact reference only in the linear Gaussian case. Do not treat transport/Gaussian/eigen likelihood estimates as unbiased likelihood estimates.

## Likelihood-gradient results

Gradient bias is ||mean(g) − g_exact||₂; variance is the trace of the sample gradient covariance; MSE is mean(||g−g_exact||₂²). The Cuthbert experiment uses its native tutorial data (T=50, q=0.4, r=0.8), 100 seeds at each N, and 21-point likelihood/gradient profile slices at N=3000. The PyTorch filter uses that same saved dataset and exact Kalman gradients. Stop-gradient estimators are assessed empirically; finite-particle unbiasedness is not assumed.

cuthbert_native

| n | stop gradient | trials | gradient bias | gradient variance | gradient mse | loglik variance | loglik bias | forward ms | forward backward ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | False | 100 | 12.98 | 44.41 | 212.5 | 0.621 | -0.2671 | 13.8 | 14.34 |
| 128 | True | 100 | 2.285 | 175.9 | 179.4 | 0.621 | -0.2671 | 13.44 | 14.08 |

torch_native

| provider | method | n | trials | gradient bias | gradient variance | gradient mse | loglik variance | loglik bias | forward ms | backward ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom | eigen | 128 | 100 | 0.5272 | 5.207 | 5.433 | 0.5643 | -0.1536 | 27.6 | 25 |
| custom | gaussian | 128 | 100 | 0.8231 | 5.602 | 6.223 | 0.86 | -0.2781 | 24 | 14.59 |
| custom | gumbel | 128 | 10 | 8.148 | 20.28 | 84.64 | 0.7297 | -2.662 | 29.09 | 11.69 |
| pydpf | ot | 128 | 10 | 14.21 | 714.6 | 844.9 | 0.3098 | -0.4472 | 609.8 | 34.68 |
| pydpf | soft | 128 | 100 | 6.675 | 51.45 | 95.5 | 1.081 | -0.3859 | 34.43 | 17.76 |
| pydpf | systematic | 128 | 100 | 11.27 | 41.82 | 168.4 | 0.6556 | -0.2953 | 23.99 | 10.13 |
| pydpf | systematic_stop | 128 | 100 | 1.748 | 150.6 | 152.2 | 0.6556 | -0.2953 | 27.19 | 13.36 |
| torchfilter | soft | 128 | 100 | 6.675 | 51.45 | 95.5 | 1.081 | -0.3859 | 43.14 | 14.7 |

## Stochastic-volatility parameter learning

The known state model is x_t=0.95 x_(t−1)+0.2 ε_t; y_t=s exp(x_t/2) η_t, with true s=1. Fit s by maximizing the particle likelihood on {0.6,0.8,1.0,1.2,1.4}. Ten independent datasets and common random numbers are used. These are coarse-grid estimates, not continuous optimizer convergence claims. The likelihood-score metric uses central differences in log(s), step 0.01. The independent reference is a numerical grid filter on [−4,4] with 1001 cells; a 2001-cell likelihood cross-check is stored per dataset. This reference is approximate, unlike the exact Kalman reference.

| provider | method | trials | parameter absolute error | mle reference error | score rmse |
| --- | --- | --- | --- | --- | --- |
| custom | eigen | 10 | 0.26 | 0.06 | 2.112 |
| custom | gaussian | 10 | 0.22 | 0.06 | 5.252 |
| custom | gumbel | 10 | 0.26 | 0.1 | 4.606 |
| cuthbert | systematic_stop | 10 | 0.22 | 0.02 | 7.468 |
| filterflow | ot | 10 | 0.28 | 0.08 | 1.307 |
| julia | systematic | 10 | 0.28 | 0.12 | 7.394 |
| particles | systematic | 10 | 0.22 | 0.02 | 8.793 |
| pfjax | ot | 10 | 0.26 | 0.06 | 1.179 |
| pydpf | ot | 10 | 0.26 | 0.06 | 1.146 |
| stonesoup | systematic | 10 | 0.24 | 0.08 | 10.19 |
| torchfilter | soft | 10 | 0.2 | 0.08 | 10.14 |

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

## Artifacts

- `report.html`: self-contained interactive plots; open in a browser.
- `distribution_vs_runtime.png` and `filter_rmse.png`: exportable static plots.
- `*_summary.csv`: means, 95% intervals, gradient bias/variance, and conditional likelihood variance.
- `../<provider>/*.csv`: raw trial measurements; `status.jsonl` records successful and failed cells.
- `../../environment-main.txt`, `../../environment-tf.txt`, `../../hardware.txt`: dependency and hardware records.
- `../../../README.md`: setup and rerun commands.
