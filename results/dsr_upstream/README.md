# DSR upstream reproduction

Checks of [fcumlin/differentiable-systematic-resampling](https://github.com/fcumlin/differentiable-systematic-resampling) at commit `39a86e5` against the paper, *Differentiable Systematic Resampling for Variational Sequential Monte Carlo* (Cumlin & Chatterjee, NeurIPS 2026, [arXiv:2610.12094](https://arxiv.org/abs/2610.12094)). The source is checked out unmodified in `vendor/differentiable-systematic-resampling`.

Environment: `.venv-dsr`, Python 3.10.22, torch 2.1.0+cpu, numpy 1.26.4, scipy 1.14.1. This matches the repository's stated CPU test setup ([environment-dsr.txt](environment-dsr.txt)). Everything ran on CPU (i7-6700HQ, about 4 GB of free RAM).

`scripts/dsr_check_claims.py` compares every number mechanically and writes [claims.csv](claims.csv).

## Outcome

| Paper result | How it was run | Outcome |
|---|---|---|
| Fig. 1, transport plans (τ=0.1, N=25) | upstream `transport_maps --taus 0.1` | **Pixel-identical** to `figures/transport_maps.png` (723×1682 RGBA, max pixel difference 0). The PNG bytes differ only in metadata. [png](transport_maps_tau0.1.png) |
| Fig. 5, transport plans for all τ | upstream `transport_maps` | Ran. No reference image in the repo to compare against. [png](transport_maps_all_taus.png) |
| Table 6, bias/variance/MSE of resampled estimates | `scripts/dsr_bias_variance_chunked.py` (see note 1) | **42/42** numbers match at the paper's 2 significant figures. [log](bias_variance.log) |
| Table 6 harness self-test | upstream `--self-test` | Pass. Multinomial/systematic variance ratio 4.76 > 1. [log](bias_variance_selftest.log) |
| Table 13, inter-modal mass | `scripts/dsr_intermodal_chunked.py` (see note 1) | **25/25** means match at 3 decimals. Standard deviations also match. [log](intermodal_mass.log) |
| Table 15, DSR vs unbiased VSMC gradient | upstream `vsmc_gradient_bias`, unmodified | **36/36** match at every printed decimal. The bit-exact claim holds. [log](vsmc_gradient_bias.log) |
| Table 15 estimator self-test | upstream `--self-test` | Pass. The pmf sums to 1 within 2e-16, and Monte Carlo agrees with exact enumeration (max \|z\| < 1.5). [log](vsmc_gradient_bias_selftest.log) |
| Relaxation metrics (N_eff, correlation, L2; App. E, Table 8 L2 column) | upstream `relaxation_metrics`, unmodified | **55/60** match at all 4 decimals of `published_relaxation_metrics.json`, and 58/60 within 1e-3. The worst case is N=200, τ=0.5: L2 0.5839 vs 0.5852 and N_eff 81.63 vs 81.68, which is ≤0.1σ of the published spread. The repo calls this experiment "seeded and deterministic". It reproduces numerically, but not bit-for-bit, on this CPU and torch build. [json](relaxation_metrics.json) |
| Table 14 estimator self-test | upstream `marginal_likelihood --self-test` | Pass. The Jensen gap is positive and shrinks with N. [log](marginal_likelihood_selftest.log) |

Note 1: The upstream Table 6 and Table 13 scripts resample all 20 000 or 10 000 draws in a single batch. Their Slurm scripts request 64 GB, and the unmodified Table 13 script drove this machine into full swap, so it was stopped. The chunked drivers import the upstream particle set, resamplers and estimators unchanged and only accumulate over chunks of draws. That makes them statistically equivalent but not bit-exact, because the RNG streams differ. The repo claims these two experiments are bit-exact. That claim was not tested.

## Not run

- **Training results** (Tables 1–5 and 7–12, gradient norms, Lorenz/character figures). These need GPU-scale compute. The paper reports 2.5 h per Lorenz-63 run on a T4. Here, one linear-SSM epoch took about 90 s on CPU, which is about 12 h per 500-epoch run, and the paper uses 10 seeds × 4 methods per configuration. One epoch of `configs/linear/dsr_10db.gin` was run as a pipeline check, and it trains and evaluates without error.
- **Table 14 itself**. It needs the trained linear-SSM models.

## Inconsistencies found in the paper text

1. Section 5.5 says "DSR adds only 9.6% overhead over Hard (2.55 h vs. 2.49 h)". Those totals give 2.4%, and the per-epoch times in Table 5 (18.39 s vs 17.94 s) give 2.5%.
2. Section 5.3 says "The large MSE-5-step variance of EROT at 0 dB (57.60)". Tables 2 and 12 report the standard deviation as 60.72.
3. The Section 5.3 text and Table 2 give DSR NMSE at 0 dB as −18.28 dB, but Table 11 (Morton row) gives −18.27 dB.
4. Appendix G says EROT "for d ≥ 3 places all resampled mass between the modes", but Table 13 and this rerun give 0.747 at d=3. The mass is 1.000 only from d=5.
5. Appendix E.5 says DSR's variance for f(x)=x is "two to five orders of magnitude below hard systematic resampling". At τ=0.01 the gap is 1.2e-3 vs 2.9e-5, about 1.6 orders.
6. The Table 1 caption says "mean ± std over 3 runs". The README's `run_paper.sh` default is 10 runs per configuration.

The qualitative claims attached to the reproduced tables hold:
- DSR's gradient bias is below Hard SR's at every τ in Table 15.
- τ=0.1 has the lowest MSE at d=4 and the second lowest at d=2.
- Morton sorting cuts inter-modal mass from 1.0 to about 0.28 for every d.
- DSR has lower MSE than Hard SR for f(x)=x at every τ.
