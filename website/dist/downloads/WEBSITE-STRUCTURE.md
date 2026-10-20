# Resampler Lab: learning path and website structure

Audience: college students who know basic probability and matrix multiplication. The first lessons also explain these concepts. Write mainly in simplified technical English. Define technical terms when they first appear. Put longer proofs in expandable sections.

The persistent left sidebar follows this order. Every lesson has a learning goal, a visual or interactive example, precise definitions/equations, a question with explanatory feedback, and sources. Previous/next controls make the learning order clear. Search, a glossary, and a page outline support later reference use.

| Part | Lesson | Interactive or visual component | Main question |
|---|---|---|---|
| Start | 01 Start here | Clickable learning path and claims overview | What are we trying to build? |
| Foundations | 02 Particles and weights | Adjustable particle weights, mean, and effective sample size | How do weighted points represent uncertainty? |
| Foundations | 03 Why resample? | Linear filtering simulator with truth, observations, exact Kalman reference, uncertainty, and ESS | Where does resampling fit in a particle filter? |
| Foundations | 04 Classical methods | CDF selection, sample ancestors, repeat trials, and reorder particles | What do copying methods preserve in expectation? |
| Design | 05 Our research claims | Worked two-particle expectation table; live contract lab; claim-to-test matrix | What would justify a differentiable, explicit resampler? |
| Design | 06 Build the eigen method | Complete scalar hand calculation, then 2D center/whiten/color/shift stages | How do we match mean and covariance? |
| Design | 07 Test its limits | Exact cubic counterexample; adjustable bimodal distribution and shape errors | Why can correct moments hide a wrong distribution? |
| Design | 08 Understand gradients | Hand-calculated slope, two-point unequal-gradient counterexample, analytic/finite-difference lab | Which objective does a gradient describe? |
| Evaluation | 09 Choose metrics | Worked W1/MMD/covariance/RMSE/coverage/bias calculations; live metric playground | Which measurement tests each claim? |
| Evaluation | 10 Run benchmarks | Six first-principles task explanations with toy cases, metric reasons, pinned sources, and a task selector | Which open-source tasks should we use? |
| Evaluation | 11 Explore our results | Eight paper-style figures: one question, one metric, caption, best observed mean, and finding; optional full explorer | What do the completed runs actually show? |
| Reference | 12 Read and reproduce | Question-led literature map, repo-to-task table, complete experiment plan, tested smoke command, glossary | How do I check and extend this work? |

Three central claims drive the site:

1. **Explicit construction:** a finite matrix formula without an N-by-N particle transport optimization; spectral roots still need numerical linear algebra.
2. **Differentiability:** first-order derivatives of the implemented map, away from singular covariance/floor kinks; derivative agreement does not prove an accurate gradient of the true filtering likelihood.
3. **Useful uncertainty:** mean/covariance preservation is established under stated conditions; distribution shape, calibration, and downstream likelihood accuracy remain separate empirical tests.

The current eigen implementation is an affine moment-matching baseline. The site must not claim novelty, universal speed superiority, unbiased likelihood estimates, or general distribution preservation. It must show CPU-only measurements, reduced expensive-method grids, source-model differences, and the observed upstream failures.

Data layers are distinct: browser teaching simulations; the shared-model benchmark suite; native source-model checks; and existing literature. Plots and download labels identify their layer. Each numerical claim links to a measurement or a derivation. Actual results are exported from existing CSV files, not transcribed by hand.

## Lessons 5–12: first-principles revision

Each lesson starts with a probability or measurement principle, defines the new terms, works through small numbers, and connects the claim to a metric, task, and source. The recurring two-particle example has positions [−1,1], weights [0.25,0.75], target mean 0.5, and variance 0.75. Its affine output preserves these moments while changing its cubic expectation from 0.5 to 1.25. This counterexample motivates the shape and nonlinear-expectation metrics.

Interactive components include the two-particle contract lab, the shift/collapse/affine metric playground, and the claim-to-task selector. Lesson 11 now uses a paper-style figure walkthrough, with the full explorer available on request. All eight revised lessons contain worked examples. The original measured results retain their provenance and are distinct from the teaching calculations.

## Lesson 11: one figure, one finding

The default result walkthrough has eight fixed figures at N=128: covariance error, mixture sliced W1, linear tracking RMSE, UNGM RMSE, native likelihood-gradient MSE, volatility scale error, bearing-gap recovery fraction, and CPU forward time. Each figure gives the question, a plain metric definition, one plot, a full experimental caption, the best observed mean among all eligible implementations, the eigen result, and a bounded finding. Close or overlapping uncertainty intervals do not establish a superiority claim.

To keep each plot readable, show the five best observed means plus eigen if it is outside that group. State this selection visibly and retain every eligible method, interval, trial count, and failure in an expandable table and CSV. Offer SVG, PDF, and PNG downloads. Narrow screens use separately typeset versions of the same plots. Keep the general explorer in a closed, optional section.

Figures are generated by `website/generate_figures.py` from the original CSV summaries. Both figures and the web export preserve nonzero errors near floating-point precision; small values must not be rounded to zero during serialization.
