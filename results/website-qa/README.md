# Tutorial website verification

The website uses small JavaScript teaching models and exports the completed Python benchmark summaries. Its tests check the teaching calculations and website controls. They do not replace or rerun the benchmark suite.

- Scientific checks: **18 passed** with Node 22.16.0 (`website/tests/science.test.mjs`).
- Browser checks: **32 groups passed** with Playwright 1.55.0 and local Google Chrome; see [the recorded report](browser-report.json).
- Browser errors: **0**. Failed local requests: **0**.
- All 12 lessons were checked at 1440px and 390px. Checks cover live controls, all measured-data tabs and metrics, CSV export, saved progress, glossary, quizzes, keyboard search, mobile navigation, and local downloads.
- Independent fixtures use the project PyTorch eigen implementation, SciPy Wasserstein distance, NumPy Gaussian-kernel sums, and a separate Kalman recursion. Cases include full rank, rank one, and repeated covariance eigenvalues. Analytic 1D gradients are compared with central finite differences.

Screenshots: [desktop introduction](desktop-start.png), [desktop results](desktop-results.png), and [mobile eigen lesson](mobile-eigen.png).

The browser verification ran against the local static preview. The hosted private site is verified separately through its deployment status.

The lessons 5–12 revision adds checks for the two-particle contract, all metric-playground cases, all six task-guide selections, the measured-evidence tour, and first-principles/worked-example content in every revised lesson. The documented eigen smoke command completed 16 cells without execution or numerical failures. Outputs are in `results/website-lesson-smoke/`; they are verification data, separate from the completed standard benchmark export.

The paper-style lesson 11 adds eight fixed, single-metric figures and five source/data integrity tests. Those tests verify the best means against the complete original CSV slices, source hashes, full result tables, unsuccessful attempts, native-score separation, artifact formats, and exact floating-point preservation in the web export. Desktop and mobile plots have separate typesetting and share the same measurements. The latest browser report has 32 passed groups with no console or local-request failures.
