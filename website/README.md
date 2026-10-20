# Resampler Lab tutorial website

The site teaches particle resampling from first principles and connects each research claim to a test. Its twelve lessons are listed in [the site structure](../docs/WEBSITE-STRUCTURE.md).

The buildless site is in `dist/`. No account, database, or API key is needed for its lessons. All calculations run in the browser. Teaching simulations are explicitly separate from the exported Python benchmark results. Math rendering assets are bundled locally with their license.

Refresh the measured data and downloads from the workspace:

```bash
../.venv/bin/python export_data.py
```

Preview from this folder:

```bash
python -m http.server 8765 --directory dist --bind 127.0.0.1
```

Open `http://127.0.0.1:8765`. Run the JavaScript scientific checks with Node:

```bash
node --test tests/science.test.mjs
```

Browser checks cover the lesson routes, lab controls, result filters, keyboard navigation, narrow screens, and console/network errors. Publication uses the Sites skill and the project identity in `.openai/hosting.json`.

The current checks pass: 18 scientific JavaScript tests and 32 browser check groups. Independent fixtures compare the browser transform with `resampler_bench/eigen.py`, empirical W1 with SciPy, kernel MMD with NumPy, and filtering with an independent Kalman update. Rank-one and repeated-eigenvalue cases are included. The browser check covers all 12 lessons at desktop and 390px mobile widths and records console/network failures.

From the workspace root, the browser check uses the separate QA environment and a running preview server:

```bash
.tools/site-qa/bin/python website/tests/browser_check.py
```

QA artifacts are in `results/website-qa/`. Fixture generation is in `website/tests/make_fixtures.py`; it needs the project's Python environment and runs as a module from the workspace root. Neither the website nor its scientific labs need Python at runtime.

Lessons 5–12 are written in `dist/advanced.js`. Their four added interactions are in `dist/advanced-labs.js`: a two-particle claim lab, metric playground, task selector, and recorded result walkthrough. Each revised lesson starts from a principle and contains a worked example. The smoke command in lesson 12 was executed successfully: all 16 cells completed; see `results/website-lesson-smoke/` in the workspace.

## Paper-style result figures

Lesson 11 uses eight figures rather than making readers configure a plot. Each has a single metric, an experimental caption, best observed mean, eigen comparison, and a plain finding. All methods and unsuccessful attempts remain in expandable tables. The original results explorer is optional and closed by default.

Regenerate shareable SVG/PDF/PNG figures and their source CSVs from the workspace root:

```bash
.venv/bin/python website/generate_figures.py
python -m unittest discover -s website/tests -p 'test_paper_results.py' -v
```

The five figure/data tests check source hashes, all eligible methods, full ranking, retained failures, artifact formats, the separation of native score references, and exact CSV-to-JSON numeric preservation. The browser suite has 32 check groups, including the figures, downloads, optional explorer, and responsive figure versions. Tiny covariance errors remain nonzero in the data export.
