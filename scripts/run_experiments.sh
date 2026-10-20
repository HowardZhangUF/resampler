#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
unset PYTHONPATH
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/check_shared_inputs.py --write results/input-fixtures.npz
.venv-tf/bin/python scripts/check_shared_inputs.py --check results/input-fixtures.npz
.venv/bin/python scripts/run_all.py --profile smoke --output results/smoke
.venv/bin/python scripts/run_all.py --profile standard --jobs 2 --output results/standard
.venv/bin/python scripts/retry_failed_cells.py
.venv/bin/python scripts/cuthbert_gradient.py
.venv/bin/python scripts/torch_gradient.py
.venv/bin/python scripts/run_learning.py
# Timing is deliberately measured after accuracy workers finish.
.venv/bin/python scripts/run_all.py --profile standard --jobs 1 --isolate-methods --stages timing gradient --output results/performance
.venv/bin/python scripts/illustrate_moments.py
.venv/bin/python scripts/validate_results.py
.venv/bin/python scripts/report.py
