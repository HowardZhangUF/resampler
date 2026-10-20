# Native task and upstream-test checks

These results supplement the shared benchmark suite. Source models and native filters are identified in the main [README](../../README.md). Settings, commands, revisions, and script hashes are in [summary.json](summary.json).

| Check | Result |
|---|---|
| gaussian_cloud | Passed; 8,400 trials |
| stonesoup_tutorial | Passed |
| stonesoup_bearing | Passed |
| particles | Passed; 2,880 trials |
| filterflow | 30 trials attempted; 1 numerical failures |
| pydpf | Passed |
| filterflow_diagnostics | Passed |
| stonesoup-tests | 32 passed; 0 failed; 0 errors; 0 skipped |
| torchfilter-tests | 5 passed; 0 failed; 0 errors; 0 skipped |
| pfjax-tests | 5 passed; 0 failed; 0 errors; 0 skipped |
| filterflow-tests | 13 passed; 4 failed; 0 errors; 0 skipped |

## Observed failures

- FilterFlow: two failures come from an obsolete positional `State` constructor in upstream tests.
- FilterFlow: the upstream POT comparison uses different coordinate scaling and regularization conventions. The independent comparison with matching native cost and epsilon is in `filterflow_diagnostics.json`.
- FilterFlow: the original weight-gradient test fails. The native backward clips transport cotangents to [-1,1]. Scaling the objective to avoid clipping and using a larger finite-difference step passes the same tolerances; the original test remains failed.
- FilterFlow native covariance-corrected OT fails on one volatility trial with a singular Riccati solve; that row retains its error and missing metrics in `filterflow_native_trials.csv`.

Native `particles` results use ten datasets × three PF seeds per model/method/N, N=32,128,512,2048, and T=50. The native bearing state order is [vx,vy,px,py]; RMSE uses px and py. Linear errors use the native exact Kalman filter. See `particles_native_summary.csv` and `particles_native_trials.csv`.

The native weighted-Gaussian comparison uses the original book loop and TV metric with N=1024 and 21 tau values (100 repetitions), rather than the original N=10000 and 500 values. `gaussian_cloud_trials.csv` has the raw measurements.

PyDPF uses native replication functions with explicit size bounds because the pinned script does not apply `--smoke` consistently: 50 trajectories, 21 generated states, batch size 16, time_extent=10, reference PF=512. Linear candidates use 25/100 particles; volatility candidates use 100. These are pipeline checks, not paper-scale accuracy or performance reproductions.

The full training/maze tasks and the full Julia statistical stress suite are outside this supplemental check. Earlier Cuthbert native-gradient results and selected Julia checks are linked in the main README.
