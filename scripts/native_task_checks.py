"""Run source-defined tasks independently of the shared benchmark models."""
import argparse
import ast
import csv
import json
from pathlib import Path
import os
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resampler_bench.adapters import paths

paths()


def gaussian_cloud_task(out):
    """Execute the particles book's weighted-Gaussian TV comparison, bounded grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = ROOT / 'vendor/particles/book/resampling/compare_tv_distance_resampling.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    changed = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = getattr(statement.targets[0], 'id', '')
            if name in ('N', 'taus'):
                expression = '1024' if name == 'N' else 'np.linspace(0., 10., 21)'
                statement.value = ast.parse(expression, mode='eval').body
                changed.append(name)
    assert sorted(changed) == ['N', 'taus']
    ast.fix_missing_locations(tree)
    np.random.seed(20261011)
    ns = {'__name__': '__main__', '__file__': str(source)}
    previous = Path.cwd()
    try:
        os.chdir(out)
        exec(compile(tree, str(source), 'exec'), ns)
    finally:
        os.chdir(previous)
    rows = []
    for method, values in ns['results'].items():
        assert values.shape == (100, 21)
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        if method != 'multinomial':
            np.testing.assert_allclose(values[:, 0], 0., atol=1e-12)
        for trial, distances in enumerate(values):
            for tau, distance in zip(ns['taus'], distances):
                rows.append(dict(method=method, trial=trial, tau=tau, n=1024, tv_distance=distance))
    with (out / 'gaussian_cloud_trials.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    plt.close('all')
    return dict(passed=True, trials=len(rows), source=str(source.relative_to(ROOT)),
                settings='native book loop and TV metric; N=1024; 21 tau values in [0,10]; original 100 repetitions; fixed seed 20261011',
                notes='only N and tau-grid assignments bounded; original source uses N=10000 and 500 tau values; original plot percentile conventions retained')


def stone_tutorial(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.random.seed(20261011)
    source = ROOT / 'vendor/Stone-Soup/docs/tutorials/sampling/ResamplingTutorial.py'
    ns = runpy.run_path(str(source))
    for name, n in [('resampled_particles', 5), ('upsampled_particles', 10)]:
        particles = ns[name]
        assert len(particles) == n
        np.testing.assert_allclose(np.asarray(particles.weight, float), 1 / n)
        assert set(np.asarray(particles.state_vector).ravel()).issubset(range(5))
    # The tutorial's equal-weight ESS example should retain its input cloud.
    retained = ns['resampler'].resample(ns['particles'])
    np.testing.assert_array_equal(retained.state_vector, np.arange(5)[None])
    for i, number in enumerate(plt.get_fignums()):
        plt.figure(number).savefig(out / f'stonesoup_tutorial_{i:02d}.png', dpi=100)
    plt.close('all')
    return dict(source=str(source.relative_to(ROOT)), passed=True,
                assertions='native multinomial sizes/weights/support and ESS retention')


def stone_bearing(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = ROOT / 'vendor/Stone-Soup/docs/examples/filters/bearing_only_example.py'
    ns = runpy.run_path(str(source))
    truths = [np.asarray(point, float) for path in ns['groundtruth_paths'].values() for point in path]
    estimates = [np.asarray(point, float) for path in ns['kalman_tracks'].values() for point in path]
    assert len(truths) == 100 and len(estimates) > 0
    assert np.isfinite(truths).all() and np.isfinite(estimates).all()
    plt.close('all')
    return dict(passed=True, source=str(source.relative_to(ROOT)),
                ground_truth_steps=len(truths), tracked_states=len(estimates),
                notes='unmodified moving-sensor bearing-only EKF example; task execution check, not a particle-resampler comparison')


def particles_tasks(out):
    import particles
    from particles import kalman, state_space_models as ssm
    import torch
    from resampler_bench.eigen import eigen_resample

    torch.set_num_threads(1)

    class EigenSMC(particles.SMC):
        """Change only resample_move for native bootstrap filters without history."""
        def resample_move(self):
            assert not self.fk.isAPF and self.hist is None
            self.rs_flag = self.fk.time_to_resample(self)
            self.A = None  # A transform has no discrete ancestors.
            if self.rs_flag:
                shape = self.X.shape
                x = torch.as_tensor(self.X.reshape(self.N, -1), dtype=torch.float64)
                lw = torch.as_tensor(self.aux.lw, dtype=torch.float64)
                with torch.no_grad():
                    self.Xp = eigen_resample(x, lw).numpy().reshape(shape)
                self.reset_weights()
            else:
                self.Xp = self.X
            self.X = self.fk.M(self.t, self.Xp)

    models = dict(linear=kalman.LinearGauss, ungm=ssm.Gordon_etal,
                  volatility=ssm.StochVol, bearing=ssm.BearingsOnly)
    methods = ['multinomial', 'systematic', 'stratified', 'residual', 'ssp', 'eigen']
    records = []
    with (out / 'particles_native_trials.csv').open('w', newline='') as handle:
        writer = None
        for model_name, constructor in models.items():
            model = constructor()  # Unmodified upstream defaults, including indexing.
            for data_seed in range(10):
                np.random.seed(51000 + data_seed)
                truth, observations = model.simulate(50)
                truth = np.asarray(truth).reshape(50, -1)
                exact = None
                if model_name == 'linear':
                    exact = kalman.Kalman(ssm=model, data=observations)
                    exact.filter()
                np.savez(out / f'particles_{model_name}_data_{data_seed}.npz',
                         states=truth, observations=np.asarray(observations))
                for n in [32, 128, 512, 2048]:
                    for method in methods:
                        for pf_seed in range(3):
                            np.random.seed(61000 + 100 * data_seed + pf_seed)
                            cls = EigenSMC if method == 'eigen' else particles.SMC
                            pf = cls(fk=ssm.Bootstrap(ssm=model, data=observations),
                                     N=n, resampling='systematic' if method == 'eigen' else method,
                                     ESSrmin=.5, collect='off', store_history=False)
                            means, variances, ess, flags = [], [], [], []
                            start = time.perf_counter()
                            for _ in pf:
                                x = pf.X.reshape(n, -1)
                                w = pf.wgts.W
                                mean = w @ x
                                means.append(mean)
                                variances.append(w @ (x - mean) ** 2)
                                ess.append(pf.wgts.ESS)
                                flags.append(pf.rs_flag)
                                assert np.isfinite(x).all() and np.isfinite(w).all()
                                np.testing.assert_allclose(w.sum(), 1., atol=1e-10)
                            elapsed = time.perf_counter() - start
                            means, variances = np.asarray(means), np.asarray(variances)
                            assert len(means) == 50 and np.isfinite(pf.logLt)
                            dims = [2, 3] if model_name == 'bearing' else [0]
                            row = dict(model=model_name, method=method, n=n,
                                       data_seed=data_seed, pf_seed=pf_seed, steps=50,
                                       rmse=float(np.sqrt(np.mean((means[:, dims] - truth[:, dims]) ** 2))),
                                       loglik=float(pf.logLt), mean_ess=float(np.mean(ess)),
                                       resampling_events=sum(flags), filter_ms=elapsed * 1000,
                                       kalman_mean_rmse=np.nan, kalman_variance_rmse=np.nan,
                                       kalman_loglik_error=np.nan)
                            if exact is not None:
                                em = np.array([f.mean.item() for f in exact.filt])
                                ev = np.array([f.cov.item() for f in exact.filt])
                                row.update(kalman_mean_rmse=float(np.sqrt(np.mean((means[:, 0]-em)**2))),
                                           kalman_variance_rmse=float(np.sqrt(np.mean((variances[:, 0]-ev)**2))),
                                           kalman_loglik_error=float(pf.logLt-np.sum(exact.logpyt)))
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=row)
                                writer.writeheader()
                            writer.writerow(row)
                            records.append(row)
                handle.flush()
            print(f'Native particles {model_name}: complete', flush=True)
    import pandas as pd
    frame = pd.DataFrame(records)
    assert len(frame) == 2880
    assert not frame.duplicated(['model', 'method', 'n', 'data_seed', 'pf_seed']).any()
    assert (frame.groupby(['model', 'method', 'n']).size() == 30).all()
    frame.groupby(['model', 'method', 'n']).agg(
        trials=('rmse', 'size'), rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        loglik_mean=('loglik', 'mean'), kalman_mean_rmse=('kalman_mean_rmse', 'mean'),
        kalman_variance_rmse=('kalman_variance_rmse', 'mean'),
        kalman_loglik_error=('kalman_loglik_error', 'mean')
    ).to_csv(out / 'particles_native_summary.csv')
    return dict(passed=True, trials=len(frame), datasets=40, models=list(models),
                methods=methods, particle_counts=[32, 128, 512, 2048],
                settings='upstream defaults; T=50; ten datasets x three PF seeds; ESS=0.5N')


def filterflow_tasks(out):
    import tensorflow as tf
    from filterflow.base import State
    from filterflow.models.simple_linear_gaussian import make_filter as linear_filter
    from filterflow.models.stochastic_volatility import make_filter as sv_filter
    from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler
    from filterflow.resampling.differentiable import RegularisedTransform, CorrectedRegularizedTransform
    from filterflow.resampling.criterion import NeffCriterion
    from resampler_bench.data import sequence, kalman

    tf.config.set_visible_devices([], 'GPU')
    rows = []
    # Native SMC.update avoids the legacy tf.data iterator API in full-run wrappers.
    # Model parameters and observations are the suite's scalar shared variants.
    for kind in ['linear', 'volatility']:
        for method, constructor in [('multinomial', MultinomialResampler),
                                    ('systematic', SystematicResampler),
                                    ('stratified', StratifiedResampler),
                                    ('ot', RegularisedTransform),
                                    ('corrected_ot', CorrectedRegularizedTransform)]:
            for data_seed in range(3):
                n = 128 if method not in ('ot', 'corrected_ot') else 32
                tf.random.set_seed(77000 + data_seed)
                rng = np.random.default_rng(88000 + data_seed)
                truth, ys, _ = sequence(kind, data_seed, steps=30)
                resampler = constructor(epsilon=.1, max_iter=100) if 'ot' in method else constructor()
                matrix = lambda x: tf.constant([[x]], tf.float32)
                if kind == 'linear':
                    pf = linear_filter(matrix(1.), matrix(.8), matrix(.8), matrix(.5),
                                       resampler, NeffCriterion(.5, True))
                    sd = 1.
                else:
                    pf = sv_filter(tf.constant([0.]), matrix(.95), matrix(.2), matrix(1.),
                                   resampler, NeffCriterion(.5, True))
                    sd = .2 / np.sqrt(1-.95**2)
                st = State(particles=tf.constant(rng.normal(0, sd, [1, n, 1]), tf.float32))
                estimates = []
                failure = ''
                for t, y in enumerate(ys):
                    seed1 = tf.constant([data_seed, 2*t+1], tf.int32)
                    seed2 = tf.constant([data_seed, 2*t+2], tf.int32)
                    try:
                        st = pf.update(st, tf.constant([y], tf.float32), tf.constant([0.]), seed1, seed2)
                    except tf.errors.OpError as error:
                        failure = type(error).__name__ + ': ' + str(error)
                        break
                    weights = st.weights.numpy()[0]
                    np.testing.assert_allclose(weights.sum(), 1., atol=1e-5)
                    assert np.isfinite(st.particles.numpy()).all()
                    estimates.append(weights @ st.particles.numpy()[0, :, 0])
                ll = float(st.log_likelihoods.numpy()[0]) if not failure else np.nan
                if not failure:
                    assert np.isfinite(ll)
                row = dict(model=kind, method=method, n=n, data_seed=data_seed,
                           steps_completed=len(estimates),
                           rmse=float(np.sqrt(np.mean((np.array(estimates)-truth[:, 0])**2))) if not failure else np.nan,
                           loglik=ll, kalman_mean_rmse=np.nan, kalman_loglik_error=np.nan, error=failure)
                if kind == 'linear' and not failure:
                    em, _, el = kalman(ys)
                    row.update(kalman_mean_rmse=float(np.sqrt(np.mean((estimates-em)**2))),
                               kalman_loglik_error=ll-el)
                rows.append(row)
                with (out / 'filterflow_native_trials.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
            print(f'Native FilterFlow {kind}/{method}: complete', flush=True)
    with (out / 'filterflow_native_trials.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(bool(row['error']) for row in rows)
    return dict(completed=True, passed=not failures, trials=len(rows), numerical_failures=failures,
                settings='native SMC.update and model classes; shared scalar parameters; T=30; three datasets',
                notes='SV observations generated by the shared model, not native SVObservationModel.sample')


def pydpf_tasks(out):
    import pandas as pd
    import pydpf
    from pydpf.filtering import SIS
    import torch

    torch.set_num_threads(1)
    original_simulate = pydpf.simulate_and_save
    original_forward = SIS.forward

    def bounded_simulate(*args, **kwargs):
        kwargs.update(n_trajectories=50, time_extent=20, batch_size=16)
        return original_simulate(*args, **kwargs)

    def bounded_forward(self, n_particles, time_extent, *args, **kwargs):
        return original_forward(self, min(n_particles, 512), min(time_extent, 10), *args, **kwargs)

    # The pinned replication script does not apply --smoke to these expensive
    # dimensions. Bound only sizes; keep its models, PF, resamplers, and metrics.
    pydpf.simulate_and_save = bounded_simulate
    SIS.forward = bounded_forward
    source = ROOT / 'vendor/pydpf/replication_materials/run_experiments.py'
    ns = runpy.run_path(str(source), run_name='native_pydpf_replication')
    saved_argv = sys.argv
    sys.argv = [str(source), '--smoke', '--device', 'cpu', '--batch-size', '16',
                '--experiment', 'kalman', 'sv_filtering',
                '--models', '25', '100', 'DPF', 'Soft', 'Stop-Gradient', 'Optimal Transport',
                '--data-dir', str(out / 'pydpf-bounded-data'),
                '--results-dir', str(out / 'pydpf')]
    try:
        ns['main']()
    finally:
        sys.argv = saved_argv
        pydpf.simulate_and_save = original_simulate
        SIS.forward = original_forward
    # Upstream catches experiment errors without a failing exit code. Check CSVs.
    linear = pd.read_csv(out / 'pydpf/Kalman_comparison_results_smoke.csv', index_col=0)
    linear = linear.loc[['Kalman Filter', 'PF K = 25', 'PF K = 100'],
                        ['Time CPU (s)', 'epsilon x', 'epsilon y']]
    sv = pd.read_csv(out / 'pydpf/fully_specified_results_smoke.csv', index_col=0)
    sv = sv.loc[['DPF', 'Soft', 'Stop-Gradient', 'Optimal Transport'], ['e_x', 'e_l', 'time']]
    assert np.isfinite(linear.to_numpy()).all() and np.isfinite(sv.to_numpy()).all()
    return dict(passed=True, source=str(source.relative_to(ROOT)), result_rows=7,
                settings='bounded native replication smoke; 50 trajectories x 21 states; batch=16; time_extent=10; reference PF=512; candidate SV PF=100; linear PF=25,100',
                notes='runtime wrappers bound sizes missing from upstream smoke handling; no paper-scale or GPU reproduction')


def filterflow_diagnostics(out):
    """Keep original failed tests; independently check their numerical conventions."""
    import tensorflow as tf
    import ot
    from filterflow.resampling.differentiable.regularized_transport.plan import transport
    from filterflow.resampling.differentiable.regularized_transport.utils import diameter, cost
    from filterflow.utils import normalize

    np.random.seed(42)
    x_np = np.random.uniform(-1, 1, [3, 25, 2]).astype(np.float32)
    w = np.random.uniform(0, 1, [3, 25]).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    x, lw = tf.constant(x_np), tf.constant(np.log(w))
    plan = transport(x, lw, tf.constant(.25), tf.constant(.75), tf.constant(1e-5),
                     tf.constant(100), tf.constant(25)).numpy()
    scaled = (x_np - x_np.mean(axis=1, keepdims=True)) / (
        diameter(x, x).numpy()[:, None, None] * np.sqrt(2))
    matching_cost = cost(tf.constant(scaled), tf.constant(scaled)).numpy()
    reference = ot.sinkhorn(np.ones(25)/25, w[0].astype(float)/w[0].sum(),
                            matching_cost[0].astype(float), .25,
                            numItermax=10000, stopThr=1e-10) * 25
    plan_error = float(np.max(np.abs(plan[0] - reference)))
    row_error = float(np.max(np.abs(plan.sum(axis=2) - 1)))
    col_error = float(np.max(np.abs(plan.sum(axis=1) - 25*w)))
    np.testing.assert_allclose(plan[0], reference, atol=1e-3, rtol=1e-3)

    @tf.function
    def objective(logw, objective_scale=1.):
        normalized = normalize(logw, 1, tf.constant(25))
        matrix = transport(x, normalized, tf.constant(.25), tf.constant(.9),
                           tf.constant(1e-5), tf.constant(100), tf.constant(25))
        return objective_scale * tf.reduce_sum(matrix @ x)

    gradient_checks = []
    for objective_scale in [1., .1]:
        for step in [1e-5, 1e-4, 1e-3, 1e-2]:
            analytic, numerical = tf.test.compute_gradient(
                lambda logw: objective(logw, objective_scale), [lw], delta=step)
            error = np.abs(analytic[0]-numerical[0])
            gradient_checks.append(dict(objective_scale=objective_scale, step=step,
                                        max_absolute_error=float(error.max()),
                                        rmse=float(np.sqrt(np.mean(error**2))),
                                        passes_upstream_tolerance=bool(np.allclose(
                                            analytic[0], numerical[0], atol=1e-2, rtol=1e-2))))
    return dict(passed=True, matching_cost_plan_max_error=plan_error,
                row_marginal_max_error=row_error, column_marginal_max_error=col_error,
                weight_gradient_step_sweep=gradient_checks,
                notes='upstream tests remain unchanged and failing; POT comparison here uses native cost, dimension scaling, and epsilon')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['gaussian_cloud', 'stonesoup_tutorial', 'stonesoup_bearing', 'particles', 'filterflow', 'pydpf', 'filterflow_diagnostics'])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/upstream_tasks')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'gaussian_cloud': gaussian_cloud_task, 'stonesoup_tutorial': stone_tutorial,
              'stonesoup_bearing': stone_bearing, 'particles': particles_tasks,
              'filterflow': filterflow_tasks, 'pydpf': pydpf_tasks,
              'filterflow_diagnostics': filterflow_diagnostics}[args.task](args.output)
    (args.output / f'{args.task}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
