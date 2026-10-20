"""Run and report source-defined tasks and selected upstream tests on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/upstream_tasks')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    main_python = ROOT / '.venv/bin/python'
    tf_python = ROOT / '.venv-tf/bin/python'
    environment = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                       TF_CPP_MIN_LOG_LEVEL='2', MPLBACKEND='Agg', JAX_PLATFORMS='cpu',
                       JAX_ENABLE_X64='true', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                       PYTHONUNBUFFERED='1', PYTHONPATH='')
    commands = {}
    artifacts = {}
    for task in ['gaussian_cloud', 'stonesoup_tutorial', 'stonesoup_bearing', 'particles', 'filterflow', 'pydpf', 'filterflow_diagnostics']:
        python = tf_python if task.startswith('filterflow') else main_python
        commands[task] = [str(python), str(ROOT / 'scripts/native_task_checks.py'), task, '--output', str(out)]
        artifacts[task] = out / f'{task}.json'
    tests = {
        'stonesoup': ('Stone-Soup', ['vendor/Stone-Soup/stonesoup/resampler/tests/test_particle.py']),
        'torchfilter': ('torchfilter', ['vendor/torchfilter/tests/test_filters.py', '-k', 'particle_filter']),
        'pfjax': ('pfjax/src', ['vendor/pfjax/tests/test_resample_mvn.py', 'vendor/pfjax/tests/test_resample_ot.py']),
        'filterflow': ('filterflow', ['vendor/filterflow/tests/resampling/standard',
                                    'vendor/filterflow/tests/resampling/test_criterion.py',
                                    'vendor/filterflow/tests/resampling/differentiable/optimal_transport/test_sinkhorn.py',
                                    '-k', 'not penalty']),
    }
    for provider, (_, selections) in tests.items():
        name = provider + '-tests'
        python = tf_python if provider == 'filterflow' else main_python
        commands[name] = [str(python), '-m', 'pytest', '-q', *selections,
                          f'--junitxml={out / (name + ".xml")}']
        artifacts[name] = out / (name + '.xml')
    if not args.summarize_only and not args.resume:
        occupied = [str(path) for path in artifacts.values() if path.exists()]
        if occupied:
            raise SystemExit('Choose a new output folder, or use --resume: ' + ', '.join(occupied))
    statuses = {}
    for name, command in commands.items():
        if args.summarize_only or (args.resume and artifacts[name].exists()):
            continue
        env = os.environ.copy()
        env.update(environment)
        if name.endswith('-tests'):
            provider = name.removesuffix('-tests')
            env['PYTHONPATH'] = str(ROOT / 'vendor' / tests[provider][0])
            if provider == 'torchfilter':
                env['PYTHONPATH'] += os.pathsep + str(ROOT / 'vendor/torchfilter/tests')
        print(f'Running {name}', flush=True)
        with (out / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        statuses[name] = result.returncode
        # A failing upstream test must not prevent later checks from running.
    outcomes = {}
    failures = []
    for name, artifact in artifacts.items():
        if not artifact.exists():
            outcomes[name] = dict(completed=False, returncode=statuses.get(name))
            failures.append(name + ': missing result')
        elif name.endswith('-tests'):
            suites = ET.parse(artifact).getroot().iter('testsuite')
            counts = {key: 0 for key in ['tests', 'failures', 'errors', 'skipped']}
            for suite in suites:
                for key in counts:
                    counts[key] += int(suite.get(key, '0'))
            counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
            outcomes[name] = dict(completed=True, **counts)
            if counts['failures'] or counts['errors']:
                failures.append(name)
        else:
            outcomes[name] = json.loads(artifact.read_text())
            if not outcomes[name].get('passed', False):
                failures.append(name)
    sources = json.loads((ROOT / 'sources.lock.json').read_text())
    for source in sources:
        actual = subprocess.check_output(['git', '-C', str(ROOT / source['path']), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != source['commit']:
            failures.append(source['repository'] + ': wrong revision')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ['scripts/native_task_checks.py', 'scripts/check_upstream_tasks.py']}
    manifest = dict(outcomes=outcomes, checks_with_failures=failures, sources=sources,
                    commands=commands, environment=environment, script_sha256=hashes,
                    notes='CPU only; bounded native tasks supplement the separate shared standard suite. Test failures remain failures.')
    (out / 'summary.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# Native task and upstream-test checks', '',
             'These results supplement the shared benchmark suite. Source models and native filters are identified in the main [README](../../README.md). Settings, commands, revisions, and script hashes are in [summary.json](summary.json).', '',
             '| Check | Result |', '|---|---|']
    for name, outcome in outcomes.items():
        if name.endswith('-tests') and outcome.get('completed'):
            description = f"{outcome['passed']} passed; {outcome['failures']} failed; {outcome['errors']} errors; {outcome['skipped']} skipped"
        elif 'numerical_failures' in outcome:
            description = f"{outcome['trials']} trials attempted; {outcome['numerical_failures']} numerical failures"
        else:
            description = ('Passed' if outcome.get('passed') else 'Incomplete/failed')
            if 'trials' in outcome:
                description += f"; {outcome['trials']:,} trials"
        lines.append(f'| {name} | {description} |')
    lines.extend(['', '## Observed failures', '',
                  '- FilterFlow: two failures come from an obsolete positional `State` constructor in upstream tests.',
                  '- FilterFlow: the upstream POT comparison uses different coordinate scaling and regularization conventions. The independent comparison with matching native cost and epsilon is in `filterflow_diagnostics.json`.',
                  '- FilterFlow: the original weight-gradient test fails. The native backward clips transport cotangents to [-1,1]. Scaling the objective to avoid clipping and using a larger finite-difference step passes the same tolerances; the original test remains failed.',
                  '- FilterFlow native covariance-corrected OT fails on one volatility trial with a singular Riccati solve; that row retains its error and missing metrics in `filterflow_native_trials.csv`.', '',
                  'Native `particles` results use ten datasets × three PF seeds per model/method/N, N=32,128,512,2048, and T=50. The native bearing state order is [vx,vy,px,py]; RMSE uses px and py. Linear errors use the native exact Kalman filter. See `particles_native_summary.csv` and `particles_native_trials.csv`.', '',
                  'The native weighted-Gaussian comparison uses the original book loop and TV metric with N=1024 and 21 tau values (100 repetitions), rather than the original N=10000 and 500 values. `gaussian_cloud_trials.csv` has the raw measurements.', '',
                  'PyDPF uses native replication functions with explicit size bounds because the pinned script does not apply `--smoke` consistently: 50 trajectories, 21 generated states, batch size 16, time_extent=10, reference PF=512. Linear candidates use 25/100 particles; volatility candidates use 100. These are pipeline checks, not paper-scale accuracy or performance reproductions.', '',
                  'The full training/maze tasks and the full Julia statistical stress suite are outside this supplemental check. Earlier Cuthbert native-gradient results and selected Julia checks are linked in the main README.'])
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(outcomes=outcomes, checks_with_failures=failures), indent=2))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
