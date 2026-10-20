"""Export measured summaries and reproducibility artifacts for the tutorial."""
import hashlib
import json
from pathlib import Path
import shutil
from summary_io import read_summary

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'website/dist'
DATA = DIST / 'data'
DOWNLOADS = DIST / 'downloads'
DATA.mkdir(parents=True, exist_ok=True)
DOWNLOADS.mkdir(parents=True, exist_ok=True)
names = ['static_summary', 'filter_summary', 'gradient_summary', 'learning_summary',
         'cuthbert_native_summary', 'torch_native_summary']
tables = {}
provenance = {}
for name in names:
    source = ROOT / 'results/standard/report' / (name + '.csv')
    rows = read_summary(source)
    tables[name] = rows
    provenance[name] = {'file': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': len(rows)}
    shutil.copy2(source, DOWNLOADS / source.name)
for name in ['particles_native_summary', 'filterflow_native_trials']:
    source = ROOT / 'results/upstream_tasks' / (name + '.csv')
    rows = read_summary(source)
    for row in rows:
        if 'error' in row:
            row['error'] = 'Native singular-matrix error' if row['error'] else ''
    tables[name] = rows
    provenance[name] = {'file': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': len(rows)}
    shutil.copy2(source, DOWNLOADS / source.name)
audit = json.loads((ROOT / 'results/standard/report/audit.json').read_text())
upstream = json.loads((ROOT / 'results/upstream_tasks/summary.json').read_text())
payload = dict(tables=tables, provenance=provenance, sources=json.loads((ROOT / 'sources.lock.json').read_text()),
               audit=dict(completed_cells=audit['completed_cells'], execution_failures=len(audit['failures']),
                          numerical_failures=len(audit['numerical_trial_failures']), exclusions=len(audit['skips'])),
               upstream=upstream['outcomes'], validation=json.loads((ROOT / 'results/standard/validation.json').read_text()),
               scope='CPU benchmarks; shared and native task results are separate. Static timings are from the serial timing pass.')
(DATA / 'results.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
for source, filename in [('resampler_bench/eigen.py', 'eigen.py'), ('resampler_bench/data.py', 'models.py'),
                         ('resampler_bench/metrics.py', 'metrics.py'), ('README.md', 'PROJECT-README.md'),
                         ('results/standard/report/REPORT.md', 'BENCHMARK-REPORT.md'),
                         ('results/upstream_tasks/README.md', 'NATIVE-CHECKS.md'),
                         ('sources.lock.json', 'sources.lock.json'), ('docs/WEBSITE-STRUCTURE.md', 'WEBSITE-STRUCTURE.md')]:
    shutil.copy2(ROOT / source, DOWNLOADS / filename)
print(json.dumps(dict(tables={key:len(value) for key,value in tables.items()}, bytes=(DATA/'results.json').stat().st_size), indent=2))
