"""Finish an already running experiment pipeline and audit all artifacts."""
import argparse
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
if a.wait_pid:
    print('Waiting for accuracy/native/learning pipeline PID',a.wait_pid,flush=True)
    while True:
        try:os.kill(a.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
env=os.environ.copy();env.pop('PYTHONPATH',None)
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',CUDA_VISIBLE_DEVICES='')
python=ROOT/'.venv/bin/python'
commands=[['scripts/retry_failed_cells.py'],
          ['scripts/run_all.py','--profile','standard','--jobs','1','--isolate-methods','--stages','timing','gradient','--output','results/performance'],
          ['scripts/illustrate_moments.py'],['scripts/report.py'],['scripts/validate_results.py']]
for command in commands:
    print('Finishing:',*command,flush=True)
    subprocess.run([str(python),*command],cwd=ROOT,env=env,check=True)
print('All experiments, reports, and validation complete.',flush=True)
