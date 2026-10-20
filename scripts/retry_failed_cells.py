"""Retry failed cells while retaining successful cells and source algorithms.

The current runner records per-trial numerical failures, so one non-finite draw
does not discard every successful trial from the same configuration.
"""
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.pop('PYTHONPATH',None)
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',CUDA_VISIBLE_DEVICES='')
for folder in sorted((ROOT/'results/standard').iterdir()):
    path=folder/'status.jsonl'
    if not path.exists():continue
    states={}
    for line in path.read_text().splitlines():
        row=json.loads(line);states[(row['method'],row['stage'],row['case'])]=row
    groups={(r['method'],r['stage']) for r in states.values() if r['status']=='failed' and r['stage'] in ('static','filter','gradient')}
    for method,stage in sorted(groups):
        print('Retrying incomplete cells:',folder.name,method,stage,flush=True)
        python=ROOT/('.venv-tf' if folder.name=='filterflow' else '.venv')/'bin/python'
        with (ROOT/'results'/f'retry-{folder.name}-{method}-{stage}.log').open('w') as f:
            subprocess.run([str(python),'-m','resampler_bench.run','--provider',folder.name,'--methods',method,'--stages',stage,'--resume','--output',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
