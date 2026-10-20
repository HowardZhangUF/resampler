import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.pop('PYTHONPATH',None)
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',CUDA_VISIBLE_DEVICES='')
for provider in ['particles','stonesoup','cuthbert','pfjax','pydpf','torchfilter','filterflow','julia','custom','variational','dsr']:
    python=ROOT/('.venv-tf' if provider=='filterflow' else '.venv')/'bin/python'
    print('Learning:',provider,flush=True)
    with (ROOT/'results'/f'learning-{provider}.log').open('w') as f:
        subprocess.run([str(python),'-m','resampler_bench.learning','--provider',provider],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
