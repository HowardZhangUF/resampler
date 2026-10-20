"""Run providers sequentially to keep memory and timing interference low."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resampler_bench.adapters import METHODS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['smoke','standard'],default='standard')
    p.add_argument('--providers',nargs='+',default=list(METHODS))
    p.add_argument('--stages',nargs='+',default=['static','filter','gradient'])
    p.add_argument('--models',nargs='+',help='Filtering models (default: all models in resampler_bench.run)')
    p.add_argument('--output',default='results/standard')
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--isolate-methods',action='store_true',help='Fresh interpreter per method, useful for process-memory measurements')
    a=p.parse_args()
    out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
    occupied=[provider for provider in a.providers if any((out/provider).glob('*.csv'))]
    if occupied:
        p.error('Output contains trials for '+', '.join(occupied)+'. Choose a fresh --output folder to prevent duplicate observations.')
    env=os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',
               XLA_PYTHON_CLIENT_PREALLOCATE='false',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',CUDA_VISIBLE_DEVICES='')
    previous=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
    manifest=dict(profile=a.profile,providers=list(dict.fromkeys(previous.get('providers',[])+a.providers)),stages=a.stages,jobs=a.jobs,isolate_methods=a.isolate_methods,python=sys.version,platform=platform.platform(),
                  cpu=platform.processor(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),settings=dict(
                      particle_counts=[32,128,512,2048],dimensions=[1,2,4],fast_repeats=100,expensive_repeats=10,
                      expensive_scaling_repeats=3,filter_steps=50,ess_threshold=.5,ot_epsilon=.1,ot_max_iterations=100,
                      gumbel_temperature=.5,soft_alpha=.5,expensive_filter_max_n=128,corrected_ot_max_n=128,
                      gradient_max_n=512,expensive_gradient_max_n=128,device='cpu',timing_includes_adapter=True))
    manifest['settings']['slow_cpu_ot_scaling']='pydpf/filterflow OT at N=2048: Gaussian D=2 only; full static grid through N=512'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def run_provider(provider):
        python=ROOT/('.venv-tf' if provider=='filterflow' else '.venv')/'bin/python'
        log=out/(provider+'.log')
        print('Running',provider,'->',log,flush=True)
        with log.open('w') as f:
            groups=[[method] for method in METHODS[provider]] if a.isolate_methods else [None]
            for methods in groups:
                command=[str(python),'-m','resampler_bench.run','--provider',provider,'--profile',a.profile,
                         '--output',str(out/provider),'--stages',*a.stages]
                if a.models:command+=['--models',*a.models]
                if methods:command+=['--methods',*methods]
                r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                if r.returncode:break
        print(provider,'exit',r.returncode,flush=True)
        if r.returncode: (out/(provider+'-process-failure.json')).write_text(json.dumps(dict(returncode=r.returncode)))
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run_provider,a.providers))
    manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
