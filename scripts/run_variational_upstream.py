"""Run the upstream Variational-Resampling experiments without editing vendor files.

The toy experiment runs unchanged; plt.show() saves each figure instead of opening a window.
main.py runs unchanged except for the truth-particle count. Its 50,000-particle reference keeps
about 3 GB of (N, T) arrays in memory, so the default here is 10,000; pass --truth-particles 50000
to reproduce the original setting on a machine with enough free memory.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'vendor/Variational-Resampling'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/variational_upstream')
    p.add_argument('--truth-particles',type=int,default=10000)
    p.add_argument('--skip-main',action='store_true')
    a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REPO))
    os.chdir(out)

    print('== tests.py ==',flush=True)
    runpy.run_path(str(REPO/'tests.py'),run_name='__main__')

    print('\n== toy_experiment_visualization.py ==',flush=True)
    figures=iter(['toy_samples','toy_lb_gamma','toy_lb_w','toy_multinomial','toy_systematic','toy_stratified'])
    plt.show=lambda *args,**kwargs:(plt.gcf().savefig(next(figures)+'.png',dpi=120,bbox_inches='tight'),plt.close('all'))
    runpy.run_path(str(REPO/'toy_experiment_visualization.py'),run_name='__main__')

    if a.skip_main:return
    print(f'\n== main.py (truth_particles={a.truth_particles}) ==',flush=True)
    source=(REPO/'main.py').read_text()
    original='truth_particles = 50000'
    if source.count(original)!=1:raise SystemExit('main.py changed upstream; truth_particles override no longer applies')
    source=source.replace(original,f'truth_particles = {a.truth_particles}')
    for scheme in ['kl','kl-iw','multinomial','systematic','stratified']:(out/'results'/scheme).mkdir(parents=True,exist_ok=True)
    exec(compile(source,str(REPO/'main.py'),'exec'),{'__name__':'__main__'})


if __name__=='__main__':main()
