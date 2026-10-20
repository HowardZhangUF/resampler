"""Run the native Cuthbert tutorial against its exact native Kalman filter.

Extract only the named definition cells from the pinned upstream Markdown.
No approximate gradient reference or local substitute PF is used here.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/cuthbert'))
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',default='results/standard/cuthbert_native');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    md=(ROOT/'vendor/cuthbert/docs/examples/diff_pf_resampling.md').read_text()
    cells=dict(re.findall(r'^```\{\.python #([^}]+)\}\n(.*?)\n^```',md,re.S|re.M))
    ns={}
    for cell in ['pf_grad_bias-imports','pf_grad_bias-simulate','pf_grad_bias-kalman_mll','pf_grad_bias-pf_mll']:
        exec(compile(cells[cell],str(ROOT/'vendor/cuthbert/docs/examples/diff_pf_resampling.md'), 'exec'),ns)
    theta,ys=ns['theta_true'],ns['ys'];base=dict(m0=ns['m0'],chol_P0=ns['chol_P0'])
    exact_ll=float(ns['kalman_mll'](theta,ys,**base));exact=np.array(ns['kalman_grad'](theta,ys,**base))
    np.savez(out/'data.npz',theta=theta,ys=ys,xs=ns['xs'],exact_gradient=exact,exact_loglik=exact_ll)
    rows=[]
    for n in ([32] if a.smoke else [32,128,512,2048]):
        for diff in [False,True]:
            def fn(th,k):return ns['pf_mll'](th,ys,key=k,n_particles=n,ess_threshold=.7,differentiable_resampling=diff,**base)
            value=jax.jit(fn); value_grad=jax.jit(jax.value_and_grad(fn))
            key=jax.random.PRNGKey(999)
            value(theta,key).block_until_ready();jax.block_until_ready(value_grad(theta,key))
            for seed in range(3 if a.smoke else 100):
                key=jax.random.PRNGKey(seed)
                start=time.perf_counter();value(theta,key).block_until_ready();forward_ms=1000*(time.perf_counter()-start)
                start=time.perf_counter();ll,g=value_grad(theta,key);jax.block_until_ready(g);total_ms=1000*(time.perf_counter()-start)
                g=np.array(g)
                rows.append(dict(n=n,stop_gradient=diff,seed=seed,loglik=float(ll),exact_loglik=exact_ll,
                                 grad_logq=g[0],grad_logr=g[1],exact_grad_logq=exact[0],exact_grad_logr=exact[1],
                                 gradient_error=np.linalg.norm(g-exact),forward_ms=forward_ms,forward_backward_ms=total_ms))
            print('Native Cuthbert gradient complete',n,diff,flush=True)
    with (out/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    # Native tutorial also compares profile likelihood and gradient slices.
    profiles=[]
    if not a.smoke:
        for dimension in range(2):
            grid=jnp.linspace(theta[dimension]-1.2,theta[dimension]+1.2,21)
            exact_fn=jax.jit(jax.value_and_grad(lambda th:ns['kalman_mll'](th,ys,**base)))
            for diff in [False,True]:
                def native(th,k):return ns['pf_mll'](th,ys,key=k,n_particles=3000,ess_threshold=.7,differentiable_resampling=diff,**base)
                native=jax.jit(jax.value_and_grad(native))
                for v in grid:
                    th=theta.at[dimension].set(v); el,eg=exact_fn(th);ll,g=native(th,jax.random.PRNGKey(41))
                    profiles.append(dict(parameter=['logq','logr'][dimension],value=float(v),stop_gradient=diff,loglik=float(ll),exact_loglik=float(el),gradient=float(g[dimension]),exact_gradient=float(eg[dimension]),n=3000))
        with (out/'profiles.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=profiles[0]);writer.writeheader();writer.writerows(profiles)


if __name__=='__main__':main()
