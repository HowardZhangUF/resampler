"""Likelihood gradients of differentiable filters against exact Kalman gradients."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resampler_bench.adapters import Adapter
torch.set_num_threads(1)


def kalman(theta,ys):
    q,r=theta.exp();m=theta.new_tensor(0.);p=theta.new_tensor(1.);ll=theta.new_tensor(0.)
    for y in ys:
        pp=p+q*q;s=pp+r*r;residual=y-m
        ll=ll-.5*(torch.log(2*torch.pi*s)+residual*residual/s)
        m=m+pp/s*residual;p=pp*r*r/s
    return ll


def pf(adapter,theta,ys,n,seed):
    torch.manual_seed(seed)
    # Draw all transition noise before the adapter reseeds framework RNGs.
    noise=torch.randn((len(ys)+1,n,1),dtype=theta.dtype)
    x=noise[0];lw=torch.full((n,),-np.log(n),dtype=theta.dtype);ll=theta.new_tensor(0.)
    q,r=theta.exp()
    for i,y in enumerate(ys):
        x=x+q*noise[i+1]
        updated=lw-.5*((y-x[:,0])**2/(r*r)+2*torch.log(r)+np.log(2*np.pi))
        normalizer=torch.logsumexp(updated,0);ll=ll+normalizer;lw=updated-normalizer
        if float(1/lw.detach().exp().square().sum()) < .7*n and i<len(ys)-1:
            x,lw=adapter.torch_call(x,lw,seed*1000+i+715)
            lw=lw-torch.logsumexp(lw,0)
    return ll


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',default='results/standard/torch_native')
    p.add_argument('--methods',nargs='+',help='provider/method pairs; defaults to the original eight');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    # Use the exact same dataset as the native Cuthbert experiment when available.
    native=ROOT/'results/standard/cuthbert_native/data.npz'
    if native.exists():ys=np.load(native)['ys'][:,0]
    else:
        rng=np.random.default_rng(812);x=rng.normal();ys=[]
        for _ in range(50):x+=.4*rng.normal();ys.append(x+.8*rng.normal())
        ys=np.array(ys)
    ys=torch.tensor(ys,dtype=torch.float64)
    true=np.log([.4,.8]);th=torch.tensor(true,requires_grad=True)
    exact_ll=kalman(th,ys);exact=torch.autograd.grad(exact_ll,th)[0].numpy()
    np.savez(out/'data.npz',ys=ys,theta=true,exact_gradient=exact,exact_loglik=exact_ll.detach())
    methods=[('custom','eigen'),('custom','gaussian'),('custom','gumbel'),('pydpf','systematic'),('pydpf','systematic_stop'),('pydpf','soft'),('pydpf','ot'),('torchfilter','soft')]
    if a.methods:methods=[tuple(m.split('/',1)) for m in a.methods]
    rows=[]
    for provider,method in methods:
        adapter=Adapter(provider,method)
        for n in ([32] if a.smoke else ([32,128] if adapter.expensive else [32,128,512,2048])):
            for seed in range(3 if a.smoke else (10 if adapter.expensive else 100)):
                th=torch.tensor(true,requires_grad=True)
                start=time.perf_counter();ll=pf(adapter,th,ys,n,seed);fm=1000*(time.perf_counter()-start)
                start=time.perf_counter();g=torch.autograd.grad(ll,th)[0];bm=1000*(time.perf_counter()-start)
                if not torch.isfinite(g).all():raise FloatingPointError(f'{provider}/{method}: nonfinite gradient')
                g=g.numpy()
                rows.append(dict(provider=provider,method=method,n=n,seed=seed,loglik=float(ll.detach()),exact_loglik=float(exact_ll.detach()),grad_logq=g[0],grad_logr=g[1],exact_grad_logq=exact[0],exact_grad_logr=exact[1],gradient_error=np.linalg.norm(g-exact),forward_ms=fm,backward_ms=bm))
            # Checkpoint after each completed cell.
            with (out/'trials.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            print('Torch PF gradient complete',provider,method,n,flush=True)


if __name__=='__main__':main()
