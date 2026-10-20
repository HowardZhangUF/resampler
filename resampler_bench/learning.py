"""Stochastic-volatility parameter estimation and finite-difference scores.

All providers use the same observation-scale grid and common random numbers.
A numerical grid filter supplies an independent, approximate likelihood reference.
"""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from .adapters import Adapter
from .data import sequence,initial,propagate,log_observation

REPRESENTATIVE={'particles':['systematic'],'stonesoup':['systematic'],'cuthbert':['systematic_stop'],
                'pfjax':['ot'],'pydpf':['ot'],'torchfilter':['soft'],'filterflow':['ot'],
                'julia':['systematic'],'custom':['eigen','gaussian','gumbel'],'variational':['kl'],'dsr':['dsr']}


def grid_likelihood(ys, scale, size=1001):
    z=np.linspace(-4.,4.,size);dx=z[1]-z[0]
    transition=norm.pdf((z[:,None]-.95*z[None,:])/.2)/.2*dx
    p=norm.pdf(z,scale=.2/np.sqrt(1-.95**2))*dx;p/=p.sum()
    ll=0.
    for y in ys:
        p=transition@p
        lw=np.log(np.maximum(p,np.finfo(float).tiny))+log_observation('volatility',z[:,None],y/scale)-np.log(scale)
        total=logsumexp(lw);ll+=total;p=np.exp(lw-total)
    return ll


def likelihood(adapter,ys,scale,seed,n=128):
    rng=np.random.default_rng(9110+seed);x=initial('volatility',rng,n);w=np.ones(n)/n;ll=0.
    for i,y in enumerate(ys):
        x=propagate('volatility',x,rng,i+1)
        lw=np.log(np.maximum(w,np.finfo(float).tiny))+log_observation('volatility',x,y/scale)-np.log(scale)
        total=logsumexp(lw);ll+=total;w=np.exp(lw-total)
        if 1/np.sum(w*w)<.5*n:x,w=adapter(x,w,seed*1000+i+917)
    return ll


def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',required=True);p.add_argument('--output',default='results/standard');a=p.parse_args()
    out=Path(a.output)/a.provider;out.mkdir(parents=True,exist_ok=True)
    grid=np.array([.6,.8,1.,1.2,1.4]);delta=.01
    rows=[];profiles=[]
    for method in REPRESENTATIVE[a.provider]:
        adapter=Adapter(a.provider,method)
        adapter.record_ancestors=False
        wx=initial('volatility',np.random.default_rng(0),128);adapter(wx,np.ones(128)/128,0)
        for seed in range(10):
            _,ys,_=sequence('volatility',seed,50)
            # Cache numerical reference once so all repositories share it.
            cache=Path(a.output)/'volatility_reference';cache.mkdir(exist_ok=True)
            reference=cache/f'{seed}.npz'
            if reference.exists():data=np.load(reference);reference_ll=data['loglik'];ref_score=float(data['score']);grid_check=float(data['grid_check'])
            else:
                reference_ll=np.array([grid_likelihood(ys,s) for s in grid])
                ref_score=(grid_likelihood(ys,np.exp(delta))-grid_likelihood(ys,np.exp(-delta)))/(2*delta)
                grid_check=grid_likelihood(ys,1.,2001)-reference_ll[2]
                np.savez(reference,loglik=reference_ll,score=ref_score,grid_check=grid_check,grid=grid,ys=ys)
            start=time.perf_counter();ll=np.array([likelihood(adapter,ys,s,seed) for s in grid])
            score=(likelihood(adapter,ys,np.exp(delta),seed)-likelihood(adapter,ys,np.exp(-delta),seed))/(2*delta)
            estimate=grid[np.argmax(ll)];reference_estimate=grid[np.argmax(reference_ll)]
            rows.append(dict(provider=a.provider,method=method,n=128,data_seed=seed,scale_estimate=estimate,true_scale=1.,
                             parameter_absolute_error=abs(estimate-1.),reference_grid_mle=reference_estimate,mle_reference_error=abs(estimate-reference_estimate),
                             likelihood_score=score,reference_score=ref_score,score_error=score-ref_score,reference_grid_check=grid_check,learning_ms=1000*(time.perf_counter()-start)))
            for s,v,r in zip(grid,ll,reference_ll):profiles.append(dict(provider=a.provider,method=method,n=128,data_seed=seed,scale=s,loglik=v,reference_loglik=r))
            for name,values in [('learning.csv',rows),('learning_profiles.csv',profiles)]:
                with (out/name).open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=values[0]);writer.writeheader();writer.writerows(values)
            print(a.provider,method,'SV learning complete',seed,flush=True)
        if a.provider=='julia':adapter.process.stdin.close();adapter.process.wait()


if __name__=='__main__':main()
