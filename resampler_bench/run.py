"""Run each repository in its own process; checkpoint raw trials after each cell."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2
from .adapters import Adapter, METHODS, ROOT
from .data import cloud, initial, propagate, log_observation, sequence, kalman
from .metrics import distribution

MODELS=['linear','ungm','volatility','bearing','lorenz63','lorenz96']
STATE_DIM={'bearing':4,'lorenz63':3,'lorenz96':10}


def append_csv(path, rows):
    if not rows: return
    exists = path.exists()
    fields=list(rows[0])
    if exists:
        with path.open(newline='') as f:fields=next(csv.reader(f))
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerows(rows)


def status(out, provider, method, stage, case, error=None):
    row = dict(provider=provider, method=method, stage=stage, case=case, status='failed' if error else 'complete', error=str(error) if error else '')
    with (out/'status.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def numerical_failure(out,adapter,stage,case,seed,error):
    with (out/'numerical_failures.jsonl').open('a') as f:
        f.write(json.dumps(dict(provider=adapter.provider,method=adapter.method,stage=stage,case=case,seed=seed,error=str(error)))+'\n')


def already_done(args,adapter,stage,case):
    return (adapter.method,stage,case) in args.done


def repetitions(adapter, n, args):
    if args.profile=='smoke': return 3
    if adapter.expensive: return 10 if n<=128 else 3
    return 100


def static(adapter, args, out):
    adapter.record_ancestors=True
    for n in ([32] if args.profile=='smoke' else [32, 128, 512, 2048]):
        for dim in ([1, 2] if args.profile=='smoke' else [1, 2, 4]):
            for kind in ['fixed', 'gaussian', 'mixture', 'skewed']:
                case = f'{kind}/N{n}/D{dim}'
                if already_done(args,adapter,'static',case):continue
                if adapter.provider in ('pydpf','filterflow') and adapter.method=='ot' and n==2048 and (dim!=2 or kind!='gaussian'):
                    with (out/'skips.jsonl').open('a') as f:
                        f.write(json.dumps(dict(provider=adapter.provider,method=adapter.method,stage='static',case=case,reason='Dense CPU OT at N=2048 uses Gaussian D=2 as a scaling case; full distribution grid stops at N=512.'))+'\n')
                    continue
                # Corrected transport solves dense 2N x 2N matrix equations.
                if adapter.method=='corrected_ot' and n>128:
                    with (out/'skips.jsonl').open('a') as f:
                        f.write(json.dumps(dict(provider=adapter.provider, method=adapter.method, stage='static', case=case, reason='Covariance-corrected OT capped at N=128; cubic dense matrix solver.'))+'\n')
                    continue
                try:
                    x, w = cloud(kind, n, dim)
                    try:adapter(x, w, 999)  # Compile/warm up outside timing.
                    except FloatingPointError:pass
                    rows, counts = [], []
                    for seed in range(repetitions(adapter, n, args)):
                        start=time.perf_counter()
                        try:
                            y,v=adapter(x,w,seed);elapsed=time.perf_counter()-start
                            metrics=distribution(x,w,y,v,kind)
                        except FloatingPointError as e:
                            elapsed=time.perf_counter()-start
                            numerical_failure(out,adapter,'static',case,seed,e)
                            metrics=dict.fromkeys(['sliced_w1','rff_mmd2','mean_error','covariance_error','eigenvalue_error','estimator_mse_x','estimator_mse_x2','estimator_mse_sin','mode_coverage','mode_mass_error'],np.nan)
                        row=dict(provider=adapter.provider, method=adapter.method, cloud=kind, n=n, dim=dim, seed=seed,
                                 forward_ms=1000*elapsed, process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                                 **metrics)
                        rows.append(row)
                        if adapter.last_indices is not None and np.isfinite(metrics['mean_error']): counts.append(np.bincount(adapter.last_indices, minlength=n))
                    append_csv(out/'static.csv', rows)
                    if counts:
                        c=np.asarray(counts)
                        append_csv(out/'offspring.csv', [dict(provider=adapter.provider, method=adapter.method, cloud=kind,n=n,dim=dim,repetitions=len(c),
                                   mean_count_variance=np.var(c,axis=0,ddof=1).mean(), count_bias_rmse=np.sqrt(np.mean((c.mean(axis=0)-n*(.5*w+.5/n if adapter.method=='soft' else w))**2)),
                                   target='soft proposal' if adapter.method=='soft' else 'input weights')])
                    status(out, adapter.provider, adapter.method, 'static', case)
                except Exception as e:
                    traceback.print_exc(); status(out, adapter.provider, adapter.method, 'static', case, e)


def filter_once(adapter, kind, n, seed, steps):
    # 10 distinct observation sequences, repeated filter random seeds per sequence.
    data_seed = seed//10
    truth, ys, mask=sequence(kind, data_seed, steps)
    rng=np.random.default_rng(181000+seed)
    x=initial(kind, rng, n); w=np.full(n,1/n)
    means, variances, ess, covered, counts, step_errors = [], [], [], [], [], []
    total_ll, resamples=0.,0
    start=time.perf_counter()
    for i, y in enumerate(ys):
        x=propagate(kind, x, rng, i+1)
        if mask[i]:
            lw=np.log(np.maximum(w,np.finfo(float).tiny))+log_observation(kind,x,y)
            norm=logsumexp(lw); total_ll += norm
            w=np.exp(lw-norm)
        m=w@x; a=x-m; cov=a.T@(w[:, None]*a)
        means.append(m); variances.append(cov)
        e=truth[i]-m
        # Joint ellipsoid coverage; population covariance. Pseudoinverse ridge only here.
        covered.append(e@np.linalg.solve(cov+1e-8*np.eye(len(m)),e)<=chi2.ppf(.95,len(m)))
        step_errors.append(np.linalg.norm(e[[0, 2]]) if kind=='bearing' else abs(e[0]))
        neff=1/np.sum(w*w); ess.append(neff)
        if neff < .5*n:
            x,w=adapter(x,w,seed*10000+i+900); resamples+=1
    elapsed=time.perf_counter()-start
    means=np.asarray(means); variances=np.asarray(variances)
    row=dict(provider=adapter.provider,method=adapter.method,model=kind,n=n,seed=seed,data_seed=data_seed,steps=steps,
             rmse=np.sqrt(np.mean((means[:, [0,2]]-truth[:, [0,2]])**2)) if kind=='bearing' else np.sqrt(np.mean((means-truth)**2)),
             loglik=total_ll, predictive_nll=-total_ll/np.sum(mask),coverage95=np.mean(covered),
             mean_ess_fraction=np.mean(ess)/n,resampling_steps=resamples,filter_ms=1000*elapsed,
             posterior_mean_rmse=np.nan,posterior_covariance_error=np.nan,loglik_error=np.nan,
             recovery_steps=np.nan,recovered=np.nan,post_occlusion_rmse=np.nan)
    if kind=='linear':
        exact_m, exact_p, exact_ll=kalman(ys)
        row.update(posterior_mean_rmse=np.sqrt(np.mean((means[:,0]-exact_m)**2)),posterior_covariance_error=np.mean(np.abs(variances[:,0,0]-exact_p)),loglik_error=total_ll-exact_ll)
    if kind=='bearing' and steps>32:
        e=np.asarray(step_errors); recovery=steps-30; recovered=0
        for j in range(30, steps-2):
            if np.all(e[j:j+3]<2.): recovery=j-30; recovered=1; break
        row.update(recovery_steps=recovery,recovered=recovered,post_occlusion_rmse=np.sqrt(np.mean(e[30:]**2)))
    return row


def filtering(adapter,args,out):
    adapter.record_ancestors=False
    ns = [32] if args.profile=='smoke' else ([32,128] if adapter.expensive else [32,128,512,2048])
    for n in ns:
        for kind in args.models:
            case=f'{kind}/N{n}'
            if already_done(args,adapter,'filter',case):continue
            try:
                # Warm the actual dimensions, including the 4D tracker.
                wx,ww=cloud('gaussian',n,STATE_DIM.get(kind,1)); adapter(wx,ww,999)
                rows=[]
                for raw_seed in range(repetitions(adapter,n,args)):
                    seed=raw_seed*10 if adapter.expensive and args.profile!='smoke' else raw_seed
                    steps=12 if args.profile=='smoke' else 50
                    start=time.perf_counter()
                    try:row=filter_once(adapter,kind,n,seed,steps)
                    except FloatingPointError as e:
                        numerical_failure(out,adapter,'filter',case,seed,e)
                        row=dict(provider=adapter.provider,method=adapter.method,model=kind,n=n,seed=seed,data_seed=seed//10,steps=steps,
                                 **dict.fromkeys(['rmse','loglik','predictive_nll','coverage95','mean_ess_fraction','resampling_steps','filter_ms',
                                                  'posterior_mean_rmse','posterior_covariance_error','loglik_error','recovery_steps','recovered','post_occlusion_rmse'],np.nan))
                        row['filter_ms']=1000*(time.perf_counter()-start)
                    rows.append(row)
                append_csv(out/'filter.csv',rows); status(out,adapter.provider,adapter.method,'filter',case)
            except Exception as e:
                traceback.print_exc(); status(out,adapter.provider,adapter.method,'filter',case,e)


def gradients(adapter,args,out):
    if adapter.provider not in ('custom','cuthbert','pfjax','pydpf','torchfilter','filterflow','dsr'): return
    for n in ([32] if args.profile=='smoke' else [32,128,512]):
        if adapter.expensive and n>128: continue
        for kind in ['gaussian','mixture']:
            x,w=cloud(kind,n,2)
            for fn in ['x2','sin']:
                case=f'{kind}/{fn}/N{n}'
                if already_done(args,adapter,'gradient',case):continue
                try:
                    values=np.sum(x*x,axis=1) if fn=='x2' else np.sum(np.sin(x),axis=1)
                    exact=np.sum(w*(values-w@values)*x[:,0])
                    base=np.log(w)
                    if adapter.provider in ('cuthbert','pfjax'):
                        jax,jnp=adapter.jax,adapter.jnp
                        def loss(theta,key):
                            lw=jnp.asarray(base)+theta*jnp.asarray(x[:,0]); lw=lw-jax.scipy.special.logsumexp(lw)
                            y,v=adapter.jax_fn(key,jnp.asarray(x),lw)
                            f=jnp.sum(y*y,axis=1) if fn=='x2' else jnp.sum(jnp.sin(y),axis=1)
                            return jnp.sum(jax.nn.softmax(v)*f)
                        jf=jax.jit(loss); jg=jax.jit(jax.grad(loss))
                        def compute(seed):
                            key=jax.random.PRNGKey(seed)
                            t=time.perf_counter(); jf(0.,key).block_until_ready(); fm=1000*(time.perf_counter()-t)
                            t=time.perf_counter(); g=jg(0.,key); g.block_until_ready(); bm=1000*(time.perf_counter()-t)
                            return float(g),fm,bm,'forward_and_backward'
                    elif adapter.provider=='filterflow':
                        tf=adapter.tf
                        def compute(seed):
                            theta=tf.Variable(0.,dtype=tf.float32)
                            with tf.GradientTape() as tape:
                                t=time.perf_counter()
                                lw=tf.convert_to_tensor(base,tf.float32)+theta*tf.convert_to_tensor(x[:,0],tf.float32)
                                lw=lw-tf.reduce_logsumexp(lw)
                                y,v=adapter.tf_fn(tf.convert_to_tensor(x,tf.float32),lw,tf.constant([seed,0],tf.int32))
                                f=tf.reduce_sum(y*y,axis=1) if fn=='x2' else tf.reduce_sum(tf.sin(y),axis=1)
                                value=tf.reduce_sum(tf.nn.softmax(v)*f); value.numpy(); fm=1000*(time.perf_counter()-t)
                            t=time.perf_counter(); g=tape.gradient(value,theta); bm=1000*(time.perf_counter()-t)
                            return float(g) if g is not None else 0.,fm,bm,'backward_only'
                    else:
                        torch=adapter.torch
                        def compute(seed):
                            theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
                            t=time.perf_counter()
                            lw=torch.as_tensor(base)+theta*torch.as_tensor(x[:,0]); lw=lw-torch.logsumexp(lw,0)
                            y,v=adapter.torch_call(torch.as_tensor(x),lw,seed)
                            f=(y*y).sum(1) if fn=='x2' else y.sin().sum(1)
                            value=(v.softmax(0)*f).sum(); fm=1000*(time.perf_counter()-t)
                            t=time.perf_counter()
                            g=torch.autograd.grad(value,theta,allow_unused=True)[0] if value.requires_grad else None
                            bm=1000*(time.perf_counter()-t)
                            return float(g) if g is not None else 0.,fm,bm,'backward_only'
                    compute(999)
                    rows=[]
                    for seed in range(repetitions(adapter,n,args)):
                        g,fm,bm,scope=compute(seed)
                        if not np.isfinite(g): raise FloatingPointError('Non-finite gradient')
                        rows.append(dict(provider=adapter.provider,method=adapter.method,cloud=kind,function=fn,n=n,seed=seed,
                                         gradient=g,exact_gradient=exact,error=g-exact,squared_error=(g-exact)**2,forward_ms=fm,backward_ms=bm,backward_scope=scope))
                    append_csv(out/'gradient.csv',rows); status(out,adapter.provider,adapter.method,'gradient',case)
                except Exception as e:
                    traceback.print_exc(); status(out,adapter.provider,adapter.method,'gradient',case,e)


def timing(adapter,args,out):
    """A serial pass that measures each fixed-cloud wrapper without metrics."""
    adapter.record_ancestors=False
    for n in [32,128,512,2048]:
        if adapter.method=='corrected_ot' and n>128:continue
        for dim in [1,2,4]:
            for kind in ['fixed','gaussian','mixture','skewed']:
                if adapter.provider in ('pydpf','filterflow') and adapter.method=='ot' and n==2048 and (dim!=2 or kind!='gaussian'):continue
                x,w=cloud(kind,n,dim)
                adapter(x,w,900)
                rows=[]
                for seed in range(3 if adapter.expensive and n>128 else (10 if adapter.expensive else 100)):
                    start=time.perf_counter();adapter(x,w,seed);ms=1000*(time.perf_counter()-start)
                    row=dict(provider=adapter.provider,method=adapter.method,cloud=kind,n=n,dim=dim,seed=seed,
                             forward_ms=ms,process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                    if adapter.provider=='julia':
                        lines=Path(f'/proc/{adapter.process.pid}/status').read_text().splitlines()
                        row['worker_peak_rss_mb']=float(next(line.split()[1] for line in lines if line.startswith('VmHWM:')))/1024
                    rows.append(row)
                append_csv(out/'timing.csv',rows)
                status(out,adapter.provider,adapter.method,'timing',f'{kind}/N{n}/D{dim}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--provider',choices=list(METHODS),required=True)
    parser.add_argument('--profile',choices=['smoke','standard'],default='standard')
    parser.add_argument('--stages',nargs='+',choices=['static','filter','gradient','timing'],default=['static','filter','gradient'])
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS,help='Filtering models of the filter stage')
    parser.add_argument('--output',default='results/standard')
    parser.add_argument('--resume',action='store_true',help='Skip cells with a final complete status in the output folder')
    args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    args.done=set()
    if args.resume and (out/'status.jsonl').exists():
        states={}
        for line in (out/'status.jsonl').read_text().splitlines():
            row=json.loads(line);states[(row['method'],row['stage'],row['case'])]=row['status']
        args.done={key for key,value in states.items() if value=='complete'}
    for method in args.methods or METHODS[args.provider]:
        try:
            adapter=Adapter(args.provider,method)
            for stage in args.stages: {'static':static,'filter':filtering,'gradient':gradients,'timing':timing}[stage](adapter,args,out)
            if args.provider=='julia': adapter.process.stdin.close(); adapter.process.wait()
        except Exception as e:
            traceback.print_exc(); status(out,args.provider,method,'initialization','all',e)


if __name__=='__main__': main()
