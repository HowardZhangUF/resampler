"""Fail if expected standard cells, repetitions, or numerical checks are missing."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resampler_bench.adapters import METHODS

root=ROOT/'results/standard';problems=[];counts={}
for provider,methods in METHODS.items():
    for stage in ['static','filter','gradient']:
        required_gradient=provider in ('custom','cuthbert','pfjax','pydpf','torchfilter','filterflow')
        if stage=='gradient' and not required_gradient:continue
        path=root/provider/(stage+'.csv')
        if not path.exists():problems.append(str(path)+' missing');continue
        df=pd.read_csv(path);counts[provider+'/'+stage]=len(df)
        unique=['provider','method','n','seed']+({'static':['cloud','dim'],'filter':['model'],'gradient':['cloud','function']}[stage])
        if df.duplicated(unique).any():problems.append(provider+'/'+stage+': duplicate trial identifiers')
        essential={'static':['sliced_w1','mean_error','covariance_error','rff_mmd2'],'filter':['rmse','loglik','coverage95'],'gradient':['gradient','exact_gradient','squared_error']}[stage]
        missing=~np.isfinite(df[essential].to_numpy()).all(axis=1)
        failures=root/provider/'numerical_failures.jsonl'
        documented=[json.loads(line) for line in failures.read_text().splitlines()] if failures.exists() else []
        for _,row in df[missing].iterrows():
            if not any(r['method']==row.method and r['stage']==stage and r['seed']==row.seed and f'N{row.n}' in r['case'] and str(row.get('model',row.get('cloud',''))) in r['case'] for r in documented):
                problems.append(provider+'/'+stage+': undocumented nonfinite required metrics')
        for method in methods:
            expensive=method in ('ot','corrected_ot','gumbel')
            ns=[32,128,512,2048] if stage=='static' or (stage=='filter' and not expensive) else [32,128]
            if stage=='gradient' and not expensive:ns=[32,128,512]
            if stage=='static' and method=='corrected_ot':ns=[32,128]
            keys=({'static':['cloud','dim'],'filter':['model'],'gradient':['cloud','function']}[stage])
            expected_groups={'static':12,'filter':4,'gradient':4}[stage]
            for n in ns:
                part=df[(df.method==method)&(df.n==n)]
                reps=(10 if n<=128 else 3) if expensive else 100
                sizes=part.groupby(keys).size()
                sparse=(stage=='static' and provider in ('pydpf','filterflow') and method=='ot' and n==2048)
                if sparse:
                    if ('gaussian',2) not in sizes.index or sizes.loc[('gaussian',2)]!=reps or not (sizes==reps).all():problems.append(f'{provider}/{method}/{stage}/N{n}: Gaussian D=2 scaling trials missing')
                elif len(sizes)!=expected_groups or not (sizes==reps).all():problems.append(f'{provider}/{method}/{stage}/N{n}: expected {expected_groups} groups of {reps}, got {sizes.to_dict()}')
                if stage=='filter':
                    datasets=part.groupby('model').data_seed.nunique()
                    if not (datasets==10).all():problems.append(f'{provider}/{method}/filter/N{n}: expected 10 observation sequences, got {datasets.to_dict()}')
    path=root/provider/'learning.csv'
    expected=30 if provider=='custom' else 10
    if not path.exists() or len(pd.read_csv(path))!=expected:problems.append(provider+': parameter-learning trials missing')
for folder,expected in [('cuthbert_native',800),('torch_native',2440)]:
    path=root/folder/'trials.csv'
    if not path.exists():problems.append(folder+': missing');continue
    df=pd.read_csv(path)
    if len(df)!=expected:problems.append(f'{folder}: expected {expected} trials, got {len(df)}')
    if folder=='cuthbert_native':
        pivot=df.pivot(index=['n','seed'],columns='stop_gradient',values='loglik')
        if not np.allclose(pivot[False],pivot[True],atol=1e-9,rtol=0):problems.append('Cuthbert stop-gradient changed forward likelihood')
profile=root/'cuthbert_native/profiles.csv'
if not profile.exists() or len(pd.read_csv(profile))!=84:problems.append('Cuthbert profile slices missing')
native=root/'cuthbert_native/data.npz';torch=root/'torch_native/data.npz'
if native.exists() and torch.exists():
    a,b=np.load(native),np.load(torch)
    if not np.allclose(a['ys'][:,0],b['ys'],atol=0,rtol=0):problems.append('Native gradient datasets differ')
    if not np.allclose(a['exact_gradient'],b['exact_gradient'],atol=1e-9,rtol=0):problems.append('Independent exact Kalman gradients differ')
eigen=root/'custom/static.csv'
if eigen.exists():
    df=pd.read_csv(eigen);df=df[df.method=='eigen']
    if df.mean_error.max()>1e-8 or df.covariance_error.max()>1e-8:problems.append('Eigen population moment errors exceed 1e-8')
performance=ROOT/'results/performance'
for provider in METHODS:
    path=performance/provider/'timing.csv'
    if not path.exists():problems.append(provider+': serial forward timing missing')
    else:
        df=pd.read_csv(path)
        if set(df.method)!=set(METHODS[provider]):problems.append(provider+': serial timing methods missing')
        if df.duplicated(['method','cloud','n','dim','seed']).any():problems.append(provider+': duplicate serial timing trials')
        if not np.isfinite(df[['forward_ms','process_peak_rss_mb']].to_numpy()).all():problems.append(provider+': nonfinite serial timing metrics')
        if provider=='julia' and ('worker_peak_rss_mb' not in df or not (df.worker_peak_rss_mb>0).all()):problems.append('Julia worker memory measurement missing')
        for method in METHODS[provider]:
            expected=240 if method=='corrected_ot' else (312 if method in ('ot','gumbel') else 4800)
            if provider in ('pydpf','filterflow') and method=='ot':expected=279
            actual=len(df[df.method==method])
            if actual!=expected:problems.append(f'{provider}/{method}: serial timing expected {expected} measurements, got {actual}')
result=dict(ok=not problems,problems=problems,raw_counts=counts)
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(bool(problems))
