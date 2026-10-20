"""Independent references from the project PyTorch method, SciPy, and a Kalman update."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from resampler_bench.eigen import eigen_resample
rng=np.random.default_rng(2026)
cases=[]
for label,x in [('full-rank',rng.normal(size=(12,2))@np.array([[1,.4],[.3,1.7]])),('rank-one',np.array([[i,2*i] for i in [-2,-1,0,1,2]],float)),('repeated-eigenvalues',np.array([[1,0],[-1,0],[0,1],[0,-1]],float))]:
    w=rng.uniform(.2,1,len(x));w/=sum(w)
    y=eigen_resample(torch.tensor(x),torch.tensor(np.log(w))).numpy()
    cases.append(dict(label=label,x=x.tolist(),w=w.tolist(),y=y.tolist()))
x=[-3,-1,0,2,4];w=[.1,.3,.2,.15,.25];y=[-2,-2,1,3];v=[.1,.3,.2,.4]
bw=np.sqrt(np.sum(np.array(w)*(np.array(x)-np.dot(x,w))**2))
k=lambda a,b:np.exp(-.5*((np.array(a)[:,None]-np.array(b)[None,:])/bw)**2)
mmd=np.array(w)@k(x,x)@w+np.array(v)@k(y,y)@v-2*np.array(w)@k(x,y)@v
obs=[.2,-.5,1.2,.1,-.3];m=0.;p=1.;ll=0.;ms=[];ps=[]
for a in obs:
    m=.8*m;p=.64*p+.25;s=p+.64
    ll+=-.5*(np.log(2*np.pi*s)+(a-m)**2/s)
    m=m+p/s*(a-m);p=p*.64/s;ms.append(m);ps.append(p)
Path('website/tests/fixtures.json').write_text(json.dumps(dict(eigen=cases,distribution=dict(x=x,w=w,y=y,v=v,w1=wasserstein_distance(x,y,w,v),mmd=mmd),kalman=dict(observations=obs,means=ms,variances=ps,loglik=ll)),indent=2)+'\n')
