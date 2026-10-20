"""Show why matching moments does not ensure mixture preservation."""
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resampler_bench.data import cloud
from resampler_bench.eigen import eigen_resample

out=ROOT/'results/standard/report';out.mkdir(parents=True,exist_ok=True)
x,w=cloud('mixture',2048,1)
y=eigen_resample(torch.tensor(x),torch.tensor(np.log(w))).numpy()
fig,ax=plt.subplots(figsize=(9,4))
bins=np.linspace(-5,6,70)
ax.hist(x[:,0],bins=bins,weights=w,density=True,histtype='step',lw=2,label='Weighted input (35% / 65% modes)')
ax.hist(y[:,0],bins=bins,density=True,histtype='step',lw=2,label='Eigen affine output (equal particle weights)')
input_mean=float(w@x[:,0]);output_mean=float(y.mean())
input_variance=float(w@(x[:,0]-input_mean)**2);output_variance=float(np.var(y))
ax.set(xlabel='State',ylabel='Density',title='Same mean and variance; different mixture shape')
ax.text(.02,.95,f'Input / output mean: {input_mean:.6f} / {output_mean:.6f}\nInput / output variance: {input_variance:.6f} / {output_variance:.6f}',transform=ax.transAxes,va='top',fontsize=9)
ax.legend(loc='upper right',fontsize=8);fig.tight_layout()
fig.savefig(out/'eigen_mixture_moments.png',dpi=180)
