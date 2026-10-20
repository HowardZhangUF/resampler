"""Check seeded clouds/model data across the NumPy and TensorFlow environments."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resampler_bench.data import cloud,sequence
p=argparse.ArgumentParser();p.add_argument('--write');p.add_argument('--check');a=p.parse_args()
values={}
for n in [32,128,512,2048]:
    for d in [1,2,4]:
        for kind in ['fixed','gaussian','mixture','skewed']:
            x,w=cloud(kind,n,d);name=f'{kind}_{n}_{d}'
            values[name+'_x']=x;values[name+'_w']=w
for kind in ['linear','ungm','volatility','bearing']:
    for seed in range(10):
        x,y,mask=sequence(kind,seed)
        values[f'{kind}_{seed}_states']=x;values[f'{kind}_{seed}_observations']=y;values[f'{kind}_{seed}_mask']=mask
if a.write:
    np.savez_compressed(a.write,**values);print('Wrote',len(values),'input arrays; NumPy',np.__version__)
elif a.check:
    expected=np.load(a.check);maximum=0.;identical=0
    for name,value in values.items():
        np.testing.assert_allclose(value,expected[name],rtol=0,atol=1e-12)
        if value.dtype!=bool:maximum=max(maximum,float(np.max(np.abs(value-expected[name]))))
        identical+=int(np.array_equal(value,expected[name]))
    print(f'{len(values)} input arrays agree within 1e-12; {identical} bitwise identical; maximum difference {maximum:.3g}; NumPy {np.__version__}')
else:p.error('Specify --write or --check')
