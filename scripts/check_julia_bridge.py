"""Check binary protocol equivalence and repeated-seed reproducibility."""
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resampler_bench.adapters import Adapter,METHODS
from resampler_bench.data import cloud
x,w=cloud('mixture',128,2)
for method in METHODS['julia']:
    reference=subprocess.Popen([str(ROOT/'.tools/julia-1.10.10/bin/julia'),'--startup-file=no',str(ROOT/'scripts/julia_worker.jl')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    line,_=reference.communicate(method+'\t91\t'+','.join(map(str,w))+'\n')
    expected=np.fromstring(line,sep=',',dtype=int)-1
    adapter=Adapter('julia',method)
    for _ in range(2):
        adapter(x,w,91);np.testing.assert_array_equal(adapter.last_indices,expected)
    adapter.process.stdin.close();adapter.process.wait()
    print(method,'protocol equivalence and reproducibility passed',flush=True)
