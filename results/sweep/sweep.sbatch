#!/bin/bash
#SBATCH --job-name=ess-tau-sweep
#SBATCH --account=jshin2
#SBATCH --qos=jshin2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=results/sweep/slurm-%j.log
# Dose-response sweep: E_2 < tau N for tau in {0.1, 0.25, 0.5, 0.75, 1.0}; filter stage, 50 steps,
# 14 methods (pfjax/ot and filterflow/ot need JAX / TensorFlow, not in zhz14). --resume makes resubmission safe.
cd /home/zhangzihao/blue_jshin2/zhangzihao/resampler
E=/blue/jshin2/zhangzihao/conda/envs/zhz14/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date; hostname; $E -V
$E scripts/run_triggers.py --python $E --jobs $SLURM_CPUS_PER_TASK --output results/sweep \
  --triggers ess:0.1 ess:0.25 ess:0.5 ess:0.75 ess:1.0 \
  --methods dsr/dsr dsr/dsr_unsorted dsr/dsr_tau0.5 dsr/dsr_scaled particles/systematic particles/multinomial \
            pydpf/systematic_stop pydpf/soft torchfilter/soft pydpf/ot custom/eigen custom/gaussian custom/gumbel variational/kl
date
