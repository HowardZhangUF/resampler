"""Adapters call the checked-out upstream code, without modifying vendor files."""
import os
from pathlib import Path
import subprocess
import struct
import sys
from types import SimpleNamespace
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def paths():
    for p in ['particles', 'Stone-Soup', 'cuthbert', 'pfjax/src', 'pydpf/pydpf', 'torchfilter', 'filterflow']:
        sys.path.insert(0, str(ROOT/'vendor'/p))


class Adapter:
    def __init__(self, provider, method):
        paths()
        self.provider, self.method = provider, method
        self.expensive = method in ('ot', 'corrected_ot', 'gumbel')
        self.last_indices = None
        self.record_ancestors = True
        if provider == 'particles':
            from particles import resampling
            self.fn = lambda x, w: x[resampling.resampling(method, w)]
        elif provider == 'stonesoup':
            from stonesoup.resampler import particle
            self.impl = getattr(particle, method.capitalize()+'Resampler')()
        elif provider in ('cuthbert', 'pfjax'):
            import jax
            import jax.numpy as jnp
            jax.config.update('jax_enable_x64', True)
            self.jax, self.jnp = jax, jnp
            if provider == 'cuthbert':
                from cuthbertlib.resampling import multinomial, systematic, autodiff
                fn = {'multinomial':multinomial, 'systematic':systematic}[method.split('_')[0]].resampling
                if method.endswith('_stop'): fn = autodiff.stop_gradient_decorator(fn)
                self.jax_fn = lambda k, x, lw: fn(k, lw, x, x.shape[0])[1:][::-1]
            else:
                from pfjax import particle_resamplers as pr
                fn = getattr(pr, 'resample_'+method)
                if method=='ot':
                    self.jax_fn = lambda k, x, lw: (fn(k, x, lw, pointcloud_kwargs={'epsilon':.1}, sinkhorn_kwargs={'max_iterations':100, 'threshold':1e-3})['x_particles'], jnp.zeros_like(lw))
                else:
                    self.jax_fn = lambda k, x, lw: (fn(k, x, lw)['x_particles'], jnp.zeros_like(lw))
            self.jax_fn = jax.jit(self.jax_fn)
        elif provider in ('pydpf', 'torchfilter', 'custom', 'dsr'):
            import torch
            torch.set_num_threads(1)
            self.torch = torch
            self.generator = torch.Generator()
            if provider=='dsr':
                # Upstream module is a top-level resampling.py; load by path to avoid name clashes.
                import importlib.util
                spec=importlib.util.spec_from_file_location('dsr_resampling',ROOT/'vendor/differentiable-systematic-resampling/resampling.py')
                self.dsr=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.dsr)
                # Paper default: Morton sorting, tau=0.1, transport gradient clamped to [-1, 1].
                # dsr_scaled is a local diagnostic, not from the paper: tau=2.5/N keeps tau*N at the paper's N=25 value.
                self.dsr_options=dict(dsr=(True,.1,True),dsr_unsorted=(False,.1,True),dsr_noclip=(True,.1,False),dsr_scaled=(True,None,True),**{'dsr_tau0.5':(True,.5,True)})[method]
            elif provider=='pydpf':
                from pydpf import resampling as r
                if method=='ot': self.impl = r.OptimalTransportResampler(.1, .75, 1e-3, 100, 1.)
                else:
                    base = r.SystematicResampler(self.generator) if method.startswith('systematic') else r.MultinomialResampler(self.generator)
                    if method=='soft': base = r.SoftResampler(.5, base, torch.device('cpu'))
                    elif method.endswith('_stop'): base = r.StopGradientResampler(base)
                    self.impl = base
            elif provider=='torchfilter':
                from torchfilter.filters import ParticleFilter
                self.resample_fn = ParticleFilter._resample
        elif provider=='filterflow':
            import tensorflow as tf
            from filterflow.base import State
            from filterflow.resampling.differentiable import RegularisedTransform, CorrectedRegularizedTransform
            from filterflow.resampling.standard import MultinomialResampler, SystematicResampler, StratifiedResampler
            tf.config.set_visible_devices([], 'GPU')
            self.tf, self.State = tf, State
            cls = {'ot':RegularisedTransform, 'corrected_ot':CorrectedRegularizedTransform,
                   'systematic':SystematicResampler, 'stratified':StratifiedResampler, 'multinomial':MultinomialResampler}[method]
            self.impl = cls(epsilon=.1, max_iter=100) if self.expensive else cls()
            @tf.function
            def call(x, lw, seed):
                st = self.impl.apply(State(particles=x[None], log_weights=lw[None]), tf.constant([True]), seed=seed)
                return st.particles[0], st.log_weights[0]
            self.tf_fn = call
        elif provider=='variational':
            # Upstream module is a top-level resampling.py; load by path to avoid name clashes.
            import importlib.util
            spec=importlib.util.spec_from_file_location('variational_resampling',ROOT/'vendor/Variational-Resampling/resampling.py')
            vr=importlib.util.module_from_spec(spec);spec.loader.exec_module(vr)
            # KL greedy allocation on log importance weights (main.py's 'kl-iw'); TV is deterministic rounding.
            self.fn={'kl':lambda w,lw:vr.kl(lw),'tv':lambda w,lw:vr.tv(w)}[method]
        elif provider=='julia':
            self.process = subprocess.Popen([str(ROOT/'.tools/julia-1.10.10/bin/julia'), '--startup-file=no', str(ROOT/'scripts/julia_worker_binary.jl')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(ROOT/'results/julia-worker.log', 'a'), bufsize=0)
        else: raise ValueError(provider)

    def torch_call(self, x, lw, seed):
        """Native differentiable call; returns particles and log weights."""
        t = self.torch
        self.generator.manual_seed(seed)
        t.manual_seed(seed)
        if self.provider=='pydpf':
            y, out_lw = self.impl(x[None], lw[None])
            return y[0], out_lw[0]
        if self.provider=='torchfilter':
            alpha = .5 if self.method=='soft' else 1.
            state = SimpleNamespace(particle_states=x[None], particle_log_weights=lw[None], num_particles=len(x), soft_resample_alpha=alpha)
            self.resample_fn(state)
            if alpha<1:
                # Upstream _resample leaves correction weights in input order.
                # Recover sampled ancestors and gather those weights explicitly.
                idx = cKDTree(x.detach().numpy()).query(state.particle_states[0].detach().numpy())[1]
                out_lw = state.particle_log_weights[0][t.as_tensor(idx)]
            else: out_lw = state.particle_log_weights[0]
            return state.particle_states[0], out_lw
        if self.provider=='dsr':
            # Upstream draws the comb offset from the global torch RNG, seeded above.
            morton,tau,clip=self.dsr_options
            if tau is None:tau=2.5/len(x)
            fn=self.dsr.resampler_systematic_morton if morton else self.dsr.resampler_systematic
            # The clamp is a tensor hook, which needs a graph; skip it under no_grad.
            y,_=fn(x[None],lw.softmax(-1)[None],tau=tau,requires_grad=clip and t.is_grad_enabled() and (x.requires_grad or lw.requires_grad))
            return y[0],t.full_like(lw,-np.log(len(x)))
        from .eigen import eigen_resample
        if self.method=='eigen': y = eigen_resample(x, lw)
        elif self.method=='gaussian':
            w = lw.softmax(-1); mu = w @ x; z = x-mu
            cov = z.T @ (w[:, None]*z)
            root = t.linalg.cholesky(cov+1e-10*t.eye(x.shape[1], dtype=x.dtype))
            y = mu+t.randn(x.shape, generator=self.generator, dtype=x.dtype) @ root.T
        elif self.method=='gumbel':
            u = t.rand((len(x), len(x)), generator=self.generator, dtype=x.dtype).clamp(1e-12, 1-1e-12)
            coefficients = t.softmax((lw[None]-t.log(-t.log(u)))/.5, dim=-1)
            y = coefficients @ x
        else: raise ValueError(self.method)
        return y, t.full_like(lw, -np.log(len(x)))

    def __call__(self, x, w, seed):
        self.last_indices = None
        np.random.seed(seed)
        lw = np.log(np.maximum(w, np.finfo(float).tiny))
        if self.provider=='particles':
            from particles import resampling
            idx = resampling.resampling(self.method, w)
            y, out_lw = x[idx], np.zeros(len(x)); self.last_indices = idx
        elif self.provider=='variational':
            idx=np.asarray(self.fn(w,lw),dtype=int)
            y, out_lw = x[idx], np.zeros(len(x)); self.last_indices = idx
        elif self.provider=='stonesoup':
            from stonesoup.types.state import ParticleState
            # Include an ancestor index as a state coordinate (sorting is by weight).
            from stonesoup.types.array import StateVectors
            state = ParticleState(state_vector=StateVectors(np.vstack([x.T, np.arange(len(x))])), log_weight=lw)
            out = self.impl.resample(state)
            y, out_lw = np.asarray(out.state_vector[:-1].T), np.asarray(out.log_weight)
            self.last_indices = np.asarray(out.state_vector[-1], dtype=int)
        elif self.provider in ('cuthbert', 'pfjax'):
            y, out_lw = self.jax_fn(self.jax.random.PRNGKey(seed), self.jnp.asarray(x), self.jnp.asarray(lw))
            y.block_until_ready(); y, out_lw = np.asarray(y), np.asarray(out_lw)
        elif self.provider in ('pydpf', 'torchfilter', 'custom', 'dsr'):
            with self.torch.no_grad():
                y, out_lw = self.torch_call(self.torch.as_tensor(x), self.torch.as_tensor(lw), seed)
            y, out_lw = y.numpy(), out_lw.numpy()
        elif self.provider=='filterflow':
            y, out_lw = self.tf_fn(self.tf.convert_to_tensor(x, dtype=self.tf.float32), self.tf.convert_to_tensor(lw, dtype=self.tf.float32), self.tf.constant([seed, 0], self.tf.int32))
            y, out_lw = y.numpy().astype(float), out_lw.numpy().astype(float)
        elif self.provider=='julia':
            methods=['multinomial','systematic','stratified','residual','ssp','killing']
            payload=struct.pack('<IIq',methods.index(self.method),len(w),seed)+np.ascontiguousarray(w,dtype='<f8').tobytes()
            # Raw pipes can perform short writes/reads for larger clouds.
            offset=0
            while offset<len(payload):offset+=self.process.stdin.write(payload[offset:])
            self.process.stdin.flush()
            chunks=[];remaining=8*len(w)
            while remaining:
                chunk=self.process.stdout.read(remaining)
                if not chunk:raise RuntimeError('Julia worker exited')
                chunks.append(chunk);remaining-=len(chunk)
            idx=np.frombuffer(b''.join(chunks),dtype='<i8')-1
            y, out_lw = x[idx], np.zeros(len(x)); self.last_indices=idx
        if not np.isfinite(y).all() or not np.isfinite(out_lw).all(): raise FloatingPointError('Non-finite resampler output')
        v = np.exp(out_lw-logsumexp(out_lw))
        # Methods that move particles (convex combinations, moment maps) have no ancestor counts.
        if self.record_ancestors and self.last_indices is None and self.method not in ('ot', 'corrected_ot', 'mvn', 'gaussian', 'eigen', 'gumbel') and self.provider!='dsr':
            self.last_indices = cKDTree(x).query(y)[1]
        return y, v


METHODS = {
    'particles':['multinomial', 'systematic', 'stratified', 'residual', 'ssp'],
    'stonesoup':['multinomial', 'systematic', 'stratified', 'residual'],
    'cuthbert':['multinomial', 'systematic', 'systematic_stop'],
    'pfjax':['multinomial', 'mvn', 'ot'],
    'pydpf':['multinomial', 'systematic', 'systematic_stop', 'soft', 'ot'],
    'torchfilter':['multinomial', 'soft'],
    'filterflow':['multinomial', 'systematic', 'stratified', 'ot', 'corrected_ot'],
    'julia':['multinomial', 'systematic', 'stratified', 'residual', 'ssp', 'killing'],
    'custom':['eigen', 'gaussian', 'gumbel'],
    'variational':['kl', 'tv'],
    'dsr':['dsr', 'dsr_unsorted', 'dsr_noclip', 'dsr_tau0.5', 'dsr_scaled'],
}
