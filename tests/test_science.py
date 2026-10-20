import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance
from resampler_bench.eigen import eigen_resample
from resampler_bench.data import cloud, kalman
from resampler_bench.metrics import distribution


@pytest.mark.parametrize('dim',[1,2,4])
def test_eigen_preserves_population_moments(dim):
    x,w=cloud('mixture',128,dim)
    x=torch.tensor(x);lw=torch.tensor(np.log(w))
    y=eigen_resample(x,lw)
    mu=(torch.tensor(w)[:,None]*x).sum(0); centered=x-mu
    cov=centered.T@(torch.tensor(w)[:,None]*centered)
    yc=y-y.mean(0)
    torch.testing.assert_close(y.mean(0),mu,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(yc.T@yc/len(y),cov,rtol=1e-9,atol=1e-9)


def test_eigen_derivatives_match_finite_difference():
    x,w=cloud('gaussian',32,2)
    x=torch.tensor(x);lw=torch.tensor(np.log(w),requires_grad=True)
    assert torch.autograd.gradcheck(lambda a:eigen_resample(x,a), (lw,), eps=1e-6,atol=1e-5,rtol=1e-4)


def test_rank_deficient_eigen_remains_finite():
    x=torch.zeros(32,4,dtype=torch.float64)
    y=eigen_resample(x,torch.zeros(32,dtype=torch.float64))
    assert torch.isfinite(y).all()


def test_eigen_repeated_eigenvalue_gradients():
    x=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],dtype=torch.float64,requires_grad=True)
    lw=torch.zeros(4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(eigen_resample,(x,lw),eps=1e-6,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('inverse',[False,True])
def test_regularized_root_gradients(inverse):
    from resampler_bench.eigen import _SymmetricRoot
    matrix=torch.diag(torch.tensor([1e-12,1.],dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(lambda a:_SymmetricRoot.apply(a,inverse,1e-4),(matrix,),eps=1e-7,atol=1e-4,rtol=1e-4)


def test_identity_metrics_and_one_dimensional_wasserstein():
    x,w=cloud('gaussian',32,1)
    result=distribution(x,w,x,w,'gaussian')
    assert result['covariance_error']==0
    assert result['rff_mmd2']==0
    y=x+.5; v=np.ones(32)/32
    result=distribution(x,w,y,v,'gaussian')
    assert result['sliced_w1']==pytest.approx(wasserstein_distance(x[:,0],y[:,0],w,v))


def test_batched_projected_wasserstein_matches_scipy():
    from resampler_bench.metrics import prepare
    x,w=cloud('mixture',128,4);y=x[::2];v=np.ones(len(y))/len(y)
    ref=prepare(x,w);xp=x@ref['directions'];yp=y@ref['directions']
    expected=np.mean([wasserstein_distance(xp[:,i],yp[:,i],w,v) for i in range(16)])
    assert distribution(x,w,y,v,'mixture')['sliced_w1']==pytest.approx(expected,abs=1e-12)


def test_kalman_matches_dense_gaussian_likelihood():
    from scipy.stats import multivariate_normal
    ys=np.array([.1,-.3,.5,1.,-.2]);n=len(ys)
    transition=.8; q=.5**2; r=.8**2
    cov=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            cov[i,j]=transition**(i+j+2)+q*sum(transition**(i+j-2*k) for k in range(min(i,j)+1))+(r if i==j else 0)
    _,_,ll=kalman(ys)
    assert ll==pytest.approx(multivariate_normal.logpdf(ys,mean=np.zeros(n),cov=cov),abs=1e-12)


@pytest.mark.parametrize('method',['kl','tv'])
def test_variational_resamplers_are_deterministic_count_allocations(method):
    from resampler_bench.adapters import Adapter
    adapter=Adapter('variational',method)
    x,w=cloud('mixture',128,2)
    y,v=adapter(x,w,0);counts=np.bincount(adapter.last_indices,minlength=128)
    y2,_=adapter(x,w,1)
    assert counts.sum()==128 and np.array_equal(y,y2) and np.allclose(v,1/128)
    if method=='tv':assert np.abs(counts-128*w).max()<1


def test_dsr_adapter_is_seeded_convex_and_tends_to_systematic():
    from resampler_bench.adapters import Adapter
    adapter=Adapter('dsr','dsr')
    x,w=cloud('gaussian',64,2)
    y,v=adapter(x,w,3);y2,_=adapter(x,w,3)
    assert np.array_equal(y,y2) and np.allclose(v,1/64)
    # Rows of the transport plan are convex weights, so outputs stay in the input bounding box.
    assert (y.min(0)>=x.min(0)-1e-9).all() and (y.max(0)<=x.max(0)+1e-9).all()
    # As tau -> 0 every output particle is (numerically) a copy of an input particle.
    adapter.dsr_options=(True,1e-6,False)
    y,_=adapter(x,w,3)
    assert np.abs(y[:,None]-x[None]).sum(-1).min(1).max()<1e-6


@pytest.mark.parametrize('method',['dsr','dsr_noclip','dsr_unsorted'])
def test_dsr_gradients_reach_weights(method):
    from resampler_bench.adapters import Adapter
    adapter=Adapter('dsr',method)
    x,w=cloud('gaussian',32,2)
    lw=torch.tensor(np.log(w),requires_grad=True)
    y,v=adapter.torch_call(torch.tensor(x),lw,0)
    g=torch.autograd.grad((v.softmax(0)[:,None]*y).sum(),lw)[0]
    assert torch.isfinite(g).all() and g.abs().max()>0


@pytest.mark.parametrize('kind,dim,obs_dim',[('lorenz63',3,64),('lorenz96',10,10)])
def test_lorenz_models_are_seeded_and_finite(kind,dim,obs_dim):
    from resampler_bench.data import sequence, initial, propagate, log_observation
    states,ys,mask=sequence(kind,3,50); states2,ys2,_=sequence(kind,3,50)
    assert states.shape==(50,dim) and ys.shape==(50,obs_dim) and mask.all()
    assert np.array_equal(states,states2) and np.array_equal(ys,ys2)
    x=propagate(kind,initial(kind,np.random.default_rng(0),64),np.random.default_rng(1),1)
    lw=log_observation(kind,x,ys[0])
    assert x.shape==(64,dim) and lw.shape==(64,) and np.isfinite(lw).all()


def test_lorenz96_matches_dense_gaussian_likelihood():
    from scipy.stats import multivariate_normal
    from resampler_bench.data import log_observation, L96_R
    x=np.random.default_rng(2).normal(size=(5,10));y=np.random.default_rng(3).normal(size=10)
    expected=[multivariate_normal.logpdf(y,mean=xi,cov=L96_R*np.eye(10)) for xi in x]
    assert np.allclose(log_observation('lorenz96',x,y),expected)
