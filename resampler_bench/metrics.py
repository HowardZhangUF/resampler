"""Metrics always compare against the input weighted empirical measure."""
import numpy as np
from scipy.stats import wasserstein_distance

_cached = None


def prepare(x,w):
    """Reuse fixed-cloud reference features across Monte Carlo repetitions."""
    global _cached
    if _cached is not None and _cached['x'] is x and _cached['w'] is w:
        return _cached
    mean=w@x;a=x-mean;cov=a.T@(w[:,None]*a)
    rng=np.random.default_rng(903)
    directions=rng.normal(size=(x.shape[1],16));directions/=np.linalg.norm(directions,axis=0)
    bandwidth=max(np.sqrt(np.trace(cov)),1e-6)
    frequencies=rng.normal(size=(x.shape[1],128))/bandwidth
    phases=rng.uniform(0,2*np.pi,128)
    _cached=dict(x=x,w=w,mean=mean,cov=cov,directions=directions,xp=x@directions,
                 frequencies=frequencies,phases=phases,feature_mean=w@np.cos(x@frequencies+phases),
                 eigenvalues=np.linalg.eigvalsh(cov))
    return _cached


def distribution(x, w, y, v, kind):
    ref=prepare(x,w)
    mean,cov=ref['mean'],ref['cov'];new_mean=v@y;b=y-new_mean
    new_cov=b.T@(v[:,None]*b)
    # Integrate |F-G| exactly over the union of the two projected supports.
    points=np.concatenate([ref['xp'],y@ref['directions']],axis=0)
    order=np.argsort(points,axis=0)
    sorted_points=np.take_along_axis(points,order,axis=0)
    masses=np.broadcast_to(np.concatenate([w,-v])[:,None],points.shape)
    cdf_diff=np.cumsum(np.take_along_axis(masses,order,axis=0),axis=0)[:-1]
    sw=np.mean(np.sum(np.abs(cdf_diff)*np.diff(sorted_points,axis=0),axis=0))
    # Random Fourier features approximate Gaussian-kernel MMD squared.
    fy=np.cos(y@ref['frequencies']+ref['phases'])
    mmd2=2/128*np.sum((ref['feature_mean']-v@fy)**2)
    return dict(sliced_w1=sw, rff_mmd2=mmd2,
                mean_error=np.linalg.norm(new_mean-mean),
                covariance_error=np.linalg.norm(new_cov-cov, 'fro'),
                eigenvalue_error=np.linalg.norm(np.linalg.eigvalsh(new_cov)-ref['eigenvalues']),
                estimator_mse_x=np.mean((new_mean-mean)**2),
                estimator_mse_x2=np.mean((v@(y*y)-w@(x*x))**2),
                estimator_mse_sin=np.mean((v@np.sin(y)-w@np.sin(x))**2),
                mode_coverage=(float(np.sum(v[np.abs(y[:,0]+3.)<1.35])>.01)+float(np.sum(v[np.abs(y[:,0]-3.)<1.35])>.01))/2 if kind=='mixture' else np.nan,
                mode_mass_error=abs(np.sum(v[y[:, 0]<0])-np.sum(w[x[:, 0]<0])) if kind=='mixture' else np.nan)
