"""Shared seeded clouds and state-space models, independent of providers."""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp

# Stochastic Lorenz-63 as in Cumlin & Chatterjee (DSR, NeurIPS 2026): x_{t+1} = F(x_t) x_t + e_t with
# F(x) = 5th-order Taylor series of exp(A(x_1) delta), delta = 0.02, Ce = 0.1 I (-10 dB), and the
# 8x8 camera observation of vendor/differentiable-systematic-resampling/lorenz_attractor.py at SMNR 10 dB.
L63_DELTA, L63_Q, L63_SMNR_DB = .02, .1, 10.
# Stochastic Lorenz-96: d=10, F=8, RK4 over dt=0.05 per step, additive noise variance 0.1, y = x + N(0, I).
L96_DIM, L96_FORCING, L96_DT, L96_Q, L96_R = 10, 8., .05, .1, 1.
_GRID = np.stack(np.meshgrid(np.linspace(-30, 30, 8), np.linspace(-40, 40, 8), indexing='ij'), axis=-1).reshape(-1, 2)


def l63_mean(x):
    """F(x) x with the Taylor series applied term by term; x is (n, 3)."""
    z = x[:, 0]  # A depends on the first coordinate of the current state only.
    acc, v = x.copy(), x.copy()
    for j in range(1, 6):
        v = np.stack([-10*v[:, 0]+10*v[:, 1], 28*v[:, 0]-v[:, 1]-z*v[:, 2], z*v[:, 1]-8/3*v[:, 2]], axis=1)*L63_DELTA/j
        acc = acc + v
    return acc


def camera(x):
    """8x8 Gaussian point-spread image of the state; x is (n, 3), returns (n, 64)."""
    sq = ((_GRID[None] - x[:, None, :2])**2).sum(-1)
    return 10.*np.exp(-.5*sq/(1e-12+np.clip(x[:, None, 2], -6.9, np.inf)+7))


def l96_drift(x):
    return (np.roll(x, -1, axis=1)-np.roll(x, 2, axis=1))*np.roll(x, 1, axis=1)-x+L96_FORCING


def l96_step(x):
    k1 = l96_drift(x); k2 = l96_drift(x+.5*L96_DT*k1); k3 = l96_drift(x+.5*L96_DT*k2); k4 = l96_drift(x+L96_DT*k3)
    return x+L96_DT/6*(k1+2*k2+2*k3+k4)


@lru_cache(None)
def l63_obs_variance():
    """Camera noise variance at SMNR 10 dB: signal power of a long reference run started at the origin."""
    rng = np.random.default_rng(6300); x = np.zeros((1, 3)); frames = []
    for _ in range(20000):
        x = l63_mean(x)+np.sqrt(L63_Q)*rng.normal(size=x.shape); frames.append(camera(x)[0])
    return float(np.var(np.array(frames)))/10**(L63_SMNR_DB/10)


@lru_cache(None)
def l96_climatology():
    """Mean and per-component variance of the stochastic L96 attractor (long run after spin-up)."""
    rng = np.random.default_rng(9600); x = L96_FORCING+rng.normal(size=(1, L96_DIM)); states = []
    for t in range(21000):
        x = l96_step(x)+np.sqrt(L96_Q)*rng.normal(size=x.shape)
        if t >= 1000: states.append(x[0].copy())
    states = np.array(states); return float(states.mean()), float(states.var())


def cloud(kind, n, dim):
    rng = np.random.default_rng(4100 + n * 7 + dim)
    x = rng.normal(size=(n, dim))
    if kind == 'mixture':
        x[:, 0] = x[:, 0] * .45 + np.where(np.arange(n) % 2, 3., -3.)
    elif kind == 'fixed':
        x[:, 0] = np.linspace(-4., 4., n)
    lw = .6 * x[:, 0]
    if kind == 'skewed':
        lw = np.linspace(-12., 0., n)
    elif kind == 'mixture':
        lw = np.where(x[:, 0] < 0, np.log(.35), np.log(.65))
    return x, np.exp(lw - logsumexp(lw))


def initial(kind, rng, n):
    if kind == 'lorenz63':
        return rng.normal(size=(n, 3))
    if kind == 'lorenz96':
        mean, var = l96_climatology()
        return mean+np.sqrt(var)*rng.normal(size=(n, L96_DIM))
    if kind == 'bearing':
        return rng.multivariate_normal([10., .35, -6., .25], np.diag([4., .04, 4., .04]), n)
    sd = np.sqrt(5.) if kind == 'ungm' else (.2 / np.sqrt(1 - .95**2) if kind == 'volatility' else 1.)
    return rng.normal(0, sd, (n, 1))


def propagate(kind, x, rng, t):
    if kind == 'linear':
        return .8 * x + .5 * rng.normal(size=x.shape)
    if kind == 'ungm':
        return .5*x + 25*x/(1+x*x) + 8*np.cos(1.2*t) + np.sqrt(10)*rng.normal(size=x.shape)
    if kind == 'volatility':
        return .95*x + .2*rng.normal(size=x.shape)
    if kind == 'lorenz63':
        return l63_mean(x) + np.sqrt(L63_Q)*rng.normal(size=x.shape)
    if kind == 'lorenz96':
        return l96_step(x) + np.sqrt(L96_Q)*rng.normal(size=x.shape)
    out = x.copy()
    out[:, 0] += x[:, 1]; out[:, 2] += x[:, 3]
    accel = .03*rng.normal(size=(len(x), 2))
    out[:, [0, 2]] += .5*accel; out[:, [1, 3]] += accel
    return out


def observe(kind, x, rng):
    if kind == 'linear': return x[0] + .8*rng.normal()
    if kind == 'ungm': return x[0]**2/20 + rng.normal()
    if kind == 'volatility': return np.exp(x[0]/2)*rng.normal()
    if kind == 'lorenz63': return camera(x[None])[0] + np.sqrt(l63_obs_variance())*rng.normal(size=64)
    if kind == 'lorenz96': return x + np.sqrt(L96_R)*rng.normal(size=L96_DIM)
    return np.arctan2(x[2], x[0]) + .05*rng.normal()


def log_observation(kind, x, y):
    if kind == 'linear': residual, variance = y-x[:, 0], .8**2
    elif kind == 'ungm': residual, variance = y-x[:, 0]**2/20, 1.
    elif kind == 'volatility': residual, variance = y, np.exp(np.clip(x[:, 0], -50, 50))
    elif kind in ('lorenz63', 'lorenz96'):
        # Isotropic Gaussian noise: sum the per-dimension log densities.
        variance = l63_obs_variance() if kind == 'lorenz63' else L96_R
        residual = y[None]-(camera(x) if kind == 'lorenz63' else x)
        return -.5*(residual.shape[1]*np.log(2*np.pi*variance) + (residual**2).sum(1)/variance)
    else:
        residual = (y-np.arctan2(x[:, 2], x[:, 0])+np.pi) % (2*np.pi) - np.pi
        variance = .05**2
    return -.5*(np.log(2*np.pi*variance) + residual**2/variance)


def sequence(kind, seed, steps=50):
    rng = np.random.default_rng(seed + 71200)
    x = initial(kind, rng, 1)
    states, observations = [], []
    for t in range(1, steps+1):
        x = propagate(kind, x, rng, t)
        states.append(x[0].copy()); observations.append(observe(kind, x[0], rng))
    mask = np.ones(steps, dtype=bool)
    if kind == 'bearing': mask[20:30] = False
    return np.array(states), np.array(observations), mask


def kalman(ys):
    m, p, ll = 0., 1., 0.
    means, variances = [], []
    for y in ys:
        mp, pp = .8*m, .8**2*p + .5**2
        s = pp + .8**2
        ll += -.5*(np.log(2*np.pi*s)+(y-mp)**2/s)
        k = pp/s
        m, p = mp+k*(y-mp), (1-k)*pp
        means.append(m); variances.append(p)
    return np.array(means), np.array(variances), ll
