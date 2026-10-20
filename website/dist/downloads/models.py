"""Shared seeded clouds and state-space models, independent of providers."""
import numpy as np
from scipy.special import logsumexp


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
    out = x.copy()
    out[:, 0] += x[:, 1]; out[:, 2] += x[:, 3]
    accel = .03*rng.normal(size=(len(x), 2))
    out[:, [0, 2]] += .5*accel; out[:, [1, 3]] += accel
    return out


def observe(kind, x, rng):
    if kind == 'linear': return x[0] + .8*rng.normal()
    if kind == 'ungm': return x[0]**2/20 + rng.normal()
    if kind == 'volatility': return np.exp(x[0]/2)*rng.normal()
    return np.arctan2(x[2], x[0]) + .05*rng.normal()


def log_observation(kind, x, y):
    if kind == 'linear': residual, variance = y-x[:, 0], .8**2
    elif kind == 'ungm': residual, variance = y-x[:, 0]**2/20, 1.
    elif kind == 'volatility': residual, variance = y, np.exp(np.clip(x[:, 0], -50, 50))
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
