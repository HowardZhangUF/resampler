"""Adaptive resampling triggers: a degeneracy score of the normalized weights and its threshold rule.

Each trigger maps normalized weights w (summing to 1) to True when the filter should resample.
'ess' is the benchmark default (ESS < 0.5N) and reproduces the published runs exactly.
"""
import numpy as np


def ess2(w):
    """Standard ESS, 1/sum w^2."""
    return 1/np.sum(w*w)


def cv2(w):
    """Squared coefficient of variation of N w; equals N/ESS - 1."""
    return np.mean((len(w)*w-1)**2)


def perplexity(w):
    """Entropy ESS exp(H), with 0 log 0 = 0."""
    p = w[w > 0]
    return np.exp(-np.sum(p*np.log(p)))


def inverse_max(w):
    """1/max w."""
    return 1/np.max(w)


# name: (score, threshold, resample when score < threshold*N if 'below' else score > threshold).
TRIGGERS = {
    'ess': (ess2, .5, 'below'),
    'cv2': (cv2, 1., 'above'),  # CV^2 > 1 <=> ESS < N/2: same rule as 'ess', kept as a consistency check.
    'entropy': (perplexity, .5, 'below'),
    'maxweight': (inverse_max, .5, 'below'),
}
DESCRIPTIONS = {
    'ess': 'E_2 = 1/sum w^2 < 0.5 N',
    'cv2': 'CV^2 = mean((N w - 1)^2) > 1',
    'entropy': 'E_H = exp(-sum w log w) < 0.5 N',
    'maxweight': 'E_inf = 1/max w < 0.5 N',
}


def parse(spec):
    """'entropy' uses the default threshold; 'entropy:0.62' overrides it."""
    name, _, value = spec.partition(':')
    if name not in TRIGGERS: raise ValueError(f'unknown trigger {name!r}')
    return name, float(value) if value else TRIGGERS[name][1]


def should_resample(spec, w):
    name, threshold = parse(spec)
    score, _, side = TRIGGERS[name]
    s = score(w)
    return s < threshold*len(w) if side == 'below' else s > threshold
