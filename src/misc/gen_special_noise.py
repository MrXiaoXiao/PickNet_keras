import numpy as np
from numpy.fft import irfft, rfftfreq
from numpy import sqrt, newaxis
from numpy.random import normal

def gen_colored_noise(alpha, length, dt = 0.01, fmin=0):
    # calculate freq
    f = rfftfreq(length, dt)
    # scaling factor
    s_scale = f
    fmin = max(fmin, 1./(length*dt))
    cutoff_index   = np.sum(s_scale < fmin)
    if cutoff_index and cutoff_index < len(s_scale):
        s_scale[:cutoff_index] = s_scale[cutoff_index]
    s_scale = s_scale**(-alpha/2.)
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (length % 2)) / 2.
    sigma = 2 * sqrt(np.sum(w**2)) / length
    size = [len(f)]
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)
    if not (length % 2): si[...,-1] = 0
    si[...,0] = 0
    s  = sr + 1J * si
    y = irfft(s, n=length, axis=-1) / sigma
    return y

def gen_cosine_noise(alpha, length, dt = 0.01, fmin=0):
    return