# funnel functions
from functools import partial

import jax.numpy as np
from jax import jit, grad
from jax.numpy import linalg

def distance(r, cell_size):
    diff = r[1:] - r[0]
    diff = diff - np.round(diff / cell_size) * cell_size
    return np.linalg.norm(diff,axis=1)

def switch(r, r0, k,):
    return (1+np.exp(k*(r-r0)))**-1

def coordnum_exp(r, cell_size, r0s, ks, locations, coeffs):
    total = 0
    for i in range(len(r0s)):
        r0 = r0s[i]
        k = ks[i]
        total += coeffs[i] * switch(distance(r[locations[i]:locations[i+1]],cell_size), r0, k).sum()
    return total

def energy(r, cell_size, r0s, ks, locations, coeffs, c_min, k2):
    val = coordnum_exp(r, cell_size, r0s, ks, locations, coeffs)
    F = np.where(val < c_min, val - c_min, 0.0)
    return 0.5 * k2 * F**2

def intermediate_funnel(pos, ids, indexes, cell_size, r0s, ks, locations, coeffs, c_min, k2):
    r = pos[ids[indexes]]
    return energy(r, cell_size, r0s, ks, locations, coeffs, c_min, k2)

def log_funnel(pos, ids, indexes, cell_size, r0s, ks, locations, coeffs):
    r = pos[ids[indexes]]
    val = coordnum_exp(r, cell_size, r0s, ks, locations, coeffs)
    return val

def external_funnel(data, indexes, cell_size, r0s, ks, locations, coeffs, c_min, k2):
    pos = data.positions[:, :3]
    ids = data.indices
    bias = grad(intermediate_funnel)(pos, ids, indexes, cell_size, r0s, ks, locations, coeffs, c_min, k2)
    proj = log_funnel(pos, ids, indexes, cell_size, r0s, ks, locations, coeffs)
    return bias, proj


def get_funnel_force(indexes, cell_size, r0s, ks, locations, coeffs, c_min, k2):
    funnel_force = partial(
        external_funnel,
        indexes=indexes, cell_size=cell_size, r0s=r0s, ks=ks,
        locations=locations, coeffs=coeffs, c_min=c_min, k2=k2,)
    return jit(funnel_force)
