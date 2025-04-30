import pysages
from pysages.colvars.core import CollectiveVariable
import jax.numpy as np

class Coordnum(CollectiveVariable):
    def __init__(self, indices, r0s=1.8, num=9, den=14, cell_size=np.asarray([19.7,19.7,19.7])):
        super().__init__(indices) #first index must be central atom
        self.r0s = r0s
        self.num = num
        self.den = den
        self.cell_size = cell_size
    @property
    def function(self):
        return lambda r: calc_coordination(r, r0=self.r0s, num=self.num, 
                                           den=self.den, cell_size=self.cell_size)

class Coordnum_comb(CollectiveVariable):
    def __init__(self, indices, r0s=None, nums=None, dens=None, locations=None, coeffs=None, cell_size=np.asarray([19.7,19.7,19.7])):
        super().__init__(indices) #first index must be central atom
        self.r0s = r0s
        self.nums = nums
        self.dens = dens
        self.locations = [0] + locations + [len(indices)]
        self.coeffs = coeffs
        self.cell_size = cell_size
    @property
    def function(self):
        return lambda r: coordnum_comb(r, r0s=self.r0s, nums=self.nums, dens=self.dens, locations=self.locations, coeffs=self.coeffs, cell_size=self.cell_size)

class Coordnum_exp(CollectiveVariable):
    def __init__(self, indices, r0s=None, ks=None, locations=None, coeffs=None, cell_size=np.asarray([19.7,19.7,19.7])):
        super().__init__(indices) #first index must be central atom
        self.r0s = r0s
        self.ks=ks
        self.locations = [0] + locations + [len(indices)]
        self.coeffs = coeffs
        self.cell_size = cell_size
    @property
    def function(self):
        return lambda r: coordnum_exp(r, r0s=self.r0s, ks=self.ks, locations=self.locations, coeffs=self.coeffs, cell_size=self.cell_size)

def coordnum_comb(r, r0s=None, nums=None, dens=None, locations=None, coeffs=None, cell_size=None):
    total = 0
    for i in range(len(r0s)):
        r0 = r0s[i]
        num = nums[i]
        den = dens[i]
        print(r[locations[i]:locations[i+1]])
        total += coeffs[i] * calc_coordination(r[locations[i]:locations[i+1]], r0=r0, num=num, den=den, cell_size=cell_size)
    return total

def coordnum_exp(r, r0s=None, ks=None, locations=None, coeffs=None, cell_size=None):
    total = 0
    for i in range(len(r0s)):
        r0 = r0s[i]
        k = ks[i]
        total += coeffs[i] * calc_coordination_exp(r[locations[i]:locations[i+1]], r0=r0, k=k, cell_size=cell_size)
    return total

def distance(r, cell_size=None):
    diff = r[1:] - r[0]
    diff = diff - np.round(diff / cell_size) * cell_size
    return np.linalg.norm(diff,axis=1)

def switch(arr, r0=None, num=None, den=None):
    return ((1-(arr/r0)**num)/(1-(arr/r0)**den))

def calc_coordination(r, r0=None, num=None, den=None, cell_size=None):
    return switch(distance(r,cell_size=cell_size), r0=r0, num=num, den=den).sum()

def calc_coordination_exp(r, r0=None, k=None, cell_size=None):
    return switch_exp(distance(r,cell_size=cell_size), r0=r0, k=k).sum()

def switch_exp(arr, r0=None, k=None,):
    return (1+np.exp(k*(arr-r0)))**-1
