#!/usr/bin/env python3

"""
Example SpectralABF simulation with pysages and lammps.

For a list of possible options for running the script pass `-h` as argument from the
command line, or call `get_args(["-h"])` if the module was loaded interactively.
"""

# %%
import argparse
import sys
import os
import dill as pickle
import importlib
import numpy as onp

import pysages
import jax
from jax import numpy as np
from functools import partial

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from xtb_ase import XTB
from xtb_ase.calculator import XTBProfile

def generate_simulation(log_steps=100):
    masses = read('mace.data',format='lammps-data',atom_style='atomic')
    atoms = read('mace.xyz',format='extxyz')
    atoms.set_masses(masses.get_masses())
    atoms.pbc = False

    atoms.calc = XTB(profile=XTBProfile(["xtb","--sp","--silent","--alpb","water","--etemp",350,"--iterations",350]),method='gfn1-xtb',spinpol=True,)
    box_center = np.asarray([10,10,10])
    
    md = Langevin(atoms, 0.5 * units.fs, temperature_K=600, friction=0.01,)
    
    def write_frame():
        md.atoms.write('out.xyz', append=True, format='extxyz')
    md.attach(write_frame, interval=log_steps)
    md.attach(MDLogger(md, atoms, 'md.log', header=False, stress=False, peratom=True, mode="a"), interval=log_steps)

    def recenter():
        com = md.atoms.get_center_of_mass()
        md.atoms.positions = md.atoms.positions - com + box_center
    md.attach(recenter, interval=log_steps)

    MaxwellBoltzmannDistribution(atoms, temperature_K=600)
    return md
