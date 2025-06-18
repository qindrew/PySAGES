import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Union

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from jax import numpy as np
import numpy as onp
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_model
from ase.io import read
from functools import partial
from tqdm import tqdm
import torch.nn as nn
from pysages.colvars.core import CollectiveVariable


class Committor_NN(nn.Module):
    def __init__(self, mace_path, device='cuda', input_dim=64, h1=32, h2=16, h3=8, output_dim=1, batch_size=5,
                 sig_k=3,):
        super().__init__()
        self.sig_k = sig_k
        mace_model = torch.load(mace_path,map_location=device)
        mace_model.to(device)
        if device == 'cuda':
            mace_model = run_e3nn_to_cueq(mace_model, device=device).to(device)
        
        for param in mace_model.parameters():
            param.requires_grad = False
            
        self.mace_model = mace_model
        self.z_table = utils.AtomicNumberTable([int(z) for z in mace_model.atomic_numbers])
        self.keyspec = data.KeySpecification(info_keys={}, arrays_keys={"charges": "Qs"})
        irreps_out = o3.Irreps(str(mace_model.products[0].linear.irreps_out))
        self.l_max = irreps_out.lmax
        self.num_invariant_features = irreps_out.dim // (irreps_out.lmax + 1) ** 2
        self.num_layers = mace_model.num_interactions
        self.atom_mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, output_dim),
        ).to(device)
        self.batch_size = batch_size
    def sigmoid_like(self, x):
        return torch.sigmoid(self.sig_k * x)
    def forward(self, batch, training=False):
        batchdict = batch.to_dict()
        descriptor = self.mace_model(
            batchdict,
            training=training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )["node_feats"]
        invariant = extract_invariant(
            descriptor,
            num_layers=self.num_layers,
            num_features=self.num_invariant_features,
            l_max=self.l_max,)                                                 # [batchsize * Natoms, input_dim]
        atom_outputs = self.atom_mlp(invariant)                                # [batchsize * Natoms, 1]
        batch_output = atom_outputs.reshape(self.batch_size,-1,1).sum(dim=-2)  # [batchsize, Natoms, 1] -> [batchsize, 1]
        if training:
            return self.sigmoid_like(batch_output).squeeze(), batchdict
        else:
            return self.sigmoid_like(batch_output).squeeze()



def full_function(positions, atomic_numbers, pbc, cell, z_table, mace_model, num_layers, 
                  num_invariant_features, l_max, atom_mlp):

    config = data.utils.Configuration(
        atomic_numbers=atomic_numbers, positions=positions, properties={},
        weight=1.0, property_weights={}, head="Default", config_type="Default",
        pbc=pbc, cell=cell,
    )

    dataset = data.AtomicData.from_config(config, z_table=z_table, cutoff=4.)

    dataloader = torch_geometric.dataloader.DataLoader(
            dataset=[dataset],
            batch_size=1,
            shuffle=True,
            drop_last=False,
        )
    
    batchdict = next(iter(dataloader)).to('cuda').to_dict()
    descriptor = mace_model(
        batchdict,
        training=True,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=False,
        compute_hessian=False,
        compute_edge_forces=False,
        compute_atomic_stresses=False,
    )["node_feats"]
    
    invariant = extract_invariant(
        descriptor,
        num_layers=num_layers,
        num_features=num_invariant_features,
        l_max=l_max,)                                                 # [batchsize * Natoms, input_dim]
    atom_outputs = atom_mlp(invariant)                                # [batchsize * Natoms, 1]
    batch_output = atom_outputs.sum(dim=-2)  # [batchsize, Natoms, 1] -> [batchsize, 1]
    return batch_output.squeeze(), batchdict['positions']
