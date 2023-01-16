import os
import wandb
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from argparse import ArgumentParser
from typing import List
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from PIL import Image

from utils.chanset import * 

class Qcalc(nn.Module):
    def __init__(
        self,
        ghost_atoms: List[int] = [286, 936, 1707, 67, 1163, 1469], 
        key_atoms: List[int] = [2086, 2145, 2204, 2264, 2328, 2379], 
        alpha: float = 1.80654,
        beta: float = 0.0949,
        stdval: torch.Tensor = torch.tensor(1),
    ):
        super(Qcalc, self).__init__()
        ghost_atoms = torch.as_tensor(ghost_atoms, dtype=torch.long)
        key_atoms = torch.as_tensor(key_atoms, dtype=torch.long)
        alpha = torch.as_tensor(alpha)
        beta = torch.as_tensor(beta)

        self.register_buffer("ghost_atoms", ghost_atoms)
        self.register_buffer("key_atoms", key_atoms)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("stdval", stdval)
        
    def forward(self, xyz):
        """
        xyz - tensor with coords shape (B, 3, N) in angstrom 
              (non-normalized, if stdval param is not passed to constructor)
        """
        xyz *= self.stdval * 0.1 # angstrom to nm conversion

        ghost_coords = xyz[:, :, self.ghost_atoms]
        key_coords = xyz[:, :, self.key_atoms]

        mean_z = ghost_coords[:, 2].mean(dim=1) # B
        key_z = key_coords[:, 2] # (B, N)
        mean_z = mean_z.unsqueeze(1) # (B,1)
        z = (key_z - mean_z)
        
        deltas = torch.sigmoid(self.alpha * (z - self.beta)) # (B, N)
        Qtot = deltas.sum(dim=1) # B 
        return Qtot
