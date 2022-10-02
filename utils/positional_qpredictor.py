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

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from PIL import Image

from utils.chanset import * 

class PositionalQPredictor(nn.Module):
    def __init__(self, num_unique_atoms, embedding_dim=16, depth=4, scale=2, channels=32, res_n=2, droprate=None, batch_norm=True):
        super(PositionalQPredictor, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=num_unique_atoms, embedding_dim=embedding_dim)
        self.positional_enconding = PositionalEncoding(embedding_dim)

        layers = []
        layers.append(nn.Conv1d(3+embedding_dim, channels, 4, 2, 1, bias=False))

        if batch_norm:
            layers.append(nn.BatchNorm1d(channels))
            
        if droprate is not None:
            layers.append(nn.Dropout(p=droprate))

        layers.append(nn.ReLU(inplace=True))

        for i in range(depth):
            in_channels = int(channels*scale**i)
            out_channels = int(channels*scale**(i+1))

            layers.append(nn.Conv1d(in_channels, out_channels, 4, 2, 1, bias=False))

            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))

            if droprate is not None:
                layers.append(nn.Dropout(p=droprate))

            layers.append(nn.ReLU(inplace=True))
            for j in range(res_n):
                layers.append(ResidualBlock(out_channels))

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_channels, 1, bias=True))
        layers.append(nn.Flatten(0, 1))
        self.layers = nn.Sequential(*layers)
    

    def forward(self, xyz, atom_names): 
        atom_names = self.embeddings(atom_names)
        atom_names = atom_names.swapaxes(1,2)
        atom_names = self.positional_enconding(atom_names)
        x = torch.cat([xyz, atom_names], dim=1)
        x = self.layers(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # max_len=N (1, N)
        position = torch.arange(max_len).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(1) # (D/2, 1)
        pe = torch.zeros(1, d_model, max_len) # (1, D, N)  
        pe[0, 0::2, :] = torch.sin(position * div_term) # broadcasting -> (1,N)*(D/2, 1) = (D/2, N) 
        pe[0, 1::2, :] = torch.cos(position * div_term) # (D/2, N) 
        self.register_buffer('pe', pe) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim, seq_len] (B,D,N'<N)
        """
        x = x + self.pe[:,:,:x.shape[-1]] # (B,D,N')+(1,D,N')=(B,D,N')
        return self.dropout(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        res_layers = []
        res_layers.append(nn.Conv1d(channels, channels, 3, stride=1, padding=1, bias=False))
        res_layers.append(nn.BatchNorm1d(channels))
        res_layers.append(nn.ReLU(inplace=True)) 
        res_layers.append(nn.Conv1d(channels, channels, 3, stride=1, padding=1, bias=False)) 
        res_layers.append(nn.BatchNorm1d(channels))

        self.conv_block = nn.Sequential(*res_layers)

    def forward(self, x):
        return x + self.conv_block(x)