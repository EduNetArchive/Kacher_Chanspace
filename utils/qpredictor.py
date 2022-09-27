import os
import wandb
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

class QPredictor(nn.Sequential):
    def __init__(self, depth=4, scale=2, channels=32, res_n=2, droprate=None, batch_norm=True):
        layers = []
        layers.append(nn.Conv1d(3, channels, 4, 2, 1, bias=False))

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

        super(QPredictor, self).__init__(*layers)


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