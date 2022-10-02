import os
import wandb

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tensorflow as tf

import utils.chanset as chanset
from utils.positional_qpredictor import PositionalQPredictor

from tqdm import tqdm, trange
from argparse import ArgumentParser

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

from matplotlib import pyplot as plt
from PIL import Image

        
def create_dirs(args):
    root = os.path.join(args.output, args.experiment_name)

    if not os.path.exists(root):
        os.mkdir(root)

    checkpoints_dir = os.path.join(root, 'checkpoints')
    checkpoints_extra_dir = os.path.join(root, 'checkpoints_extra')

    if not os.path.exists(checkpoints_extra_dir):
        os.mkdir(checkpoints_extra_dir)

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
        
    return root, checkpoints_dir, checkpoints_extra_dir

def train_epoch(loss_function, train_loader, model, optimizer, device):
    for batch in train_loader:
        xyz, atoms, qcharges = batch
        xyz = xyz.to(device, torch.float)
        atoms = atoms.to(device, torch.long)
        qcharges = qcharges.to(device, torch.float)

        xyz = xyz.transpose(1,2)
        output = model(xyz, atoms)
        loss = loss_function(output, qcharges)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.verbose: 
            wandb.log({"MSE on train": loss})


def val_epoch(loss_function, val_loader, model, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0

        for batch in val_loader:
            xyz, atoms, qcharges = batch
            xyz = xyz.to(device, torch.float)
            atoms = atoms.to(device, torch.long)
            qcharges = qcharges.to(device, torch.float)

            xyz = xyz.transpose(1,2)
            output = model(xyz, atoms)
            loss = loss_function(output, qcharges)
            total_loss += loss.item()

        total_loss /= len(val_loader)

        if args.verbose:
            wandb.log({"MSE on validation": total_loss})

    return total_loss

def main(args):
    root, checkpoints_dir, checkpoints_extra_dir = create_dirs(args)
    if args.extra:
        checkpoints_dir = checkpoints_extra_dir

    # QDataset(self, frames, qcharge_path=None, topology=None, atoms='protein and not name H*', 
            # skip=1, start=0, stop=-1, calculate_statistics=True, meanval=0, stdval=1, in_memory=False, 
            # transform=False, plane='xy', root='/home/ebam/kacher1/molearn/DATA', exp_name='new_exp', extra=False):

    train_dataset = chanset.QDataset(
                            frames = args.traj_path,
                            qcharge_path = args.qcharge_path,
                            topology = args.topology_path, 
                            atoms = args.atoms, 
                            skip = args.train_skip,
                            start = args.train_start,
                            stop = args.train_stop,
                            transform = args.transform,
                            plane = args.plane,
                            root = args.output,
                            exp_name = args.experiment_name,
                            extra = args.extra,
                            )

    # Validation dataset comprises frames that were not included in train_dataset 
    # but are extracted from the same MD simulations.

    val_dataset = chanset.QDataset(frames = args.traj_path,
                            qcharge_path = args.qcharge_path,
                            topology = args.topology_path, 
                            atoms = args.atoms, 
                            skip = args.val_skip, # skipping frames in order not to cross with training dataset
                            start = args.val_start, # or just start from a different point 
                            stop = args.val_stop, # or finish somewhere else
                            transform = args.transform,
                            plane = args.plane,
                            root = args.output,
                            exp_name = args.experiment_name, 
                            extra = args.extra,
                            )

    #Initialization 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = args.num_workers

    batch_size = args.batchsize 
    epochs = args.epochs
    lr = args.lr


    num_unique_atoms = len(train_dataset.atomname_to_label)

    model = PositionalQPredictor(num_unique_atoms=num_unique_atoms, embedding_dim=16, depth=4, scale=2, channels=32, res_n=2, droprate=None, batch_norm=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_function = nn.MSELoss()


    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                    batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    cfg = args.__dict__

    if args.extra_path:
        model.load_state_dict(torch.load(args.extra_path),strict=False)
    

    if args.verbose:
        wandb.init(project="molearn", entity="jk", name=args.experiment_name)
        wandb.config = cfg

    # Training loop 
    best_loss = torch.inf
    best_loss_name = None
        
    for epoch in trange(epochs, desc="Training..."):
        train_epoch(loss_function, train_loader, model, optimizer, device)
        if epoch%5: 
            loss = val_epoch(loss_function, val_loader, model, device)
            if  loss < best_loss: 
                best_loss = loss
                if best_loss_name is not None: 
                    os.remove(best_loss_name)
                best_loss_name = f'{checkpoints_dir}/epoch_{epoch}_{loss:.5}.pth'
                torch.save(model.state_dict(), best_loss_name) 
        torch.save(model.state_dict(), f'{checkpoints_dir}/last_checkpoint.pth')
            
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--output", "-o", type=str, default="/home/ebam/kacher1/molearn/DATA")
    parser.add_argument("--experiment_name", "-e", type=str, default="Q_eps24_alpha35")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--extra", "-d", action="store_true", default=False)
    parser.add_argument("--extra_path", type=str, default=None)

    parser.add_argument("--traj_path", "-i", type=str, default="/home/ebam/kacher1/Files/Kv1.2.VSD/preprocessed_pdb_traj/eps24_alpha35_together_aligned.pdb")
    parser.add_argument("--topology_path", "-t", type=str, default=None)
    parser.add_argument("--qcharge_path", "-q", type=str, default="/home/ebam/kacher1/Files/Kv1.2.VSD/qcharges/eps24_alpha35_slice.csv")
    parser.add_argument("--atoms", "-a", type=str, default="protein and not name H*")

    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_stop", type=int, default=-1)
    parser.add_argument("--train_skip", "-s", type=int, default=100)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--plane", type=str, default="xy")
    parser.add_argument("--val_start", type=int, default=3)
    parser.add_argument("--val_stop", type=int, default=-1)
    parser.add_argument("--val_skip", "-k", type=int, default=100)

    parser.add_argument("--num_workers", "-w", type=int, default=32)
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)

    args = parser.parse_args()
    main(args)
